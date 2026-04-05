"""Evaluate FinanceBench questions against existing retrieval indexes.

This runner samples rows from the Hugging Face dataset
``PatronusAI/financebench``, writes the sampled answers to disk before any
model calls, then evaluates a 3 x 3 matrix:

- methods: ``bm25``, ``traditional-rag``, ``agentic-rag``
- chunking: ``fixed``, ``semantic``, ``agentic``

Important:
- Reuses only the existing chunk caches and Chroma collections from
  ``evals/evaluate.py``.
- Refuses to regenerate caches or vector stores if they are missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import evaluate as retrieval_eval
import llm_as_judge as judge_utils


EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
TEXTS_DIR = retrieval_eval.TEXTS_DIR

DEFAULT_SAMPLE_SIZE = 50
DEFAULT_SEED = 42
DEFAULT_MAX_WORKERS = 6
DEFAULT_DATASET = "PatronusAI/financebench"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_PREFIX = "financebench"

# Add case-insensitive substrings here to exclude FinanceBench rows whose
# `doc_name` or evidence document names match files you do not have locally.
EXCLUDED_ROW_STRINGS: list[str] = [
    # "apple",
    # "2023_q4",
]

CHUNKINGS = ("fixed", "semantic", "agentic")
METHODS = ("bm25", "traditional-rag", "agentic-rag")


@dataclass(frozen=True)
class JobSpec:
    chunking: str
    method: str

    @property
    def label(self) -> str:
        return f"{self.chunking}+{self.method}"


@dataclass
class ResourceBundle:
    chunks_by: dict[str, list[Document]]
    stores_by: dict[str, Chroma]
    bm25_by: dict[str, retrieval_eval.BM25Index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FinanceBench evaluation over existing retrieval indexes."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of FinanceBench rows to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Hugging Face dataset split to load.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset identifier.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=retrieval_eval.DEFAULT_K,
        help="Top-k retrieval cutoff.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum parallel row-method jobs.",
    )
    parser.add_argument(
        "--judge-model",
        default=judge_utils.DEFAULT_JUDGE_MODEL,
        help="Model used by the LLM judge.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for sample and results files.",
    )
    return parser.parse_args()


def validate_environment() -> None:
    retrieval_eval.load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set. Check your .env file.")


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def tokenize_text(value: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", value.lower()))
    stop_words = {
        "annual",
        "report",
        "quarterly",
        "current",
        "form",
        "corp",
        "section",
        "sections",
        "statement",
        "income",
        "q",
        "k",
    }
    return {token for token in tokens if token not in stop_words}


def build_local_source_catalog() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for path in sorted(TEXTS_DIR.glob("*.md")):
        stem = path.stem
        catalog.append(
            {
                "filename": path.name,
                "normalized": normalize_text(stem),
                "tokens": tokenize_text(stem),
            }
        )
    return catalog


def find_local_source_matches(doc_name: str, catalog: list[dict[str, Any]]) -> list[str]:
    if not doc_name:
        return []

    normalized_doc = normalize_text(doc_name)
    doc_tokens = tokenize_text(doc_name)
    matches: list[str] = []

    for item in catalog:
        if normalized_doc and (
            normalized_doc in item["normalized"] or item["normalized"] in normalized_doc
        ):
            matches.append(item["filename"])
            continue

        overlap = doc_tokens & item["tokens"]
        if len(overlap) >= 3:
            matches.append(item["filename"])

    return sorted(set(matches))


def row_matches_exclusion(doc_name: str, evidence_doc_names: list[str]) -> bool:
    excluded = [item.strip().lower()
                for item in EXCLUDED_ROW_STRINGS if item.strip()]
    if not excluded:
        return False

    haystacks = [doc_name, *evidence_doc_names]
    for value in haystacks:
        lowered = value.lower()
        if any(excluded_text in lowered for excluded_text in excluded):
            return True
    return False


def load_financebench_rows(
    dataset_name: str,
    split: str,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_name, split=split)
    catalog = build_local_source_catalog()

    usable_rows: list[dict[str, Any]] = []
    for row in dataset:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            continue

        subset_label = str(row.get("dataset_subset_label") or "")
        if subset_label and subset_label != "OPEN_SOURCE":
            continue

        evidence = row.get("evidence") or []
        evidence_doc_names = sorted(
            {
                str(item.get("doc_name") or item.get(
                    "evidence_doc_name") or "").strip()
                for item in evidence
                if str(item.get("doc_name") or item.get("evidence_doc_name") or "").strip()
            }
        )

        doc_name = str(row.get("doc_name") or "").strip()
        if row_matches_exclusion(doc_name, evidence_doc_names):
            continue
        local_matches = find_local_source_matches(doc_name, catalog)

        usable_rows.append(
            {
                "financebench_id": str(row.get("financebench_id") or ""),
                "company": str(row.get("company") or ""),
                "doc_name": doc_name,
                "question": question,
                "answer": answer,
                "dataset_subset_label": subset_label,
                "question_type": str(row.get("question_type") or ""),
                "question_reasoning": str(row.get("question_reasoning") or ""),
                "doc_type": str(row.get("doc_type") or ""),
                "doc_period": row.get("doc_period"),
                "doc_link": str(row.get("doc_link") or ""),
                "evidence_doc_names": evidence_doc_names,
                "local_source_matches": local_matches,
                "has_confident_local_source_match": len(local_matches) == 1,
            }
        )

    if len(usable_rows) < sample_size:
        raise ValueError(
            f"Requested {sample_size} rows but only found {len(usable_rows)} usable rows."
        )

    rng = random.Random(seed)
    sampled = rng.sample(usable_rows, sample_size)
    sampled.sort(key=lambda row: row["financebench_id"])
    return sampled


def save_sample(rows: list[dict[str, Any]], prefix: str) -> Path:
    sample_path = EVAL_DIR / f"{prefix}_sample.json"
    payload = {
        "sample_size": len(rows),
        "rows": rows,
    }
    sample_path.write_text(json.dumps(
        payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return sample_path


def open_existing_store(
    strategy: str,
    embeddings: OpenAIEmbeddings,
) -> Chroma:
    collection_name = f"eval_{strategy}_chunks"
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(retrieval_eval.DB_DIR),
    )
    count = store._collection.count()
    if count <= 0:
        raise FileNotFoundError(
            f"Required Chroma collection '{collection_name}' is missing or empty. "
            "Refusing to regenerate indexes."
        )
    return store


def load_existing_resources() -> ResourceBundle:
    chunks_by: dict[str, list[Document]] = {}
    stores_by: dict[str, Chroma] = {}
    bm25_by: dict[str, retrieval_eval.BM25Index] = {}

    embeddings = OpenAIEmbeddings(model=retrieval_eval.EMBEDDING_MODEL)

    for strategy in CHUNKINGS:
        cache_name = f"chunks_{strategy}"
        chunks = retrieval_eval.load_chunk_cache(cache_name)
        if chunks is None:
            raise FileNotFoundError(
                f"Required chunk cache '{cache_name}.json' is missing. "
                "Refusing to regenerate chunks."
            )

        chunks_by[strategy] = chunks
        bm25_by[strategy] = retrieval_eval.BM25Index(chunks)
        stores_by[strategy] = open_existing_store(strategy, embeddings)

    return ResourceBundle(chunks_by=chunks_by, stores_by=stores_by, bm25_by=bm25_by)


def dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[tuple[str, Any, str]] = set()
    unique_docs: list[Document] = []
    for doc in documents:
        key = (
            str(doc.metadata.get("source", "")),
            doc.metadata.get("chunk_index"),
            doc.page_content,
        )
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)
    return unique_docs


def extract_sources(documents: list[Document]) -> list[str]:
    return [str(doc.metadata.get("source", "")) for doc in documents]


def run_bm25_method(
    query: str,
    chunking: str,
    resources: ResourceBundle,
    k: int,
) -> dict[str, Any]:
    documents = retrieval_eval.retrieve_bm25(
        query, resources.bm25_by[chunking], k)
    contexts = [doc.page_content for doc in documents]
    llm = ChatOpenAI(model=retrieval_eval.LLM_MODEL, temperature=0)
    answer = retrieval_eval.generate_answer(query, contexts, llm)
    return {
        "answer": answer,
        "retrieved_contexts": contexts,
        "retrieved_sources": extract_sources(documents),
    }


def run_traditional_method(
    query: str,
    chunking: str,
    resources: ResourceBundle,
    k: int,
) -> dict[str, Any]:
    documents = retrieval_eval.retrieve_traditional(
        query, resources.stores_by[chunking], k)
    contexts = [doc.page_content for doc in documents]
    llm = ChatOpenAI(model=retrieval_eval.LLM_MODEL, temperature=0)
    answer = retrieval_eval.generate_answer(query, contexts, llm)
    return {
        "answer": answer,
        "retrieved_contexts": contexts,
        "retrieved_sources": extract_sources(documents),
    }


def extract_agent_answer(agent_result: dict[str, Any]) -> str:
    messages = agent_result.get("messages", [])
    if not messages:
        return str(agent_result)
    content = getattr(messages[-1], "content", "")
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content)


def build_agent_tools(
    vector_store: Chroma,
    captured_docs: list[Document],
    lock: threading.Lock,
    k: int,
) -> list[Any]:
    @tool
    def search_knowledge_base(search_query: str, top_k: int = k) -> str:
        """Search the shared Chroma collection for relevant financial evidence."""
        results = vector_store.similarity_search_with_score(
            search_query, k=top_k)
        if not results:
            return "No relevant chunks found."

        docs = [doc for doc, _ in results]
        with lock:
            captured_docs.extend(docs)

        lines = ["Top retrieved chunks:"]
        for idx, (doc, score) in enumerate(results, start=1):
            lines.append(f"[{idx}] score={score:.4f}")
            lines.append(f"source={doc.metadata.get('source', 'unknown')}")
            lines.append(doc.page_content[:1200])
            lines.append("-")
        return "\n".join(lines)

    @tool
    def get_chunk_count() -> str:
        """Return how many chunks exist in the current shared Chroma collection."""
        return f"Chunks in vector store: {vector_store._collection.count()}"

    return [search_knowledge_base, get_chunk_count]


def run_agentic_method(
    query: str,
    chunking: str,
    resources: ResourceBundle,
    k: int,
) -> dict[str, Any]:
    vector_store = resources.stores_by[chunking]
    captured_docs: list[Document] = []
    lock = threading.Lock()

    tools = build_agent_tools(vector_store, captured_docs, lock, k)
    llm = ChatOpenAI(model=retrieval_eval.LLM_MODEL, temperature=0)
    system_prompt = (
        "You are a financial-document QA agent. "
        "Always search the knowledge base before answering. "
        "Use only retrieved evidence. Perform calculations carefully. "
        "If the answer is not supported by the retrieved text, reply with "
        "'Not found in context.'"
    )
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    answer = extract_agent_answer(result)

    documents = dedupe_documents(captured_docs)
    if not documents:
        documents = retrieval_eval.retrieve_traditional(query, vector_store, k)

    return {
        "answer": answer,
        "retrieved_contexts": [doc.page_content for doc in documents],
        "retrieved_sources": extract_sources(documents),
    }


def run_method(
    job: JobSpec,
    query: str,
    resources: ResourceBundle,
    k: int,
) -> dict[str, Any]:
    if job.method == "bm25":
        return run_bm25_method(query, job.chunking, resources, k)
    if job.method == "traditional-rag":
        return run_traditional_method(query, job.chunking, resources, k)
    if job.method == "agentic-rag":
        return run_agentic_method(query, job.chunking, resources, k)
    raise ValueError(f"Unsupported method: {job.method}")


def judge_output(
    judge_fn: Any,
    question: str,
    reference_answer: str,
    output: dict[str, Any],
) -> dict[str, Any]:
    metrics = judge_fn(
        {"query": question},
        output,
        {"reference_answer": reference_answer},
    )

    scored: dict[str, Any] = {}
    for item in metrics:
        key = item["key"]
        if "score" in item:
            scored[key] = item["score"]
        else:
            scored[key] = item.get("value")
    return scored


def run_single_job(
    row: dict[str, Any],
    job: JobSpec,
    resources: ResourceBundle,
    judge_fn: Any,
    k: int,
) -> dict[str, Any]:
    started = time.time()
    question = row["question"]
    reference_answer = row["answer"]

    result = run_method(job, question, resources, k)
    judge_scores = judge_output(judge_fn, question, reference_answer, result)
    elapsed = time.time() - started

    return {
        "financebench_id": row["financebench_id"],
        "company": row["company"],
        "doc_name": row["doc_name"],
        "question": question,
        "reference_answer": reference_answer,
        "chunking": job.chunking,
        "method": job.method,
        "label": job.label,
        "answer": result["answer"],
        "retrieved_sources": result["retrieved_sources"],
        "retrieved_contexts": result["retrieved_contexts"],
        "judge_correctness": judge_scores.get("judge_correctness"),
        "judge_groundedness": judge_scores.get("judge_groundedness"),
        "judge_relevance": judge_scores.get("judge_relevance"),
        "judge_pass": judge_scores.get("judge_pass"),
        "judge_reasoning": judge_scores.get("judge_reasoning"),
        "latency_seconds": round(elapsed, 3),
        "local_source_matches": row["local_source_matches"],
        "has_confident_local_source_match": row["has_confident_local_source_match"],
    }


def run_all_jobs(
    rows: list[dict[str, Any]],
    resources: ResourceBundle,
    judge_model: str,
    k: int,
    max_workers: int,
) -> list[dict[str, Any]]:
    judge_fn = judge_utils.build_judge(judge_model)
    jobs = [JobSpec(chunking=chunking, method=method)
            for chunking in CHUNKINGS for method in METHODS]

    results: list[dict[str, Any]] = []
    future_map: dict[Any, tuple[dict[str, Any], JobSpec]] = {}
    total = len(rows) * len(jobs)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for row in rows:
            for job in jobs:
                future = executor.submit(
                    run_single_job, row, job, resources, judge_fn, k)
                future_map[future] = (row, job)
        for future in as_completed(future_map):
            row, job = future_map[future]
            completed += 1
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "financebench_id": row["financebench_id"],
                        "company": row["company"],
                        "doc_name": row["doc_name"],
                        "question": row["question"],
                        "reference_answer": row["answer"],
                        "chunking": job.chunking,
                        "method": job.method,
                        "label": job.label,
                        "answer": None,
                        "retrieved_sources": [],
                        "retrieved_contexts": [],
                        "judge_correctness": None,
                        "judge_groundedness": None,
                        "judge_relevance": None,
                        "judge_pass": None,
                        "judge_reasoning": f"ERROR: {exc}",
                        "latency_seconds": None,
                        "local_source_matches": row["local_source_matches"],
                        "has_confident_local_source_match": row["has_confident_local_source_match"],
                        "error": str(exc),
                    }
                )
            if completed % 10 == 0 or completed == total:
                print(f"Completed {completed}/{total} jobs")

    results.sort(
        key=lambda row: (
            row.get("financebench_id") or "",
            row.get("chunking") or "",
            row.get("method") or "",
        )
    )
    return results


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for row in results:
        chunking = row.get("chunking")
        method = row.get("method")
        if not chunking or not method:
            continue

        key = (chunking, method)
        bucket = grouped.setdefault(
            key,
            {
                "chunking": chunking,
                "method": method,
                "label": f"{chunking}+{method}",
                "rows": 0,
                "judge_correctness": 0.0,
                "judge_groundedness": 0.0,
                "judge_relevance": 0.0,
                "judge_pass": 0.0,
                "latency_seconds": 0.0,
            },
        )

        bucket["rows"] += 1
        for metric in (
            "judge_correctness",
            "judge_groundedness",
            "judge_relevance",
            "judge_pass",
            "latency_seconds",
        ):
            value = row.get(metric)
            if isinstance(value, (int, float)):
                bucket[metric] += float(value)

    summary: list[dict[str, Any]] = []
    for bucket in grouped.values():
        rows = max(bucket["rows"], 1)
        summary.append(
            {
                "chunking": bucket["chunking"],
                "method": bucket["method"],
                "label": bucket["label"],
                "rows": bucket["rows"],
                "avg_judge_correctness": round(bucket["judge_correctness"] / rows, 4),
                "avg_judge_groundedness": round(bucket["judge_groundedness"] / rows, 4),
                "avg_judge_relevance": round(bucket["judge_relevance"] / rows, 4),
                "avg_judge_pass": round(bucket["judge_pass"] / rows, 4),
                "avg_latency_seconds": round(bucket["latency_seconds"] / rows, 4),
            }
        )

    summary.sort(key=lambda item: (-item["avg_judge_pass"], item["label"]))
    return summary


def save_results(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    prefix: str,
) -> tuple[Path, Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_json_path = RESULTS_DIR / f"{prefix}_raw_results.json"
    raw_json_path.write_text(
        json.dumps({"results": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_json_path = RESULTS_DIR / f"{prefix}_summary.json"
    summary_json_path.write_text(
        json.dumps({"summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_csv_path = RESULTS_DIR / f"{prefix}_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "chunking",
                "method",
                "label",
                "rows",
                "avg_judge_correctness",
                "avg_judge_groundedness",
                "avg_judge_relevance",
                "avg_judge_pass",
                "avg_latency_seconds",
            ],
        )
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    return raw_json_path, summary_json_path, summary_csv_path


def print_mapping_summary(rows: list[dict[str, Any]]) -> None:
    confident = sum(
        1 for row in rows if row["has_confident_local_source_match"])
    ambiguous = sum(1 for row in rows if len(row["local_source_matches"]) > 1)
    unmatched = len(rows) - confident - ambiguous
    print(
        "Local source mapping summary: "
        f"{confident} confident, {ambiguous} ambiguous, {unmatched} unmatched"
    )


def main() -> None:
    args = parse_args()
    validate_environment()

    print(f"Loading {args.dataset_name}:{args.split}")
    sampled_rows = load_financebench_rows(
        dataset_name=args.dataset_name,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"Sampled {len(sampled_rows)} FinanceBench rows")
    print_mapping_summary(sampled_rows)

    sample_path = save_sample(sampled_rows, args.output_prefix)
    print(f"Saved sample to {sample_path}")

    print("Loading existing chunk caches and Chroma collections")
    resources = load_existing_resources()

    print(
        "Running "
        f"{len(sampled_rows) * len(CHUNKINGS) * len(METHODS)} jobs "
        f"with max_workers={args.max_workers}"
    )
    raw_results = run_all_jobs(
        rows=sampled_rows,
        resources=resources,
        judge_model=args.judge_model,
        k=args.k,
        max_workers=args.max_workers,
    )

    summary = aggregate_results(raw_results)
    raw_path, summary_json_path, summary_csv_path = save_results(
        raw_results,
        summary,
        args.output_prefix,
    )

    print(f"Raw results: {raw_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print("Top configurations:")
    for row in summary[:5]:
        print(
            f"  {row['label']}: "
            f"pass={row['avg_judge_pass']:.4f}, "
            f"correctness={row['avg_judge_correctness']:.4f}, "
            f"groundedness={row['avg_judge_groundedness']:.4f}, "
            f"relevance={row['avg_judge_relevance']:.4f}"
        )


if __name__ == "__main__":
    main()
