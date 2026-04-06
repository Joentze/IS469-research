"""Evaluate FinanceBench questions against existing retrieval indexes.

Samples rows from ``PatronusAI/financebench``, then evaluates a 3×3 matrix:
    chunking : fixed | semantic | agentic
    methods  : bm25 | traditional-rag | agentic-rag

Reuses chunk caches and Chroma collections built by ``evals/evaluate.py``.
Run evaluate.py first if caches are missing.
"""

from __future__ import annotations

import argparse
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
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

import sys
PROJECT_ROOT = Path(__file__).parents[1]
RETRIEVAL_ROOT = PROJECT_ROOT / "retrieval"
if str(RETRIEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_ROOT))

from indexes.bm25 import BM25Index
import evaluate as retrieval_eval
import llm_as_judge as judge_utils
import eval_utils

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
TEXTS_DIR = retrieval_eval.TEXTS_DIR
DB_DIR = retrieval_eval.DB_DIR
EMBEDDING_MODEL = retrieval_eval.EMBEDDING_MODEL
EMBEDDING_DIM = retrieval_eval.EMBEDDING_DIM
LLM_MODEL = retrieval_eval.LLM_MODEL

DEFAULT_SAMPLE_SIZE = 50
DEFAULT_SEED = 42
DEFAULT_MAX_WORKERS = 6
DEFAULT_DATASET = "PatronusAI/financebench"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_PREFIX = "financebench"

EXCLUDED_ROW_STRINGS: list[str] = []
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
    stores_by: dict[str, Chroma]
    bm25_by: dict[str, BM25Index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FinanceBench evaluation over existing retrieval indexes.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=retrieval_eval.DEFAULT_K)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--judge-model", default=judge_utils.DEFAULT_JUDGE_MODEL)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def validate_environment() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def tokenize_text(value: str) -> set[str]:
    stop_words = {"annual", "report", "quarterly", "current", "form", "corp",
                  "section", "sections", "statement", "income", "q", "k"}
    return {t for t in re.findall(r"[a-z0-9]+", value.lower()) if t not in stop_words}


def build_local_source_catalog() -> list[dict[str, Any]]:
    return [
        {"filename": p.name, "normalized": normalize_text(p.stem), "tokens": tokenize_text(p.stem)}
        for p in sorted(TEXTS_DIR.glob("*.md"))
    ]


def find_local_source_matches(doc_name: str, catalog: list[dict[str, Any]]) -> list[str]:
    if not doc_name:
        return []
    nd = normalize_text(doc_name)
    dt = tokenize_text(doc_name)
    matches = [
        item["filename"] for item in catalog
        if (nd and (nd in item["normalized"] or item["normalized"] in nd))
        or len(dt & item["tokens"]) >= 3
    ]
    return sorted(set(matches))


def row_matches_exclusion(doc_name: str, evidence_doc_names: list[str]) -> bool:
    excluded = [e.strip().lower() for e in EXCLUDED_ROW_STRINGS if e.strip()]
    return any(e in v.lower() for e in excluded for v in [doc_name, *evidence_doc_names]) if excluded else False


def load_financebench_rows(dataset_name: str, split: str, sample_size: int, seed: int) -> list[dict[str, Any]]:
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
        evidence_doc_names = sorted({
            str(item.get("doc_name") or item.get("evidence_doc_name") or "").strip()
            for item in evidence
            if str(item.get("doc_name") or item.get("evidence_doc_name") or "").strip()
        })
        doc_name = str(row.get("doc_name") or "").strip()
        if row_matches_exclusion(doc_name, evidence_doc_names):
            continue

        local_matches = find_local_source_matches(doc_name, catalog)
        usable_rows.append({
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
        })

    if len(usable_rows) < sample_size:
        raise ValueError(f"Requested {sample_size} rows but only {len(usable_rows)} available.")

    rng = random.Random(seed)
    sampled = rng.sample(usable_rows, sample_size)
    sampled.sort(key=lambda r: r["financebench_id"])
    return sampled


def save_sample(rows: list[dict[str, Any]], prefix: str) -> Path:
    sample_path = EVAL_DIR / f"{prefix}_sample.json"
    sample_path.write_text(
        json.dumps({"sample_size": len(rows), "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return sample_path


def open_existing_store(strategy: str, embeddings: OpenAIEmbeddings) -> Chroma:
    collection_name = f"eval_{strategy}_chunks"
    store = Chroma(collection_name=collection_name, embedding_function=embeddings,
                   persist_directory=str(DB_DIR))
    if store._collection.count() <= 0:
        raise FileNotFoundError(f"Chroma collection '{collection_name}' is empty. Run evaluate.py first.")
    return store


def load_existing_resources() -> ResourceBundle:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM)
    stores_by: dict[str, Chroma] = {}
    bm25_by: dict[str, BM25Index] = {}

    for strategy in CHUNKINGS:
        chunks = retrieval_eval.load_chunk_cache(f"chunks_{strategy}")
        if chunks is None:
            raise FileNotFoundError(f"Chunk cache 'chunks_{strategy}.json' missing. Run evaluate.py first.")
        idx_bm25 = BM25Index()
        idx_bm25.add_documents(chunks)
        bm25_by[strategy] = idx_bm25
        stores_by[strategy] = open_existing_store(strategy, embeddings)

    return ResourceBundle(stores_by=stores_by, bm25_by=bm25_by)


def run_method(job: JobSpec, query: str, resources: ResourceBundle, k: int) -> dict[str, Any]:
    if job.method == "bm25":
        return eval_utils.run_bm25(query, resources.bm25_by[job.chunking], k)
    if job.method == "traditional-rag":
        return eval_utils.run_traditional_rag(query, resources.stores_by[job.chunking], k)
    if job.method == "agentic-rag":
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        return eval_utils.run_agentic_rag(query, resources.stores_by[job.chunking], k, llm)
    raise ValueError(f"Unsupported method: {job.method}")


def run_single_job(row: dict[str, Any], job: JobSpec, resources: ResourceBundle,
                   judge_fn: Any, k: int) -> dict[str, Any]:
    started = time.time()
    question, reference_answer = row["question"], row["answer"]
    result = run_method(job, question, resources, k)
    judge_scores = judge_utils.judge_output(judge_fn, question, reference_answer, result)
    elapsed = time.time() - started
    return {
        "financebench_id": row["financebench_id"], "company": row["company"],
        "doc_name": row["doc_name"], "question": question, "reference_answer": reference_answer,
        "chunking": job.chunking, "method": job.method, "label": job.label,
        "answer": result["answer"], "retrieved_sources": result["retrieved_sources"],
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


def run_all_jobs(rows: list[dict[str, Any]], resources: ResourceBundle,
                 judge_model: str, k: int, max_workers: int) -> list[dict[str, Any]]:
    judge_fn = judge_utils.build_judge(judge_model)
    jobs = [JobSpec(chunking=c, method=m) for c in CHUNKINGS for m in METHODS]
    results: list[dict[str, Any]] = []
    future_map: dict[Any, tuple[dict[str, Any], JobSpec]] = {}
    total, completed = len(rows) * len(jobs), 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for row in rows:
            for job in jobs:
                future_map[executor.submit(run_single_job, row, job, resources, judge_fn, k)] = (row, job)
        for future in as_completed(future_map):
            row, job = future_map[future]
            completed += 1
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "financebench_id": row["financebench_id"], "company": row["company"],
                    "doc_name": row["doc_name"], "question": row["question"],
                    "reference_answer": row["answer"], "chunking": job.chunking,
                    "method": job.method, "label": job.label, "answer": None,
                    "retrieved_sources": [], "retrieved_contexts": [],
                    "judge_correctness": None, "judge_groundedness": None,
                    "judge_relevance": None, "judge_pass": None,
                    "judge_reasoning": f"ERROR: {exc}", "latency_seconds": None,
                    "local_source_matches": row["local_source_matches"],
                    "has_confident_local_source_match": row["has_confident_local_source_match"],
                    "error": str(exc),
                })
            if completed % 10 == 0 or completed == total:
                print(f"Completed {completed}/{total} jobs")

    results.sort(key=lambda r: (r.get("financebench_id") or "", r.get("chunking") or "", r.get("method") or ""))
    return results


def main() -> None:
    args = parse_args()
    validate_environment()

    print(f"Loading {args.dataset_name}:{args.split}")
    sampled_rows = load_financebench_rows(args.dataset_name, args.split, args.sample_size, args.seed)
    confident = sum(1 for r in sampled_rows if r["has_confident_local_source_match"])
    print(f"Sampled {len(sampled_rows)} rows | {confident} with confident local source match")

    sample_path = save_sample(sampled_rows, args.output_prefix)
    print(f"Saved sample to {sample_path}")

    print("Loading existing chunk caches and Chroma collections")
    resources = load_existing_resources()

    total_jobs = len(sampled_rows) * len(CHUNKINGS) * len(METHODS)
    print(f"Running {total_jobs} jobs with max_workers={args.max_workers}")
    raw_results = run_all_jobs(sampled_rows, resources, args.judge_model, args.k, args.max_workers)

    summary = eval_utils.aggregate_results(raw_results)
    raw_path, summary_json_path, summary_csv_path = eval_utils.save_judge_results(
        raw_results, summary, RESULTS_DIR, args.output_prefix
    )

    print(f"Raw results: {raw_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print("Top configurations:")
    for row in summary[:5]:
        print(f"  {row['label']}: pass={row['avg_judge_pass']:.4f}, "
              f"correctness={row['avg_judge_correctness']:.4f}, "
              f"groundedness={row['avg_judge_groundedness']:.4f}, "
              f"relevance={row['avg_judge_relevance']:.4f}")


if __name__ == "__main__":
    main()
