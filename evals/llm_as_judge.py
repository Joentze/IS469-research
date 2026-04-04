"""Run LLM-as-a-judge evaluation over FinanceBench questions.

Evaluates 5 retrieval methods across 3 chunking strategies + orchestrator:

    Methods  : traditional-rag | bm25 | hyde | agentic-rag
    Chunking : fixed | semantic | agentic
    + orchestrator (multi-agent, standalone)

Total: 13 configurations (4 methods x 3 chunking + 1 orchestrator).

Questions and reference answers are sourced from ``evals/financebench_sample.json``.
Results (raw JSON + summary CSV/JSON) are saved to ``evals/results/``.

Usage:
    python evals/llm_as_judge.py
    python evals/llm_as_judge.py --max-workers 8 --k 5
    python evals/llm_as_judge.py --methods traditional-rag hyde agentic-rag --chunkings fixed semantic
    python evals/llm_as_judge.py --resume          # skip already-completed evaluations
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from pydantic import BaseModel, Field

import evaluate as retrieval_eval

EVAL_DIR = Path(__file__).parent
PROJECT_ROOT = EVAL_DIR.parent
RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_SAMPLE_PATH = EVAL_DIR / "financebench_sample.json"
DEFAULT_JUDGE_MODEL = "gpt-5-mini"
DEFAULT_MAX_WORKERS = 8
DEFAULT_OUTPUT_PREFIX = "llm_judge"

CHUNKINGS = ("fixed", "semantic", "agentic")
RETRIEVAL_METHODS = ("traditional-rag", "bm25", "hyde", "agentic-rag")

_judge_client: OpenAI | None = None


def _get_judge_client() -> OpenAI:
    global _judge_client
    if _judge_client is None:
        _judge_client = OpenAI()
    return _judge_client


# ---------------------------------------------------------------------------
# Structured judge response
# ---------------------------------------------------------------------------

class JudgeResponse(BaseModel):
    """Structured response returned by the judge model."""

    correctness: int = Field(
        ge=0, le=1,
        description="1 if the answer matches the reference answer, else 0.",
    )
    groundedness: int = Field(
        ge=0, le=1,
        description="1 if the answer is supported by the retrieved context, else 0.",
    )
    relevance: int = Field(
        ge=0, le=1,
        description="1 if the answer addresses the user's question, else 0.",
    )
    reasoning: str = Field(
        description="A short explanation justifying the scores.",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-judge evaluation over FinanceBench questions."
    )
    parser.add_argument(
        "--sample-path",
        default=str(DEFAULT_SAMPLE_PATH),
        help="Path to the FinanceBench sample JSON file.",
    )
    parser.add_argument(
        "--k", type=int, default=retrieval_eval.DEFAULT_K,
        help="Top-k retrieval cutoff.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help="Maximum parallel jobs.",
    )
    parser.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL,
        help="Model used by the LLM judge.",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help=f"Retrieval methods to evaluate. Available: {list(RETRIEVAL_METHODS)}. "
        "Omit to run all. Add 'orchestrator' to include orchestrator.",
    )
    parser.add_argument(
        "--chunkings", nargs="+", default=None,
        help=f"Chunking strategies. Available: {list(CHUNKINGS)}. Omit for all.",
    )
    parser.add_argument(
        "--no-orchestrator", action="store_true",
        help="Skip the orchestrator evaluation.",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Rebuild chunk caches and vector stores.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous run: skip already-completed evaluations.",
    )
    parser.add_argument(
        "--output-prefix", default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for output files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment & data loading
# ---------------------------------------------------------------------------

def validate_environment() -> None:
    retrieval_eval.load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set. Check your .env file.")


def load_sample(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rows" in data:
        return data["rows"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected format in {path}")


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_prior_results(prefix: str) -> list[dict[str, Any]]:
    raw_path = RESULTS_DIR / f"{prefix}_raw_results.json"
    if not raw_path.exists():
        return []
    try:
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        results = data.get("results", [])
        return [r for r in results if r.get("answer") is not None]
    except Exception:
        return []


def build_completed_keys(results: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        (r.get("financebench_id", ""), r.get("chunking", ""), r.get("method", ""))
        for r in results
        if r.get("answer") is not None
    }


# ---------------------------------------------------------------------------
# Job specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JobSpec:
    chunking: str
    method: str

    @property
    def label(self) -> str:
        if self.method == "orchestrator":
            return "orchestrator+multi-agent"
        return f"{self.chunking}+{self.method}"


# ---------------------------------------------------------------------------
# Resource bundle (shared indexes + reusable clients)
# ---------------------------------------------------------------------------

@dataclass
class ResourceBundle:
    chunks_by: dict[str, list[Document]]
    stores_by: dict[str, Chroma]
    bm25_by: dict[str, retrieval_eval.BM25Index]
    llm: ChatOpenAI | None = field(default=None)
    embeddings: OpenAIEmbeddings | None = field(default=None)


def load_or_build_resources(
    chunkings: tuple[str, ...],
    methods: tuple[str, ...],
    force_rebuild: bool,
) -> ResourceBundle:
    needs_vector = any(m in ("traditional-rag", "hyde",
                       "agentic-rag") for m in methods)
    needs_bm25 = "bm25" in methods

    pipeline_configs: list[retrieval_eval.PipelineConfig] = []
    for chunking in chunkings:
        if needs_vector:
            pipeline_configs.append(
                retrieval_eval.PipelineConfig(f"{chunking}+traditional", chunking, "traditional"))
        if needs_bm25:
            pipeline_configs.append(
                retrieval_eval.PipelineConfig(f"{chunking}+bm25", chunking, "bm25"))

    embeddings = OpenAIEmbeddings(model=retrieval_eval.EMBEDDING_MODEL)
    llm = ChatOpenAI(model=retrieval_eval.LLM_MODEL, temperature=0)

    if not pipeline_configs:
        return ResourceBundle(
            chunks_by={}, stores_by={}, bm25_by={}, llm=llm, embeddings=embeddings)

    chunks_by, stores_by, bm25_by = retrieval_eval.build_indexes(
        pipeline_configs, embeddings, llm, force_rebuild,
    )
    return ResourceBundle(
        chunks_by=chunks_by, stores_by=stores_by, bm25_by=bm25_by,
        llm=llm, embeddings=embeddings,
    )


# ---------------------------------------------------------------------------
# Retrieval methods
# ---------------------------------------------------------------------------

def run_traditional_rag(
    query: str,
    chunking: str,
    resources: ResourceBundle,
    k: int,
) -> dict[str, Any]:
    store = resources.stores_by[chunking]
    docs = retrieval_eval.retrieve_traditional(query, store, k)
    contexts = [doc.page_content for doc in docs]
    sources = [str(doc.metadata.get("source", "")) for doc in docs]
    answer = retrieval_eval.generate_answer(query, contexts, resources.llm)
    return {"answer": answer, "retrieved_contexts": contexts, "retrieved_sources": sources}


def run_bm25(
    query: str,
    chunking: str,
    resources: ResourceBundle,
    k: int,
) -> dict[str, Any]:
    index = resources.bm25_by[chunking]
    docs = retrieval_eval.retrieve_bm25(query, index, k)
    contexts = [doc.page_content for doc in docs]
    sources = [str(doc.metadata.get("source", "")) for doc in docs]
    answer = retrieval_eval.generate_answer(query, contexts, resources.llm)
    return {"answer": answer, "retrieved_contexts": contexts, "retrieved_sources": sources}


def run_hyde(
    query: str,
    chunking: str,
    resources: ResourceBundle,
    k: int,
) -> dict[str, Any]:
    store = resources.stores_by[chunking]
    docs = retrieval_eval.retrieve_hyde(
        query, store, resources.embeddings, resources.llm, k)
    contexts = [doc.page_content for doc in docs]
    sources = [str(doc.metadata.get("source", "")) for doc in docs]
    answer = retrieval_eval.generate_answer(query, contexts, resources.llm)
    return {"answer": answer, "retrieved_contexts": contexts, "retrieved_sources": sources}


def _extract_agent_answer(agent_result: dict[str, Any]) -> str:
    messages = agent_result.get("messages", [])
    if not messages:
        return str(agent_result)
    content = getattr(messages[-1], "content", "")
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content)


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[tuple[str, Any, str]] = set()
    unique: list[Document] = []
    for doc in documents:
        key = (
            str(doc.metadata.get("source", "")),
            doc.metadata.get("chunk_index"),
            doc.page_content,
        )
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def _build_agent_tools(
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


def run_agentic_rag(
    query: str,
    chunking: str,
    resources: ResourceBundle,
    k: int,
) -> dict[str, Any]:
    vector_store = resources.stores_by[chunking]
    captured_docs: list[Document] = []
    lock = threading.Lock()

    tools = _build_agent_tools(vector_store, captured_docs, lock, k)
    agent = create_agent(
        model=resources.llm,
        tools=tools,
        system_prompt=(
            "You are a financial-document QA agent. "
            "Always search the knowledge base before answering. "
            "Use only retrieved evidence. Perform calculations carefully. "
            "If the answer is not supported by the retrieved text, reply with "
            "'Not found in context.'"
        ),
    )
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    answer = _extract_agent_answer(result)

    documents = _dedupe_documents(captured_docs)
    if not documents:
        documents = retrieval_eval.retrieve_traditional(query, vector_store, k)

    return {
        "answer": answer,
        "retrieved_contexts": [doc.page_content for doc in documents],
        "retrieved_sources": [str(doc.metadata.get("source", "")) for doc in documents],
    }


def run_orchestrator(query: str) -> dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT / "agentic"))
    import orchestrator
    answer = orchestrator.run(query)
    return {"answer": answer, "retrieved_contexts": [], "retrieved_sources": []}


def run_method(
    job: JobSpec,
    query: str,
    resources: ResourceBundle,
    k: int,
) -> dict[str, Any]:
    if job.method == "traditional-rag":
        return run_traditional_rag(query, job.chunking, resources, k)
    if job.method == "bm25":
        return run_bm25(query, job.chunking, resources, k)
    if job.method == "hyde":
        return run_hyde(query, job.chunking, resources, k)
    if job.method == "agentic-rag":
        return run_agentic_rag(query, job.chunking, resources, k)
    if job.method == "orchestrator":
        return run_orchestrator(query)
    raise ValueError(f"Unknown method: {job.method}")


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def format_contexts(contexts: list[str], limit: int = 3, chars: int = 1200) -> str:
    trimmed = []
    for i, text in enumerate(contexts[:limit], start=1):
        snippet = retrieval_eval._sanitize(text)[:chars]
        trimmed.append(f"[Context {i}]\n{snippet}")
    return "\n\n".join(trimmed) if trimmed else "[No retrieved context]"


def build_judge(judge_model: str):
    def llm_judge(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        instructions = (
            "You are grading an answer produced by a financial-document QA system.\n\n"
            "Score each field as either 0 or 1.\n"
            "- correctness: 1 only if the answer matches the reference answer in meaning.\n"
            "- groundedness: 1 only if the answer is supported by the retrieved context.\n"
            "- relevance: 1 only if the answer actually addresses the user's question.\n\n"
            "Be strict. If the answer is vague, unsupported, or materially incomplete, give 0.\n"
            "Return JSON that matches the provided schema."
        )

        message = (
            f"Question: {inputs.get('query', '')}\n\n"
            f"Reference answer: {reference_outputs.get('reference_answer', '')}\n\n"
            f"Model answer: {outputs.get('answer', '')}\n\n"
            f"Retrieved sources: {outputs.get('retrieved_sources', [])}\n\n"
            f"Retrieved context:\n{format_contexts(outputs.get('retrieved_contexts', []))}"
        )

        client = _get_judge_client()
        response = client.beta.chat.completions.parse(
            model=judge_model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": message},
            ],
            response_format=JudgeResponse,
        )
        parsed = response.choices[0].message.parsed
        overall = int(
            parsed.correctness == 1
            and parsed.groundedness == 1
            and parsed.relevance == 1
        )

        return [
            {"key": "judge_correctness", "score": parsed.correctness},
            {"key": "judge_groundedness", "score": parsed.groundedness},
            {"key": "judge_relevance", "score": parsed.relevance},
            {"key": "judge_pass", "score": overall},
            {"key": "judge_reasoning", "value": parsed.reasoning},
        ]

    return llm_judge


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
        scored[key] = item.get(
            "score") if "score" in item else item.get("value")
    return scored


# ---------------------------------------------------------------------------
# Single job runner
# ---------------------------------------------------------------------------

def _build_result_row(
    row: dict[str, Any],
    job: JobSpec,
    result: dict[str, Any] | None,
    judge_scores: dict[str, Any] | None,
    elapsed: float,
    error: str | None = None,
) -> dict[str, Any]:
    base = {
        "financebench_id": row.get("financebench_id", ""),
        "company": row.get("company", ""),
        "doc_name": row.get("doc_name", ""),
        "question": row["question"],
        "reference_answer": row["answer"],
        "chunking": job.chunking,
        "method": job.method,
        "label": job.label,
        "latency_seconds": round(elapsed, 3),
    }
    if error is not None:
        base.update({
            "answer": None, "retrieved_sources": [], "retrieved_contexts": [],
            "judge_correctness": None, "judge_groundedness": None,
            "judge_relevance": None, "judge_pass": None,
            "judge_reasoning": f"ERROR: {error}", "error": error,
        })
    else:
        base.update({
            "answer": result["answer"],
            "retrieved_sources": result["retrieved_sources"],
            "retrieved_contexts": result["retrieved_contexts"],
            "judge_correctness": judge_scores.get("judge_correctness"),
            "judge_groundedness": judge_scores.get("judge_groundedness"),
            "judge_relevance": judge_scores.get("judge_relevance"),
            "judge_pass": judge_scores.get("judge_pass"),
            "judge_reasoning": judge_scores.get("judge_reasoning"),
        })
    return base


def run_single_job(
    row: dict[str, Any],
    job: JobSpec,
    resources: ResourceBundle,
    judge_fn: Any,
    k: int,
) -> dict[str, Any]:
    started = time.time()
    try:
        result = run_method(job, row["question"], resources, k)
        judge_scores = judge_output(
            judge_fn, row["question"], row["answer"], result)
        return _build_result_row(row, job, result, judge_scores, time.time() - started)
    except Exception as exc:
        return _build_result_row(
            row, job, None, None, time.time() - started, error=str(exc))


# ---------------------------------------------------------------------------
# Incremental save (crash-safe)
# ---------------------------------------------------------------------------

_save_lock = threading.Lock()


def _incremental_save(results: list[dict[str, Any]], prefix: str) -> None:
    with _save_lock:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{prefix}_raw_results.json"
        path.write_text(
            json.dumps({"results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Orchestrator runner (sequential — agent state is not thread-safe)
# ---------------------------------------------------------------------------

def run_orchestrator_jobs(
    rows: list[dict[str, Any]],
    judge_fn: Any,
    completed_keys: set[tuple[str, str, str]],
    collected: list[dict[str, Any]],
    prefix: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    job = JobSpec(chunking="orchestrator", method="orchestrator")
    empty_bundle = ResourceBundle({}, {}, {})
    total = len(rows)

    for i, row in enumerate(rows):
        key = (row.get("financebench_id", ""), "orchestrator", "orchestrator")
        if key in completed_keys:
            continue

        result = run_single_job(row, job, empty_bundle, judge_fn, k=0)
        results.append(result)
        collected.append(result)

        if (i + 1) % 5 == 0 or (i + 1) == total:
            print(f"  orchestrator: {i + 1}/{total} questions done")
            _incremental_save(collected, prefix)

    return results


# ---------------------------------------------------------------------------
# All jobs runner
# ---------------------------------------------------------------------------

def run_all_jobs(
    rows: list[dict[str, Any]],
    resources: ResourceBundle,
    jobs: list[JobSpec],
    judge_model: str,
    k: int,
    max_workers: int,
    run_orch: bool,
    prior_results: list[dict[str, Any]],
    prefix: str,
) -> list[dict[str, Any]]:
    judge_fn = build_judge(judge_model)
    completed_keys = build_completed_keys(prior_results)
    collected: list[dict[str, Any]] = list(prior_results)

    retrieval_jobs = [j for j in jobs if j.method != "orchestrator"]

    if retrieval_jobs:
        pending: list[tuple[dict[str, Any], JobSpec]] = []
        for row in rows:
            for job in retrieval_jobs:
                key = (row.get("financebench_id", ""),
                       job.chunking, job.method)
                if key not in completed_keys:
                    pending.append((row, job))

        skipped = len(rows) * len(retrieval_jobs) - len(pending)
        if skipped:
            print(
                f"\nResuming: {skipped} already done, {len(pending)} remaining")

        if pending:
            completed = 0
            total = len(pending)
            future_map: dict[Any, tuple[dict[str, Any], JobSpec]] = {}

            print(
                f"\nRunning {total} retrieval jobs (max_workers={max_workers})...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for row, job in pending:
                    future = executor.submit(
                        run_single_job, row, job, resources, judge_fn, k)
                    future_map[future] = (row, job)

                for future in as_completed(future_map):
                    completed += 1
                    result = future.result()
                    collected.append(result)

                    if completed % 10 == 0 or completed == total:
                        print(
                            f"  Completed {completed}/{total} retrieval jobs")
                        _incremental_save(collected, prefix)

            _incremental_save(collected, prefix)

    if run_orch:
        orch_pending = sum(
            1 for r in rows
            if (r.get("financebench_id", ""), "orchestrator", "orchestrator") not in completed_keys
        )
        if orch_pending:
            print(
                f"\nRunning orchestrator on {orch_pending} questions (sequential)...")
            run_orchestrator_jobs(
                rows, judge_fn, completed_keys, collected, prefix)
            _incremental_save(collected, prefix)
        else:
            print("\nOrchestrator: all questions already completed")

    collected.sort(key=lambda r: (
        r.get("financebench_id", ""),
        r.get("chunking", ""),
        r.get("method", ""),
    ))
    return collected


# ---------------------------------------------------------------------------
# Aggregation & output
# ---------------------------------------------------------------------------

def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for row in results:
        chunking = row.get("chunking")
        method = row.get("method")
        if not chunking or not method:
            continue

        key = (chunking, method)
        bucket = grouped.setdefault(key, {
            "chunking": chunking,
            "method": method,
            "label": row.get("label", f"{chunking}+{method}"),
            "rows": 0,
            "judge_correctness": 0.0,
            "judge_groundedness": 0.0,
            "judge_relevance": 0.0,
            "judge_pass": 0.0,
            "latency_seconds": 0.0,
        })

        bucket["rows"] += 1
        for metric in ("judge_correctness", "judge_groundedness",
                       "judge_relevance", "judge_pass", "latency_seconds"):
            value = row.get(metric)
            if isinstance(value, (int, float)):
                bucket[metric] += float(value)

    summary: list[dict[str, Any]] = []
    for bucket in grouped.values():
        n = max(bucket["rows"], 1)
        summary.append({
            "chunking": bucket["chunking"],
            "method": bucket["method"],
            "label": bucket["label"],
            "rows": bucket["rows"],
            "avg_judge_correctness": round(bucket["judge_correctness"] / n, 4),
            "avg_judge_groundedness": round(bucket["judge_groundedness"] / n, 4),
            "avg_judge_relevance": round(bucket["judge_relevance"] / n, 4),
            "avg_judge_pass": round(bucket["judge_pass"] / n, 4),
            "avg_latency_seconds": round(bucket["latency_seconds"] / n, 4),
        })

    summary.sort(key=lambda s: (-s["avg_judge_pass"], s["label"]))
    return summary


SUMMARY_FIELDS = [
    "chunking", "method", "label", "rows",
    "avg_judge_correctness", "avg_judge_groundedness",
    "avg_judge_relevance", "avg_judge_pass", "avg_latency_seconds",
]


def save_results(
    raw: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    prefix: str,
) -> tuple[Path, Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RESULTS_DIR / f"{prefix}_raw_results.json"
    raw_path.write_text(
        json.dumps({"results": raw}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_json_path = RESULTS_DIR / f"{prefix}_summary.json"
    summary_json_path.write_text(
        json.dumps({"summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_csv_path = RESULTS_DIR / f"{prefix}_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    return raw_path, summary_json_path, summary_csv_path


def print_summary(summary: list[dict[str, Any]]) -> None:
    col_w = 14
    name_w = 30
    headers = ["correctness", "groundedness", "relevance", "pass", "latency"]
    header_line = f"{'Label':<{name_w}}" + \
        "".join(f"{h:>{col_w}}" for h in headers)
    sep = "=" * (name_w + col_w * len(headers))

    print(f"\n{sep}")
    print("LLM-AS-A-JUDGE SUMMARY")
    print(sep)
    print(header_line)
    print("-" * len(sep))
    for row in summary:
        print(
            f"{row['label']:<{name_w}}"
            f"{row['avg_judge_correctness']:>{col_w}.4f}"
            f"{row['avg_judge_groundedness']:>{col_w}.4f}"
            f"{row['avg_judge_relevance']:>{col_w}.4f}"
            f"{row['avg_judge_pass']:>{col_w}.4f}"
            f"{row['avg_latency_seconds']:>{col_w}.1f}s"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    validate_environment()

    sample_path = Path(args.sample_path)
    rows = load_sample(sample_path)
    print(f"Loaded {len(rows)} questions from {sample_path.name}")

    chunkings = tuple(args.chunkings) if args.chunkings else CHUNKINGS
    requested_methods = args.methods if args.methods else list(
        RETRIEVAL_METHODS)

    retrieval_methods = [m for m in requested_methods if m != "orchestrator"]
    run_orch = ("orchestrator" in requested_methods) or (
        args.methods is None and not args.no_orchestrator
    )

    unknown_methods = set(retrieval_methods) - set(RETRIEVAL_METHODS)
    if unknown_methods:
        print(f"Unknown methods: {sorted(unknown_methods)}")
        print(f"Available: {list(RETRIEVAL_METHODS)} + orchestrator")
        return

    unknown_chunkings = set(chunkings) - set(CHUNKINGS)
    if unknown_chunkings:
        print(f"Unknown chunking strategies: {sorted(unknown_chunkings)}")
        print(f"Available: {list(CHUNKINGS)}")
        return

    prior_results: list[dict[str, Any]] = []
    if args.resume:
        prior_results = load_prior_results(args.output_prefix)
        if prior_results:
            print(f"Resuming: loaded {len(prior_results)} prior results")

    jobs: list[JobSpec] = []
    for chunking in chunkings:
        for method in retrieval_methods:
            jobs.append(JobSpec(chunking=chunking, method=method))

    n_configs = len(jobs) + (1 if run_orch else 0)
    print(
        f"Configurations: {n_configs} ({len(jobs)} retrieval + {'1 orchestrator' if run_orch else 'no orchestrator'})")
    print(f"Questions: {len(rows)}")
    print(f"Total evaluations: {len(rows) * n_configs}")
    print(f"Judge model: {args.judge_model}")
    print(f"Max workers: {args.max_workers}")

    resources = load_or_build_resources(
        chunkings=chunkings,
        methods=tuple(retrieval_methods),
        force_rebuild=args.force_rebuild,
    )

    all_results = run_all_jobs(
        rows=rows,
        resources=resources,
        jobs=jobs,
        judge_model=args.judge_model,
        k=args.k,
        max_workers=args.max_workers,
        run_orch=run_orch,
        prior_results=prior_results,
        prefix=args.output_prefix,
    )

    summary = aggregate_results(all_results)
    print_summary(summary)

    raw_path, summary_json_path, summary_csv_path = save_results(
        all_results, summary, args.output_prefix,
    )
    print(f"\nRaw results  : {raw_path}")
    print(f"Summary JSON : {summary_json_path}")
    print(f"Summary CSV  : {summary_csv_path}")


if __name__ == "__main__":
    main()
