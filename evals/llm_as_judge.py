"""Run LLM-as-a-judge evaluation over FinanceBench questions.

Evaluates retrieval methods across chunking strategies + orchestrator:

    Methods  : traditional-rag | bm25 | hyde | agentic-rag
    Chunking : fixed | semantic | agentic
    + orchestrator (multi-agent, standalone)

Questions and reference answers are sourced from ``evals/financebench_sample.json``.
Results (raw JSON + summary CSV/JSON) are saved to ``evals/results/``.

Usage:
    python evals/llm_as_judge.py
    python evals/llm_as_judge.py --max-workers 8 --k 5
    python evals/llm_as_judge.py --methods traditional-rag hyde agentic-rag --chunkings fixed semantic
    python evals/llm_as_judge.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).parents[1]
RETRIEVAL_ROOT = PROJECT_ROOT / "retrieval"
if str(RETRIEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_ROOT))

from indexes.bm25 import BM25Index
import evaluate as retrieval_eval
import eval_utils

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_SAMPLE_PATH = EVAL_DIR / "financebench_sample.json"
DEFAULT_JUDGE_MODEL = "gpt-5-mini"
DEFAULT_MAX_WORKERS = 8
DEFAULT_OUTPUT_PREFIX = "llm_judge"

EMBEDDING_MODEL = retrieval_eval.EMBEDDING_MODEL
EMBEDDING_DIM = retrieval_eval.EMBEDDING_DIM
LLM_MODEL = retrieval_eval.LLM_MODEL
DB_DIR = retrieval_eval.DB_DIR

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
    correctness: int = Field(ge=0, le=1, description="1 if answer matches reference, else 0.")
    groundedness: int = Field(ge=0, le=1, description="1 if answer is supported by context, else 0.")
    relevance: int = Field(ge=0, le=1, description="1 if answer addresses the question, else 0.")
    reasoning: str = Field(description="Short explanation justifying the scores.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge evaluation over FinanceBench questions.")
    parser.add_argument("--sample-path", default=str(DEFAULT_SAMPLE_PATH))
    parser.add_argument("--k", type=int, default=retrieval_eval.DEFAULT_K)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Available: {list(RETRIEVAL_METHODS)} + orchestrator")
    parser.add_argument("--chunkings", nargs="+", default=None,
                        help=f"Available: {list(CHUNKINGS)}")
    parser.add_argument("--no-orchestrator", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment & data loading
# ---------------------------------------------------------------------------

def validate_environment() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")


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
        return [r for r in json.loads(raw_path.read_text(encoding="utf-8")).get("results", [])
                if r.get("answer") is not None]
    except Exception:
        return []


def build_completed_keys(results: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {(r.get("financebench_id", ""), r.get("chunking", ""), r.get("method", ""))
            for r in results if r.get("answer") is not None}


# ---------------------------------------------------------------------------
# Job specification & resource bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JobSpec:
    chunking: str
    method: str

    @property
    def label(self) -> str:
        return "orchestrator+multi-agent" if self.method == "orchestrator" else f"{self.chunking}+{self.method}"


@dataclass
class ResourceBundle:
    stores_by: dict[str, Chroma]
    bm25_by: dict[str, BM25Index]
    llm: ChatOpenAI | None = field(default=None)
    embeddings: OpenAIEmbeddings | None = field(default=None)


def _open_existing_store(strategy: str, embeddings: OpenAIEmbeddings) -> Chroma:
    collection_name = f"eval_{strategy}_chunks"
    store = Chroma(collection_name=collection_name, embedding_function=embeddings,
                   persist_directory=str(DB_DIR))
    if store._collection.count() <= 0:
        raise FileNotFoundError(f"Chroma collection '{collection_name}' is empty. Run evaluate.py first.")
    return store


def load_or_build_resources(
    chunkings: tuple[str, ...],
    methods: tuple[str, ...],
    force_rebuild: bool,
) -> ResourceBundle:
    needs_vector = any(m in ("traditional-rag", "hyde", "agentic-rag") for m in methods)
    needs_bm25 = "bm25" in methods

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    stores_by: dict[str, Chroma] = {}
    bm25_by: dict[str, BM25Index] = {}

    for strategy in chunkings:
        cache_name = f"chunks_{strategy}"
        chunks = retrieval_eval.load_chunk_cache(cache_name)
        if chunks is None:
            if force_rebuild:
                pipeline_configs = []
                if needs_vector:
                    pipeline_configs.append(
                        retrieval_eval.PipelineConfig(f"{strategy}+traditional", strategy, "traditional"))
                if needs_bm25:
                    pipeline_configs.append(
                        retrieval_eval.PipelineConfig(f"{strategy}+bm25", strategy, "bm25"))
                retrieval_eval.build_indexes(pipeline_configs, force_rebuild=True)
                chunks = retrieval_eval.load_chunk_cache(cache_name)
            if chunks is None:
                raise FileNotFoundError(f"Chunk cache '{cache_name}.json' not found. Run evaluate.py first.")

        if needs_bm25:
            idx_bm25 = BM25Index()
            idx_bm25.add_documents(chunks)
            bm25_by[strategy] = idx_bm25
        if needs_vector:
            stores_by[strategy] = _open_existing_store(strategy, embeddings)

    return ResourceBundle(stores_by=stores_by, bm25_by=bm25_by, llm=llm, embeddings=embeddings)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def format_contexts(contexts: list[str], limit: int = 3, chars: int = 1200) -> str:
    trimmed = [f"[Context {i}]\n{eval_utils.sanitize(text)[:chars]}"
               for i, text in enumerate(contexts[:limit], start=1)]
    return "\n\n".join(trimmed) if trimmed else "[No retrieved context]"


def build_judge(judge_model: str):
    def llm_judge(inputs: dict[str, Any], outputs: dict[str, Any],
                  reference_outputs: dict[str, Any]) -> list[dict[str, Any]]:
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
        response = _get_judge_client().beta.chat.completions.parse(
            model=judge_model,
            messages=[{"role": "system", "content": instructions}, {"role": "user", "content": message}],
            response_format=JudgeResponse,
        )
        parsed = response.choices[0].message.parsed
        overall = int(parsed.correctness == 1 and parsed.groundedness == 1 and parsed.relevance == 1)
        return [
            {"key": "judge_correctness", "score": parsed.correctness},
            {"key": "judge_groundedness", "score": parsed.groundedness},
            {"key": "judge_relevance", "score": parsed.relevance},
            {"key": "judge_pass", "score": overall},
            {"key": "judge_reasoning", "value": parsed.reasoning},
        ]
    return llm_judge


def judge_output(judge_fn: Any, question: str, reference_answer: str, output: dict[str, Any]) -> dict[str, Any]:
    metrics = judge_fn({"query": question}, output, {"reference_answer": reference_answer})
    return {item["key"]: item.get("score") if "score" in item else item.get("value") for item in metrics}


# ---------------------------------------------------------------------------
# Single job runner
# ---------------------------------------------------------------------------

def _build_result_row(row: dict[str, Any], job: JobSpec, result: dict[str, Any] | None,
                      judge_scores: dict[str, Any] | None, elapsed: float,
                      error: str | None = None) -> dict[str, Any]:
    base = {
        "financebench_id": row.get("financebench_id", ""), "company": row.get("company", ""),
        "doc_name": row.get("doc_name", ""), "question": row["question"],
        "reference_answer": row["answer"], "chunking": job.chunking,
        "method": job.method, "label": job.label, "latency_seconds": round(elapsed, 3),
    }
    if error is not None:
        base.update({"answer": None, "retrieved_sources": [], "retrieved_contexts": [],
                     "judge_correctness": None, "judge_groundedness": None,
                     "judge_relevance": None, "judge_pass": None,
                     "judge_reasoning": f"ERROR: {error}", "error": error})
    else:
        base.update({
            "answer": result["answer"], "retrieved_sources": result["retrieved_sources"],
            "retrieved_contexts": result["retrieved_contexts"],
            "judge_correctness": judge_scores.get("judge_correctness"),
            "judge_groundedness": judge_scores.get("judge_groundedness"),
            "judge_relevance": judge_scores.get("judge_relevance"),
            "judge_pass": judge_scores.get("judge_pass"),
            "judge_reasoning": judge_scores.get("judge_reasoning"),
        })
    return base


def run_method(job: JobSpec, query: str, resources: ResourceBundle, k: int) -> dict[str, Any]:
    if job.method == "traditional-rag":
        return eval_utils.run_traditional_rag(query, resources.stores_by[job.chunking], k)
    if job.method == "bm25":
        return eval_utils.run_bm25(query, resources.bm25_by[job.chunking], k)
    if job.method == "hyde":
        return eval_utils.run_hyde(query, resources.stores_by[job.chunking], k)
    if job.method == "agentic-rag":
        return eval_utils.run_agentic_rag(query, resources.stores_by[job.chunking], k, resources.llm)
    if job.method == "orchestrator":
        sys.path.insert(0, str(PROJECT_ROOT / "agentic"))
        import orchestrator
        answer = orchestrator.run(query)
        return {"answer": answer, "retrieved_contexts": [], "retrieved_sources": []}
    raise ValueError(f"Unknown method: {job.method}")


def run_single_job(row: dict[str, Any], job: JobSpec, resources: ResourceBundle,
                   judge_fn: Any, k: int) -> dict[str, Any]:
    started = time.time()
    try:
        result = run_method(job, row["question"], resources, k)
        scores = judge_output(judge_fn, row["question"], row["answer"], result)
        return _build_result_row(row, job, result, scores, time.time() - started)
    except Exception as exc:
        return _build_result_row(row, job, None, None, time.time() - started, error=str(exc))


# ---------------------------------------------------------------------------
# Incremental save
# ---------------------------------------------------------------------------

_save_lock = threading.Lock()


def _incremental_save(results: list[dict[str, Any]], prefix: str) -> None:
    with _save_lock:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{prefix}_raw_results.json"
        path.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestrator runner (sequential)
# ---------------------------------------------------------------------------

def run_orchestrator_jobs(rows: list[dict[str, Any]], judge_fn: Any,
                          completed_keys: set[tuple[str, str, str]],
                          collected: list[dict[str, Any]], prefix: str) -> None:
    job = JobSpec(chunking="orchestrator", method="orchestrator")
    empty = ResourceBundle({}, {})
    for i, row in enumerate(rows):
        if (row.get("financebench_id", ""), "orchestrator", "orchestrator") in completed_keys:
            continue
        result = run_single_job(row, job, empty, judge_fn, k=0)
        collected.append(result)
        if (i + 1) % 5 == 0 or (i + 1) == len(rows):
            print(f"  orchestrator: {i + 1}/{len(rows)} done")
            _incremental_save(collected, prefix)


# ---------------------------------------------------------------------------
# All jobs runner
# ---------------------------------------------------------------------------

def run_all_jobs(rows: list[dict[str, Any]], resources: ResourceBundle, jobs: list[JobSpec],
                 judge_model: str, k: int, max_workers: int, run_orch: bool,
                 prior_results: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    judge_fn = build_judge(judge_model)
    completed_keys = build_completed_keys(prior_results)
    collected: list[dict[str, Any]] = list(prior_results)

    retrieval_jobs = [j for j in jobs if j.method != "orchestrator"]
    pending = [(row, job) for row in rows for job in retrieval_jobs
               if (row.get("financebench_id", ""), job.chunking, job.method) not in completed_keys]

    if pending:
        skipped = len(rows) * len(retrieval_jobs) - len(pending)
        if skipped:
            print(f"\nResuming: {skipped} already done, {len(pending)} remaining")

        completed, total = 0, len(pending)
        print(f"\nRunning {total} retrieval jobs (max_workers={max_workers})...")
        future_map: dict[Any, tuple[dict[str, Any], JobSpec]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for row, job in pending:
                future_map[executor.submit(run_single_job, row, job, resources, judge_fn, k)] = (row, job)
            for future in as_completed(future_map):
                completed += 1
                collected.append(future.result())
                if completed % 10 == 0 or completed == total:
                    print(f"  Completed {completed}/{total} retrieval jobs")
                    _incremental_save(collected, prefix)
        _incremental_save(collected, prefix)

    if run_orch:
        orch_pending = sum(1 for r in rows
                           if (r.get("financebench_id", ""), "orchestrator", "orchestrator") not in completed_keys)
        if orch_pending:
            print(f"\nRunning orchestrator on {orch_pending} questions (sequential)...")
            run_orchestrator_jobs(rows, judge_fn, completed_keys, collected, prefix)
            _incremental_save(collected, prefix)
        else:
            print("\nOrchestrator: all questions already completed")

    collected.sort(key=lambda r: (r.get("financebench_id", ""), r.get("chunking", ""), r.get("method", "")))
    return collected


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(summary: list[dict[str, Any]]) -> None:
    col_w, name_w = 14, 30
    headers = ["correctness", "groundedness", "relevance", "pass", "latency"]
    sep = "=" * (name_w + col_w * len(headers))
    print(f"\n{sep}\nLLM-AS-A-JUDGE SUMMARY\n{sep}")
    print(f"{'Label':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in headers))
    print("-" * len(sep))
    for row in summary:
        print(f"{row['label']:<{name_w}}"
              f"{row['avg_judge_correctness']:>{col_w}.4f}"
              f"{row['avg_judge_groundedness']:>{col_w}.4f}"
              f"{row['avg_judge_relevance']:>{col_w}.4f}"
              f"{row['avg_judge_pass']:>{col_w}.4f}"
              f"{row['avg_latency_seconds']:>{col_w}.1f}s")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    validate_environment()

    rows = load_sample(Path(args.sample_path))
    print(f"Loaded {len(rows)} questions from {Path(args.sample_path).name}")

    chunkings = tuple(args.chunkings) if args.chunkings else CHUNKINGS
    requested_methods = args.methods if args.methods else list(RETRIEVAL_METHODS)
    retrieval_methods = [m for m in requested_methods if m != "orchestrator"]
    run_orch = "orchestrator" in requested_methods or (args.methods is None and not args.no_orchestrator)

    unknown_methods = set(retrieval_methods) - set(RETRIEVAL_METHODS)
    unknown_chunkings = set(chunkings) - set(CHUNKINGS)
    if unknown_methods:
        print(f"Unknown methods: {sorted(unknown_methods)}. Available: {list(RETRIEVAL_METHODS)}")
        return
    if unknown_chunkings:
        print(f"Unknown chunkings: {sorted(unknown_chunkings)}. Available: {list(CHUNKINGS)}")
        return

    prior_results = load_prior_results(args.output_prefix) if args.resume else []
    if prior_results:
        print(f"Resuming: loaded {len(prior_results)} prior results")

    jobs = [JobSpec(chunking=c, method=m) for c in chunkings for m in retrieval_methods]
    n_configs = len(jobs) + (1 if run_orch else 0)
    print(f"Configurations: {n_configs} | Questions: {len(rows)} | Total: {len(rows) * n_configs}")
    print(f"Judge model: {args.judge_model} | Max workers: {args.max_workers}")

    resources = load_or_build_resources(
        chunkings=chunkings, methods=tuple(retrieval_methods), force_rebuild=args.force_rebuild,
    )

    all_results = run_all_jobs(
        rows=rows, resources=resources, jobs=jobs,
        judge_model=args.judge_model, k=args.k,
        max_workers=args.max_workers, run_orch=run_orch,
        prior_results=prior_results, prefix=args.output_prefix,
    )

    summary = eval_utils.aggregate_results(all_results)
    print_summary(summary)

    raw_path, summary_json_path, summary_csv_path = eval_utils.save_judge_results(
        all_results, summary, RESULTS_DIR, args.output_prefix
    )
    print(f"\nRaw results  : {raw_path}")
    print(f"Summary JSON : {summary_json_path}")
    print(f"Summary CSV  : {summary_csv_path}")


if __name__ == "__main__":
    main()
