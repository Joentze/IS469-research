"""
Unified evaluation harness for all RAG retrieval methods.

Evaluates 9 pipeline combinations:
    Chunking  : fixed | semantic | agentic
    Retrieval : traditional (vector) | bm25 | hyde

Retrieval metrics (computed per query, averaged across the test set):
    hit_rate@k  — 1 if relevant source document appears anywhere in top-k results
    mrr         — 1 / rank of the first result from the relevant source
    precision@k — fraction of top-k results that come from the relevant source
    ndcg@k      — normalised discounted cumulative gain (1 relevant document)

Usage:
    # First generate the test set (once):
    python evals/generate_testset.py

    # Retrieval metrics:
    python evals/evaluate.py

    # Evaluate specific pipelines only:
    python evals/evaluate.py --methods fixed+bm25 semantic+traditional

    # Force rebuild of all indexes:
    python evals/evaluate.py --force-rebuild

    # Change k:
    python evals/evaluate.py --k 10
"""

from __future__ import annotations

import argparse, csv, json, math, os, sys, time
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Ensure retrieval/ is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parents[1]
RETRIEVAL_ROOT = PROJECT_ROOT / "retrieval"
if str(RETRIEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_ROOT))

from dotenv import load_dotenv; load_dotenv()

from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table
from rich import box

# retrieval/ strategy imports
# Don't mind the resolve errors. They're fixed with the above sys.path hack
from chunkers.fixed import FixedChunker
from chunkers.semantic import SemanticChunker
from chunkers.agentic import AgenticChunker
from embeddings.openai import OpenAIEmbedder
from indexes.chroma import ChromaIndex
from indexes.bm25 import BM25Index
from retrievers.chroma import ChromaRetriever
from retrievers.bm25 import BM25Retriever
from query_transformers.hyde import HyDETransformer
from query_transformers.identity import IdentityTransformer
from utils.markdown_loader import load_markdown_documents

console = Console()


# ============================================================================
# PATHS & CONFIGURATION
# ============================================================================

TEXTS_DIR = PROJECT_ROOT / "texts"
EVAL_DIR = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "database"
DB_DIR = PROJECT_ROOT / "database"
RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_TESTSET = EVAL_DIR / "testset.json"

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
DEFAULT_K = 5

FIXED_CHUNK_SIZE = 512
FIXED_CHUNK_OVERLAP = 64

SEMANTIC_MAX_SENTENCES = 8
SEMANTIC_MAX_TOKENS = 512
SEMANTIC_OVERLAP = 2


# ============================================================================
# CHUNK CACHE (JSON serialisation)
# ============================================================================


def _chunks_to_json(chunks: list[Document]) -> str:
    return json.dumps(
        [{"content": c.page_content, "metadata": c.metadata} for c in chunks],
        ensure_ascii=False,
    )


def _chunks_from_json(raw: str) -> list[Document]:
    return [Document(page_content=d["content"], metadata=d["metadata"]) for d in json.loads(raw)]


def save_chunk_cache(chunks: list[Document], name: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{name}.json").write_text(_chunks_to_json(chunks), encoding="utf-8")


def load_chunk_cache(name: str) -> list[Document] | None:
    path = CACHE_DIR / f"{name}.json"
    if path.exists():
        return _chunks_from_json(path.read_text(encoding="utf-8"))
    return None


# ============================================================================
# METRICS
# ============================================================================


def compute_metrics(retrieved_sources: list[str], relevant_source: str, k: int) -> dict[str, float]:
    top_k = retrieved_sources[:k]
    hits = [1 if s == relevant_source else 0 for s in top_k]
    precision_k = sum(hits) / k if k else 0.0
    hit_rate_k = 1.0 if any(hits) else 0.0
    mrr = next((1.0 / r for r, s in enumerate(retrieved_sources, 1) if s == relevant_source), 0.0)
    # nDCG@k with a single relevant document: IDCG = 1/log2(2) = 1.0
    dcg = sum(h / math.log2(r + 1) for r, h in enumerate(hits, start=1))
    return {"hit_rate@k": hit_rate_k, "mrr": mrr, "precision@k": precision_k, "ndcg@k": dcg}


def average_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not all_metrics:
        return {}
    keys = list(all_metrics[0].keys())
    return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}


# ============================================================================
# PIPELINE REGISTRY
# ============================================================================


class PipelineConfig(NamedTuple):
    name: str
    chunking: str   # "fixed" | "semantic" | "agentic"
    retrieval: str  # "traditional" | "bm25" | "hyde"


ALL_PIPELINES: list[PipelineConfig] = [
    PipelineConfig("fixed+traditional",    "fixed",    "traditional"),
    PipelineConfig("fixed+bm25",           "fixed",    "bm25"),
    PipelineConfig("fixed+hyde",           "fixed",    "hyde"),
    PipelineConfig("semantic+traditional", "semantic", "traditional"),
    PipelineConfig("semantic+bm25",        "semantic", "bm25"),
    PipelineConfig("semantic+hyde",        "semantic", "hyde"),
    PipelineConfig("agentic+traditional",  "agentic",  "traditional"),
    PipelineConfig("agentic+bm25",         "agentic",  "bm25"),
    PipelineConfig("agentic+hyde",         "agentic",  "hyde"),
]


# ============================================================================
# INDEX BUILDING
# ============================================================================


def _build_chunks(strategy: str, docs: list[Document], force_rebuild: bool) -> list[Document]:
    cache_name = f"chunks_{strategy}"
    if not force_rebuild:
        cached = load_chunk_cache(cache_name)
        if cached is not None:
            console.print(f"  [dim]Loaded {len(cached)} chunks from cache ({strategy})[/dim]")
            return cached

    t0 = time.time()
    embedder = OpenAIEmbedder(model=EMBEDDING_MODEL)

    if strategy == "fixed":
        chunks = FixedChunker(
            chunk_size=FIXED_CHUNK_SIZE,
            chunk_overlap=FIXED_CHUNK_OVERLAP,
        ).chunk(docs)

    elif strategy == "semantic":
        chunks = SemanticChunker(
            embedder=embedder,
            max_sentences_per_chunk=SEMANTIC_MAX_SENTENCES,
            max_tokens_per_chunk=SEMANTIC_MAX_TOKENS,
            overlap_sentences=SEMANTIC_OVERLAP,
        ).chunk(docs)

    elif strategy == "agentic":
        chunks = AgenticChunker(llm_model=LLM_MODEL).chunk(docs)

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    elapsed = time.time() - t0
    save_chunk_cache(chunks, cache_name)
    console.print(f"  [dim]Built {len(chunks)} chunks in {elapsed:.1f}s (cached)[/dim]")
    return chunks


def build_indexes(
    pipelines: list[PipelineConfig],
    force_rebuild: bool,
) -> tuple[dict[str, list[Document]], dict[str, ChromaIndex], dict[str, BM25Index]]:
    """Build (or load) all chunk lists, ChromaIndexes, and BM25Indexes needed."""
    strategies = {p.chunking for p in pipelines}
    needs_vector = {p.chunking for p in pipelines if p.retrieval in ("traditional", "hyde")}
    needs_bm25 = {p.chunking for p in pipelines if p.retrieval == "bm25"}

    console.print(f"\nLoading [bold]{TEXTS_DIR.name}[/bold] documents...")
    docs = load_markdown_documents(TEXTS_DIR)
    console.print(f"  Loaded [bold]{len(docs)}[/bold] documents")

    chunks_by: dict[str, list[Document]] = {}
    chroma_by: dict[str, ChromaIndex] = {}
    bm25_by: dict[str, BM25Index] = {}
    embedder = OpenAIEmbedder(model=EMBEDDING_MODEL)

    for strategy in sorted(strategies):
        console.print(f"\n[bold]Chunker: {strategy}[/bold]")
        chunks = _build_chunks(strategy, docs, force_rebuild)
        chunks_by[strategy] = chunks

        if strategy in needs_vector:
            idx = ChromaIndex(
                collection_name=f"eval_{strategy}_chunks",
                persist_directory=str(DB_DIR),
                force_rebuild=force_rebuild,
            )
            idx.add_documents(chunks, embeddings=embedder)
            chroma_by[strategy] = idx

        if strategy in needs_bm25:
            idx_bm25 = BM25Index()
            idx_bm25.add_documents(chunks)
            bm25_by[strategy] = idx_bm25

    return chunks_by, chroma_by, bm25_by


# ============================================================================
# EVALUATION LOOP
# ============================================================================


def run_evaluation(
    testset: list[dict],
    pipelines: list[PipelineConfig],
    k: int,
    force_rebuild: bool,
) -> dict[str, dict[str, float]]:

    _, chroma_by, bm25_by = build_indexes(pipelines, force_rebuild)

    console.print(
        f"\nEvaluating [bold]{len(pipelines)}[/bold] pipelines × "
        f"[bold]{len(testset)}[/bold] queries (k={k})"
    )

    results: dict[str, dict[str, float]] = {}

    for pipeline in pipelines:
        console.print(f"\n[bold cyan]{pipeline.name}[/bold cyan]")
        per_query: list[dict[str, float]] = []

        chroma_idx = chroma_by.get(pipeline.chunking)
        bm25_idx = bm25_by.get(pipeline.chunking)

        if pipeline.retrieval == "traditional":
            retriever = ChromaRetriever(chroma_idx, query_transformer=IdentityTransformer())
        elif pipeline.retrieval == "bm25":
            retriever = BM25Retriever(bm25_idx)
        elif pipeline.retrieval == "hyde":
            retriever = ChromaRetriever(chroma_idx, query_transformer=HyDETransformer())
        else:
            raise ValueError(f"Unknown retrieval method: {pipeline.retrieval}")

        for qi, item in enumerate(testset):
            query: str = item["query"]
            relevant_source: str = item["source"]

            try:
                retrieved_docs = retriever.retrieve(query, k=k)
            except Exception as exc:
                console.print(f"  [yellow]Query {qi} failed:[/yellow] {exc}")
                per_query.append({m: 0.0 for m in ["hit_rate@k", "mrr", "precision@k", "ndcg@k"]})
                continue

            retrieved_sources = [r.metadata.get("source", "") for r in retrieved_docs]
            per_query.append(compute_metrics(retrieved_sources, relevant_source, k))

            if (qi + 1) % 10 == 0:
                console.print(f"  {qi + 1}/{len(testset)} queries", end="\r")

        avg = average_metrics(per_query)
        results[pipeline.name] = avg
        console.print(
            f"  hit_rate@{k}=[green]{avg['hit_rate@k']:.3f}[/green]  "
            f"mrr=[green]{avg['mrr']:.3f}[/green]  "
            f"precision@{k}={avg['precision@k']:.3f}  "
            f"ndcg@{k}={avg['ndcg@k']:.3f}"
        )

    return results


# ============================================================================
# OUTPUT
# ============================================================================

RETRIEVAL_METRIC_LABELS = ["hit_rate@k", "mrr", "precision@k", "ndcg@k"]


def print_results_table(results: dict[str, dict[str, float]], k: int) -> None:
    table = Table(
        title=f"Retrieval Metrics  (k={k},  {len(results)} pipelines)",
        box=box.SIMPLE_HEAVY,
        highlight=True,
    )
    table.add_column("Pipeline", style="bold", no_wrap=True)
    for lbl in RETRIEVAL_METRIC_LABELS:
        table.add_column(lbl.replace("@k", f"@{k}"), justify="right")
    for name, scores in sorted(results.items(), key=lambda x: -x[1].get("hit_rate@k", 0)):
        table.add_row(name, *[f"{scores.get(m, 0):.4f}" for m in RETRIEVAL_METRIC_LABELS])
    console.print()
    console.print(table)


def save_results(results: dict[str, dict[str, float]], k: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "evaluation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", *RETRIEVAL_METRIC_LABELS, "k"])
        for name, scores in sorted(results.items()):
            writer.writerow([name, *[f"{scores.get(m, 0):.6f}" for m in RETRIEVAL_METRIC_LABELS], k])
    console.print(f"\n[dim]CSV  → {csv_path}[/dim]")

    json_path = RESULTS_DIR / "evaluation_results.json"
    json_path.write_text(
        json.dumps({"k": k, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"[dim]JSON → {json_path}[/dim]")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval methods on a financial document corpus"
    )
    parser.add_argument("--testset", default=str(DEFAULT_TESTSET), help="Path to testset.json (default: evaluation/testset.json)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k chunks to retrieve (default: 5)")
    parser.add_argument(
        "--methods", nargs="+", metavar="PIPELINE",
        help=f"Run only these pipelines. Available: {[p.name for p in ALL_PIPELINES]}",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild chunk caches and vector stores from scratch")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set. Check your .env file.")

    testset_path = Path(args.testset)
    if not testset_path.exists():
        console.print(
            f"[bold red]Test set not found:[/bold red] {testset_path}\n"
            "Run:  python evals/generate_testset.py  first."
        )
        return

    testset: list[dict] = json.loads(testset_path.read_text(encoding="utf-8"))
    console.print(f"Loaded [bold]{len(testset)}[/bold] test queries from {testset_path}")

    pipelines = ALL_PIPELINES
    if args.methods:
        name_set = set(args.methods)
        pipelines = [p for p in ALL_PIPELINES if p.name in name_set]
        unknown = name_set - {p.name for p in pipelines}
        if unknown:
            console.print(f"[bold red]Unknown pipeline(s):[/bold red] {sorted(unknown)}")
            console.print(f"Available: {[p.name for p in ALL_PIPELINES]}")
            return
        if not pipelines:
            return

    final_results = run_evaluation(testset, pipelines, args.k, args.force_rebuild)
    print_results_table(final_results, args.k)
    save_results(final_results, args.k)


if __name__ == "__main__":
    main()
