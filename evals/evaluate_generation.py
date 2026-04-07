"""
Generation quality evaluation using RAGAS.

Loads a retrieval pipeline (by name), runs queries from the test set through
the full RAG pipeline (retrieval + generation), then scores the outputs with
RAGAS metrics:
    context_precision   — are retrieved chunks relevant to the reference answer?
    context_recall      — does retrieved context cover the reference answer?
    faithfulness        — is the generated answer grounded in the retrieved context?
    answer_relevancy    — is the generated answer relevant to the question?

Usage:
    # Evaluate a specific pipeline:
    python evals/evaluate_generation.py --pipeline fixed+traditional

    # Evaluate multiple pipelines:
    python evals/evaluate_generation.py --pipeline fixed+traditional semantic+hyde

    # Change k (number of retrieved docs):
    python evals/evaluate_generation.py --pipeline fixed+traditional --k 5
"""

from __future__ import annotations

import argparse, csv, json, os, re, sys, warnings
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Ensure retrieval/ is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parents[1]
RETRIEVAL_ROOT = PROJECT_ROOT / "retrieval"
if str(RETRIEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_ROOT))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rich.console import Console
from rich.table import Table
from rich import box

try:
    from openai import OpenAI as _OpenAIClient
    from ragas import EvaluationDataset, SingleTurnSample, evaluate as ragas_evaluate
    import ragas.metrics as _ragas_metrics
    from ragas.llms import llm_factory as ragas_llm_factory
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

# retrieval/ strategy imports
from chunkers.fixed import FixedChunker
from chunkers.semantic import SemanticChunker
from chunkers.agentic import AgenticChunker
from embeddings.openai import OpenAIEmbedder
from indexes.chroma import ChromaIndex
from indexes.bm25 import BM25Index
from retrievers.chroma import ChromaRetriever
from retrievers.bm25 import BM25Retriever
from generators.traditional import TraditionalGenerator
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
EMBEDDING_DIM = 512
LLM_MODEL = "gpt-4o-mini"
DEFAULT_K = 5

FIXED_CHUNK_SIZE = 512
FIXED_CHUNK_OVERLAP = 64

SEMANTIC_MAX_SENTENCES = 8
SEMANTIC_MAX_TOKENS = 512
SEMANTIC_OVERLAP = 2

RAGAS_METRIC_LABELS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
]


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

_PIPELINE_BY_NAME = {p.name: p for p in ALL_PIPELINES}


# ============================================================================
# CHUNK CACHE (shared with evaluate.py)
# ============================================================================


def _chunks_from_json(raw: str) -> list[Document]:
    return [Document(page_content=d["content"], metadata=d["metadata"]) for d in json.loads(raw)]


def load_chunk_cache(name: str) -> list[Document] | None:
    path = CACHE_DIR / f"{name}.json"
    if path.exists():
        return _chunks_from_json(path.read_text(encoding="utf-8"))
    return None


# ============================================================================
# INDEX BUILDING
# ============================================================================


def _build_retriever(pipeline: PipelineConfig, docs: list[Document], embedder: OpenAIEmbedder):
    """Build the retriever for a pipeline, reusing cached chunks/indexes."""
    cache_name = f"chunks_{pipeline.chunking}"
    chunks = load_chunk_cache(cache_name)
    if chunks is None:
        console.print(
            f"[bold red]No chunk cache found for '{pipeline.chunking}'.[/bold red]\n"
            "Run  python evals/evaluate.py  first to build and cache chunks."
        )
        raise FileNotFoundError(f"Chunk cache missing: {cache_name}")

    console.print(f"  [dim]Loaded {len(chunks)} cached chunks ({pipeline.chunking})[/dim]")

    if pipeline.retrieval in ("traditional", "hyde"):
        idx = ChromaIndex(
            collection_name=f"eval_{pipeline.chunking}_chunks",
            persist_directory=str(DB_DIR),
            force_rebuild=False,
        )
        idx.add_documents(chunks, embeddings=embedder)

        qt = HyDETransformer() if pipeline.retrieval == "hyde" else IdentityTransformer()
        return ChromaRetriever(idx, query_transformer=qt)

    elif pipeline.retrieval == "bm25":
        idx_bm25 = BM25Index()
        idx_bm25.add_documents(chunks)
        return BM25Retriever(idx_bm25)

    raise ValueError(f"Unknown retrieval method: {pipeline.retrieval}")


# ============================================================================
# RAGAS EVALUATION
# ============================================================================


def run_ragas(
    samples: list,
    llm_model: str,
    embedding_model: str,
    embedding_dim: int,
) -> dict[str, float]:
    if not RAGAS_AVAILABLE:
        raise ImportError("ragas is not installed. Run:  pip install ragas")

    openai_client = _OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])
    ragas_llm = ragas_llm_factory(llm_model, client=openai_client)
    embeddings = OpenAIEmbeddings(model=embedding_model, dimensions=embedding_dim)

    metric_objs = [
        _ragas_metrics.context_precision,
        _ragas_metrics.context_recall,
        _ragas_metrics.faithfulness,
        _ragas_metrics.answer_relevancy,
    ]
    for m in metric_objs:
        if hasattr(m, "llm"):
            m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    dataset = EvaluationDataset(samples=samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = ragas_evaluate(dataset=dataset, metrics=metric_objs)

    df = result.to_pandas()
    return {col: float(df[col].mean()) for col in RAGAS_METRIC_LABELS if col in df.columns}


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================


def evaluate_pipeline(
    pipeline: PipelineConfig,
    testset: list[dict],
    k: int,
) -> dict[str, float]:
    embedder = OpenAIEmbedder(model=EMBEDDING_MODEL)
    generator = TraditionalGenerator(llm_model=LLM_MODEL)

    console.print(f"\n[bold cyan]{pipeline.name}[/bold cyan]")

    docs = load_markdown_documents(TEXTS_DIR)
    retriever = _build_retriever(pipeline, docs, embedder)

    ragas_samples: list = []

    for qi, item in enumerate(testset):
        query: str = item["query"]

        try:
            retrieved_docs = retriever.retrieve(query, k=k)
        except Exception as exc:
            console.print(f"  [yellow]Query {qi} failed:[/yellow] {exc}")
            ragas_samples.append(SingleTurnSample(
                user_input=query,
                retrieved_contexts=[],
                response="",
                reference=item.get("reference_answer", ""),
            ))
            continue

        contexts = [r.page_content for r in retrieved_docs]
        answer = generator.generate(query, retrieved_docs)

        ragas_samples.append(SingleTurnSample(
            user_input=query,
            retrieved_contexts=contexts,
            response=answer,
            reference=item.get("reference_answer", ""),
        ))

        if (qi + 1) % 10 == 0:
            console.print(f"  {qi + 1}/{len(testset)} queries", end="\r")

    console.print(f"\n  Running RAGAS on {len(ragas_samples)} samples...")
    scores = run_ragas(ragas_samples, LLM_MODEL, EMBEDDING_MODEL, EMBEDDING_DIM)
    console.print(
        f"  context_precision=[green]{scores.get('context_precision', 0):.3f}[/green]  "
        f"context_recall=[green]{scores.get('context_recall', 0):.3f}[/green]  "
        f"faithfulness={scores.get('faithfulness', 0):.3f}  "
        f"answer_relevancy={scores.get('answer_relevancy', 0):.3f}"
    )
    return scores


# ============================================================================
# OUTPUT
# ============================================================================


def print_results_table(results: dict[str, dict[str, float]], k: int) -> None:
    table = Table(
        title=f"RAGAS Generation Metrics  (k={k},  {len(results)} pipelines)",
        box=box.SIMPLE_HEAVY,
        highlight=True,
    )
    table.add_column("Pipeline", style="bold", no_wrap=True)
    for lbl in RAGAS_METRIC_LABELS:
        table.add_column(lbl.replace("_", " "), justify="right")
    for name, scores in sorted(results.items(), key=lambda x: -x[1].get("faithfulness", 0)):
        table.add_row(name, *[f"{scores.get(m, 0):.4f}" for m in RAGAS_METRIC_LABELS])
    console.print()
    console.print(table)


def save_results(results: dict[str, dict[str, float]], k: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "generation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", *RAGAS_METRIC_LABELS, "k"])
        for name, scores in sorted(results.items()):
            writer.writerow([name, *[f"{scores.get(m, 0):.6f}" for m in RAGAS_METRIC_LABELS], k])
    console.print(f"\n[dim]CSV  → {csv_path}[/dim]")

    json_path = RESULTS_DIR / "generation_results.json"
    json_path.write_text(
        json.dumps({"k": k, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"[dim]JSON → {json_path}[/dim]")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    if not RAGAS_AVAILABLE:
        console.print("[bold red]ragas is not installed.[/bold red] Run:  pip install ragas")
        return

    parser = argparse.ArgumentParser(
        description="Evaluate RAG generation quality with RAGAS"
    )
    parser.add_argument("--testset", default=str(DEFAULT_TESTSET))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--pipeline", nargs="+", metavar="PIPELINE",
        help=f"Pipeline(s) to evaluate. Available: {[p.name for p in ALL_PIPELINES]}",
        default=[p.name for p in ALL_PIPELINES],
    )
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

    unknown = set(args.pipeline) - set(_PIPELINE_BY_NAME)
    if unknown:
        console.print(f"[bold red]Unknown pipeline(s):[/bold red] {sorted(unknown)}")
        console.print(f"Available: {[p.name for p in ALL_PIPELINES]}")
        return

    results: dict[str, dict[str, float]] = {}
    for name in args.pipeline:
        pipeline = _PIPELINE_BY_NAME[name]
        results[name] = evaluate_pipeline(pipeline, testset, args.k)

    print_results_table(results, args.k)
    save_results(results, args.k)


if __name__ == "__main__":
    main()
