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

RAGAS metrics (requires --ragas flag; generates answers and calls RAGAS evaluate()):
    context_precision   — are retrieved chunks relevant to the reference answer?
    context_recall      — does retrieved context cover the reference answer?
    faithfulness        — is the generated answer grounded in the retrieved context?
    response_relevancy  — is the generated answer relevant to the question?

Chunks are cached in evaluation/cache/ so expensive agentic chunking only runs once.
Vector stores are persisted in evaluation/database/ and reused on subsequent runs.

Usage:
    # First generate the test set (once):
    python evaluation/generate_testset.py

    # Retrieval metrics only (fast):
    python evaluation/evaluate.py

    # Retrieval + RAGAS metrics (slower, more LLM calls):
    python evaluation/evaluate.py --ragas

    # Evaluate specific pipelines only:
    python evaluation/evaluate.py --methods fixed+bm25 semantic+traditional

    # Force rebuild of all indexes:
    python evaluation/evaluate.py --force-rebuild

    # Change k:
    python evaluation/evaluate.py --k 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from pathlib import Path
from typing import NamedTuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

# RAGAS is optional — only imported when --ragas is passed
try:
    from openai import OpenAI as _OpenAIClient
    from ragas import EvaluationDataset, SingleTurnSample, evaluate as ragas_evaluate
    import ragas.metrics as _ragas_metrics
    from ragas.llms import llm_factory as ragas_llm_factory

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


# ============================================================================
# PATHS & CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parents[1]
TEXTS_DIR = PROJECT_ROOT / "texts"
EVAL_DIR = Path(__file__).parent
CACHE_DIR = EVAL_DIR / "cache"
DB_DIR = EVAL_DIR / "database"
RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_TESTSET = EVAL_DIR / "testset.json"

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
DEFAULT_K = 5

# Chunking parameters
FIXED_CHUNK_SIZE = 1000
FIXED_CHUNK_OVERLAP = 200

SEMANTIC_PERCENTILE = 70.0
SEMANTIC_MAX_SENTENCES = 8
SEMANTIC_OVERLAP = 2

AGENTIC_MIN_SENTENCES = 2
AGENTIC_MAX_SENTENCES = 6
AGENTIC_OVERLAP = 1
AGENTIC_WINDOW = 80        # sentences per LLM planning window
AGENTIC_WINDOW_OVERLAP = 10
AGENTIC_MAX_SENTENCE_CHARS = 500


# ============================================================================
# ENVIRONMENT
# ============================================================================


def load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
    if "OPENAI_API_KEY" not in os.environ and "API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]


# ============================================================================
# DOCUMENT LOADING
# ============================================================================


def load_documents() -> list[Document]:
    docs = []
    for fp in sorted(TEXTS_DIR.glob("*.md")):
        content = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if content:
            docs.append(Document(page_content=content, metadata={"source": fp.name}))
    if not docs:
        raise ValueError(f"No .md files found in {TEXTS_DIR}")
    return docs


# ============================================================================
# CHUNKING UTILITIES
# ============================================================================


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def percentile_val(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    ordered = sorted(vals)
    rank = (len(ordered) - 1) * (q / 100)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


# ============================================================================
# CHUNKING STRATEGIES
# ============================================================================


def chunk_fixed(docs: list[Document]) -> list[Document]:
    """Character sliding window — no API calls required."""
    chunks: list[Document] = []
    step = FIXED_CHUNK_SIZE - FIXED_CHUNK_OVERLAP
    for doc in docs:
        text = doc.page_content
        idx = 0
        for start in range(0, len(text), step):
            chunk = text[start : start + FIXED_CHUNK_SIZE].strip()
            if chunk:
                chunks.append(Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk_index": idx, "chunk_method": "fixed"},
                ))
                idx += 1
            if start + FIXED_CHUNK_SIZE >= len(text):
                break
    return chunks


def chunk_semantic(docs: list[Document], embeddings: OpenAIEmbeddings) -> list[Document]:
    """Sentence embedding similarity — requires embedding API calls."""
    chunked: list[Document] = []
    for doc in docs:
        sents = split_sentences(doc.page_content)
        if len(sents) <= 1:
            chunked.append(Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "chunk_index": 0, "chunk_method": "semantic"},
            ))
            continue

        vecs = embeddings.embed_documents(sents)
        sims = [cosine_sim(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
        threshold = percentile_val(sims, SEMANTIC_PERCENTILE)
        breaks = {i for i, s in enumerate(sims) if s <= threshold}

        current: list[str] = []
        chunk_idx = 0
        for i, sent in enumerate(sents):
            current.append(sent)
            if (i in breaks and len(current) >= 2) or len(current) >= SEMANTIC_MAX_SENTENCES:
                chunked.append(Document(
                    page_content=" ".join(current).strip(),
                    metadata={**doc.metadata, "chunk_index": chunk_idx, "chunk_method": "semantic"},
                ))
                chunk_idx += 1
                current = current[-SEMANTIC_OVERLAP:] if SEMANTIC_OVERLAP > 0 else []
        if current:
            chunked.append(Document(
                page_content=" ".join(current).strip(),
                metadata={**doc.metadata, "chunk_index": chunk_idx, "chunk_method": "semantic"},
            ))
    return chunked


def chunk_agentic(docs: list[Document], llm: ChatOpenAI) -> list[Document]:
    """LLM-proposed topic boundaries — requires one LLM call per sentence window."""
    chunked: list[Document] = []
    step = AGENTIC_WINDOW - AGENTIC_WINDOW_OVERLAP

    for doc_num, doc in enumerate(docs):
        sents = split_sentences(doc.page_content)
        if len(sents) <= 2:
            chunked.append(Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "chunk_index": 0, "chunk_method": "agentic"},
            ))
            continue

        break_indices: set[int] = set()
        for start in range(0, len(sents), step):
            window = sents[start : start + AGENTIC_WINDOW]
            numbered = "\n".join(
                f"{i}: {s[:AGENTIC_MAX_SENTENCE_CHARS]}" for i, s in enumerate(window)
            )
            prompt = (
                f"You are chunking a financial document. Identify sentence indices where a "
                f"chunk break should occur so each chunk is {AGENTIC_MIN_SENTENCES}–"
                f"{AGENTIC_MAX_SENTENCES} sentences and covers one coherent topic.\n\n"
                f"Sentences:\n{numbered}\n\n"
                f'Respond with JSON only: {{"break_after": [list of 0-based indices]}}'
            )
            try:
                resp = llm.invoke(prompt)
                content = str(getattr(resp, "content", "")).strip()
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                data = json.loads(content.strip())
                for idx in data.get("break_after", []):
                    if isinstance(idx, int) and 0 <= idx < len(window):
                        break_indices.add(start + idx)
            except Exception:
                pass

        current: list[str] = []
        chunk_idx = 0
        for i, sent in enumerate(sents):
            current.append(sent)
            if (i in break_indices and len(current) >= AGENTIC_MIN_SENTENCES) or len(current) >= AGENTIC_MAX_SENTENCES:
                chunked.append(Document(
                    page_content=" ".join(current).strip(),
                    metadata={**doc.metadata, "chunk_index": chunk_idx, "chunk_method": "agentic"},
                ))
                chunk_idx += 1
                current = current[-AGENTIC_OVERLAP:] if AGENTIC_OVERLAP > 0 else []
        if current:
            chunked.append(Document(
                page_content=" ".join(current).strip(),
                metadata={**doc.metadata, "chunk_index": chunk_idx, "chunk_method": "agentic"},
            ))

        if (doc_num + 1) % 5 == 0:
            print(f"    Agentic chunking: {doc_num + 1}/{len(docs)} docs done")

    return chunked


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
# VECTOR STORE
# ============================================================================


def build_or_load_vector_store(
    collection_name: str,
    chunks: list[Document],
    embeddings: OpenAIEmbeddings,
    force_rebuild: bool = False,
) -> Chroma:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )
    existing = store._collection.count()

    if existing > 0 and not force_rebuild:
        print(f"    Reusing '{collection_name}' ({existing} chunks already indexed)")
        return store

    if force_rebuild and existing > 0:
        print(f"    Rebuilding '{collection_name}'...")
        store.delete_collection()
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(DB_DIR),
        )

    print(f"    Indexing {len(chunks)} chunks into '{collection_name}'...")
    for start in range(0, len(chunks), 5000):
        store.add_documents(chunks[start : start + 5000])
    print(f"    Done: {store._collection.count()} chunks indexed")
    return store


# ============================================================================
# BM25 INDEX
# ============================================================================


class BM25Index:
    """Lightweight wrapper around BM25Okapi built once per chunking strategy."""

    _TOKENIZE = re.compile(r"\b[a-z0-9]+\b")

    def __init__(self, chunks: list[Document]) -> None:
        self.chunks = chunks
        corpus = [self._TOKENIZE.findall(c.page_content.lower()) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, k: int) -> list[Document]:
        tokens = self._TOKENIZE.findall(query.lower())
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top_idx]


# ============================================================================
# RETRIEVAL METHODS
# ============================================================================


def retrieve_traditional(query: str, store: Chroma, k: int) -> list[Document]:
    return store.similarity_search(query, k=k)


def retrieve_bm25(query: str, index: BM25Index, k: int) -> list[Document]:
    return index.retrieve(query, k)


def retrieve_hyde(
    query: str,
    store: Chroma,
    embeddings: OpenAIEmbeddings,
    llm: ChatOpenAI,
    k: int,
) -> list[Document]:
    prompt = (
        "Write a short factual passage (2–3 sentences) that directly answers the "
        f"following question. Do not mention that it is hypothetical.\n\n"
        f"Question: {query}\n\nPassage:"
    )
    resp = llm.invoke(prompt)
    hyp_doc = str(getattr(resp, "content", "")).strip()
    hyp_emb = embeddings.embed_query(hyp_doc)
    return store.similarity_search_by_vector(hyp_emb, k=k)


def _sanitize(text: str) -> str:
    """Remove null bytes and ASCII control characters that break JSON serialisation."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text).strip()


def generate_answer(query: str, contexts: list[str], llm: ChatOpenAI) -> str:
    """Generate an answer grounded in the retrieved context chunks."""
    clean_contexts = [_sanitize(c)[:2000] for c in contexts]  # cap each chunk at 2000 chars
    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(clean_contexts))
    prompt = (
        "Answer the following question using only the provided context. "
        "If the answer is not present in the context, respond with "
        "'Not found in context.'\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    resp = llm.invoke(prompt)
    return str(getattr(resp, "content", "")).strip()


# ============================================================================
# METRICS
# ============================================================================


def compute_metrics(retrieved_sources: list[str], relevant_source: str, k: int) -> dict[str, float]:
    """
    Compute retrieval metrics treating the relevant source document as the ground truth.

    retrieved_sources: ordered list of source filenames from top-k results
    relevant_source:   the ground-truth source filename
    k:                 cutoff
    """
    top_k = retrieved_sources[:k]
    hits = [1 if s == relevant_source else 0 for s in top_k]

    precision_k = sum(hits) / k if k else 0.0
    hit_rate_k = 1.0 if any(hits) else 0.0

    mrr = 0.0
    for rank, s in enumerate(retrieved_sources, start=1):
        if s == relevant_source:
            mrr = 1.0 / rank
            break

    # nDCG@k with a single relevant document: IDCG = 1/log2(2) = 1.0
    dcg = sum(h / math.log2(r + 1) for r, h in enumerate(hits, start=1))
    idcg = 1.0  # ideal: relevant doc at rank 1
    ndcg_k = dcg / idcg

    return {
        "hit_rate@k": hit_rate_k,
        "mrr": mrr,
        "precision@k": precision_k,
        "ndcg@k": ndcg_k,
    }


def average_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not all_metrics:
        return {}
    keys = list(all_metrics[0].keys())
    return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}


# ============================================================================
# RAGAS EVALUATION
# ============================================================================

_RAGAS_METRICS_DEF = [
    ("context_precision", "context_precision"),
    ("context_recall",    "context_recall"),
    ("faithfulness",      "faithfulness"),
    ("answer_relevancy",  "answer_relevancy"),
]

RAGAS_METRIC_LABELS = [col for _, col in _RAGAS_METRICS_DEF]


def run_ragas_evaluation(
    samples: list,          # list[SingleTurnSample] — typed loosely to avoid NameError when RAGAS is absent
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
) -> dict[str, float]:
    """
    Run RAGAS evaluate() on a list of SingleTurnSamples and return per-metric averages.

    Metrics:
      context_precision  — LLMContextPrecisionWithReference
      context_recall     — LLMContextRecall
      faithfulness       — Faithfulness
      response_relevancy — ResponseRelevancy
    """
    if not RAGAS_AVAILABLE:
        raise ImportError(
            "ragas is not installed. Run:  pip install ragas"
        )

    openai_client = _OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])
    ragas_llm = ragas_llm_factory(LLM_MODEL, client=openai_client)

    # Use singleton metric objects (old-style API that evaluate() accepts)
    # LangChain embeddings needed for answer_relevancy's embed_query interface
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
            m.embeddings = embeddings  # LangChain OpenAIEmbeddings has embed_query

    dataset = EvaluationDataset(samples=samples)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = ragas_evaluate(dataset=dataset, metrics=metric_objs)

    df = result.to_pandas()
    scores: dict[str, float] = {}
    for _, col in _RAGAS_METRICS_DEF:
        if col in df.columns:
            scores[f"ragas_{col}"] = float(df[col].mean())
    return scores


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
# EVALUATION LOOP
# ============================================================================


def build_indexes(
    pipelines: list[PipelineConfig],
    embeddings: OpenAIEmbeddings,
    llm: ChatOpenAI,
    force_rebuild: bool,
) -> tuple[dict[str, list[Document]], dict[str, Chroma], dict[str, BM25Index]]:
    """Build (or load) all chunk lists, vector stores, and BM25 indexes needed."""
    strategies = {p.chunking for p in pipelines}
    needs_vector = {p.chunking for p in pipelines if p.retrieval in ("traditional", "hyde")}
    needs_bm25 = {p.chunking for p in pipelines if p.retrieval == "bm25"}

    print("\n[1] Loading source documents...")
    docs = load_documents()
    print(f"    Loaded {len(docs)} documents from {TEXTS_DIR}")

    chunks_by: dict[str, list[Document]] = {}
    stores_by: dict[str, Chroma] = {}
    bm25_by: dict[str, BM25Index] = {}

    for strategy in sorted(strategies):
        print(f"\n[2] Chunking strategy: {strategy}")
        cache_name = f"chunks_{strategy}"

        # --- Load or build chunks -----------------------------------------
        cached = None if force_rebuild else load_chunk_cache(cache_name)
        if cached is not None:
            print(f"    Loaded {len(cached)} chunks from cache")
            chunks = cached
        else:
            t0 = time.time()
            if strategy == "fixed":
                chunks = chunk_fixed(docs)
            elif strategy == "semantic":
                print("    Embedding sentences for semantic chunking...")
                chunks = chunk_semantic(docs, embeddings)
            elif strategy == "agentic":
                print("    Running LLM-based agentic chunking (slow, cached after first run)...")
                chunks = chunk_agentic(docs, llm)
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy}")
            elapsed = time.time() - t0
            save_chunk_cache(chunks, cache_name)
            print(f"    Created {len(chunks)} chunks in {elapsed:.1f}s  (cached to evaluation/cache/)")

        chunks_by[strategy] = chunks

        # --- Vector store (for traditional + hyde) -------------------------
        if strategy in needs_vector:
            collection = f"eval_{strategy}_chunks"
            stores_by[strategy] = build_or_load_vector_store(
                collection, chunks, embeddings, force_rebuild
            )

        # --- BM25 index (built in-memory, fast) ----------------------------
        if strategy in needs_bm25:
            print(f"    Building BM25 index...")
            bm25_by[strategy] = BM25Index(chunks)
            print(f"    BM25 index ready")

    return chunks_by, stores_by, bm25_by


def run_evaluation(
    testset: list[dict],
    pipelines: list[PipelineConfig],
    k: int,
    embeddings: OpenAIEmbeddings,
    llm: ChatOpenAI,
    force_rebuild: bool,
    run_ragas: bool = False,
) -> dict[str, dict[str, float]]:

    _, stores_by, bm25_by = build_indexes(pipelines, embeddings, llm, force_rebuild)

    ragas_label = " + RAGAS" if run_ragas else ""
    print(f"\n[3] Evaluating {len(pipelines)} pipelines × {len(testset)} queries (k={k}{ragas_label})")
    print("-" * 60)

    results: dict[str, dict[str, float]] = {}

    for pipeline in pipelines:
        print(f"\n  {pipeline.name}")
        per_query: list[dict[str, float]] = []
        ragas_samples: list = []

        store = stores_by.get(pipeline.chunking)
        bm25 = bm25_by.get(pipeline.chunking)

        for qi, item in enumerate(testset):
            query: str = item["query"]
            relevant_source: str = item["source"]

            if pipeline.retrieval == "traditional":
                results_docs = retrieve_traditional(query, store, k)
            elif pipeline.retrieval == "bm25":
                results_docs = retrieve_bm25(query, bm25, k)
            elif pipeline.retrieval == "hyde":
                results_docs = retrieve_hyde(query, store, embeddings, llm, k)
            else:
                raise ValueError(f"Unknown retrieval method: {pipeline.retrieval}")

            retrieved_sources = [r.metadata.get("source", "") for r in results_docs]
            per_query.append(compute_metrics(retrieved_sources, relevant_source, k))

            if run_ragas:
                contexts = [r.page_content for r in results_docs]
                answer = generate_answer(query, contexts, llm)
                ragas_samples.append(SingleTurnSample(
                    user_input=query,
                    retrieved_contexts=contexts,
                    response=answer,
                    reference=item.get("reference_answer", ""),
                ))

            if (qi + 1) % 10 == 0:
                print(f"    {qi + 1}/{len(testset)} queries", end="\r", flush=True)

        avg = average_metrics(per_query)

        if run_ragas and ragas_samples:
            print(f"\n    Running RAGAS on {len(ragas_samples)} samples...")
            ragas_scores = run_ragas_evaluation(ragas_samples, llm, embeddings)
            avg.update(ragas_scores)
            print(
                f"    context_precision={ragas_scores.get('ragas_context_precision', 0):.3f}  "
                f"context_recall={ragas_scores.get('ragas_context_recall', 0):.3f}  "
                f"faithfulness={ragas_scores.get('ragas_faithfulness', 0):.3f}  "
                f"answer_relevancy={ragas_scores.get('ragas_answer_relevancy', 0):.3f}"
            )

        results[pipeline.name] = avg
        print(
            f"    hit_rate@{k}={avg['hit_rate@k']:.3f}  "
            f"mrr={avg['mrr']:.3f}  "
            f"precision@{k}={avg['precision@k']:.3f}  "
            f"ndcg@{k}={avg['ndcg@k']:.3f}"
        )

    return results


# ============================================================================
# OUTPUT
# ============================================================================

RETRIEVAL_METRIC_LABELS = ["hit_rate@k", "mrr", "precision@k", "ndcg@k"]
RAGAS_OUTPUT_LABELS = [f"ragas_{c}" for c in RAGAS_METRIC_LABELS]


def _print_section(
    title: str,
    results: dict[str, dict[str, float]],
    metric_keys: list[str],
    header_labels: list[str],
    sort_key: str,
) -> None:
    col_w = 16
    name_w = 28
    total_w = name_w + col_w * len(metric_keys)
    sep = "=" * total_w

    print(f"\n{sep}")
    print(title)
    print(sep)
    print(f"{'Pipeline':<{name_w}}" + "".join(f"{lbl:>{col_w}}" for lbl in header_labels))
    print("-" * total_w)

    for name, scores in sorted(results.items(), key=lambda x: -x[1].get(sort_key, 0)):
        row = f"{name:<{name_w}}" + "".join(
            f"{scores.get(m, 0):.4f}".rjust(col_w) for m in metric_keys
        )
        print(row)

    print(sep)


def print_results_table(results: dict[str, dict[str, float]], k: int) -> None:
    has_ragas = any(
        any(key.startswith("ragas_") for key in scores)
        for scores in results.values()
    )

    _print_section(
        title=f"RETRIEVAL METRICS  (k={k},  {len(results)} pipelines)",
        results=results,
        metric_keys=RETRIEVAL_METRIC_LABELS,
        header_labels=[m.replace("@k", f"@{k}") for m in RETRIEVAL_METRIC_LABELS],
        sort_key="hit_rate@k",
    )

    if has_ragas:
        _print_section(
            title=f"RAGAS METRICS  (k={k},  {len(results)} pipelines)",
            results=results,
            metric_keys=RAGAS_OUTPUT_LABELS,
            header_labels=[c.replace("ragas_", "") for c in RAGAS_OUTPUT_LABELS],
            sort_key="ragas_faithfulness",
        )


def save_results(results: dict[str, dict[str, float]], k: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    has_ragas = any(
        any(key.startswith("ragas_") for key in scores)
        for scores in results.values()
    )
    all_labels = RETRIEVAL_METRIC_LABELS + (RAGAS_OUTPUT_LABELS if has_ragas else [])

    csv_path = RESULTS_DIR / "evaluation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", *all_labels, "k"])
        for name, scores in sorted(results.items()):
            writer.writerow([name, *[f"{scores.get(m, 0):.6f}" for m in all_labels], k])
    print(f"\nCSV  → {csv_path}")

    json_path = RESULTS_DIR / "evaluation_results.json"
    json_path.write_text(
        json.dumps({"k": k, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"JSON → {json_path}")


# ============================================================================
# RAGAS-ONLY MODE
# ============================================================================


def run_ragas_only(
    testset: list[dict],
    pipelines: list[PipelineConfig],
    k: int,
    embeddings: OpenAIEmbeddings,
    llm: ChatOpenAI,
    prior_results: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Re-run retrieval (to get contexts) then score with RAGAS only.
    Merges RAGAS scores into the prior retrieval results.
    """
    _, stores_by, bm25_by = build_indexes(pipelines, embeddings, llm, force_rebuild=False)

    print(f"\n[RAGAS-ONLY] Scoring {len(pipelines)} pipelines × {len(testset)} queries")
    print("-" * 60)

    results = {name: dict(scores) for name, scores in prior_results.items()}

    for pipeline in pipelines:
        print(f"\n  {pipeline.name}")
        ragas_samples: list = []

        store = stores_by.get(pipeline.chunking)
        bm25 = bm25_by.get(pipeline.chunking)

        for qi, item in enumerate(testset):
            query: str = item["query"]

            if pipeline.retrieval == "traditional":
                results_docs = retrieve_traditional(query, store, k)
            elif pipeline.retrieval == "bm25":
                results_docs = retrieve_bm25(query, bm25, k)
            elif pipeline.retrieval == "hyde":
                results_docs = retrieve_hyde(query, store, embeddings, llm, k)
            else:
                raise ValueError(f"Unknown retrieval method: {pipeline.retrieval}")

            contexts = [r.page_content for r in results_docs]
            answer = generate_answer(query, contexts, llm)
            ragas_samples.append(SingleTurnSample(
                user_input=query,
                retrieved_contexts=contexts,
                response=answer,
                reference=item.get("reference_answer", ""),
            ))

            if (qi + 1) % 10 == 0:
                print(f"    {qi + 1}/{len(testset)} queries", end="\r", flush=True)

        print(f"\n    Running RAGAS on {len(ragas_samples)} samples...")
        ragas_scores = run_ragas_evaluation(ragas_samples, llm, embeddings)

        if pipeline.name not in results:
            results[pipeline.name] = {}
        results[pipeline.name].update(ragas_scores)

        print(
            f"    context_precision={ragas_scores.get('ragas_context_precision', 0):.3f}  "
            f"context_recall={ragas_scores.get('ragas_context_recall', 0):.3f}  "
            f"faithfulness={ragas_scores.get('ragas_faithfulness', 0):.3f}  "
            f"answer_relevancy={ragas_scores.get('ragas_answer_relevancy', 0):.3f}"
        )

    return results


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval methods on a financial document corpus"
    )
    parser.add_argument(
        "--testset", default=str(DEFAULT_TESTSET),
        help="Path to testset.json (default: evaluation/testset.json)"
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help="Top-k chunks to retrieve (default: 5)"
    )
    parser.add_argument(
        "--methods", nargs="+", metavar="PIPELINE",
        help=(
            "Run only these pipelines, e.g. --methods fixed+bm25 semantic+traditional. "
            f"Available: {[p.name for p in ALL_PIPELINES]}"
        ),
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Rebuild chunk caches and vector stores from scratch"
    )
    parser.add_argument(
        "--ragas", action="store_true",
        help=(
            "Also run RAGAS metrics (context_precision, context_recall, "
            "faithfulness, answer_relevancy). Requires ragas to be installed."
        ),
    )
    parser.add_argument(
        "--ragas-only", action="store_true",
        help=(
            "Skip retrieval evaluation and only run RAGAS metrics on top of "
            "the existing evaluation_results.json. Merges scores into that file."
        ),
    )
    args = parser.parse_args()

    if (args.ragas or args.ragas_only) and not RAGAS_AVAILABLE:
        print(
            "RAGAS is not installed. Install it with:\n"
            "  pip install ragas"
        )
        return

    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set. Check your .env file.")

    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(
            f"Test set not found at '{testset_path}'.\n"
            "Run:  python evaluation/generate_testset.py  first."
        )
        return

    testset: list[dict] = json.loads(testset_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(testset)} test queries from {testset_path}")

    pipelines = ALL_PIPELINES
    if args.methods:
        name_set = set(args.methods)
        pipelines = [p for p in ALL_PIPELINES if p.name in name_set]
        unknown = name_set - {p.name for p in pipelines}
        if unknown:
            print(f"Unknown pipeline(s): {sorted(unknown)}")
            print(f"Available: {[p.name for p in ALL_PIPELINES]}")
            return
        if not pipelines:
            return

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    if args.ragas_only:
        # Load existing retrieval results and add RAGAS scores on top
        existing_json = RESULTS_DIR / "evaluation_results.json"
        if not existing_json.exists():
            print(
                f"No existing results found at '{existing_json}'.\n"
                "Run without --ragas-only first to generate retrieval results."
            )
            return
        existing = json.loads(existing_json.read_text(encoding="utf-8"))
        k = existing.get("k", args.k)
        prior_results: dict[str, dict[str, float]] = existing["results"]
        final_results = run_ragas_only(testset, pipelines, k, embeddings, llm, prior_results)
    else:
        final_results = run_evaluation(
            testset, pipelines, args.k, embeddings, llm, args.force_rebuild,
            run_ragas=args.ragas,
        )

    print_results_table(final_results, args.k)
    save_results(final_results, args.k)


if __name__ == "__main__":
    main()
