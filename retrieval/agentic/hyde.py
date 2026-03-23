"""
HyDE (Hypothetical Document Embeddings) with agentic chunking.

Instead of embedding the raw query, HyDE asks an LLM to generate a hypothetical
document that would answer the query, then embeds that document for retrieval.
Agentic chunking uses a second LLM call to choose topic-aware chunk boundaries
before indexing.

References:
- HyDE paper: https://arxiv.org/abs/2212.10496
- ChromaDB: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
"""

import json
import logging
import math
import os
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
VECTOR_STORE_DIR = "./database"
COLLECTION_NAME = "hyde_agentic_chunks_collection"

MAX_SENTENCES_PER_CHUNK = 6
MIN_SENTENCES_PER_CHUNK = 2
OVERLAP_SENTENCES = 1
TOP_K = 5

RELEVANT_DOC_IDS = {4, 6}

DOCS = [
    "Apple Inc. reported a 10% increase in quarterly revenue, driven by strong iPhone sales.",
    "Tesla announced the launch of its new Model Y, targeting the mid-range EV market.",
    "Federal Reserve raised interest rates by 0.25%, citing inflation concerns.",
    "Goldman Sachs forecasts a bullish trend in the stock market for 2026.",
    "Amazon's cloud division, AWS, saw a 15% growth in Q1 2026.",
    "JP Morgan reported higher profits due to increased trading activity.",
    "Microsoft's Azure revenue grew by 20% year-over-year.",
    "Meta Platforms faces regulatory scrutiny over data privacy issues.",
    "Bank of America expects mortgage rates to remain stable throughout 2026.",
    "Nvidia's GPU sales surge as demand for AI chips increases.",
]

QUERY = "Which company reported the highest cloud revenue growth in 2026?"


# ============================================================================
# AGENTIC CHUNKING
# ============================================================================


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s and s.strip()]


def _normalize_breakpoints(raw_breakpoints: list[int], sentence_count: int) -> list[int]:
    if sentence_count <= 1:
        return []
    return sorted({idx for idx in raw_breakpoints if 0 <= idx < sentence_count - 1})


def _fallback_breakpoints(sentence_count: int) -> list[int]:
    if sentence_count <= MAX_SENTENCES_PER_CHUNK:
        return []
    breaks: list[int] = []
    idx = MAX_SENTENCES_PER_CHUNK - 1
    while idx < sentence_count - 1:
        breaks.append(idx)
        idx += MAX_SENTENCES_PER_CHUNK
    return breaks


def choose_chunk_breakpoints_with_agent(sentences: list[str], llm: ChatOpenAI) -> list[int]:
    """Ask an LLM to propose chunk boundaries as sentence-end indexes."""
    if len(sentences) <= MAX_SENTENCES_PER_CHUNK:
        return []

    numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
    prompt = (
        "You are a chunking planner for a RAG system.\n"
        "Group adjacent sentences into coherent chunks by topic/idea shifts.\n"
        "Rules:\n"
        f"- Every chunk should have {MIN_SENTENCES_PER_CHUNK} to {MAX_SENTENCES_PER_CHUNK} sentences when possible.\n"
        "- Prefer boundaries where topic changes.\n"
        "- Return ONLY valid JSON with this exact schema:\n"
        '{\"break_after\": [int, int, ...]}\n'
        "- Do not include markdown or explanations.\n\n"
        f"Sentences:\n{numbered}"
    )

    try:
        response = llm.invoke(prompt)
        text = str(getattr(response, "content", "")).strip()
        parsed = json.loads(text)
        raw = parsed.get("break_after", [])
        if not isinstance(raw, list):
            return _fallback_breakpoints(len(sentences))
        return _normalize_breakpoints([int(x) for x in raw], len(sentences))
    except Exception:
        return _fallback_breakpoints(len(sentences))


def enforce_chunk_size_rules(breaks: list[int], sentence_count: int) -> list[int]:
    if sentence_count <= 1:
        return []
    all_breaks = _normalize_breakpoints(breaks, sentence_count)
    if not all_breaks:
        return _fallback_breakpoints(sentence_count)

    filtered: list[int] = []
    start = 0
    for brk in all_breaks:
        if brk - start + 1 >= MIN_SENTENCES_PER_CHUNK:
            filtered.append(brk)
            start = brk + 1

    with_caps: list[int] = []
    start = 0
    for brk in filtered + [sentence_count - 1]:
        while brk - start + 1 > MAX_SENTENCES_PER_CHUNK:
            cap = start + MAX_SENTENCES_PER_CHUNK - 1
            if cap < sentence_count - 1:
                with_caps.append(cap)
            start = cap + 1
        if brk < sentence_count - 1:
            with_caps.append(brk)
        start = brk + 1

    return _normalize_breakpoints(with_caps, sentence_count)


def build_chunks_from_breakpoints(
    sentences: list[str],
    breakpoints: list[int],
    doc_id: int,
) -> list[Document]:
    chunks: list[Document] = []
    start = 0
    chunk_index = 0

    for brk in breakpoints + [len(sentences) - 1]:
        if brk < start:
            continue
        chunk_sentences = sentences[start : brk + 1]
        if not chunk_sentences:
            continue
        chunks.append(
            Document(
                page_content=" ".join(chunk_sentences).strip(),
                metadata={
                    "source_doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "chunk_method": "agentic",
                },
            )
        )
        chunk_index += 1
        start = max(brk + 1 - OVERLAP_SENTENCES, 0) if OVERLAP_SENTENCES > 0 else brk + 1
        if start >= len(sentences):
            break

    return chunks


def agentic_chunk_documents(docs: list[str], llm: ChatOpenAI) -> list[Document]:
    """Chunk each doc with LLM-selected topic boundaries."""
    all_chunks: list[Document] = []
    for doc_id, doc_text in enumerate(docs):
        sentences = split_into_sentences(doc_text)
        if not sentences:
            continue
        if len(sentences) <= MAX_SENTENCES_PER_CHUNK:
            all_chunks.append(
                Document(
                    page_content=" ".join(sentences),
                    metadata={
                        "source_doc_id": doc_id,
                        "chunk_index": 0,
                        "chunk_method": "agentic",
                    },
                )
            )
            continue
        raw_breaks = choose_chunk_breakpoints_with_agent(sentences, llm)
        breaks = enforce_chunk_size_rules(raw_breaks, len(sentences))
        all_chunks.extend(build_chunks_from_breakpoints(sentences, breaks, doc_id))
    return all_chunks


# ============================================================================
# VECTOR STORE
# ============================================================================


def initialize_vector_store(
    documents: list[Document],
    embeddings: OpenAIEmbeddings,
) -> Chroma:
    """Build a fresh Chroma collection from the given documents."""
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR,
    )
    try:
        vector_store.delete_collection()
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTOR_STORE_DIR,
        )
    except Exception:
        pass

    if documents:
        vector_store.add_documents(documents)
    return vector_store


# ============================================================================
# HYDE RETRIEVAL
# ============================================================================


def generate_hypothetical_document(query: str, llm: ChatOpenAI) -> str:
    """
    Ask the LLM to write a short passage that would answer the query.
    This hypothetical document is then embedded for retrieval.
    """
    prompt = (
        "Write a short, factual passage (2-3 sentences) that directly answers "
        "the following question. Do not mention that it is hypothetical.\n\n"
        f"Question: {query}\n\nPassage:"
    )
    response = llm.invoke(prompt)
    return str(getattr(response, "content", "")).strip()


def hyde_retrieve(
    query: str,
    vector_store: Chroma,
    embeddings: OpenAIEmbeddings,
    llm: ChatOpenAI,
    k: int = TOP_K,
) -> list[Document]:
    """
    HyDE retrieval: embed a hypothetical document instead of the raw query.
    Returns the top-k chunks most similar to the hypothetical document.
    """
    hypothetical_doc = generate_hypothetical_document(query, llm)
    logger.info("Hypothetical document: %s", hypothetical_doc)
    print(f"[HyDE] Hypothetical document:\n  {hypothetical_doc}\n")

    hyp_embedding = embeddings.embed_query(hypothetical_doc)
    results = vector_store.similarity_search_by_vector(hyp_embedding, k=k)
    return results


# ============================================================================
# EVALUATION
# ============================================================================


def dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def evaluate_retrieval(
    ranked_doc_ids: list[int],
    relevant_doc_ids: set[int],
    k: int,
) -> dict[str, float]:
    """Compute precision@k, recall@k, hit_rate@k, MRR, and nDCG@k."""
    ranked_at_k = ranked_doc_ids[:k]
    hits = [1 if doc_id in relevant_doc_ids else 0 for doc_id in ranked_at_k]

    precision_at_k = sum(hits) / k if k else 0.0
    recall_at_k = sum(hits) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
    hit_rate_at_k = 1.0 if any(hits) else 0.0

    mrr = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            mrr = 1.0 / rank
            break

    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(hits, start=1) if rel)
    ideal_hits = [1] * min(len(relevant_doc_ids), k)
    idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_hits, start=1))
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

    return {
        "precision@k": precision_at_k,
        "recall@k": recall_at_k,
        "hit_rate@k": hit_rate_at_k,
        "mrr": mrr,
        "ndcg@k": ndcg_at_k,
    }


# ============================================================================
# PIPELINE
# ============================================================================


def run_hyde_pipeline(query: str = QUERY) -> dict[str, float]:
    """Full HyDE pipeline: agentic chunk -> index -> HyDE retrieval -> evaluate."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    print("=" * 60)
    print("HyDE PIPELINE - AGENTIC CHUNKING")
    print("=" * 60)

    print("\n[1] Building agentic chunks...")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    langchain_docs = agentic_chunk_documents(DOCS, llm)
    print(f"Created {len(langchain_docs)} chunks from {len(DOCS)} documents")
    logger.info("Chunk generation complete: %d chunks", len(langchain_docs))

    print("\n[2] Initializing ChromaDB...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = initialize_vector_store(langchain_docs, embeddings)
    print(f"Indexed {vector_store._collection.count()} chunks")

    print(f"\n[3] Query: {query}")
    results = hyde_retrieve(query, vector_store, embeddings, llm, k=TOP_K)
    print(f"Retrieved {len(results)} chunks")

    ranked_doc_ids = dedupe_preserve_order(
        [doc.metadata["source_doc_id"] for doc in results]
    )
    logger.info("Ranked source doc IDs: %s", ranked_doc_ids)

    metrics = evaluate_retrieval(
        ranked_doc_ids=ranked_doc_ids,
        relevant_doc_ids=RELEVANT_DOC_IDS,
        k=TOP_K,
    )

    print("\n[METRICS]")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\n[RESULTS]")
    for i, doc in enumerate(results, start=1):
        print(f"  [{i}] doc_id={doc.metadata['source_doc_id']} | {doc.page_content}")

    return metrics


if __name__ == "__main__":
    run_hyde_pipeline()
