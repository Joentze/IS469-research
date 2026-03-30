"""
HyDE (Hypothetical Document Embeddings) with fixed-size character chunking.

Instead of embedding the raw query, HyDE asks an LLM to generate a hypothetical
document that would answer the query, then embeds that document for retrieval.
This closes the lexical gap between short queries and longer document passages.

References:
- HyDE paper: https://arxiv.org/abs/2212.10496
- ChromaDB: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
"""

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
COLLECTION_NAME = "hyde_fixed_chunks_collection"

CHUNK_SIZE = 80
CHUNK_OVERLAP = 20
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
# CHUNKING
# ============================================================================


def fixed_size_character_sliding_window(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into fixed-size character chunks with overlap."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def build_langchain_documents(docs: list[str]) -> list[Document]:
    """Chunk source docs and return LangChain Documents with source_doc_id metadata."""
    langchain_docs: list[Document] = []
    for doc_id, doc_text in enumerate(docs):
        chunks = fixed_size_character_sliding_window(doc_text)
        for chunk_idx, chunk in enumerate(chunks):
            langchain_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source_doc_id": doc_id,
                        "chunk_index": chunk_idx,
                        "chunk_method": "fixed",
                    },
                )
            )
    return langchain_docs


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
    """Full HyDE pipeline: chunk -> index -> HyDE retrieval -> evaluate."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    print("=" * 60)
    print("HyDE PIPELINE - FIXED CHUNKING")
    print("=" * 60)

    print("\n[1] Building fixed-size chunks...")
    langchain_docs = build_langchain_documents(DOCS)
    print(f"Created {len(langchain_docs)} chunks from {len(DOCS)} documents")
    logger.info("Chunk generation complete: %d chunks", len(langchain_docs))

    print("\n[2] Initializing ChromaDB...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = initialize_vector_store(langchain_docs, embeddings)
    print(f"Indexed {vector_store._collection.count()} chunks")

    print(f"\n[3] Query: {query}")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
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
