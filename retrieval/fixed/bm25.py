"""
use bm25 in-memory store
https://docs.langchain.com/oss/python/integrations/retrievers/bm25
"""
import logging
import math
import re

from rank_bm25 import BM25Okapi


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

RELEVANT_DOC_IDS = {4, 6}

# Your documents
docs = [
    "Apple Inc. reported a 10% increase in quarterly revenue, driven by strong iPhone sales.",
    "Tesla announced the launch of its new Model Y, targeting the mid-range EV market.",
    "Federal Reserve raised interest rates by 0.25%, citing inflation concerns.",
    "Goldman Sachs forecasts a bullish trend in the stock market for 2026.",
    "Amazon's cloud division, AWS, saw a 15% growth in Q1 2026.",
    "JP Morgan reported higher profits due to increased trading activity.",
    "Microsoft's Azure revenue grew by 20% year-over-year.",
    "Meta Platforms faces regulatory scrutiny over data privacy issues.",
    "Bank of America expects mortgage rates to remain stable throughout 2026.",
    "Nvidia's GPU sales surge as demand for AI chips increases."
]

print("[START] BM25 retrieval with fixed-size character sliding window chunking")
logger.info("Loaded %d source documents", len(docs))


def fixed_size_character_sliding_window(
    text: str,
    chunk_size: int = 80,
    chunk_overlap: int = 20,
) -> list[str]:
    """Split text into fixed-size character chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    step = chunk_size - chunk_overlap
    output_chunks: list[str] = []
    logger.debug(
        "Chunking text with chunk_size=%d, chunk_overlap=%d, step=%d",
        chunk_size,
        chunk_overlap,
        step,
    )

    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            output_chunks.append(chunk_text)
        if end >= len(text):
            break

    return output_chunks


def extract_source_doc_id(chunk_text: str) -> int | None:
    """Extract source document id from `doc_i_chunk_j: ...` prefixed chunk text."""
    match = re.match(r"doc_(\d+)_chunk_\d+:", chunk_text)
    return int(match.group(1)) if match else None


def dedupe_preserve_order(values: list[int]) -> list[int]:
    """Remove duplicates while preserving first-seen order."""
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def evaluate_retrieval(
    ranked_doc_ids: list[int],
    relevant_doc_ids: set[int],
    k: int,
) -> dict[str, float]:
    """Compute standard ranking metrics at k."""
    ranked_at_k = ranked_doc_ids[:k]
    hits = [1 if doc_id in relevant_doc_ids else 0 for doc_id in ranked_at_k]

    precision_at_k = sum(hits) / k if k else 0.0
    recall_at_k = (
        sum(hits) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
    )
    hit_rate_at_k = 1.0 if any(hits) else 0.0

    mrr = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            mrr = 1.0 / rank
            break

    dcg = 0.0
    for rank, rel in enumerate(hits, start=1):
        if rel:
            dcg += rel / math.log2(rank + 1)

    ideal_hits = [1] * min(len(relevant_doc_ids), k)
    idcg = 0.0
    for rank, rel in enumerate(ideal_hits, start=1):
        idcg += rel / math.log2(rank + 1)

    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

    return {
        "precision@k": precision_at_k,
        "recall@k": recall_at_k,
        "hit_rate@k": hit_rate_at_k,
        "mrr": mrr,
        "ndcg@k": ndcg_at_k,
    }


# Apply fixed-size (character) sliding-window chunking
chunked_docs: list[str] = []
for doc_index, doc in enumerate(docs):
    chunks = fixed_size_character_sliding_window(doc, chunk_size=80, chunk_overlap=20)
    for chunk_index, chunk in enumerate(chunks):
        chunked_docs.append(f"doc_{doc_index}_chunk_{chunk_index}: {chunk}")

print(f"[INFO] Created {len(chunked_docs)} chunks from {len(docs)} documents")
logger.info("Chunk generation complete")

# Tokenize chunked documents
tokenized_docs = [doc.lower().split() for doc in chunked_docs]
print(f"[INFO] Tokenized {len(tokenized_docs)} chunked documents")
logger.info("Tokenization complete")

# Build BM25 index
bm25 = BM25Okapi(tokenized_docs)
print("[INFO] BM25 index built")
logger.info("BM25 index initialization complete")

# Example query
query = "Which company reported the highest cloud revenue growth in 2026?"
tokenized_query = query.lower().split()

# Get scores for all docs
scores = bm25.get_scores(tokenized_query)
print("[INFO] Scored all chunked documents for query")
logger.info("Scoring complete")

# Retrieve top 2 documents
top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]
results = [chunked_docs[i] for i in top_n]
print(f"[INFO] Retrieved top {len(results)} chunks")
logger.info("Retrieval complete")

ranked_source_doc_ids = []
for i in top_n:
    source_doc_id = extract_source_doc_id(chunked_docs[i])
    if source_doc_id is not None:
        ranked_source_doc_ids.append(source_doc_id)
ranked_source_doc_ids = dedupe_preserve_order(ranked_source_doc_ids)
metrics = evaluate_retrieval(
    ranked_doc_ids=ranked_source_doc_ids,
    relevant_doc_ids=RELEVANT_DOC_IDS,
    k=len(top_n),
)
print("[METRICS]")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")
logger.info("Evaluation complete: %s", metrics)

# Print results
print("[RESULTS]")
for doc in results:
    print(doc)