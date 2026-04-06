"""
use bm25 in-memory store
https://docs.langchain.com/oss/python/integrations/retrievers/bm25
"""
from __future__ import annotations

import logging
import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

TEXTS_DIR = Path(__file__).resolve().parents[2] / "texts"
DEFAULT_QUERY = "Which company reported the highest cloud revenue growth in 2026?"
DEFAULT_TOP_K = 5
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_SENTENCES_PER_CHUNK = 8
OVERLAP_SENTENCES = 2
BREAKPOINT_PERCENTILE = 70.0


@dataclass(frozen=True)
class Chunk:
    """Represents one semantic chunk and its source metadata."""

    file_name: str
    chunk_index: int
    text: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for retrieval options."""
    parser = argparse.ArgumentParser(description="BM25 retrieval on markdown files")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Search query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results to return")
    parser.add_argument(
        "--texts-dir",
        type=Path,
        default=TEXTS_DIR,
        help="Directory containing markdown files",
    )
    return parser.parse_args()


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenization for BM25 scoring (word tokens, lowercased)."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def split_into_sentences(text: str) -> list[str]:
    """Split text into simple sentence-like units."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s and s.strip()]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def percentile(values: list[float], q: float) -> float:
    """Compute percentile with linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (q / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def semantic_chunk_documents(
    docs: list[tuple[str, str]],
    embeddings: OpenAIEmbeddings,
) -> list[Chunk]:
    """Apply semantic sentence-window chunking to each source document."""
    chunked_docs: list[Chunk] = []

    for file_name, doc_text in docs:
        sentences = split_into_sentences(doc_text)
        if not sentences:
            continue
        if len(sentences) == 1:
            chunked_docs.append(Chunk(file_name=file_name, chunk_index=0, text=sentences[0]))
            continue

        sentence_vectors = embeddings.embed_documents(sentences)
        adjacent_similarities = [
            cosine_similarity(sentence_vectors[i], sentence_vectors[i + 1])
            for i in range(len(sentence_vectors) - 1)
        ]

        breakpoint_threshold = percentile(adjacent_similarities, BREAKPOINT_PERCENTILE)
        break_after_index = {
            i for i, score in enumerate(adjacent_similarities) if score <= breakpoint_threshold
        }

        current: list[str] = []
        chunk_index = 0

        for i, sentence in enumerate(sentences):
            current.append(sentence)

            reached_max = len(current) >= MAX_SENTENCES_PER_CHUNK
            semantic_break = i in break_after_index and len(current) >= 2

            if reached_max or semantic_break:
                chunked_docs.append(
                    Chunk(
                        file_name=file_name,
                        chunk_index=chunk_index,
                        text=" ".join(current).strip(),
                    )
                )
                chunk_index += 1

                overlap = current[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES > 0 else []
                current = overlap.copy()

        if current:
            chunked_docs.append(
                Chunk(
                    file_name=file_name,
                    chunk_index=chunk_index,
                    text=" ".join(current).strip(),
                )
            )

    return chunked_docs


def load_markdown_documents(texts_dir: Path) -> list[tuple[str, str]]:
    """Load markdown files from texts directory."""
    if not texts_dir.exists() or not texts_dir.is_dir():
        raise FileNotFoundError(f"Texts directory not found: {texts_dir}")

    markdown_files = sorted(texts_dir.glob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"No markdown files found in: {texts_dir}")

    docs: list[tuple[str, str]] = []
    for file_path in markdown_files:
        content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if content:
            docs.append((file_path.name, content))

    if not docs:
        raise ValueError(f"All markdown files in {texts_dir} were empty")

    return docs


def dedupe_preserve_order(values: list[str]) -> list[str]:
    """Remove duplicates while preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def main() -> None:
    args = parse_args()
    print("[START] BM25 retrieval with semantic chunking")

    docs = load_markdown_documents(args.texts_dir)
    logger.info("Loaded %d source markdown files from %s", len(docs), args.texts_dir)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    chunks = semantic_chunk_documents(docs, embeddings)
    logger.info("Sentence-window chunk generation complete: %d chunks", len(chunks))
    print(f"[INFO] Loaded {len(docs)} files, created {len(chunks)} chunks")

    tokenized_chunks = [tokenize_for_bm25(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    logger.info("BM25 index initialization complete")

    query_tokens = tokenize_for_bm25(args.query)
    if not query_tokens:
        raise ValueError("Query does not contain valid alphanumeric tokens")

    scores = bm25.get_scores(query_tokens)
    logger.info("Scoring complete")

    top_k = max(1, min(args.top_k, len(chunks)))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    ranked_chunks = [chunks[i] for i in ranked_indices]
    unique_ranked_files = dedupe_preserve_order([chunk.file_name for chunk in ranked_chunks])
    logger.info("Unique source files in top %d chunks: %d", top_k, len(unique_ranked_files))

    print(f"[QUERY] {args.query}")
    print(f"[RESULTS] top {top_k} chunks")
    for rank, idx in enumerate(ranked_indices, start=1):
        chunk = chunks[idx]
        score = scores[idx]
        preview = re.sub(r"\s+", " ", chunk.text).strip()
        preview = preview[:240] + "..." if len(preview) > 240 else preview
        print(
            f"{rank}. score={score:.4f} | file={chunk.file_name} | chunk={chunk.chunk_index}"
        )
        print(f"   {preview}")


if __name__ == "__main__":
    main()
