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

from rank_bm25 import BM25Okapi


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

TEXTS_DIR = Path(__file__).resolve().parents[2] / "texts"
DEFAULT_QUERY = "Which company reported the highest cloud revenue growth in 2026?"


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
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument(
        "--texts-dir",
        type=Path,
        default=TEXTS_DIR,
        help="Directory containing markdown files",
    )
    return parser.parse_args()


def tokenize_words(text: str) -> set[str]:
    """Tokenize into lowercased word set for similarity checks."""
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenization for BM25 scoring (word tokens, lowercased)."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute simple Jaccard similarity for adjacent semantic units."""
    tokens_a = tokenize_words(text_a)
    tokens_b = tokenize_words(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    union_size = len(tokens_a | tokens_b)
    return len(tokens_a & tokens_b) / union_size if union_size else 0.0


def split_semantic_units(text: str) -> list[str]:
    """Split text into sentence/phrase-like units."""
    units = re.split(r"(?<=[.!?;])\s+|,\s+", text.strip())
    return [unit.strip() for unit in units if unit and unit.strip()]


def semantic_chunk_text(
    text: str,
    max_chunk_chars: int = 100,
    similarity_threshold: float = 0.15,
) -> list[str]:
    """Create semantic chunks by grouping adjacent units with lexical similarity."""
    units = split_semantic_units(text)
    if not units:
        return []

    semantic_chunks: list[str] = []
    current_chunk = units[0]
    previous_unit = units[0]

    for unit in units[1:]:
        similarity = semantic_similarity(previous_unit, unit)
        candidate = f"{current_chunk} {unit}".strip()

        should_merge = (
            len(candidate) <= max_chunk_chars
            and (
                similarity >= similarity_threshold
                or len(current_chunk) < max_chunk_chars * 0.5
            )
        )

        if should_merge:
            current_chunk = candidate
        else:
            semantic_chunks.append(current_chunk)
            current_chunk = unit

        previous_unit = unit

    semantic_chunks.append(current_chunk)
    return semantic_chunks


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


def build_chunks(docs: list[tuple[str, str]]) -> list[Chunk]:
    """Apply semantic chunking to each source document."""
    chunked_docs: list[Chunk] = []
    for file_name, doc_text in docs:
        chunks = semantic_chunk_text(
            doc_text,
            max_chunk_chars=700,
            similarity_threshold=0.15,
        )
        for chunk_index, chunk_text in enumerate(chunks):
            chunked_docs.append(
                Chunk(
                    file_name=file_name,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )

    return chunked_docs


def main() -> None:
    args = parse_args()
    print("[START] BM25 retrieval with semantic chunking")

    docs = load_markdown_documents(args.texts_dir)
    logger.info("Loaded %d source markdown files from %s", len(docs), args.texts_dir)

    chunks = build_chunks(docs)
    logger.info("Semantic chunk generation complete: %d chunks", len(chunks))
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
