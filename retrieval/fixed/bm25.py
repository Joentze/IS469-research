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

# ============================================================================
# CONFIGURATION
# ============================================================================

TEXTS_DIR = Path(__file__).resolve().parents[2] / "texts"
DEFAULT_QUERY = "Which company reported the highest cloud revenue growth in 2026?"
DEFAULT_TOP_K = 5
FIXED_CHUNK_SIZE = 800
FIXED_CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class Chunk:
    """Represents one fixed-window chunk and its source metadata."""

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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=FIXED_CHUNK_SIZE,
        help="Character length of each chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=FIXED_CHUNK_OVERLAP,
        help="Character overlap between neighboring chunks",
    )
    return parser.parse_args()


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenization for BM25 scoring (word tokens, lowercased)."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


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


def fixed_size_character_sliding_window(
    text: str,
    chunk_size: int = FIXED_CHUNK_SIZE,
    chunk_overlap: int = FIXED_CHUNK_OVERLAP,
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


def dedupe_preserve_order(values: list[str]) -> list[str]:
    """Remove duplicates while preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def build_chunks(
    docs: list[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Apply fixed-size sliding-window chunking to each source document."""
    chunked_docs: list[Chunk] = []
    for file_name, doc_text in docs:
        chunks = fixed_size_character_sliding_window(
            doc_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
    print("[START] BM25 retrieval with fixed-size character sliding window chunking")

    docs = load_markdown_documents(args.texts_dir)
    logger.info("Loaded %d source markdown files from %s", len(docs), args.texts_dir)

    chunks = build_chunks(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    logger.info("Chunk generation complete: %d chunks", len(chunks))
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