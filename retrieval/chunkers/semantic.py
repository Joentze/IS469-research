"""Semantic chunking strategy based on embedding similarity."""

import re
from typing import List, Set

from langchain_core.documents import Document

from core.interfaces import Chunker, Embeddings
from utils.cache import get_global_cache


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentence-like units."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s and s.strip()]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def percentile(values: List[float], q: float) -> float:
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


class SemanticChunker(Chunker):
    """Chunk documents by semantic boundaries using embedding similarity."""

    def __init__(
        self,
        embedder: Embeddings,
        max_sentences_per_chunk: int = 8,
        overlap_sentences: int = 2,
        breakpoint_percentile: float = 70.0,
        enable_cache: bool = True,
    ):
        """
        Initialize the semantic chunker.

        Args:
            embedder: Embeddings instance to use for computing sentence similarities
            max_sentences_per_chunk: Maximum sentences in a chunk
            overlap_sentences: Number of sentences to overlap between chunks
            breakpoint_percentile: Percentile threshold for breaking (lower = more breaks)
            enable_cache: Whether to cache chunking results
        """
        self.embedder = embedder
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.breakpoint_percentile = breakpoint_percentile
        self.enable_cache = enable_cache

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents by semantic boundaries.

        We split text into sentences, embed each sentence, and insert chunk breaks
        where adjacent sentence similarity drops below a percentile threshold.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents with updated metadata
        """
        # Check cache first
        if self.enable_cache:
            cache = get_global_cache()
            cached_chunks = cache.get(self._get_chunker_name(), documents)
            if cached_chunks is not None:
                return cached_chunks

        chunked_docs: List[Document] = []

        for doc in documents:
            sentences = split_into_sentences(doc.page_content)
            if not sentences:
                continue
            if len(sentences) == 1:
                chunked_docs.append(doc)
                continue

            # Embed sentences
            sentence_vectors = self.embedder.embed_documents(sentences)
            adjacent_similarities = [
                cosine_similarity(sentence_vectors[i], sentence_vectors[i + 1])
                for i in range(len(sentence_vectors) - 1)
            ]
            breakpoint_threshold = percentile(adjacent_similarities, self.breakpoint_percentile)

            # Find breaks where similarity drops below threshold
            break_after_index: Set[int] = {
                i
                for i, score in enumerate(adjacent_similarities)
                if score <= breakpoint_threshold
            }

            current: List[str] = []
            chunk_index = 0

            for i, sentence in enumerate(sentences):
                current.append(sentence)

                reached_max = len(current) >= self.max_sentences_per_chunk
                semantic_break = i in break_after_index and len(current) >= 2

                if reached_max or semantic_break:
                    chunked_docs.append(
                        Document(
                            page_content=" ".join(current).strip(),
                            metadata={
                                **doc.metadata,
                                "chunk_index": chunk_index,
                                "chunk_method": "semantic",
                            },
                        )
                    )
                    chunk_index += 1

                    overlap = current[-self.overlap_sentences :] if self.overlap_sentences > 0 else []
                    current = overlap.copy()

            if current:
                chunked_docs.append(
                    Document(
                        page_content=" ".join(current).strip(),
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk_index,
                            "chunk_method": "semantic",
                        },
                    )
                )

        # Store in cache
        if self.enable_cache:
            cache = get_global_cache()
            cache.put(self._get_chunker_name(), chunked_docs, documents)

        return chunked_docs
