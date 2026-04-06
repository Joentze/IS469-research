"""Semantic chunking strategy based on embedding similarity."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set

from langchain_core.documents import Document
from tqdm import tqdm

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
        max_workers: int = 8,
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
        self.max_workers = max_workers

    def _chunk_single(self, doc: Document) -> List[Document]:
        """Chunk a single document. Called in parallel across documents."""
        sentences = split_into_sentences(doc.page_content)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [doc]

        sentence_vectors = self.embedder.embed_documents(sentences)
        adjacent_similarities = [
            cosine_similarity(sentence_vectors[i], sentence_vectors[i + 1])
            for i in range(len(sentence_vectors) - 1)
        ]
        breakpoint_threshold = percentile(adjacent_similarities, self.breakpoint_percentile)

        break_after_index: Set[int] = {
            i
            for i, score in enumerate(adjacent_similarities)
            if score <= breakpoint_threshold
        }

        chunks: List[Document] = []
        current: List[str] = []
        chunk_index = 0

        for i, sentence in enumerate(sentences):
            current.append(sentence)

            reached_max = len(current) >= self.max_sentences_per_chunk
            semantic_break = i in break_after_index and len(current) >= 2

            if reached_max or semantic_break:
                chunks.append(
                    Document(
                        page_content=" ".join(current).strip(),
                        metadata={**doc.metadata, "chunk_index": chunk_index, "chunk_method": "semantic"},
                    )
                )
                chunk_index += 1
                overlap = current[-self.overlap_sentences:] if self.overlap_sentences > 0 else []
                current = overlap.copy()

        if current:
            chunks.append(
                Document(
                    page_content=" ".join(current).strip(),
                    metadata={**doc.metadata, "chunk_index": chunk_index, "chunk_method": "semantic"},
                )
            )

        return chunks

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
        if self.enable_cache:
            cache = get_global_cache()
            cached_chunks = cache.get(self._get_chunker_name(), documents)
            if cached_chunks is not None:
                return cached_chunks

        results: dict[int, List[Document]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._chunk_single, doc): i for i, doc in enumerate(documents)}
            with tqdm(total=len(documents), desc="Semantic chunking", unit="doc") as pbar:
                for future in as_completed(futures):
                    i = futures[future]
                    results[i] = future.result()
                    source = documents[i].metadata.get("source", "")
                    pbar.set_postfix(doc=source, refresh=False)
                    pbar.update(1)

        chunked_docs = [chunk for i in range(len(documents)) for chunk in results.get(i, [])]

        if self.enable_cache:
            cache = get_global_cache()
            cache.put(self._get_chunker_name(), chunked_docs, documents)

        return chunked_docs
