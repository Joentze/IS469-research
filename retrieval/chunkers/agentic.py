"""Agentic chunking strategy using LLM-selected topic boundaries."""

import json
import re
from typing import Any, List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from core.interfaces import Chunker
from utils.cache import get_global_cache


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentence-like units."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s and s.strip()]


def _normalize_breakpoints(raw_breakpoints: List[int], sentence_count: int) -> List[int]:
    """Keep sorted valid sentence-end indexes for chunk boundaries."""
    if sentence_count <= 1:
        return []
    return sorted({idx for idx in raw_breakpoints if 0 <= idx < sentence_count - 1})


def _fallback_breakpoints(sentence_count: int, max_sentences: int) -> List[int]:
    """Deterministic fallback if the chunking agent output is invalid."""
    if sentence_count <= max_sentences:
        return []

    breaks: List[int] = []
    idx = max_sentences - 1
    while idx < sentence_count - 1:
        breaks.append(idx)
        idx += max_sentences
    return breaks


def _truncate_for_prompt(text: str, max_chars: int = 500) -> str:
    """Trim long sentence strings in prompts to control token usage."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _choose_breakpoints_for_window(
    window_sentences: List[str],
    llm: ChatOpenAI,
    min_sentences: int,
    max_sentences: int,
) -> List[int]:
    """Run one chunk-planning LLM call on a sentence window."""
    if len(window_sentences) <= max_sentences:
        return []

    numbered_sentences = "\n".join(
        f"{i}: {_truncate_for_prompt(sentence)}" for i, sentence in enumerate(window_sentences)
    )
    prompt = (
        "You are a chunking planner for a RAG system.\n"
        "Group adjacent sentences into coherent chunks by topic/idea shifts.\n"
        "Rules:\n"
        f"- Every chunk should have {min_sentences} to {max_sentences} sentences when possible.\n"
        "- Prefer boundaries where topic changes.\n"
        "- Return ONLY valid JSON with this exact schema:\n"
        "{\"break_after\": [int, int, ...]}\n"
        "- Do not include markdown or explanations.\n\n"
        "Sentences:\n"
        f"{numbered_sentences}"
    )

    response = llm.invoke(prompt)

    try:
        text = str(getattr(response, "content", "")).strip()
        parsed = json.loads(text)
        raw_breaks = parsed.get("break_after", [])
        if not isinstance(raw_breaks, list):
            return []
        int_breaks = [int(x) for x in raw_breaks]
        return _normalize_breakpoints(int_breaks, len(window_sentences))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def choose_chunk_breakpoints_with_agent(
    sentences: List[str],
    llm: ChatOpenAI,
    min_sentences: int,
    max_sentences: int,
    window_size: int,
    window_overlap: int,
) -> List[int]:
    """Ask an LLM to propose chunk boundaries as sentence-end indexes."""
    if len(sentences) <= max_sentences:
        return []

    actual_window_size = max(max_sentences + 1, window_size)
    actual_overlap = max(0, min(window_overlap, actual_window_size - 1))
    step = max(1, actual_window_size - actual_overlap)

    global_breaks: List[int] = []
    for start in range(0, len(sentences), step):
        end = min(start + actual_window_size, len(sentences))
        window_sentences = sentences[start:end]

        local_breaks = _choose_breakpoints_for_window(
            window_sentences, llm, min_sentences, max_sentences
        )
        for local_break in local_breaks:
            global_break = start + local_break
            if global_break < len(sentences) - 1:
                global_breaks.append(global_break)

        if end >= len(sentences):
            break

    normalized_breaks = _normalize_breakpoints(global_breaks, len(sentences))
    return (
        normalized_breaks
        if normalized_breaks
        else _fallback_breakpoints(len(sentences), max_sentences)
    )


def enforce_chunk_size_rules(
    breaks: List[int],
    sentence_count: int,
    min_sentences: int,
    max_sentences: int,
) -> List[int]:
    """Adjust boundaries to avoid tiny chunks and very large chunks."""
    if sentence_count <= 1:
        return []

    all_breaks = _normalize_breakpoints(breaks, sentence_count)
    if not all_breaks:
        return _fallback_breakpoints(sentence_count, max_sentences)

    filtered: List[int] = []
    start = 0
    for brk in all_breaks:
        chunk_len = brk - start + 1
        if chunk_len >= min_sentences:
            filtered.append(brk)
            start = brk + 1

    with_caps: List[int] = []
    start = 0
    for brk in filtered + [sentence_count - 1]:
        while brk - start + 1 > max_sentences:
            cap_break = start + max_sentences - 1
            if cap_break < sentence_count - 1:
                with_caps.append(cap_break)
            start = cap_break + 1
        if brk < sentence_count - 1:
            with_caps.append(brk)
        start = brk + 1

    return _normalize_breakpoints(with_caps, sentence_count)


def build_chunks_from_breakpoints(
    sentences: List[str],
    breakpoints: List[int],
    metadata: dict[str, Any],
    overlap_sentences: int,
) -> List[Document]:
    """Materialize chunk documents from sentence breakpoints."""
    chunks: List[Document] = []
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
                    **metadata,
                    "chunk_index": chunk_index,
                    "chunk_method": "agentic",
                },
            )
        )
        chunk_index += 1

        if overlap_sentences > 0:
            start = max(brk + 1 - overlap_sentences, 0)
        else:
            start = brk + 1

        if start >= len(sentences):
            break

    return chunks


class AgenticChunker(Chunker):
    """Chunk documents using LLM-selected topic-aware boundaries."""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        min_sentences_per_chunk: int = 2,
        max_sentences_per_chunk: int = 6,
        overlap_sentences: int = 1,
        chunk_planning_window_sentences: int = 80,
        chunk_planning_window_overlap: int = 10,
        enable_cache: bool = True,
    ):
        """
        Initialize the agentic chunker.

        Args:
            llm_model: LLM model to use for chunk planning
            min_sentences_per_chunk: Minimum sentences per chunk
            max_sentences_per_chunk: Maximum sentences per chunk
            overlap_sentences: Sentences to overlap between chunks
            chunk_planning_window_sentences: Sentence window size for LLM calls
            chunk_planning_window_overlap: Overlap of planning windows
            enable_cache: Whether to cache chunking results
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.chunk_planning_window_sentences = chunk_planning_window_sentences
        self.chunk_planning_window_overlap = chunk_planning_window_overlap
        self.enable_cache = enable_cache

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents with LLM-selected semantic/topic boundaries.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents with agentic boundaries
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
            if len(sentences) <= self.max_sentences_per_chunk:
                chunked_docs.append(
                    Document(
                        page_content=" ".join(sentences),
                        metadata={
                            **doc.metadata,
                            "chunk_index": 0,
                            "chunk_method": "agentic",
                        },
                    )
                )
                continue

            raw_breaks = choose_chunk_breakpoints_with_agent(
                sentences,
                self.llm,
                self.min_sentences_per_chunk,
                self.max_sentences_per_chunk,
                self.chunk_planning_window_sentences,
                self.chunk_planning_window_overlap,
            )
            breaks = enforce_chunk_size_rules(
                raw_breaks, len(sentences), self.min_sentences_per_chunk, self.max_sentences_per_chunk
            )
            chunked_docs.extend(
                build_chunks_from_breakpoints(sentences, breaks, doc.metadata, self.overlap_sentences)
            )

        # Store in cache
        if self.enable_cache:
            cache = get_global_cache()
            cache.put(self._get_chunker_name(), chunked_docs, documents)

        return chunked_docs
