"""
Agentic BM25 pipeline with agentic chunking.

This script uses an LLM to choose topic-aware chunk boundaries,
indexes chunks with BM25, and runs a ReAct-style retrieval agent.

References:
- BM25: https://docs.langchain.com/oss/python/integrations/retrievers/bm25
- ReAct agents: https://docs.langchain.com/oss/python/langchain/agents
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from rank_bm25 import BM25Okapi


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Models
LLM_MODEL = "gpt-4o-mini"
CHUNKING_LLM_MAX_OUTPUT_TOKENS = 150
ANSWERING_LLM_MAX_OUTPUT_TOKENS = 250

# Paths and defaults
DEFAULT_TEXTS_DIR = Path(__file__).resolve().parents[2] / "texts"
DEFAULT_QUERY = "Explain how cloud revenue trends changed across companies."
DEFAULT_TOP_K = 5

# Agentic chunking knobs
MAX_SENTENCES_PER_CHUNK = 6
MIN_SENTENCES_PER_CHUNK = 2
OVERLAP_SENTENCES = 1
CHUNK_PLANNING_WINDOW_SENTENCES = 80
CHUNK_PLANNING_WINDOW_OVERLAP = 10
CHUNK_PLANNING_MAX_SENTENCE_CHARS = 280


@dataclass(frozen=True)
class Chunk:
    """Represents one agentically chunked unit and its source metadata."""

    file_name: str
    chunk_index: int
    text: str


def load_dotenv_file() -> None:
    """Load key=value pairs from .env into process environment if not already set."""
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_texts_dir(cli_texts_dir: Path | None = None) -> Path:
    """Resolve texts directory from CLI, then env var, then default path."""
    if cli_texts_dir is not None:
        return cli_texts_dir.expanduser()

    env_texts_dir = os.getenv("TEXTS_DIR")
    if env_texts_dir:
        return Path(env_texts_dir).expanduser()

    return DEFAULT_TEXTS_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for retrieval options."""
    parser = argparse.ArgumentParser(description="Agentic BM25 retrieval on markdown files")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Chunks to return per search")
    parser.add_argument(
        "--texts-dir",
        type=Path,
        default=None,
        help="Directory containing markdown files",
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


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like units."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s and s.strip()]


def _normalize_breakpoints(raw_breakpoints: list[int], sentence_count: int) -> list[int]:
    """Keep sorted valid sentence-end indexes for chunk boundaries."""
    if sentence_count <= 1:
        return []
    return sorted({idx for idx in raw_breakpoints if 0 <= idx < sentence_count - 1})


def _fallback_breakpoints(sentence_count: int) -> list[int]:
    """Deterministic fallback if chunk-planning output is invalid."""
    if sentence_count <= MAX_SENTENCES_PER_CHUNK:
        return []

    breaks: list[int] = []
    idx = MAX_SENTENCES_PER_CHUNK - 1
    while idx < sentence_count - 1:
        breaks.append(idx)
        idx += MAX_SENTENCES_PER_CHUNK
    return breaks


def _truncate_for_prompt(text: str, max_chars: int = CHUNK_PLANNING_MAX_SENTENCE_CHARS) -> str:
    """Trim long sentence strings in prompts to control token usage."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _choose_breakpoints_for_window(window_sentences: list[str], llm: ChatOpenAI) -> list[int]:
    """Run one chunk-planning LLM call on a sentence window."""
    if len(window_sentences) <= MAX_SENTENCES_PER_CHUNK:
        return []

    numbered_sentences = "\n".join(
        f"{i}: {_truncate_for_prompt(sentence)}"
        for i, sentence in enumerate(window_sentences)
    )
    prompt = (
        "You are a chunking planner for a retrieval system.\n"
        "Group adjacent sentences into coherent chunks by topic shifts.\n"
        "Rules:\n"
        f"- Each chunk should have {MIN_SENTENCES_PER_CHUNK} to {MAX_SENTENCES_PER_CHUNK} sentences when possible.\n"
        "- Prefer boundaries where the topic changes.\n"
        "- Return ONLY valid JSON with schema: {\"break_after\": [int, int, ...]}\n"
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


def choose_chunk_breakpoints_with_agent(sentences: list[str], llm: ChatOpenAI) -> list[int]:
    """
    Ask an LLM to propose chunk boundaries.

    Returns sentence indexes where a boundary should be inserted after that index.
    """
    if len(sentences) <= MAX_SENTENCES_PER_CHUNK:
        return []

    window_size = max(MAX_SENTENCES_PER_CHUNK + 1, CHUNK_PLANNING_WINDOW_SENTENCES)
    window_overlap = max(0, min(CHUNK_PLANNING_WINDOW_OVERLAP, window_size - 1))
    step = max(1, window_size - window_overlap)

    global_breaks: list[int] = []
    for start in range(0, len(sentences), step):
        end = min(start + window_size, len(sentences))
        window_sentences = sentences[start:end]

        local_breaks = _choose_breakpoints_for_window(window_sentences, llm)
        for local_break in local_breaks:
            global_break = start + local_break
            if global_break < len(sentences) - 1:
                global_breaks.append(global_break)

        if end >= len(sentences):
            break

    normalized_breaks = _normalize_breakpoints(global_breaks, len(sentences))
    return normalized_breaks if normalized_breaks else _fallback_breakpoints(len(sentences))


def enforce_chunk_size_rules(breaks: list[int], sentence_count: int) -> list[int]:
    """Adjust boundaries to avoid tiny chunks and very large chunks."""
    if sentence_count <= 1:
        return []

    all_breaks = _normalize_breakpoints(breaks, sentence_count)
    if not all_breaks:
        return _fallback_breakpoints(sentence_count)

    filtered: list[int] = []
    start = 0
    for brk in all_breaks:
        chunk_len = brk - start + 1
        if chunk_len >= MIN_SENTENCES_PER_CHUNK:
            filtered.append(brk)
            start = brk + 1

    with_caps: list[int] = []
    start = 0
    for brk in filtered + [sentence_count - 1]:
        while brk - start + 1 > MAX_SENTENCES_PER_CHUNK:
            cap_break = start + MAX_SENTENCES_PER_CHUNK - 1
            if cap_break < sentence_count - 1:
                with_caps.append(cap_break)
            start = cap_break + 1
        if brk < sentence_count - 1:
            with_caps.append(brk)
        start = brk + 1

    return _normalize_breakpoints(with_caps, sentence_count)


def build_chunks_from_breakpoints(
    sentences: list[str],
    breakpoints: list[int],
    file_name: str,
) -> list[Chunk]:
    """Materialize chunks from sentence breakpoints with sentence overlap."""
    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    for brk in breakpoints + [len(sentences) - 1]:
        if brk < start:
            continue
        chunk_sentences = sentences[start : brk + 1]
        if not chunk_sentences:
            continue

        chunks.append(
            Chunk(
                file_name=file_name,
                chunk_index=chunk_index,
                text=" ".join(chunk_sentences).strip(),
            )
        )
        chunk_index += 1

        if OVERLAP_SENTENCES > 0:
            start = max(brk + 1 - OVERLAP_SENTENCES, 0)
        else:
            start = brk + 1

        if start >= len(sentences):
            break

    return chunks


def agentic_chunk_documents(docs: list[tuple[str, str]], llm: ChatOpenAI) -> list[Chunk]:
    """Chunk documents with LLM-selected semantic/topic boundaries."""
    chunked_docs: list[Chunk] = []

    for file_name, text in docs:
        sentences = split_into_sentences(text)
        if not sentences:
            continue

        if len(sentences) <= MAX_SENTENCES_PER_CHUNK:
            chunked_docs.append(Chunk(file_name=file_name, chunk_index=0, text=" ".join(sentences)))
            continue

        raw_breaks = choose_chunk_breakpoints_with_agent(sentences, llm)
        breaks = enforce_chunk_size_rules(raw_breaks, len(sentences))
        chunked_docs.extend(build_chunks_from_breakpoints(sentences, breaks, file_name))

    return chunked_docs


def create_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    """Create BM25 index over chunk texts."""
    tokenized_chunks = [tokenize_for_bm25(chunk.text) for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


def create_agent_tools(chunks: list[Chunk], bm25: BM25Okapi, top_k: int) -> list:
    """Create retrieval tools consumed by the ReAct agent."""

    @tool
    def search_bm25_chunks(query: str, k: int = top_k) -> str:
        """Search the BM25 index and return top matching chunks."""
        if not query.strip():
            return "Query is empty."

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return "Query does not contain valid alphanumeric tokens."

        scores = bm25.get_scores(query_tokens)
        effective_k = max(1, min(k, len(chunks)))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:effective_k]

        lines = [f"Top {effective_k} BM25 chunks:"]
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = chunks[idx]
            preview = re.sub(r"\s+", " ", chunk.text).strip()
            preview = preview[:300] + "..." if len(preview) > 300 else preview
            lines.append(
                f"[{rank}] score={scores[idx]:.4f} | source={chunk.file_name} | chunk={chunk.chunk_index}"
            )
            lines.append(preview)
            lines.append("-")

        return "\n".join(lines)

    @tool
    def get_chunk_count() -> str:
        """Return number of chunks currently indexed by BM25."""
        return f"Chunks in BM25 index: {len(chunks)}"

    return [search_bm25_chunks, get_chunk_count]


def create_rag_agent(chunks: list[Chunk], bm25: BM25Okapi, top_k: int) -> Any:
    """Build a ReAct-style answering agent with BM25 retrieval tools."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        max_tokens=ANSWERING_LLM_MAX_OUTPUT_TOKENS,
    )
    tools = create_agent_tools(chunks, bm25, top_k)
    return create_react_agent(model=llm, tools=tools)


def extract_final_text(agent_result: dict[str, Any]) -> str:
    """Extract assistant text from LangGraph agent output."""
    messages = agent_result.get("messages", [])
    if messages:
        final_message = messages[-1]
        content = getattr(final_message, "content", "")
        if isinstance(content, list):
            return "\n".join(str(part) for part in content)
        return str(content)
    return str(agent_result)


def run_agentic_bm25_pipeline(query: str, texts_dir: Path, top_k: int) -> str:
    """Run agentic chunking -> BM25 indexing -> agentic retrieval answering."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    print("=" * 60)
    print("AGENTIC BM25 PIPELINE - AGENTIC CHUNKING + AGENTIC RETRIEVAL")
    print("=" * 60)

    print("\n[1] Loading source documents...")
    docs = load_markdown_documents(texts_dir)
    print(f"Loaded {len(docs)} documents")

    print("\n[2] Building agentic chunks...")
    chunking_llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        max_tokens=CHUNKING_LLM_MAX_OUTPUT_TOKENS,
    )
    chunks = agentic_chunk_documents(docs, chunking_llm)
    print(f"Created {len(chunks)} agentic chunks")

    print("\n[3] Building BM25 index...")
    bm25 = create_bm25_index(chunks)
    print("BM25 index ready")

    print("\n[4] Creating ReAct retrieval agent...")
    rag_agent = create_rag_agent(chunks, bm25, top_k)

    print(f"\n[5] Query: {query}")
    result = rag_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use the available tools to retrieve evidence before answering. "
                        f"Question: {query}"
                    )
                )
            ]
        }
    )

    answer = extract_final_text(result)
    print("\nAnswer:")
    print(answer)
    return answer


def main() -> None:
    load_dotenv_file()
    args = parse_args()
    texts_dir = resolve_texts_dir(args.texts_dir)
    run_agentic_bm25_pipeline(args.query, texts_dir, args.top_k)


if __name__ == "__main__":
    main()