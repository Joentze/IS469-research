"""
Agentic retrieval pipeline with agentic chunking.

This script can:
- choose topic-aware chunk boundaries with an LLM,
- index chunks with BM25,
- persist chunks in ChromaDB for sharing,
- run a ReAct-style retrieval agent over BM25, Chroma, or both.

References:
- BM25: https://docs.langchain.com/oss/python/integrations/retrievers/bm25
- ChromaDB: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
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

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNKING_LLM_MAX_OUTPUT_TOKENS = 150
ANSWERING_LLM_MAX_OUTPUT_TOKENS = 250

# Paths and defaults
DEFAULT_TEXTS_DIR = Path(__file__).resolve().parents[2] / "texts"
DEFAULT_QUERY = "Explain how cloud revenue trends changed across companies."
DEFAULT_TOP_K = 5
DEFAULT_RETRIEVER_MODE = "bm25"
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parents[2] / "database" / "bm25_agentic"
DEFAULT_CHROMA_COLLECTION = "agentic_bm25_chunks_collection"
RETRIEVER_MODES = ("bm25", "chroma", "hybrid")
CHROMA_SAFE_BATCH_SIZE = 5000
CHROMA_READ_BATCH_SIZE = 5000

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


def _parse_bool_env(value: str | None) -> bool | None:
    """Parse an environment variable into bool when possible."""
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


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


def resolve_retriever_mode(cli_mode: str | None = None) -> str:
    """Resolve retriever mode from CLI, then env var, then default value."""
    if cli_mode is not None:
        return cli_mode

    env_mode = os.getenv("BM25_RETRIEVER_MODE", "").strip().lower()
    if env_mode in RETRIEVER_MODES:
        return env_mode

    return DEFAULT_RETRIEVER_MODE


def resolve_chroma_dir(cli_chroma_dir: Path | None = None) -> Path:
    """Resolve Chroma persistence directory from CLI, env var, then default path."""
    if cli_chroma_dir is not None:
        return cli_chroma_dir.expanduser()

    env_chroma_dir = os.getenv("BM25_CHROMA_DIR")
    if env_chroma_dir:
        return Path(env_chroma_dir).expanduser()

    return DEFAULT_CHROMA_DIR


def resolve_collection_name(cli_collection_name: str | None = None) -> str:
    """Resolve Chroma collection name from CLI, env var, then default value."""
    if cli_collection_name:
        return cli_collection_name

    env_collection_name = os.getenv("BM25_CHROMA_COLLECTION", "").strip()
    if env_collection_name:
        return env_collection_name

    return DEFAULT_CHROMA_COLLECTION


def resolve_reuse_existing_store(cli_reuse_existing_store: bool | None = None) -> bool:
    """Resolve Chroma reuse behavior from CLI, env var, then default value."""
    if cli_reuse_existing_store is not None:
        return cli_reuse_existing_store

    env_value = _parse_bool_env(os.getenv("BM25_REUSE_EXISTING_STORE"))
    if env_value is not None:
        return env_value

    return False


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
    parser.add_argument(
        "--retriever-mode",
        choices=RETRIEVER_MODES,
        default=None,
        help="Retrieval backend to expose to the agent: bm25, chroma, or hybrid",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=None,
        help="Directory to persist/load Chroma data",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Chroma collection name used for persisted chunks",
    )
    parser.add_argument(
        "--reuse-existing-store",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse an existing Chroma collection instead of rebuilding from source docs",
    )
    parser.add_argument(
        "--retrieval-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return retrieval outputs directly without calling the answering LLM",
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


def chunks_to_documents(chunks: list[Chunk]) -> list[Document]:
    """Convert chunk dataclasses into LangChain documents for Chroma storage."""
    return [
        Document(
            page_content=chunk.text,
            metadata={
                "source": chunk.file_name,
                "chunk_index": chunk.chunk_index,
                "chunk_method": "agentic",
            },
        )
        for chunk in chunks
    ]


def initialize_vector_store(
    documents: list[Document],
    embeddings: OpenAIEmbeddings | None,
    persist_directory: Path,
    collection_name: str,
    rebuild_collection: bool,
) -> Chroma:
    """Initialize a Chroma collection and optionally rebuild it from documents."""
    persist_directory.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    if rebuild_collection:
        vector_store.delete_collection()

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )

    if documents:
        add_documents_in_batches(vector_store, documents)

    return vector_store


def add_documents_in_batches(
    vector_store: Chroma,
    documents: list[Document],
    batch_size: int = CHROMA_SAFE_BATCH_SIZE,
) -> None:
    """Insert documents into Chroma using bounded batches to avoid upsert limits."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        vector_store.add_documents(batch)


def get_vector_store_count(vector_store: Chroma) -> int:
    """Return the number of records stored in a Chroma collection."""
    payload = vector_store.get(include=[])
    ids = payload.get("ids", [])
    return len(ids)


def load_chunks_from_vector_store(
    vector_store: Chroma,
    batch_size: int = CHROMA_READ_BATCH_SIZE,
) -> list[Chunk]:
    """Rehydrate Chunk objects from persisted Chroma documents and metadata."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    total_records = get_vector_store_count(vector_store)
    chunks: list[Chunk] = []

    for offset in range(0, total_records, batch_size):
        raw = vector_store.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])

        for idx, text in enumerate(documents):
            if not isinstance(text, str) or not text.strip():
                continue

            metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
            file_name = str(metadata.get("source", "unknown"))

            raw_chunk_index = metadata.get("chunk_index", offset + idx)
            try:
                chunk_index = int(raw_chunk_index)
            except (TypeError, ValueError):
                chunk_index = offset + idx

            chunks.append(
                Chunk(
                    file_name=file_name,
                    chunk_index=chunk_index,
                    text=text.strip(),
                )
            )

    return chunks


def create_bm25_tools(chunks: list[Chunk], bm25: BM25Okapi, top_k: int) -> list[Any]:
    """Create BM25 retrieval tools consumed by the ReAct agent."""

    def run_bm25_search(query: str, k: int) -> str:
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
    def search_bm25_chunks(query: str, k: int = top_k) -> str:
        """Search the BM25 index and return top matching chunks."""
        return run_bm25_search(query, k)

    @tool
    def get_bm25_chunk_count() -> str:
        """Return number of chunks currently indexed by BM25."""
        return f"Chunks in BM25 index: {len(chunks)}"

    setattr(search_bm25_chunks, "_direct_search", run_bm25_search)
    return [search_bm25_chunks, get_bm25_chunk_count]


def create_chroma_tools(vector_store: Chroma, top_k: int) -> list[Any]:
    """Create Chroma retrieval tools consumed by the ReAct agent."""

    def run_chroma_search(query: str, k: int) -> str:
        if not query.strip():
            return "Query is empty."

        collection_count = get_vector_store_count(vector_store)
        if collection_count <= 0:
            return "Chroma collection is empty."

        effective_k = max(1, min(k, collection_count))
        results = vector_store.similarity_search_with_score(query, k=effective_k)
        if not results:
            return "No relevant chunks found in Chroma."

        lines = [f"Top {len(results)} Chroma chunks:"]
        for rank, (doc, score) in enumerate(results, start=1):
            source = doc.metadata.get("source", "unknown")
            chunk_index = doc.metadata.get("chunk_index", "unknown")
            preview = re.sub(r"\s+", " ", doc.page_content).strip()
            preview = preview[:300] + "..." if len(preview) > 300 else preview
            lines.append(f"[{rank}] distance={score:.4f} | source={source} | chunk={chunk_index}")
            lines.append(preview)
            lines.append("-")

        return "\n".join(lines)

    @tool
    def search_chroma_chunks(query: str, k: int = top_k) -> str:
        """Search persisted Chroma chunks and return top matching results."""
        return run_chroma_search(query, k)

    @tool
    def get_chroma_chunk_count() -> str:
        """Return number of chunks currently stored in Chroma."""
        return f"Chunks in Chroma collection: {get_vector_store_count(vector_store)}"

    setattr(search_chroma_chunks, "_direct_search", run_chroma_search)
    return [search_chroma_chunks, get_chroma_chunk_count]


def create_agent_tools(
    retriever_mode: str,
    top_k: int,
    chunks: list[Chunk] | None = None,
    bm25: BM25Okapi | None = None,
    vector_store: Chroma | None = None,
) -> list[Any]:
    """Create retrieval tools consumed by the ReAct agent."""
    tools: list[Any] = []

    if retriever_mode in {"bm25", "hybrid"}:
        if not chunks or bm25 is None:
            raise ValueError("BM25 tools requested but BM25 index inputs were not initialized.")
        tools.extend(create_bm25_tools(chunks, bm25, top_k))

    if retriever_mode in {"chroma", "hybrid"}:
        if vector_store is None:
            raise ValueError("Chroma tools requested but vector store was not initialized.")
        tools.extend(create_chroma_tools(vector_store, top_k))

    return tools


def create_rag_agent(tools: list[Any]) -> Any:
    """Build a ReAct-style answering agent with configured retrieval tools."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        max_tokens=ANSWERING_LLM_MAX_OUTPUT_TOKENS,
    )
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


def run_agentic_bm25_pipeline(
    query: str,
    texts_dir: Path,
    top_k: int,
    retriever_mode: str,
    chroma_dir: Path,
    collection_name: str,
    reuse_existing_store: bool,
    retrieval_only: bool,
) -> str:
    """Run agentic chunking/indexing and query the selected retrieval backend(s)."""
    if retriever_mode not in RETRIEVER_MODES:
        raise ValueError(f"Unsupported retriever mode: {retriever_mode}")

    need_bm25 = retriever_mode in {"bm25", "hybrid"}
    need_chroma = retriever_mode in {"chroma", "hybrid"}

    def require_openai_api_key(reason: str) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(f"OPENAI_API_KEY is not set. Required for {reason}.")

    print("=" * 60)
    print("AGENTIC RETRIEVAL PIPELINE - AGENTIC CHUNKING")
    print("=" * 60)
    print(f"Retriever mode: {retriever_mode}")
    print(f"Retrieval-only mode: {retrieval_only}")

    step = 1
    chunks: list[Chunk] = []
    vector_store: Chroma | None = None
    bm25: BM25Okapi | None = None
    if need_chroma:
        require_openai_api_key("Chroma embeddings")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL) if need_chroma else None
    loaded_from_existing_store = False

    if reuse_existing_store:
        print(f"\n[{step}] Loading existing Chroma collection...")
        step += 1
        vector_store = initialize_vector_store(
            documents=[],
            embeddings=embeddings,
            persist_directory=chroma_dir,
            collection_name=collection_name,
            rebuild_collection=False,
        )
        existing_count = get_vector_store_count(vector_store)
        if existing_count > 0:
            loaded_from_existing_store = True
            print(f"Loaded {existing_count} persisted chunks from {chroma_dir}")

            if need_bm25:
                chunks = load_chunks_from_vector_store(vector_store)
                if chunks:
                    print(f"Rehydrated {len(chunks)} chunks for BM25 from persisted Chroma data")
                else:
                    loaded_from_existing_store = False
                    print("Persisted collection had no usable chunk text; rebuilding from source documents")
        else:
            print("Persisted collection is empty; rebuilding from source documents")

    if reuse_existing_store and retrieval_only and not loaded_from_existing_store:
        raise ValueError(
            "Retrieval-only reuse was requested, but no reusable persisted chunks were found. "
            "Build the store first without --reuse-existing-store."
        )

    if not loaded_from_existing_store:
        print(f"\n[{step}] Loading source documents...")
        step += 1
        docs = load_markdown_documents(texts_dir)
        print(f"Loaded {len(docs)} documents")

        print(f"\n[{step}] Building agentic chunks...")
        step += 1
        require_openai_api_key("agentic chunking")
        chunking_llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=CHUNKING_LLM_MAX_OUTPUT_TOKENS,
        )
        chunks = agentic_chunk_documents(docs, chunking_llm)
        if not chunks:
            raise ValueError("No chunks were created from source documents.")
        print(f"Created {len(chunks)} agentic chunks")

        if need_chroma:
            print(f"\n[{step}] Persisting chunks to ChromaDB...")
            step += 1
            vector_store = initialize_vector_store(
                documents=chunks_to_documents(chunks),
                embeddings=embeddings,
                persist_directory=chroma_dir,
                collection_name=collection_name,
                rebuild_collection=True,
            )
            print(
                "Indexed "
                f"{get_vector_store_count(vector_store)} chunks in collection "
                f"'{collection_name}' at {chroma_dir}"
            )

    if need_bm25:
        if not chunks:
            raise ValueError("BM25 mode requires chunks, but none were available.")
        print(f"\n[{step}] Building BM25 index...")
        step += 1
        bm25 = create_bm25_index(chunks)
        print("BM25 index ready")

    if need_chroma and vector_store is None:
        raise ValueError("Chroma mode requires a vector store, but it was not initialized.")

    bm25_tools: list[Any] = []
    chroma_tools: list[Any] = []
    if need_bm25:
        if bm25 is None:
            raise ValueError("BM25 mode requires a BM25 index, but it was not initialized.")
        bm25_tools = create_bm25_tools(chunks, bm25, top_k)
    if need_chroma:
        chroma_tools = create_chroma_tools(vector_store, top_k)

    if retrieval_only:
        print(f"\n[{step}] Running retrieval directly (no answering LLM call)...")
        outputs: list[str] = []
        if bm25_tools:
            bm25_search = bm25_tools[0]
            outputs.append(str(getattr(bm25_search, "_direct_search")(query, top_k)))
        if chroma_tools:
            chroma_search = chroma_tools[0]
            outputs.append(str(getattr(chroma_search, "_direct_search")(query, top_k)))

        answer = "\n\n".join(outputs)
        print("\nRetrieval Output:")
        print(answer)
        return answer

    print(f"\n[{step}] Creating ReAct retrieval agent...")
    step += 1
    require_openai_api_key("answering LLM")
    tools = bm25_tools + chroma_tools
    rag_agent = create_rag_agent(tools)

    print(f"\n[{step}] Query: {query}")
    mode_instruction = {
        "bm25": "Use BM25 tools to retrieve evidence before answering.",
        "chroma": "Use Chroma tools to retrieve evidence before answering.",
        "hybrid": "Use both BM25 and Chroma tools when helpful before answering.",
    }[retriever_mode]

    result = rag_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        f"{mode_instruction} "
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
    retriever_mode = resolve_retriever_mode(args.retriever_mode)
    chroma_dir = resolve_chroma_dir(args.chroma_dir)
    collection_name = resolve_collection_name(args.collection_name)
    reuse_existing_store = resolve_reuse_existing_store(args.reuse_existing_store)

    run_agentic_bm25_pipeline(
        query=args.query,
        texts_dir=texts_dir,
        top_k=args.top_k,
        retriever_mode=retriever_mode,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        reuse_existing_store=reuse_existing_store,
        retrieval_only=args.retrieval_only,
    )


if __name__ == "__main__":
    main()