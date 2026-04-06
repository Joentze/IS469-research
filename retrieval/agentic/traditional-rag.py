"""
Traditional RAG pipeline with agentic chunking.
 
Reuses the persisted Chroma vector store from  agentic chunking
pipeline, and retrieves relevant chunks
via similarity search (no agent).
 
References:
- ChromaDB handoff guide: retrieval/agentic/chroma_.md
- ChromaDB: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
"""
 
import os
import json
import re
from pathlib import Path
from typing import Any
 
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
 
 
# ============================================================================
# CONFIGURATION
# ============================================================================
 
 
def load_project_env(project_root: Path) -> None:
    """Load key-value pairs from .env into process environment."""
    env_path = project_root / ".env"
    if not env_path.exists() or not env_path.is_file():
        return
 
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
 
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
 
    if "OPENAI_API_KEY" not in os.environ and "API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]
 
 
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
COLLECTION_NAME = "agentic_bm25_chunks_collection"
PROJECT_ROOT = Path(__file__).parents[2]
load_project_env(PROJECT_ROOT)
DEFAULT_TEXTS_DIR = PROJECT_ROOT / "texts"
TEXTS_DIR = os.getenv("TEXTS_DIR", str(DEFAULT_TEXTS_DIR))
DEFAULT_VECTOR_STORE_DIR = PROJECT_ROOT / "database_agentic"
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", str(DEFAULT_VECTOR_STORE_DIR))
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "false").lower() == "true"
MAX_SENTENCES_PER_CHUNK = 6
MIN_SENTENCES_PER_CHUNK = 2
OVERLAP_SENTENCES = 1
CHUNK_PLANNING_WINDOW_SENTENCES = 80
CHUNK_PLANNING_WINDOW_OVERLAP = 10
CHUNK_PLANNING_MAX_SENTENCE_CHARS = 500
BATCH_SIZE = 5000
TOP_K = 5               # Number of chunks to retrieve


# ============================================================================
# DOCUMENTS + AGENTIC CHUNKING
# ============================================================================


def load_sample_documents() -> list[Document]:
    """Load markdown documents from the texts directory."""
    configured_path = Path(TEXTS_DIR).expanduser()
    if not configured_path.exists() or not configured_path.is_dir():
        raise FileNotFoundError(
            f"Could not find texts directory: {configured_path}. "
            "Set TEXTS_DIR to a valid folder path."
        )

    markdown_files = sorted(configured_path.glob("*.md"))
    if not markdown_files:
        raise ValueError(f"No .md files found in texts directory: {configured_path}")

    documents: list[Document] = []
    for file_path in markdown_files:
        content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={"source": file_path.name, "path": str(file_path)},
            )
        )

    return documents


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
    """Deterministic fallback if the chunking agent output is invalid."""
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
        "You are a chunking planner for a RAG system.\n"
        "Group adjacent sentences into coherent chunks by topic/idea shifts.\n"
        "Rules:\n"
        f"- Every chunk should have {MIN_SENTENCES_PER_CHUNK} to {MAX_SENTENCES_PER_CHUNK} sentences when possible.\n"
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


def choose_chunk_breakpoints_with_agent(sentences: list[str], llm: ChatOpenAI) -> list[int]:
    """Ask an LLM to propose chunk boundaries as sentence-end indexes."""
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
    metadata: dict[str, Any],
) -> list[Document]:
    """Materialize chunk documents from sentence breakpoints."""
    chunks: list[Document] = []
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

        if OVERLAP_SENTENCES > 0:
            start = max(brk + 1 - OVERLAP_SENTENCES, 0)
        else:
            start = brk + 1

        if start >= len(sentences):
            break

    return chunks


def agentic_chunk_documents(documents: list[Document], llm: ChatOpenAI) -> list[Document]:
    """Chunk documents with LLM-selected semantic/topic boundaries."""
    chunked_docs: list[Document] = []

    for doc in documents:
        sentences = split_into_sentences(doc.page_content)
        if not sentences:
            continue
        if len(sentences) <= MAX_SENTENCES_PER_CHUNK:
            chunked_docs.append(
                Document(
                    page_content=" ".join(sentences),
                    metadata={**doc.metadata, "chunk_index": 0, "chunk_method": "agentic"},
                )
            )
            continue

        raw_breaks = choose_chunk_breakpoints_with_agent(sentences, llm)
        breaks = enforce_chunk_size_rules(raw_breaks, len(sentences))
        chunked_docs.extend(build_chunks_from_breakpoints(sentences, breaks, doc.metadata))

    return chunked_docs
 
 
# ============================================================================
# VECTOR STORE
# ============================================================================
 
 
def load_vector_store() -> Chroma:
    """
    Load the persisted Chroma vector store from Gabriel's agentic chunking pipeline.
 
    No re-indexing needed — chunks are already embedded and stored.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
 
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR,
    )
 
    count = vector_store._collection.count()

    if count == 0 or FORCE_REINDEX:
        if FORCE_REINDEX and count > 0:
            print("FORCE_REINDEX=true detected. Rebuilding vector store...")
            vector_store.delete_collection()
            vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=VECTOR_STORE_DIR,
            )

        print("Vector store is empty. Building agentic chunks and indexing now...")
        docs = load_sample_documents()
        chunking_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        chunked_docs = agentic_chunk_documents(docs, chunking_llm)

        if not chunked_docs:
            raise ValueError("No chunks were created from source documents.")

        for start in range(0, len(chunked_docs), BATCH_SIZE):
            batch = chunked_docs[start : start + BATCH_SIZE]
            vector_store.add_documents(batch)

        count = vector_store._collection.count()
 
    print(f"Loaded existing vector store with {count} chunks.")
    return vector_store
 
 
# ============================================================================
# RETRIEVAL
# ============================================================================
 
 
def retrieve(vector_store: Chroma, query: str) -> list[Document]:
    """
    Retrieve the top-k most relevant chunks for a query.
 
    Unlike agentic RAG, traditional RAG retrieves directly via
    similarity search — no agent or reasoning loop involved.
 
    Args:
        vector_store: ChromaDB instance
        query: User query string
 
    Returns:
        List of top-k relevant documents
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    results = retriever.invoke(query)
    return results
 
 
# ============================================================================
# PIPELINE
# ============================================================================
 
 
def run_traditional_rag_pipeline(query: str) -> list[Document]:
    """
    Main pipeline: Load existing agentic chunks from Chroma and retrieve.
 
    Chunking and indexing are skipped — reusing Gabriel's persisted store.
 
    Args:
        query: User query to process
 
    Returns:
        List of retrieved documents
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
 
    print("=" * 60)
    print("TRADITIONAL RAG PIPELINE - AGENTIC CHUNKING")
    print("=" * 60)
 
    # Step 1: Load existing vector store (no chunking or indexing needed)
    print("\n[1] Loading agentic chunk vector store...")
    vector_store = load_vector_store()
 
    # Step 2: Retrieve relevant chunks
    print(f"\n[2] Retrieving top {TOP_K} chunks for query: '{query}'")
    print("-" * 60)
 
    results = retrieve(vector_store, query)
 
    print("\nRetrieved Chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\n[Chunk {i}]")
        print(f"Content: {doc.page_content[:500]}...")
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")
 
    return results
 
 
# ============================================================================
# ENTRY POINT
# ============================================================================
 
 
if __name__ == "__main__":
    test_query = "What is Amazon's year-over-year change in revenue from FY2016 to FY2017 (in units of percents and round to one decimal place)? Calculate what was asked by utilizing the line items clearly shown in the statement of income."
    run_traditional_rag_pipeline(test_query)
 
