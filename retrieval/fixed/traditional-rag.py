"""
use chroma db as vector store
https://docs.langchain.com/oss/python/integrations/vectorstores/chroma

"""
import os
from pathlib import Path
 
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
 
 
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
 
 
CHUNK_SIZE = 1000       # Max characters per chunk
CHUNK_OVERLAP = 100     # Overlap between chunks for context preservation
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_STORE_DIR = "./database"
COLLECTION_NAME = "trad_rag_fixed_collection"
BATCH_SIZE = 5000
TOP_K = 5               # Number of chunks to retrieve
PROJECT_ROOT = Path(__file__).parents[2]
load_project_env(PROJECT_ROOT)
DEFAULT_TEXTS_DIR = PROJECT_ROOT / "texts"
TEXTS_DIR = os.getenv("TEXTS_DIR", str(DEFAULT_TEXTS_DIR))
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "false").lower() == "true"
 
 
# ============================================================================
# DOCUMENT LOADING AND CHUNKING
# ============================================================================
 
 
def resolve_texts_directory() -> Path:
    """Resolve the texts directory from TEXTS_DIR or project default."""
    configured_path = Path(TEXTS_DIR).expanduser()
 
    if configured_path.exists() and configured_path.is_dir():
        return configured_path
 
    raise FileNotFoundError(
        f"Could not find texts directory: {configured_path}. "
        "Set TEXTS_DIR to a valid folder path."
    )
 
 
def load_documents() -> list[Document]:
    """Load markdown documents from the texts directory."""
    texts_dir = resolve_texts_directory()
    markdown_files = sorted(texts_dir.glob("*.md"))
 
    if not markdown_files:
        raise ValueError(f"No .md files found in texts directory: {texts_dir}")
 
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
 
 
def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into fixed-size chunks with sliding window overlap.

    Splits on newline characters, with overlap between chunks
    to preserve context at boundaries.
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
 
    chunked_docs = splitter.split_documents(documents)
    return chunked_docs
 
 
# ============================================================================
# VECTOR STORE SETUP
# ============================================================================
 
 
def initialize_vector_store(documents: list[Document]) -> Chroma:
    """
    Initialize ChromaDB vector store and add documents.
 
    Args:
        documents: List of chunked documents
 
    Returns:
        Chroma vector store instance
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
 
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR,
    )
 
    existing_count = vector_store._collection.count()
 
    if existing_count > 0 and not FORCE_REINDEX:
        print(
            f"Using existing vector store with {existing_count} chunks. "
            "Skipping re-indexing."
        )
        return vector_store
 
    if FORCE_REINDEX and existing_count > 0:
        print("FORCE_REINDEX=true detected. Rebuilding vector store...")
        vector_store.delete_collection()
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTOR_STORE_DIR,
        )
 
    if documents:
        for start in range(0, len(documents), BATCH_SIZE):
            batch = documents[start : start + BATCH_SIZE]
            vector_store.add_documents(batch)
        print(f"Added {len(documents)} chunks to vector store")
 
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
# MAIN PIPELINE
# ============================================================================
 
 
def run_traditional_rag_pipeline(query: str) -> list[Document]:
    """
    Main pipeline: Load docs, chunk, store, and retrieve.
 
    Args:
        query: User query to process
 
    Returns:
        List of retrieved documents
    """
    print("=" * 60)
    print("TRADITIONAL RAG PIPELINE - FIXED CHUNKING")
    print("=" * 60)
 
    # Step 1: Load documents
    print("\n[1] Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
 
    # Step 2: Chunk with character splitter
    print(f"\n[2] Chunking documents (fixed size: {CHUNK_SIZE} chars, overlap: {CHUNK_OVERLAP})...")
    chunked_docs = chunk_documents(documents)
    print(f"Created {len(chunked_docs)} chunks")
 
    # Step 3: Initialize vector store
    print("\n[3] Initializing vector store...")
    vector_store = initialize_vector_store(chunked_docs)
 
    # Step 4: Retrieve relevant chunks
    print(f"\n[4] Retrieving top {TOP_K} chunks for query: '{query}'")
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
 