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
from pathlib import Path
 
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
 
 
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
COLLECTION_NAME = "agentic_bm25_chunks_collection"
VECTOR_STORE_DIR = "retrieval/agentic/bm25_agentic"
TOP_K = 5               # Number of chunks to retrieve
PROJECT_ROOT = Path(__file__).parents[2]
load_project_env(PROJECT_ROOT)
 
 
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
    if count == 0:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' is empty. "
            "Run Gabriel's agentic chunking pipeline first, or check the path: "
            f"'{VECTOR_STORE_DIR}'."
        )
 
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
 
