"""
Agentic RAG Pipeline with Fixed Chunking

Uses fixed chunking (by character count) to split documents,
stores embeddings in ChromaDB, and implements a ReAct agent
with tools to search the vector database.

References:
- ChromaDB: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
- ReAct Agents: https://docs.langchain.com/oss/python/langchain/agents
- Fixed Chunking: https://docs.langchain.com/oss/python/modules/data_connection/document_loaders/split_code
"""

import os
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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

TOP_K = 5 
CHUNK_SIZE = 1000  # Fixed chunk size in characters
CHUNK_OVERLAP = 200  # Overlap between chunks for context preservation
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
VECTOR_STORE_DIR = "./database"
COLLECTION_NAME = "fixed_chunks_collection"
BATCH_SIZE = 5000
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


def load_sample_documents() -> list[Document]:
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


def chunk_documents_fixed(documents: list[Document]) -> list[Document]:
    """
    Split documents into fixed-sized chunks.

    Args:
        documents: List of documents to chunk

    Returns:
        List of chunked documents
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


def initialize_vector_store(
    documents: list[Document],
) -> Chroma:
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

    # Add documents in bounded batches to avoid Chroma max-batch-size errors.
    if documents:
        for start in range(0, len(documents), BATCH_SIZE):
            batch = documents[start : start + BATCH_SIZE]
            vector_store.add_documents(batch)
        print(f"Added {len(documents)} chunks to vector store")

    return vector_store


# ============================================================================
# AGENT TOOLS
# ============================================================================


def create_agent_tools(vector_store: Chroma) -> list:
    """
    Create tools for the RAG agent.

    Args:
        vector_store: ChromaDB instance

    Returns:
        List of tools for the agent
    """

    @tool
    def search_knowledge_base(query: str, k: int = TOP_K) -> str:
        """
        Search the vector store for documents relevant to the query.
        Returns the top-k most similar documents.

        Args:
            query: Search query string
            k: Number of top results to return (default: 5)

        Returns:
            Formatted string with search results
        """
        results = vector_store.similarity_search(query, k=k)

        if not results:
            return "No relevant documents found in the knowledge base."

        formatted_results = "Search Results:\n" + "=" * 50 + "\n"
        for i, doc in enumerate(results, 1):
            formatted_results += f"\n[Document {i}]\n"
            formatted_results += f"Content: {doc.page_content[:500]}...\n"
            if doc.metadata:
                formatted_results += f"Metadata: {doc.metadata}\n"

        return formatted_results

    @tool
    def get_document_count() -> str:
        """
        Get the total number of chunks in the vector store.

        Returns:
            Number of documents in the collection
        """
        collection = vector_store._collection
        count = collection.count()
        return f"Total chunks in knowledge base: {count}"

    return [search_knowledge_base, get_document_count]


# ============================================================================
# AGENT SETUP
# ============================================================================


def create_rag_agent(vector_store: Chroma) -> Any:
    """
    Create a ReAct agent with RAG tools.

    Args:
        vector_store: ChromaDB instance

    Returns:
        Compiled agent
    """
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    tools = create_agent_tools(vector_store)

    agent = create_agent(model=llm, tools=tools)
    return agent


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


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run_agentic_rag_pipeline(query: str) -> str:
    """
    Main pipeline: Load docs, chunk, store, and query with agent.

    Args:
        query: User query to process

    Returns:
        Agent response
    """
    print("=" * 60)
    print("AGENTIC RAG PIPELINE - FIXED CHUNKING")
    print("=" * 60)

    # Step 1: Load documents
    print("\n[1] Loading documents...")
    documents = load_sample_documents()
    print(f"Loaded {len(documents)} documents")

    # Step 2: Chunk with fixed size
    print(f"\n[2] Chunking documents (fixed size: {CHUNK_SIZE} chars)...")
    chunked_docs = chunk_documents_fixed(documents)
    print(f"Created {len(chunked_docs)} chunks")

    # Step 3: Initialize vector store
    print("\n[3] Initializing vector store...")
    vector_store = initialize_vector_store(chunked_docs)

    # Step 4: Create and run agent
    print("\n[4] Creating ReAct agent with RAG tools...")
    agent = create_rag_agent(vector_store)

    print(f"\n[5] Running query: '{query}'")
    print("-" * 60)

    response = agent.invoke({"messages": [HumanMessage(content=query)]})
    answer = extract_final_text(response)

    print("-" * 60)
    print("\nAgent Response:")
    print(answer)

    return answer


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    # Example query
    test_query = "What is Amazon's year-over-year change in revenue from FY2016 to FY2017 (in units of percents and round to one decimal place)? Calculate what was asked by utilizing the line items clearly shown in the statement of income."
    # Run the pipeline
    run_agentic_rag_pipeline(test_query)