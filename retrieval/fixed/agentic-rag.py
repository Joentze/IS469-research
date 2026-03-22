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

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langgraph.prebuilt import create_react_agent


# ============================================================================
# CONFIGURATION
# ============================================================================

CHUNK_SIZE = 1000  # Fixed chunk size in characters
CHUNK_OVERLAP = 100  # Overlap between chunks for context preservation
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
VECTOR_STORE_DIR = "./database"
COLLECTION_NAME = "fixed_chunks_collection"


# ============================================================================
# DOCUMENT LOADING AND CHUNKING
# ============================================================================


def load_sample_documents() -> list[Document]:
    """
    Load sample documents for testing.
    In production, replace with your actual document loading logic.
    """
    docs = [
        Document(
            page_content="""
            Python is a high-level programming language known for its simplicity and readability.
            It supports multiple programming paradigms including procedural, object-oriented, and
            functional programming. Python is widely used in web development, data science,
            artificial intelligence, and automation.
            """,
            metadata={"source": "python_intro.md", "topic": "programming"},
        ),
        Document(
            page_content="""
            LangChain is a framework for developing applications powered by language models.
            It enables the development of applications that are data-aware and agentic.
            LangChain provides integrations with various LLM providers and vector stores,
            making it easier to build RAG pipelines and agents.
            """,
            metadata={"source": "langchain_overview.md", "topic": "ai_frameworks"},
        ),
        Document(
            page_content="""
            Retrieval Augmented Generation (RAG) is a technique that combines information
            retrieval with generative models. It works by retrieving relevant documents
            from a knowledge base and using them to augment the context provided to a
            language model for generating responses.
            """,
            metadata={"source": "rag_explained.md", "topic": "ai_techniques"},
        ),
    ]
    return docs


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

    # Add documents to vector store
    if documents:
        vector_store.add_documents(documents)
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
    def search_knowledge_base(query: str, k: int = 5) -> str:
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

    agent = create_react_agent(llm, tools)
    return agent


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

    response = agent.invoke({"input": query})

    print("-" * 60)
    print("\nAgent Response:")
    print(response["output"])

    return response["output"]


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    # Example query
    test_query = "What is RAG and how does it work?"

    # Run the pipeline
    run_agentic_rag_pipeline(test_query)