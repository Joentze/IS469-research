"""
Agentic RAG pipeline with semantic chunking.

Semantic chunking groups adjacent sentences by embedding similarity,
stores chunks in ChromaDB, and uses a ReAct-style agent to answer queries
with retrieval tools.

References:
- ChromaDB: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
- ReAct agents: https://docs.langchain.com/oss/python/langchain/agents
"""

import os
import re
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
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
VECTOR_STORE_DIR = "./database"
COLLECTION_NAME = "semantic_chunks_collection"
BATCH_SIZE = 5000
PROJECT_ROOT = Path(__file__).parents[2]
load_project_env(PROJECT_ROOT)
DEFAULT_TEXTS_DIR = PROJECT_ROOT / "texts"
TEXTS_DIR = os.getenv("TEXTS_DIR", str(DEFAULT_TEXTS_DIR))
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "false").lower() == "true"

MAX_SENTENCES_PER_CHUNK = 6
OVERLAP_SENTENCES = 1
BREAKPOINT_PERCENTILE = 30.0
TOP_K = 5


# ============================================================================
# DOCUMENTS
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
        raise ValueError(
            f"No .md files found in texts directory: {configured_path}")

    documents: list[Document] = []
    for file_path in markdown_files:
        content = file_path.read_text(
            encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={"source": file_path.name, "path": str(file_path)},
            )
        )

    return documents


# ============================================================================
# SEMANTIC CHUNKING
# ============================================================================


def split_into_sentences(text: str) -> list[str]:
    """Split text into simple sentence-like units."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s and s.strip()]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def percentile(values: list[float], q: float) -> float:
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


def semantic_chunk_documents(
        documents: list[Document],
        embeddings: OpenAIEmbeddings,
) -> list[Document]:
    """
    Chunk documents by semantic boundaries.

    We split text into sentences, embed each sentence, and insert chunk breaks
    where adjacent sentence similarity drops below a percentile threshold.
    """
    chunked_docs: list[Document] = []

    for doc in documents:
        sentences = split_into_sentences(doc.page_content)
        if not sentences:
            continue
        if len(sentences) == 1:
            chunked_docs.append(doc)
            continue

        sentence_vectors = embeddings.embed_documents(sentences)
        adjacent_similarities = [
            cosine_similarity(sentence_vectors[i], sentence_vectors[i + 1])
            for i in range(len(sentence_vectors) - 1)
        ]
        breakpoint_threshold = percentile(
            adjacent_similarities, BREAKPOINT_PERCENTILE)

        break_after_index = {
            i
            for i, score in enumerate(adjacent_similarities)
            if score <= breakpoint_threshold
        }

        current: list[str] = []
        chunk_index = 0

        for i, sentence in enumerate(sentences):
            current.append(sentence)

            reached_max = len(current) >= MAX_SENTENCES_PER_CHUNK
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

                overlap = current[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES > 0 else []
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

    return chunked_docs


# ============================================================================
# VECTOR STORE
# ============================================================================


def initialize_vector_store(documents: list[Document], embeddings: OpenAIEmbeddings) -> Chroma:
    """Create a Chroma collection and load semantic chunks."""
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
            batch = documents[start: start + BATCH_SIZE]
            vector_store.add_documents(batch)
        print(f"Added {len(documents)} chunks to vector store")

    return vector_store


# ============================================================================
# TOOLS + AGENT
# ============================================================================


def create_agent_tools(vector_store: Chroma) -> list:
    """Create retrieval tools used by the ReAct agent."""

<<<<<<< Updated upstream
    @tool
    def search_knowledge_base(query: str, k: int = 4) -> str:
        """Search the semantic chunk index and return the top matching chunks."""
        results = vector_store.similarity_search_with_score(query, k=k)
        if not results:
            return "No relevant chunks found."
=======
	@tool
	def search_knowledge_base(query: str, k: int = int TOP_K) -> str:
		"""Search the semantic chunk index and return the top matching chunks."""
		results = vector_store.similarity_search_with_score(query, k=k)
		if not results:
			return "No relevant chunks found."
>>>>>>> Stashed changes

        lines = ["Top semantic chunks:"]
        for i, (doc, score) in enumerate(results, start=1):
            lines.append(f"[{i}] score={score:.4f}")
            lines.append(f"source={doc.metadata.get('source', 'unknown')}")
            lines.append(doc.page_content)
            lines.append("-")
        return "\n".join(lines)

    @tool
    def get_chunk_count() -> str:
        """Return number of chunks currently stored in Chroma."""
        return f"Chunks in vector store: {vector_store._collection.count()}"

    return [search_knowledge_base, get_chunk_count]


def create_rag_agent(vector_store: Chroma) -> Any:
    """Build a ReAct-style agent with semantic retrieval tools."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    tools = create_agent_tools(vector_store)
    return create_agent(model=llm, tools=tools)


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
# PIPELINE
# ============================================================================


def run_agentic_rag_pipeline(query: str) -> str:
    """Run semantic chunking -> indexing -> retrieval-based answering."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    print("=" * 60)
    print("AGENTIC RAG PIPELINE - SEMANTIC CHUNKING")
    print("=" * 60)

    print("\n[1] Loading source documents...")
    docs = load_sample_documents()
    print(f"Loaded {len(docs)} documents")

    print("\n[2] Building semantic chunks...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    semantic_chunks = semantic_chunk_documents(docs, embeddings)
    print(f"Created {len(semantic_chunks)} semantic chunks")

    print("\n[3] Initializing ChromaDB...")
    vector_store = initialize_vector_store(semantic_chunks, embeddings)
    print(f"Indexed {vector_store._collection.count()} chunks")

    print("\n[4] Creating ReAct agent...")
    agent = create_rag_agent(vector_store)

    print(f"\n[5] Query: {query}")
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    answer = extract_final_text(result)

    print("\nAnswer:")
    print(answer)
    return answer


if __name__ == "__main__":
    test_query = "What is Amazon's year-over-year change in revenue from FY2016 to FY2017 (in units of percents and round to one decimal place)? Calculate what was asked by utilizing the line items clearly shown in the statement of income."
    run_agentic_rag_pipeline(test_query)
