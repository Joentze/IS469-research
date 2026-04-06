from datetime import datetime
import os
from pathlib import Path

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilySearch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DB_DIR = PROJECT_ROOT / "evals" / "database"
RAG_COLLECTION_NAME = "eval_fixed_chunks"
RAG_TOP_K = 5
EMBEDDING_MODEL = "text-embedding-3-small"

_rag_vector_store: Chroma | None = None
_rag_embeddings: OpenAIEmbeddings | None = None

tavily = TavilySearch(
    max_results=5,
    topic="general",
)

# =============================================================================
# TOOLS (not yet implemented)
# =============================================================================


def load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

    if "OPENAI_API_KEY" not in os.environ and "API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]


def get_rag_vector_store() -> Chroma:
    global _rag_embeddings, _rag_vector_store

    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set. Check your .env file.")

    if _rag_vector_store is None:
        _rag_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        _rag_vector_store = Chroma(
            collection_name=RAG_COLLECTION_NAME,
            embedding_function=_rag_embeddings,
            persist_directory=str(EVAL_DB_DIR),
        )
        if _rag_vector_store._collection.count() <= 0:
            raise FileNotFoundError(
                f"Required Chroma collection '{RAG_COLLECTION_NAME}' is missing or empty."
            )

    return _rag_vector_store


@tool("query_rag", description="Query the RAG knowledge base with a question and return relevant document chunks.")
def query_rag(query: str) -> str:
    """Query the vector store / RAG pipeline and return retrieved context."""
    print(
        f"[rag_agent] Querying Chroma collection '{RAG_COLLECTION_NAME}' for: {query}")
    vector_store = get_rag_vector_store()
    results = vector_store.similarity_search_with_score(query, k=RAG_TOP_K)
    if not results:
        print("[rag_agent] No relevant chunks found.")
        return "No relevant chunks found."

    lines = ["Top retrieved chunks:"]
    for idx, (doc, score) in enumerate(results, start=1):
        lines.append(f"[{idx}] score={score:.4f}")
        lines.append(f"source={doc.metadata.get('source', 'unknown')}")
        if "chunk_index" in doc.metadata:
            lines.append(f"chunk_index={doc.metadata['chunk_index']}")
        lines.append(doc.page_content[:1200])
        lines.append("-")
    print(
        f"[rag_agent] Retrieved {len(results)} chunk(s) from the RAG database.")
    return "\n".join(lines)


@tool("search_web", description="Search the web for up-to-date information on a given query.")
def search_web(query: str) -> str:
    """Run a web search and return a summary of the top results."""
    print(f"[web_search_agent] Querying Tavily web search for: {query}")
    result = tavily.invoke({"query": query})
    print(
        f"[web_search_agent] Retrieved {len(result.get('results', []))} web result(s).")
    return result["results"]


# =============================================================================
# SUBAGENTS
# =============================================================================

rag_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[query_rag],
    system_prompt=(
        "You are a retrieval-augmented generation (RAG) specialist. "
        "When given a question, use the `query_rag` tool to fetch relevant "
        "context from the knowledge base, then synthesise a clear, accurate "
        "answer grounded in the retrieved documents. "
        "Always include your final answer in your last message so the "
        "supervisor can relay it to the user."
    ),
)

web_search_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_web],
    system_prompt=(
        "You are a web research specialist. "
        "When given a question, use the `search_web` tool to find current, "
        "authoritative information on the web, then synthesise a concise "
        "and accurate answer from the search results. "
        "Always include your final answer in your last message so the "
        "supervisor can relay it to the user."
    ),
)


# =============================================================================
# SUBAGENT TOOL WRAPPERS (expose subagents as tools for the main agent)
# =============================================================================

@tool(
    "rag_agent",
    description=(
        "Delegate a question to the RAG agent. "
        "Use this when the answer is likely in the internal knowledge base / "
        "document store (e.g. company docs, uploaded PDFs, proprietary data). "
        "Input: a natural-language question."
    ),
)
def call_rag_agent(query: str) -> str:
    result = rag_agent.invoke(
        {"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


@tool(
    "web_search_agent",
    description=(
        "Delegate a question to the web search agent. "
        "Use this when the answer requires up-to-date or publicly available "
        "information from the internet (e.g. recent news, current prices, "
        "live data). "
        "Input: a natural-language question."
    ),
)
def call_web_search_agent(query: str) -> str:
    result = web_search_agent.invoke(
        {"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# =============================================================================
# MAIN ORCHESTRATOR AGENT
# =============================================================================

now = datetime.now()
# Human-readable date text, e.g. "Thursday, 02 April 2026"
today_text = now.strftime("%A, %d %B %Y")
# Or ISO date string: now.strftime("%Y-%m-%d")
# Components: day, month, year = now.day, now.month, now.year  # integers

main_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[call_rag_agent, call_web_search_agent],
    system_prompt=(
        "You are an intelligent orchestrator that routes questions to the "
        "appropriate specialist agent.\n"
        f"Today's date: {today_text}.\n\n"
        "Available agents:\n"
        "- rag_agent: answers questions using the internal knowledge base / "
        "document store. Best for proprietary, domain-specific, or "
        "pre-loaded document content.\n"
        "- web_search_agent: answers questions by searching the live web. "
        "Best for current events, real-time data, or anything not likely "
        "covered by internal documents.\n\n"
        "Routing guidelines:\n"
        "1. If the question is about internal/proprietary content → rag_agent.\n"
        "2. If the question requires recent or publicly available info → web_search_agent.\n"
        "3. If uncertain, try rag_agent first; fall back to web_search_agent "
        "if the result is insufficient.\n"
        "4. You may call both agents and combine their answers when a question "
        "spans both domains.\n\n"
        "Always return a clear, synthesised answer to the user."
    ),
)


# =============================================================================
# ENTRYPOINT
# =============================================================================

def run(user_message: str) -> str:
    """Run the multi-agent pipeline with a user message and return the response."""
    result = main_agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    response = run(
        "Which region had the worst topline performance for MGM during FY2022?")
    print(response)
