from langchain.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

tavily = TavilySearch(
    max_results=5,
    topic="general",
)

# =============================================================================
# TOOLS (not yet implemented)
# =============================================================================


@tool("query_rag", description="Query the RAG knowledge base with a question and return relevant document chunks.")
def query_rag(query: str) -> str:
    """Query the vector store / RAG pipeline and return retrieved context."""
    # TODO: implement RAG retrieval (e.g. embed query, search vector DB, return chunks)
    raise NotImplementedError


@tool("search_web", description="Search the web for up-to-date information on a given query.")
def search_web(query: str) -> str:
    """Run a web search and return a summary of the top results."""
    # TODO: implement web search (e.g. via Tavily, SerpAPI, or Bing Search API)
    result = tavily.invoke({"query": query})
    print(result)
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
from datetime import datetime

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
    response = run("what is the latest news in iran")
    print(response)
