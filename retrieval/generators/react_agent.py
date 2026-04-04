"""ReAct Agent-based answer generation with tool use."""

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from core.interfaces import Generator


class ReactAgentGenerator(Generator):
    """ReAct agent that answers questions with retrieved documents as tools."""

    def __init__(self, llm_model: str = "gpt-4o-mini", max_tokens: int = 250):
        """
        Initialize the ReAct agent generator.

        Args:
            llm_model: LLM model to use
            max_tokens: Maximum tokens for generation
        """
        self.llm_model = llm_model
        self.max_tokens = max_tokens

    def generate(self, query: str, retrieved_docs: List[Document]) -> str:
        """
        Generate an answer using a ReAct agent with retrieved documents.

        Args:
            query: User query
            retrieved_docs: Retrieved documents

        Returns:
            Generated answer
        """
        if not retrieved_docs:
            return "No relevant documents found to answer the query."

        # Create tools from retrieved documents
        @tool
        def get_documents() -> str:
            """Get the retrieved context documents."""
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                context_parts.append(f"[Document {i}]\n{doc.page_content}")
            return "\n\n".join(context_parts)

        @tool
        def document_count() -> str:
            """Get the count of retrieved documents."""
            return f"Total retrieved documents: {len(retrieved_docs)}"

        tools = [get_documents, document_count]

        # Create and run agent
        llm = ChatOpenAI(model=self.llm_model, temperature=0, max_tokens=self.max_tokens) # type: ignore max_tokens SHOULD be supported
        agent = create_agent(model=llm, tools=tools)

        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        # Extract final message
        messages = result.get("messages", [])
        if messages:
            final_message = messages[-1]
            content = getattr(final_message, "content", "")
            if isinstance(content, list):
                return "\n".join(str(part) for part in content)
            return str(content)

        return str(result)
