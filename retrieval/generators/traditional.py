"""Traditional RAG answer generation using simple prompting."""

from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from core.interfaces import Generator


class TraditionalGenerator(Generator):
    """Simple LLM-based answer generation from retrieved documents."""

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """
        Initialize the traditional generator.

        Args:
            llm_model: LLM model to use for generation
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=0)

    def generate(self, query: str, retrieved_docs: List[Document]) -> str:
        """
        Generate an answer based on query and retrieved documents.

        Args:
            query: User query
            retrieved_docs: Retrieved documents to use as context

        Returns:
            Generated answer
        """
        if not retrieved_docs:
            return "No relevant documents found to answer the query."

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Document {i}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        prompt = (
            f"Based on the following retrieved documents, answer the question:\n\n"
            f"Question: {query}\n\n"
            f"Documents:\n{context}\n\n"
            f"Answer:"
        )

        response = self.llm.invoke(prompt)
        return str(getattr(response, "content", "")).strip()
