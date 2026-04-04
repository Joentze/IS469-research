"""HyDE (Hypothetical Document Embeddings) query transformer."""

from langchain_openai import ChatOpenAI

from core.interfaces import QueryTransformer


class HyDETransformer(QueryTransformer):
    """Generate a hypothetical document that would answer the query."""

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """
        Initialize HyDE transformer.

        Args:
            llm_model: LLM model to use for generating hypothetical documents
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=0)

    def transform(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.

        Args:
            query: User query

        Returns:
            Hypothetical document text
        """
        prompt = (
            "Write a short, factual passage (2-3 sentences) that directly answers "
            "the following question. Do not mention that it is hypothetical.\n\n"
            f"Question: {query}\n\nPassage:"
        )
        response = self.llm.invoke(prompt)
        return str(getattr(response, "content", "")).strip()
