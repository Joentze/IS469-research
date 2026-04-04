"""OpenAI embedding wrapper implementing the Embedder interface."""

from typing import List

from langchain.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


class OpenAIEmbedder(Embeddings):
    """Wrapper around LangChain's OpenAIEmbeddings."""

    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize the OpenAI embedder.

        Args:
            model: OpenAI embedding model name
        """
        self.model = model
        self._embedder = OpenAIEmbeddings(model=model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts."""
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self._embedder.embed_query(text)
