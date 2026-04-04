"""BM25 retriever."""

import re
from typing import List, Optional

from langchain_core.documents import Document

from core.interfaces import QueryTransformer, Retriever
from indexes.bm25 import BM25Index


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenization for BM25 scoring (word tokens, lowercased)."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class BM25Retriever(Retriever):
    """Retrieves documents from BM25 keyword index."""

    required_index_types: List[str] = ["bm25"]

    def __init__(self, index: BM25Index, query_transformer: Optional[QueryTransformer] = None):
        """
        Initialize BM25Retriever.

        Args:
            index: BM25Index instance
            query_transformer: Optional query transformer (default: identity)
        """
        self.index = index
        self.query_transformer = query_transformer

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve top-k documents using BM25.

        Args:
            query: User query
            k: Number of results to return

        Returns:
            List of retrieved documents
        """
        # Transform query if transformer is provided (e.g., HyDE)
        if self.query_transformer:
            query = self.query_transformer.transform(query)

        bm25 = self.index.get_bm25()
        documents = self.index.get_documents()

        if bm25 is None or not documents:
            raise RuntimeError("BM25Index has not been initialized with documents")

        # Tokenize and score
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)
        effective_k = max(1, min(k, len(documents)))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :effective_k
        ]

        return [documents[i] for i in ranked_indices]
