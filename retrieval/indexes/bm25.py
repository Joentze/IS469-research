"""BM25 keyword index."""

import re
from typing import List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from core.interfaces import Embeddings, Index


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenization for BM25 scoring (word tokens, lowercased)."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class BM25Index(Index):
    """BM25 keyword index implementation."""

    index_type: str = "bm25"

    def __init__(self):
        """Initialize BM25 index."""
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[Document] = []
        self._tokenized_docs: List[List[str]] = []

    def add_documents(self, documents: List[Document], embeddings: Optional[Embeddings] = None) -> None:
        """
        Add documents to the BM25 index.

        Note: BM25 does not use embeddings; it's a keyword-based sparse index.

        Args:
            documents: Documents to add
            embeddings: Ignored for BM25
        """
        self._documents = documents
        self._tokenized_docs = [tokenize_for_bm25(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized_docs)
        print(f"BM25Index: Built index with {len(documents)} documents")

    def get_count(self) -> int:
        """Return the number of documents in the index."""
        return len(self._documents)

    def get_bm25(self) -> Optional[BM25Okapi]:
        """Return the underlying BM25 object."""
        return self._bm25

    def get_documents(self) -> List[Document]:
        """Return the stored documents."""
        return self._documents
