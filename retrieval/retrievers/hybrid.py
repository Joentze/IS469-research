"""Hybrid/ensemble retrieval combining multiple retrievers."""

from typing import List, Optional

from langchain_core.documents import Document

from core.interfaces import EnsembleRetriever, Retriever


class HybridRetriever(EnsembleRetriever):
    """Combines BM25 and ChromaDB retrievers using Reciprocal Rank Fusion."""

    def __init__(self, retrievers: List[Retriever], fusion_method: str = "rrf"):
        """
        Initialize HybridRetriever.

        Args:
            retrievers: List of retrievers to combine (typically BM25 and Chroma)
            fusion_method: Method to fuse results (only 'rrf' supported)
        """
        super().__init__(retrievers)
        if fusion_method != "rrf":
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        self.fusion_method = fusion_method
