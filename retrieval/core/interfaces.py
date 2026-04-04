"""Abstract base classes defining the RAG pipeline strategy interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain.embeddings import Embeddings
from langchain_core.documents import Document


class Chunker(ABC):
    """Abstract chunker strategy."""

    enable_cache: bool = True

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Transform documents into chunks.

        Args:
            documents: Raw documents to chunk

        Returns:
            List of chunked documents
        """
        pass

    def _get_chunker_name(self) -> str:
        """Get the name of this chunker for cache purposes."""
        return self.__class__.__name__


class Index(ABC):
    """Abstract index/vector store strategy."""

    index_type: str = "abstract"

    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: Optional[Embeddings] = None) -> None:
        """Add documents to the index."""
        pass

    @abstractmethod
    def get_count(self) -> int:
        """Return the number of documents in the index."""
        pass


class QueryTransformer(ABC):
    """Abstract query transformation strategy."""

    @abstractmethod
    def transform(self, query: str) -> str:
        """Transform the input query (may generate hypothetical docs, etc.)."""
        pass


class Retriever(ABC):
    """Abstract retriever strategy."""

    required_index_types: List[str] = []  # List of allowed index types for this retriever

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve top-k documents for a query."""
        pass

    def validate_index_compatibility(self, indexes: List[Index]) -> None:
        """Validate that provided indexes match this retriever's requirements."""
        if not self.required_index_types:
            return  # No requirement

        index_types = [idx.index_type for idx in indexes]
        for required_type in self.required_index_types:
            if required_type not in index_types:
                raise ValueError(
                    f"Retriever {self.__class__.__name__} requires index type '{required_type}', "
                    f"but available indexes have types: {index_types}"
                )


class Generator(ABC):
    """Abstract answer generation strategy."""

    @abstractmethod
    def generate(self, query: str, retrieved_docs: List[Document]) -> str:
        """Generate an answer given a query and retrieved documents."""
        pass


class EnsembleRetriever(Retriever):
    """Combines multiple retrievers and fuses results."""

    required_index_types: List[str] = []  # Ensemble is flexible; child retrievers validate

    def __init__(self, retrievers: List[Retriever]):
        """
        Initialize with a list of child retrievers.

        Args:
            retrievers: List of retriever strategies to combine
        """
        self.retrievers = retrievers

    def validate_index_compatibility(self, indexes: List[Index]) -> None:
        """Validate each child retriever against the provided indexes."""
        for retriever in self.retrievers:
            retriever.validate_index_compatibility(indexes)

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Execute all retrievers and fuse results using Reciprocal Rank Fusion.

        Args:
            query: User query
            k: Number of results to return

        Returns:
            Fused top-k documents
        """
        # Collect results from all retrievers
        all_results = {}
        for retriever in self.retrievers:
            docs = retriever.retrieve(query, k=k)
            for rank, doc in enumerate(docs, 1):
                doc_id = id(doc)  # Use object id as key
                if doc_id not in all_results:
                    all_results[doc_id] = {"doc": doc, "scores": []}
                # Reciprocal Rank Fusion: 1 / (rank + 60)
                rrf_score = 1.0 / (rank + 60)
                all_results[doc_id]["scores"].append(rrf_score)

        # Aggregate scores and sort
        fused = []
        for doc_id, item in all_results.items():
            total_score = sum(item["scores"])
            fused.append((item["doc"], total_score))

        # Sort by combined score descending
        fused.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        return [doc for doc, _score in fused[:k]]
