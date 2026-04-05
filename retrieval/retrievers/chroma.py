"""ChromaDB retriever."""

from typing import List, Optional

from langchain_core.documents import Document

from core.interfaces import QueryTransformer, Retriever
from indexes.chroma import ChromaIndex


class ChromaRetriever(Retriever):
    """Retrieves documents from ChromaDB using similarity search."""

    required_index_types: List[str] = ["chroma"]

    def __init__(self, index: ChromaIndex, query_transformer: Optional[QueryTransformer] = None):
        """
        Initialize ChromaRetriever.

        Args:
            index: ChromaIndex instance
            query_transformer: Optional query transformer (default: identity)
        """
        self.index = index
        self.query_transformer = query_transformer

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve top-k documents from ChromaDB.

        Args:
            query: User query
            k: Number of results to return

        Returns:
            List of retrieved documents
        """
        # Transform query if transformer is provided
        if self.query_transformer:
            query = self.query_transformer.transform(query)

        vector_store = self.index.get_vector_store()
        if vector_store is None:
            raise RuntimeError("ChromaIndex has not been initialized with documents")

        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(query)
        return results
