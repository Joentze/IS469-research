"""ChromaDB vector store index."""

from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma as ChromaVectorStore
from langchain_core.documents import Document

from core.interfaces import Embeddings, Index


class ChromaIndex(Index):
    """ChromaDB vector store implementation."""

    index_type: str = "chroma"

    def __init__(
        self,
        collection_name: str = "default_collection",
        persist_directory: str = "./database",
        batch_size: int = 5000,
        force_rebuild: bool = False,
    ):
        """
        Initialize Chroma index.

        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist Chroma data
            batch_size: Batch size for adding documents
            force_rebuild: Whether to rebuild collection from scratch
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.batch_size = batch_size
        self.force_rebuild = force_rebuild
        self._vector_store: Optional[ChromaVectorStore] = None
        self._embedder: Optional[Embeddings] = None

    def add_documents(self, documents: List[Document], embeddings: Optional[Embeddings] = None) -> None:
        """
        Add documents to the Chroma index.

        Args:
            documents: Documents to add
            embeddings: Embeddings instance for computing embeddings
        """
        if embeddings is None:
            raise ValueError("ChromaIndex requires an Embeddings instance")

        self._embedder = embeddings

        # Create or load vector store
        self._vector_store = ChromaVectorStore(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_directory,
        )

        existing_count = self._vector_store._collection.count()

        # Handle force rebuild
        if self.force_rebuild and existing_count > 0:
            print(f"Force rebuild: deleting existing Chroma collection '{self.collection_name}'")
            try:
                self._vector_store.delete_collection()
                self._vector_store = ChromaVectorStore(
                    collection_name=self.collection_name,
                    embedding_function=embeddings,
                    persist_directory=self.persist_directory,
                )
            except Exception as e:
                print(f"Warning: failed to delete collection: {e}")

        # Add documents in batches
        if documents:
            for start in range(0, len(documents), self.batch_size):
                batch = documents[start : start + self.batch_size]
                self._vector_store.add_documents(batch)
            print(
                f"ChromaIndex: Added {len(documents)} documents to collection "
                f"'{self.collection_name}' at {self.persist_directory}"
            )

    def get_count(self) -> int:
        """Return the number of documents in the index."""
        if self._vector_store is None:
            return 0
        return self._vector_store._collection.count()

    def get_vector_store(self) -> Optional[ChromaVectorStore]:
        """Return the underlying Chroma vector store."""
        return self._vector_store
