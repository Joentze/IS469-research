"""ChromaDB vector store index."""

import uuid

import numpy as np
from typing import Any, Iterable, List, Optional, Union

from langchain_chroma import Chroma as ChromaVectorStore
from langchain_core.documents import Document

from core.interfaces import Embeddings, Index, MetadataAwareEmbedder


class ChromaWithMetadataAware(ChromaVectorStore):
    """Extended Chroma vector store that supports metadata-aware embeddings."""

    # This is an almost duplicate method impl from the original Chroma class, 
    # but with an additional functionality to compute embeddings using the entire document
    # if needed. As well as to just insert the documents, if embeddings were already provided.
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the `VectorStore`.

        Args:
            texts: Texts to add to the `VectorStore`.
            metadatas: Optional list of metadatas.
                    When querying, you can filter on this metadata.
            ids: Optional list of IDs. (Items without IDs will be assigned UUIDs)
            embeddings: Optional list of embeddings.
            kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the added texts.

        Raises:
            ValueError: When metadata is incorrect.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        else:
            ids = [id_ if id_ is not None else str(uuid.uuid4()) for id_ in ids]

        embeddings = None
        texts = list(texts)
        if embeddings is not None:
            if len(embeddings) != len(texts):
                raise ValueError("Number of embeddings must match number of texts")
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(texts, metadatas) # type: ignore this type warning. always use our custom metadata-aware embedder
        if metadatas:
            # fill metadatas with empty dicts if somebody
            # did not specify metadata for all texts
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids]
                    if embeddings is not None and len(embeddings) > 0
                    else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas,  # type: ignore[arg-type]
                        embeddings=embeddings_with_metadatas,  # type: ignore[arg-type]
                        documents=texts_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata from the document using "
                            "langchain_community.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg) from e
                    raise e
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self._collection.upsert(
                    embeddings=embeddings_without_metadatas,  # type: ignore[arg-type]
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            self._collection.upsert(
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=texts,
                ids=ids,
            )
        return ids


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
        self._vector_store = ChromaWithMetadataAware(
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
                self._vector_store = ChromaWithMetadataAware(
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
