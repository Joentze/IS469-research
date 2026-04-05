"""RAG Pipeline orchestrator implementing the complete workflow."""

from typing import List, Optional

from langchain_core.documents import Document

from core.interfaces import Chunker, Embeddings, Generator, Index, QueryTransformer, Retriever
from utils.markdown_loader import load_markdown_documents
from pathlib import Path


class IndexingPipeline:
    """Orchestrates the document indexing phase of the RAG pipeline."""

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embeddings,
        indexes: List[Index],
    ):
        """
        Initialize the indexing pipeline.

        Args:
            chunker: Chunking strategy
            embedder: Embedding strategy
            indexes: List of indexes to populate (supports multiple for hybrid setups)
        """
        self.chunker = chunker
        self.embedder = embedder
        self.indexes = indexes

    def index_documents(self, documents: List[Document]) -> None:
        """
        Process documents: chunk, embed, and add to all indexes.

        Args:
            documents: Raw documents to index
        """
        # Step 1: Chunk documents
        print(f"[Index] Chunking {len(documents)} documents...")
        chunked_docs = self.chunker.chunk(documents)
        print(f"[Index] Created {len(chunked_docs)} chunks")

        # Step 2: Add to all indexes (each index handles embeddings as needed)
        for idx in self.indexes:
            print(f"[Index] Adding chunks to {idx.index_type} index...")
            idx.add_documents(chunked_docs, embeddings=self.embedder)
            count = idx.get_count()
            print(f"[Index] {idx.index_type} index now contains {count} documents")

    def index_directory(self, texts_dir: Path) -> None:
        """
        Load documents from directory and index them.

        Args:
            texts_dir: Directory containing markdown files
        """
        print(f"[Index] Loading documents from {texts_dir}...")
        documents = load_markdown_documents(texts_dir)
        print(f"[Index] Loaded {len(documents)} documents")
        self.index_documents(documents)


class RetrievalPipeline:
    """Orchestrates the retrieval phase of the RAG pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        query_transformer: Optional[QueryTransformer] = None,
        generator: Optional[Generator] = None,
        indexes: Optional[List[Index]] = None,
    ):
        """
        Initialize the retrieval pipeline.

        Args:
            retriever: Retrieval strategy (can be single or ensemble)
            query_transformer: Optional query transformation
            generator: Optional answer generation
            indexes: Optional list of indexes for compatibility validation
        """
        self.retriever = retriever
        self.query_transformer = query_transformer
        self.generator = generator

        # Validate index compatibility
        if indexes:
            self.retriever.validate_index_compatibility(indexes)

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents for a query.

        Args:
            query: User query
            k: Number of results to return

        Returns:
            List of retrieved documents
        """
        print(f"[Retrieval] Retrieving top {k} documents for query: {query[:100]}...")
        results = self.retriever.retrieve(query, k=k)
        print(f"[Retrieval] Retrieved {len(results)} documents")
        return results

    def answer(self, query: str, k: int = 5) -> dict:
        """
        Full retrieval + generation pipeline.

        Args:
            query: User query
            k: Number of documents to retrieve

        Returns:
            Dictionary with 'query', 'retrieved_docs', and 'answer'
        """
        retrieved_docs = self.retrieve(query, k=k)

        answer = ""
        if self.generator:
            print("[Generation] Generating answer...")
            answer = self.generator.generate(query, retrieved_docs)
            print("[Generation] Answer generated")

        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "answer": answer,
        }
