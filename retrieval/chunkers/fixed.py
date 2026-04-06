"""Fixed-size character chunking strategy."""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from core.interfaces import Chunker
from utils.cache import get_global_cache


class FixedChunker(Chunker):
    """Fixed-size character chunking with sliding window overlap."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        enable_cache: bool = True,
    ):
        """
        Initialize the fixed chunker.

        Args:
            chunk_size: Max characters per chunk
            chunk_overlap: Overlap between chunks for context preservation
            enable_cache: Whether to cache chunking results
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_cache = enable_cache
        self.splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into fixed-size chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        # Check cache first
        if self.enable_cache:
            cache = get_global_cache()
            cached_chunks = cache.get(self._get_chunker_name(), documents)
            if cached_chunks is not None:
                return cached_chunks

        # Process chunks
        chunks = self.splitter.split_documents(documents)

        # Store in cache
        if self.enable_cache:
            cache = get_global_cache()
            cache.put(self._get_chunker_name(), chunks, documents)

        return chunks
