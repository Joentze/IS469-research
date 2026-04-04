"""
Example 1: Fixed Chunking + ChromaDB + Traditional Generation

This demonstrates:
- Simple fixed-size character chunking
- ChromaDB as vector store
- Traditional LLM-based answer generation
"""

from pathlib import Path

from chunkers.fixed import FixedChunker
from core.pipeline import IndexingPipeline, RetrievalPipeline
from embeddings.openai import OpenAIEmbedder
from generators.traditional import TraditionalGenerator
from indexes.chroma import ChromaIndex
from retrievers.chroma import ChromaRetriever
from utils.env import require_openai_api_key
from dotenv import load_dotenv; load_dotenv()


def main():
    # Setup
    require_openai_api_key("example fixed indexing")
    project_root = Path(__file__).parents[1]
    texts_dir = project_root / "texts"

    # ===== INDEXING PHASE =====
    print("=" * 60)
    print("INDEXING: Fixed Chunking + ChromaDB")
    print("=" * 60)

    # Create components
    chunker = FixedChunker(chunk_size=1000, chunk_overlap=200)
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    chroma_index = ChromaIndex(
        collection_name="example_fixed_chunks",
        persist_directory="../database/example_fixed",
        force_rebuild=False,
    )

    # Create and run indexing pipeline
    indexing_pipeline = IndexingPipeline(
        chunker=chunker,
        embedder=embedder,
        indexes=[chroma_index],
    )
    indexing_pipeline.index_directory(texts_dir)

    # ===== RETRIEVAL PHASE =====
    print("\n" + "=" * 60)
    print("RETRIEVAL: Traditional Generation")
    print("=" * 60)

    # Create components
    retriever = ChromaRetriever(index=chroma_index)
    generator = TraditionalGenerator(llm_model="gpt-4o-mini")

    # Create and run retrieval pipeline
    retrieval_pipeline = RetrievalPipeline(
        retriever=retriever,
        generator=generator,
        indexes=[chroma_index],
    )

    # Example query
    query = "What is Amazon's year-over-year change in revenue?"
    result = retrieval_pipeline.answer(query, k=5)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Query: {result['query']}")
    print(f"\nRetrieved {len(result['retrieved_docs'])} documents")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
