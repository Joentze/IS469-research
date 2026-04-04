"""
Example 3: Agentic Chunking + HyDE Query Transformation + ChromaDB

This demonstrates:
- Agentic chunking using LLM to decide topic boundaries
- HyDE query transformation (hypothetical document generation)
- ChromaDB retrieval with transformed queries
"""

from pathlib import Path

from chunkers.agentic import AgenticChunker
from core.pipeline import IndexingPipeline, RetrievalPipeline
from embeddings.openai import OpenAIEmbedder
from generators.react_agent import ReactAgentGenerator
from indexes.chroma import ChromaIndex
from query_transformers.hyde import HyDETransformer
from retrievers.chroma import ChromaRetriever
from utils.env import require_openai_api_key
from dotenv import load_dotenv; load_dotenv()


def main():
    # Setup
    require_openai_api_key("example agentic indexing")
    project_root = Path(__file__).parents[1]
    texts_dir = project_root / "texts"

    # ===== INDEXING PHASE =====
    print("=" * 60)
    print("INDEXING: Agentic Chunking + ChromaDB")
    print("=" * 60)

    # Create components
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    chunker = AgenticChunker(
        llm_model="gpt-4o-mini",
        min_sentences_per_chunk=2,
        max_sentences_per_chunk=6,
        overlap_sentences=1,
    )

    chroma_index = ChromaIndex(
        collection_name="example_agentic_chunks",
        persist_directory="../database/example_hyde",
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
    print("RETRIEVAL: HyDE Transformation + ReAct Agent")
    print("=" * 60)

    # Create components
    query_transformer = HyDETransformer(llm_model="gpt-4o-mini")
    retriever = ChromaRetriever(index=chroma_index, query_transformer=query_transformer)
    generator = ReactAgentGenerator(llm_model="gpt-4o-mini", max_tokens=250)

    # Create and run retrieval pipeline
    retrieval_pipeline = RetrievalPipeline(
        retriever=retriever,
        generator=generator,
        indexes=[chroma_index],
    )

    # Example query
    query = "What is Amazon's revenue change?"
    result = retrieval_pipeline.answer(query, k=5)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Query: {result['query']}")
    print(f"\nRetrieved {len(result['retrieved_docs'])} documents")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
