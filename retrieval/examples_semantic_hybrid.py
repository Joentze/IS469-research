"""
Example 2: Semantic Chunking + Hybrid Retrieval (BM25 + ChromaDB)

This demonstrates:
- Semantic chunking based on embedding similarity
- Multiple indexes (BM25 sparse + Chroma dense)
- Hybrid retrieval combining results from both indexes
"""
from rich import print
from pathlib import Path

from chunkers.semantic import SemanticChunker
from core.pipeline import IndexingPipeline, RetrievalPipeline
from embeddings.openai import OpenAIEmbedder
from generators.traditional import TraditionalGenerator
from indexes.bm25 import BM25Index
from indexes.chroma import ChromaIndex
from retrievers.bm25 import BM25Retriever
from retrievers.chroma import ChromaRetriever
from retrievers.hybrid import HybridRetriever
from utils.env import require_openai_api_key
from dotenv import load_dotenv; load_dotenv()


def main():
    # Setup
    require_openai_api_key("example semantic indexing")
    project_root = Path(__file__).parents[1]
    texts_dir = project_root / "texts"

    # ===== INDEXING PHASE =====
    print("=" * 60)
    print("INDEXING: Semantic Chunking + Multiple Indexes (BM25 + ChromaDB)")
    print("=" * 60)

    # Create components
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    chunker = SemanticChunker(
        embedder=embedder,
        max_sentences_per_chunk=8,
        overlap_sentences=2,
        breakpoint_percentile=70.0,
    )

    # Create multiple indexes for hybrid search
    bm25_index = BM25Index()
    chroma_index = ChromaIndex(
        collection_name="example_semantic_chunks",
        persist_directory=f"{project_root}/database/example_semantic",
        force_rebuild=False,
    )

    # Create and run indexing pipeline with multiple indexes
    indexing_pipeline = IndexingPipeline(
        chunker=chunker,
        embedder=embedder,
        indexes=[bm25_index, chroma_index],  # Chunks go to both indexes
    )
    indexing_pipeline.index_directory(texts_dir)

    # ===== RETRIEVAL PHASE =====
    print("\n" + "=" * 60)
    print("RETRIEVAL: Hybrid Search (BM25 + ChromaDB)")
    print("=" * 60)

    # Create individual retrievers
    bm25_retriever = BM25Retriever(index=bm25_index)
    chroma_retriever = ChromaRetriever(index=chroma_index)

    # Combine into hybrid retriever
    hybrid_retriever = HybridRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        fusion_method="rrf",
    )

    generator = TraditionalGenerator(llm_model="gpt-4o-mini")

    # Create and run retrieval pipeline
    retrieval_pipeline = RetrievalPipeline(
        retriever=hybrid_retriever,
        generator=generator,
        indexes=[bm25_index, chroma_index],
    )

    # Example query
    query = "Which company reported the highest cloud revenue growth?"
    result = retrieval_pipeline.answer(query, k=5)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Query: {result['query']}")
    print(f"\nRetrieved {len(result['retrieved_docs'])} documents")
    
    # Print retrieved documents
    print("\n" + "-" * 60)
    print("Retrieved Documents:")
    print("-" * 60)
    for i, doc in enumerate(result['retrieved_docs'], 1):
        print(f"\n[{i}] {doc.page_content}")
        if doc.metadata:
            print(f"    Metadata: {doc.metadata}")
    
    print("\n" + "=" * 60)
    print(f"Answer:\n{result['answer']}")


if __name__ == "__main__":
    main()
