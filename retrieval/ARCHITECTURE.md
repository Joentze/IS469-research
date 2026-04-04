"""
ARCHITECTURE REFERENCE: Modular RAG Pipeline

This document visualizes how components fit together and how data flows through the system.
"""

# ==============================================================================
# HIGH-LEVEL WORKFLOW
# ==============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE WORKFLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: INDEXING (Build the knowledge base)                               │
│  ──────────────────────────────────────────────                             │
│                                                                              │
│  1. Load Documents (loaders/markdown.py)                                    │
│     Raw .md files → List[Document]                                          │
│                                                                              │
│  2. Chunk Documents (chunkers/*.py)                                         │
│     List[Document] → Chunker.chunk() → List[Document]                      │
│     - FixedChunker: character-based sliding window                          │
│     - SemanticChunker: embedding similarity thresholds                      │
│     - AgenticChunker: LLM-guided topic boundaries                           │
│                                                                              │
│  3. Compute Embeddings (embeddings/openai.py)                               │
│     For each chunk: text → Embedder → vector                               │
│                                                                              │
│  4. Add to Indexes (indexes/*.py)                                           │
│     For each Index in [BM25Index, ChromaIndex, ...]:                       │
│       Index.add_documents(chunked_docs, embedder)                           │
│                                                                              │
│     ChromaIndex: Stores embeddings + metadata in Chroma                     │
│     BM25Index: Builds sparse BM25 index (no embeddings needed)              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ IndexingPipeline orchestrates: Loader → Chunker → Indexes      │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│                                                                              │
│  PHASE 2: RETRIEVAL (Answer user queries)                                   │
│  ─────────────────────────────────────────                                  │
│                                                                              │
│  1. User Submits Query                                                       │
│     "What is Amazon's revenue?"                                              │
│                                                                              │
│  2. Transform Query (query_transformers/*.py)                               │
│     Optional: QueryTransformer.transform(query)                             │
│     - IdentityTransformer: pass through unchanged                           │
│     - HyDETransformer: generate hypothetical document                       │
│                                                                              │
│  3. Retrieve (retrievers/*.py)                                              │
│     Pass [transformed] query to Retriever:                                  │
│                                                                              │
│     ChromaRetriever:                                                         │
│       → Embed query → Similarity search in ChromaDB → top k docs            │
│                                                                              │
│     BM25Retriever:                                                           │
│       → Tokenize query → BM25 scoring → top k docs                         │
│                                                                              │
│     HybridRetriever (Ensemble):                                              │
│       → Run all child retrievers in parallel                                │
│       → Merge results using Reciprocal Rank Fusion (RRF)                    │
│       → Return unified top-k list                                           │
│                                                                              │
│  4. Generate Answer (generators/*.py)                                       │
│     Generator.generate(query, retrieved_docs) → answer                      │
│     - TraditionalGenerator: LLM + prompt with context                       │
│     - ReactAgentGenerator: ReAct agent with tool use                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ RetrievalPipeline orchestrates: Transform → Retrieve → Generate │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# ==============================================================================
# COMPATIBILITY MATRIX
# ==============================================================================

"""
Retrievers and their allowed indexes:

┌──────────────────┬──────────────────────────────────────────┐
│ Retriever        │ Required Index Type(s)                   │
├──────────────────┼──────────────────────────────────────────┤
│ ChromaRetriever  │ "chroma"                                 │
│ BM25Retriever    │ "bm25"                                   │
│ HybridRetriever  │ Child retrievers validate themselves     │
└──────────────────┴──────────────────────────────────────────┘

Valid Multi-Index Configurations:

✅ ALLOWED:
  - IndexingPipeline with [BM25Index, ChromaIndex]
    → All chunks indexed to both
    → HybridRetriever can use both

  - IndexingPipeline with [ChromaIndex]
    → ChromaRetriever only

  - IndexingPipeline with [BM25Index]
    → BM25Retriever only

❌ NOT ALLOWED:
  - ChromaRetriever with BM25Index only
    → Error: required "chroma" but got "bm25"

  - BM25Retriever with ChromaIndex only
    → Error: required "bm25" but got "chroma"
"""

# ==============================================================================
# MULTI-INDEX INDEXING EXAMPLE
# ==============================================================================

"""
Scenario: Index with BOTH BM25 parallel keyword index AND ChromaDB vector store

┌─────────────────────────────────────────────────────────────────────┐
│                        IndexingPipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Load documents (100 files)                                      │
│                                                                     │
│  2. Chunk with SemanticChunker → 5,000 chunks created              │
│                                                                     │
│  3. For each chunk, compute embeddings                              │
│                                                                     │
│  4. Add to all indexes:                                             │
│     ┌─────────────────────────────────────────────────────┐        │
│     │ BM25Index                                           │        │
│     ├─────────────────────────────────────────────────────┤        │
│     │ - Tokenize each chunk                               │        │
│     │ - Build BM25 sparse index (5000 docs)              │        │
│     │ - No embeddings needed (keyword-based)             │        │
│     └─────────────────────────────────────────────────────┘        │
│                                                                     │
│     ┌─────────────────────────────────────────────────────┐        │
│     │ ChromaIndex                                         │        │
│     ├─────────────────────────────────────────────────────┤        │
│     │ - Persist chunks + embeddings in Chroma            │        │
│     │ - Store metadata (source, chunk_index, etc)        │        │
│     │ - Chroma collection: 5000 docs with vectors        │        │
│     └─────────────────────────────────────────────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Result: Same 5,000 chunks accessible via both sparse AND dense retrieval.

Later, at retrieval time:
  HybridRetriever([BM25Retriever, ChromaRetriever])
  → Query 1: BM25Retriever → 5 results (keyword match)
  → Query 2: ChromaRetriever → 5 results (semantic match)
  → Merge via RRF → 5-10 unique results (deduplicated, ranked)
"""

# ==============================================================================
# DATA FLOW THROUGH RETRIEVER PIPELINE
# ==============================================================================

"""
Query Processing Through HybridRetriever with RRF Fusion:

User Query: "Amazon revenue growth"

┌────────────────────────────────────────────────────────────────────┐
│ HybridRetriever.retrieve(query, k=5)                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────┐                             │
│  │ BM25Retriever.retrieve(q, k=5)   │                             │
│  ├──────────────────────────────────┤                             │
│  │ 1. Tokenize query                │                             │
│  │ 2. BM25 scoring                  │                             │
│  │ 3. Return:                        │                             │
│  │    Doc-A (score=45.2) [rank 1]   │                             │
│  │    Doc-B (score=38.9) [rank 2]   │                             │
│  │    Doc-C (score=32.1) [rank 3]   │                             │
│  │    Doc-D (score=28.5) [rank 4]   │                             │
│  │    Doc-E (score=19.3) [rank 5]   │                             │
│  └──────────────────────────────────┘                             │
│                                                                    │
│  ┌──────────────────────────────────┐                             │
│  │ ChromaRetriever.retrieve(q, k=5) │                             │
│  ├──────────────────────────────────┤                             │
│  │ 1. Embed query                   │                             │
│  │ 2. Cosine similarity search      │                             │
│  │ 3. Return:                        │                             │
│  │    Doc-B (sim=0.87) [rank 1]     │                             │
│  │    Doc-C (sim=0.84) [rank 2]     │                             │
│  │    Doc-F (sim=0.78) [rank 3]     │                             │
│  │    Doc-A (sim=0.72) [rank 4]     │                             │
│  │    Doc-G (sim=0.65) [rank 5]     │                             │
│  └──────────────────────────────────┘                             │
│                                                                    │
│  Compute RRF scores: 1 / (rank + 60)                              │
│  ┌────────────────────────────────────────┐                       │
│  │ Doc-A: 1/61 + 1/64 = 0.0164 + 0.0156 = 0.032  │                       │
│  │ Doc-B: 1/62 + 1/61 = 0.0161 + 0.0164 = 0.032  │                       │
│  │ Doc-C: 1/63 + 1/62 = 0.0159 + 0.0161 = 0.032  │                       │
│  │ Doc-F: 1/∞  + 1/63 = 0.0000 + 0.0159 = 0.016  │                       │
│  │ Doc-G: 1/∞  + 1/65 = 0.0000 + 0.0154 = 0.015  │                       │
│  └────────────────────────────────────────┘                       │
│                                                                    │
│  Sort by RRF score (descending) and return top 5:                 │
│    1. Doc-A (RRF: 0.032)                                          │
│    2. Doc-B (RRF: 0.032)                                          │
│    3. Doc-C (RRF: 0.032)                                          │
│    4. Doc-F (RRF: 0.016)                                          │
│    5. Doc-G (RRF: 0.015)                                          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

Final output: 5 documents ranked by combined sparse + dense relevance
"""

# ==============================================================================
# COMPONENT DEPENDENCY EXAMPLES
# ==============================================================================

"""
Example 1: Fixed Chunking + ChromaDB Retrieval

Chunker:  FixedChunker (no embedder dependency)
Index:    ChromaIndex (needs Embedder)
Retriever: ChromaRetriever (needs ChromaIndex)
          index_type="chroma" ✅ match

┌─────────────┐
│   Chunker   │→ chunks → ┌────────┐
└─────────────┘           │ Index  │→ ┌──────────────┐
┌──────────┐              │(chroma)│  │ Retriever    │
│ Embedder │→ embeddings →└────────┘  │ (ChromaRetri)│
└──────────┘                          └──────────────┘


Example 2: Semantic Chunking + Hybrid Retrieval

Chunker: SemanticChunker (needs Embedder)
Indexes: BM25Index + ChromaIndex
Retrievers: BM25Retriever + ChromaRetriever (via HybridRetriever)

┌──────────┐
│ Embedder │→ ┌────────────────┐
└──────────┘  │ SemanticChunker│→ chunks  ┌─────────────────┐
              └────────────────┘          │ BM25Index       │
                                          │ ChromaIndex     │
                                          └─────────────────┘
                                                   ↓
                                        ┌──────────────────────┐
                                        │ HybridRetriever      │
                                        │ [BM25, ChromaRetri]  │
                                        └──────────────────────┘
"""

# ==============================================================================
# VALIDATION FLOW AT INITIALIZATION
# ==============================================================================

"""
When you create a RetrievalPipeline, compatibility is checked:

retrieval_pipeline = RetrievalPipeline(
    retriever=HybridRetriever([...]),
    indexes=[bm25_index, chroma_index],
)

Step 1: RetrievalPipeline.__init__ calls → retriever.validate_index_compatibility(indexes)

Step 2: HybridRetriever.validate_index_compatibility([bm25, chroma])
        → For each child_retriever in self.retrievers:
          → child_retriever.validate_index_compatibility([bm25, chroma])

Step 3: BM25Retriever.validate_index_compatibility([bm25, chroma])
        → required_index_types = ["bm25"]
        → index_types = ["bm25", "chroma"]
        → "bm25" in index_types ✅ PASS

Step 4: ChromaRetriever.validate_index_compatibility([bm25, chroma])
        → required_index_types = ["chroma"]
        → index_types = ["bm25", "chroma"]
        → "chroma" in index_types ✅ PASS

✅ All validations pass → Pipeline ready to use

If configuration was wrong:
  retrieval_pipeline = RetrievalPipeline(
      retriever=BM25Retriever(...),
      indexes=[chroma_index],  # ❌ Missing BM25Index
  )

Step 3 would fail:
  ❌ ValueError: BM25Retriever requires "bm25", but available: ["chroma"]
  
Early detection = no surprises at query time!
"""

if __name__ == "__main__":
    print(__doc__)
