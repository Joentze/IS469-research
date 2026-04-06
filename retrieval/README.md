# Modular RAG Pipeline Architecture

## Overview

Implements a **Strategy Pattern** architecture, decoupling the RAG pipeline into injectable, composable components. This enables developers to:

- **Mix and match** chunking strategies, embeddings, indexes, and retrievers
- **Easily evaluate** different combinations without code duplication
- **Support hybrid approaches** (e.g., BM25 + ChromaDB together)
- **Validate compatibility** between components at initialization time

## Architecture

### Directory Structure

```
retrieval/
├── core/
│   ├── interfaces.py       # Abstract base classes for all strategies
│   └── pipeline.py         # Orchestrators: IndexingPipeline, RetrievalPipeline
├── chunkers/
│   ├── fixed.py            # Fixed-size character chunking
│   ├── semantic.py         # Embedding similarity-based chunking
│   └── agentic.py          # LLM-guided topic-aware chunking
├── embeddings/
│   └── openai.py           # OpenAI embeddings wrapper
├── indexes/
│   ├── chroma.py           # ChromaDB vector store
│   └── bm25.py             # BM25 sparse keyword index
├── query_transformers/
│   ├── identity.py         # No-op passthrough
│   └── hyde.py             # Hypothetical document generation
├── retrievers/
│   ├── chroma.py           # ChromaDB similarity search
│   ├── bm25.py             # BM25 keyword search
│   └── hybrid.py           # Ensemble combining multiple retrievers
├── generators/
│   ├── traditional.py      # Simple LLM-based answer generation
│   └── react_agent.py      # ReAct agent with tool use
└── utils/
    └── env.py              # Environment & API key management
    └── markdown_loader.py  # Load .md documents from disk

```

## Core Concepts

### 1. Strategies (Interfaces)

All components implement abstract interfaces from `core/interfaces.py`:

- **`Chunker`**: Transforms raw documents into chunks
- **`Embedder`**: Produces vector embeddings for text
- **`Index`**: Stores chunked documents and metadata
- **`QueryTransformer`**: Optionally transforms user queries (e.g., HyDE)
- **`Retriever`**: Fetches top-k relevant chunks for a query
- **`Generator`**: Produces answers from retrieved documents

### 2. Index Compatibility

Each `Index` has an `index_type` string:
- ChromaDB: `index_type = "chroma"`
- BM25: `index_type = "bm25"`

Each `Retriever` declares its `required_index_types`:
- `ChromaRetriever`: requires `["chroma"]`
- `BM25Retriever`: requires `["bm25"]`
- `HybridRetriever` (ensemble): validates all child retrievers

At initialization, retrievers validate that provided indexes are compatible, catching configuration errors early.

### 3. Multi-Index Support

The `IndexingPipeline` accepts a list of indexes. Chunks are automatically added to all:

```python
indexing_pipeline = IndexingPipeline(
    chunker=semantic_chunker,
    embedder=embedder,
    indexes=[bm25_index, chroma_index],  # Chunks go to both
)
```

### 4. Ensemble Retrieval

Multiple retrievers can be fused using `HybridRetriever` (or custom ensemble):

```python
hybrid_retriever = HybridRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    fusion_method="rrf",  # Reciprocal Rank Fusion
)
```

Results from each retriever are combined, scored, and ranked.

## Usage Examples

### Example 1: Simple Fixed Chunking + ChromaDB

```python
from chunkers.fixed import FixedChunker
from embeddings.openai import OpenAIEmbedder
from indexes.chroma import ChromaIndex
from retrievers.chroma import ChromaRetriever
from generators.traditional import TraditionalGenerator
from core.pipeline import IndexingPipeline, RetrievalPipeline

# Indexing
indexer = IndexingPipeline(
    chunker=FixedChunker(chunk_size=1000),
    embedder=OpenAIEmbedder(),
    indexes=[ChromaIndex(collection_name="my_docs")],
)
indexer.index_directory(Path("./texts"))

# Retrieval
chroma_idx = indexer.indexes[0]
retriever = ChromaRetriever(index=chroma_idx)
generator = TraditionalGenerator()

retrieval = RetrievalPipeline(
    retriever=retriever,
    generator=generator,
    indexes=[chroma_idx],
)

result = retrieval.answer("What is X?", k=5)
print(result["answer"])
```

### Example 2: Semantic Chunking + Hybrid Retrieval

```python
from chunkers.semantic import SemanticChunker
from indexes.bm25 import BM25Index
from indexing.chroma import ChromaIndex
from retrievers.bm25 import BM25Retriever
from retrievers.chroma import ChromaRetriever
from retrievers.hybrid import HybridRetriever

# Indexing to multiple backends
embedder = OpenAIEmbedder()
indexer = IndexingPipeline(
    chunker=SemanticChunker(embedder=embedder),
    embedder=embedder,
    indexes=[BM25Index(), ChromaIndex(collection_name="semantic")],
)
indexer.index_directory(Path("./texts"))

# Hybrid retrieval combining both indexes
bm25_idx, chroma_idx = indexer.indexes
hybrid = HybridRetriever(
    retrievers=[
        BM25Retriever(index=bm25_idx),
        ChromaRetriever(index=chroma_idx),
    ]
)

retrieval = RetrievalPipeline(
    retriever=hybrid,
    generator=TraditionalGenerator(),
    indexes=[bm25_idx, chroma_idx],
)

result = retrieval.answer("What is X?", k=5)
```

### Example 3: Agentic Chunking + HyDE + ReAct

```python
from chunkers.agentic import AgenticChunker
from query_transformers.hyde import HyDETransformer
from retrievers.chroma import ChromaRetriever
from generators.react_agent import ReactAgentGenerator

# Indexing
indexer = IndexingPipeline(
    chunker=AgenticChunker(llm_model="gpt-4o-mini"),
    embedder=OpenAIEmbedder(),
    indexes=[ChromaIndex(collection_name="agentic")],
)
indexer.index_directory(Path("./texts"))

# Retrieval with HyDE transformation
chroma_idx = indexer.indexes[0]
retrieval = RetrievalPipeline(
    retriever=ChromaRetriever(
        index=chroma_idx,
        query_transformer=HyDETransformer(),
    ),
    generator=ReactAgentGenerator(),
    indexes=[chroma_idx],
)

result = retrieval.answer("What is X?", k=5)
```

## Key Features

### 1. Modularity
- Each component is independent and testable
- Easier to debug and extend

### 2. Composability
- Combine any chunker with any indexer/retriever
- Support for custom implementations

### 3. Multi-Index Support
- Index same chunks into both BM25 and ChromaDB
- Query multiple indexes simultaneously (hybrid retrieval)

### 4. Compatibility Checking
- Automatic validation that retrievers match available indexes
- Prevents runtime failures due to misconfiguration

### 5. No Duplicated Code
- Environment loading (`utils/env.py`)
- Document loading (`loaders/markdown.py`)
- Chunking logic centralized per strategy
- Vector/index management consolidated

## Migration Guide

### Old Way (Monolithic)
```python
# Old: retrieval/fixed/traditional-rag.py had ALL logic in one file
# - Document loading
# - Chunking
# - Embedding
# - Vector store management
# - Retrieval
# - Generation
```

### New Way (Modular)
```python
# New: Compose only the pieces you need
indexer = IndexingPipeline(
    chunker=FixedChunker(...),
    embedder=OpenAIEmbedder(...),
    indexes=[ChromaIndex(...)],
)
retrieval = RetrievalPipeline(
    retriever=ChromaRetriever(...),
    generator=TraditionalGenerator(...),
    indexes=[...],
)
```

## Configuration for Evaluation

To evaluate different configurations:

1. **Vary chunking strategies**: `FixedChunker` vs `SemanticChunker` vs `AgenticChunker`
2. **Vary embedders**: Easy to add new embedder implementations
3. **Vary indexes**: Single index or multiple indexes
4. **Vary retrievers**: Single or ensemble, with/without transformers
5. **Vary generators**: Traditional prompting vs ReAct agent

All combinations are automatically compatible as long as retriever requirements are met.

## Future Extensions

- Custom chunkers (e.g., sliding window, recursive)
- Custom embedders (e.g., OSS models)
- Custom indexes (e.g., Pinecone, Weaviate)
- Custom retrievers (e.g., MMR, threshold filtering)
- Custom generators (e.g., prompt engineering variants, fine-tuned models)

Each new implementation just needs to inherit from the appropriate interface in `core/interfaces.py`.

## Environment Setup

Required environment variables (set in `.env` or shell):

```
OPENAI_API_KEY=sk-...
TEXTS_DIR=./texts  # (optional, defaults to project root)
```

Load via:
```python
from utils.env import load_project_env, require_openai_api_key

load_project_env(Path(__file__).parents[1])
require_openai_api_key("my operation")
```

## References

- **Strategy Pattern**: Gang of Four design pattern for encapsulating interchangeable algorithms
- **Multi-Index Retrieval**: Hybrid search combining sparse (BM25) and dense (vectors) retrieval
- **Reciprocal Rank Fusion**: Simple effective method for combining ranked lists
