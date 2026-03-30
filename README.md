# IS469-research

Research on various document retrieval methods.

## Prerequisites

- **Python 3.12+** — [python.org/downloads](https://www.python.org/downloads/)
- **Poetry** — install via pip:

```bash
pip install poetry
```

## Getting Started

1. **Set the Poetry environment to use your Python installation:**

```bash
poetry env use python
```

2. **Activate the virtual environment:**

```bash
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
poetry install
```

## Vector Store (ChromaDB)

This project uses [ChromaDB](https://docs.trychroma.com/getting-started) via the [`langchain-chroma`](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma) integration to store and query vector embeddings.

Documents are converted into vector embeddings using an embedding model and stored in a Chroma collection. At query time, the input is embedded with the same model and Chroma performs a similarity search against the stored vectors to retrieve the most relevant documents.

We use a **persistent local directory** (`./database`) so that embeddings survive across runs without needing to re-index:

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./database",
)
```

The `database/` directory is git-ignored, so each collaborator maintains their own local copy of the vector store and won't overwrite each other's data.

## Sharing Agentic BM25 Chunks via Chroma

The BM25 pipeline (`retrieval/agentic/bm25.py`) now supports three retrieval modes:

- `bm25`: lexical BM25 only
- `chroma`: vector retrieval from persisted Chroma chunks
- `hybrid`: expose both BM25 and Chroma tools to the agent

### Build and persist a shareable chunk store

```bash
poetry run python retrieval/agentic/bm25.py \
    --retriever-mode chroma \
    --chroma-dir ./database/bm25_agentic \
    --collection-name agentic_bm25_chunks_collection
```

This rebuilds the collection from source markdown files and persists it in `./database/bm25_agentic`.

### Share with groupmates

1. Zip the `./database/bm25_agentic` folder.
2. Send it to your groupmates.
3. They unzip it anywhere locally and point `--chroma-dir` to that path.

### Reuse an existing shared store (no re-chunking)

```bash
poetry run python retrieval/agentic/bm25.py \
    --retriever-mode hybrid \
    --reuse-existing-store \
    --chroma-dir ./database/bm25_agentic \
    --collection-name agentic_bm25_chunks_collection \
    --query "Explain how cloud revenue trends changed across companies."
```

### Compatibility notes

- Use the same collection name and Chroma directory path.
- Use the same embedding model for indexing and querying.
- Keep dependency versions reasonably aligned across teammates.
- `OPENAI_API_KEY` is still required for chunk planning and query-time embeddings.
