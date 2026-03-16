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
