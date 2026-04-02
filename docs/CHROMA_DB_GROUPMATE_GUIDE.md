# Chroma DB Handoff Guide (Groupmates)

This guide explains how to reuse the persisted Chroma store from this project in your own retrieval implementations.

## 1) What to Share

Share this folder:

- `retrieval/agentic/bm25_agentic`

It contains the local Chroma database files, including the persisted collection data.

Current default collection name used by this project:

- `agentic_bm25_chunks_collection`

## 2) Expected Setup

Install dependencies in your environment:

```bash
poetry install
```

Or at minimum, ensure these are available:

- `chromadb`
- `langchain-chroma`
- `langchain-openai` (only if doing vector similarity with OpenAI embeddings)

## 3) Quick Health Check (Collection Exists and Has Data)

Run from repo root:

```bash
poetry run python -c "from langchain_chroma import Chroma; db=Chroma(collection_name='agentic_bm25_chunks_collection', persist_directory='retrieval/agentic/bm25_agentic'); print(len(db.get(include=[]).get('ids', [])))"
```

If count is greater than 0, the shared store is populated.

## 4) Use Case A: Reuse Chunks for BM25/Keyword Retrieval (No OpenAI Needed)

If your implementation is lexical/BM25-based, just load documents + metadata from Chroma and build your own index.

```python
import chromadb
from pathlib import Path

persist_dir = Path("retrieval/agentic/bm25_agentic").resolve()
client = chromadb.PersistentClient(path=str(persist_dir))
collection = client.get_collection("agentic_bm25_chunks_collection")

# Read in pages to avoid SQLite variable limits on very large collections.
batch_size = 5000
offset = 0
all_docs = []

while True:
    page = collection.get(include=["documents", "metadatas"], limit=batch_size, offset=offset)
    docs = page.get("documents", [])
    metas = page.get("metadatas", [])
    if not docs:
        break

    for i, text in enumerate(docs):
        if isinstance(text, str) and text.strip():
            meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            all_docs.append(
                {
                    "text": text,
                    "source": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", i),
                }
            )

    offset += batch_size

print("Loaded chunks:", len(all_docs))
```

## 5) Use Case B: Vector Similarity Retrieval From Existing Store

If your implementation does semantic retrieval, connect with the same collection and the same embedding model used to build/query it.

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name="agentic_bm25_chunks_collection",
    embedding_function=embeddings,
    persist_directory="retrieval/agentic/bm25_agentic",
)

results = vector_store.similarity_search_with_score("cloud revenue trend", k=5)
for rank, (doc, score) in enumerate(results, start=1):
    print(rank, score, doc.metadata.get("source"), doc.metadata.get("chunk_index"))
```

Notes:

- This path usually requires `OPENAI_API_KEY` for query embedding.
- Keep embedding model consistent (`text-embedding-3-small`) for best compatibility.

## 6) Path and Collaboration Rules

- Use one agreed path and one agreed collection name across the team.
- Avoid multiple people writing to the same exact folder copy at the same time.
- For safety, distribute read-only copies (zip and unzip per user machine).

## 7) Common Pitfalls

- Wrong path: `bm25_agentic` (local working dir) vs `retrieval/agentic/bm25_agentic` (repo-root relative).
- Empty collection: DB files exist but count is 0.
- Large collection fetch errors: use paged `limit` and `offset` reads.

## 8) Reference Command in This Project

No-LLM reuse path for BM25 retrieval:

```bash
poetry run python retrieval/agentic/bm25.py --retriever-mode bm25 --reuse-existing-store --retrieval-only --chroma-dir retrieval/agentic/bm25_agentic --collection-name agentic_bm25_chunks_collection --query "Explain how cloud revenue trends changed across companies."
```
