# BM25 Retrieval Script: Top-Down Explanation

This document explains how [bm25.py](retrieval/semantic/bm25.py) works from top to bottom.

## 1. Module Setup And Imports

- The file starts with a short docstring describing BM25 usage in [bm25.py](retrieval/semantic/bm25.py#L1).
- Imports are declared for logging, CLI parsing, regex tokenization, dataclasses, and paths in [bm25.py](retrieval/semantic/bm25.py#L5).
- `BM25Okapi` from `rank_bm25` is imported in [bm25.py](retrieval/semantic/bm25.py#L13).

## 2. Logging And Defaults

- Logging is configured once with timestamped INFO-level messages in [bm25.py](retrieval/semantic/bm25.py#L16).
- `TEXTS_DIR` points to the project-level texts folder in [bm25.py](retrieval/semantic/bm25.py#L22).
- `DEFAULT_QUERY` provides a fallback search query in [bm25.py](retrieval/semantic/bm25.py#L23).

## 3. Chunk Data Model

- `Chunk` is a frozen dataclass in [bm25.py](retrieval/semantic/bm25.py#L26).
- Each chunk stores:
  - source markdown filename (`file_name`),
  - local chunk number inside that file (`chunk_index`),
  - chunk text content (`text`).

## 4. Command-Line Arguments

- `parse_args()` defines runtime options in [bm25.py](retrieval/semantic/bm25.py#L35).
- Supported flags:
  - `--query`: text query to search.
  - `--top-k`: number of top chunk results.
  - `--texts-dir`: override input markdown directory.

## 5. Tokenization Helpers

- `tokenize_words()` returns a unique token set for similarity calculations in [bm25.py](retrieval/semantic/bm25.py#L49).
- `tokenize_for_bm25()` returns token lists for BM25 scoring in [bm25.py](retrieval/semantic/bm25.py#L54).

## 6. Semantic Similarity + Chunking

- `semantic_similarity()` computes Jaccard overlap between adjacent text units in [bm25.py](retrieval/semantic/bm25.py#L59).
- `split_semantic_units()` splits text by sentence punctuation and commas in [bm25.py](retrieval/semantic/bm25.py#L69).
- `semantic_chunk_text()` merges neighboring units when:
  - merged size stays under `max_chunk_chars`, and
  - either unit similarity passes threshold or current chunk is still short.
- This logic lives in [bm25.py](retrieval/semantic/bm25.py#L75) and produces a list of chunk strings.

## 7. Reading Markdown Corpus

- `load_markdown_documents()` validates directory existence and gathers all `.md` files in [bm25.py](retrieval/semantic/bm25.py#L113).
- Each file is read as UTF-8 (ignoring undecodable bytes), stripped, and included if non-empty in [bm25.py](retrieval/semantic/bm25.py#L123).
- Explicit exceptions are raised for missing directory, no markdown files, or all-empty files.

## 8. Utility: Stable Deduplication

- `dedupe_preserve_order()` removes duplicates while preserving first appearance order in [bm25.py](retrieval/semantic/bm25.py#L134).
- It is used to count distinct source files represented in the top-ranked chunks.

## 9. Chunk Expansion For All Documents

- `build_chunks()` applies semantic chunking to every loaded markdown document in [bm25.py](retrieval/semantic/bm25.py#L145).
- It creates `Chunk` objects with `max_chunk_chars=700` and `similarity_threshold=0.15` in [bm25.py](retrieval/semantic/bm25.py#L149).

## 10. Main Retrieval Pipeline

- `main()` orchestrates end-to-end retrieval in [bm25.py](retrieval/semantic/bm25.py#L166):
  1. Parse CLI args.
  2. Load markdown files.
  3. Build semantic chunks.
  4. Tokenize chunks.
  5. Build BM25 index.
  6. Tokenize query and validate non-empty.
  7. Score all chunks.
  8. Select top-k indices by descending score.
  9. Compute unique source-file count among ranked chunks.
  10. Print ranked output with score, source file, chunk id, and preview text.

## 11. Script Entry Point

- The script runs `main()` only when executed directly in [bm25.py](retrieval/semantic/bm25.py#L208).

## Summary

The script loads all markdown reports from the texts folder, semantically chunks them, builds a BM25 index over chunks, and returns the highest-scoring chunks for a user query with source-file metadata for traceability.
