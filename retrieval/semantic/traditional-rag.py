"""
Traditional RAG pipeline with semantic chunking.

Semantic chunking groups adjacent sentences by embedding similarity,
stores chunks in ChromaDB, and retrieves relevant chunks via
similarity search (no agent).

References:
- ChromaDB: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
- Semantic chunking: https://medium.com/@anixlynch/7-chunking-strategies-for-langchain-b50dac194813
"""

import os
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# ============================================================================
# CONFIGURATION
# ============================================================================


def load_project_env(project_root: Path) -> None:
	"""Load key-value pairs from .env into process environment."""
	env_path = project_root / ".env"
	if not env_path.exists() or not env_path.is_file():
		return

	for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue

		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if key and key not in os.environ:
			os.environ[key] = value

	if "OPENAI_API_KEY" not in os.environ and "API_KEY" in os.environ:
		os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]


EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_STORE_DIR = "./database"
COLLECTION_NAME = "trad_rag_semantic_collection"
BATCH_SIZE = 5000
TOP_K = 5               # Number of chunks to retrieve
PROJECT_ROOT = Path(__file__).parents[2]
load_project_env(PROJECT_ROOT)
DEFAULT_TEXTS_DIR = PROJECT_ROOT / "texts"
TEXTS_DIR = os.getenv("TEXTS_DIR", str(DEFAULT_TEXTS_DIR))
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "false").lower() == "true"

MAX_SENTENCES_PER_CHUNK = 8
OVERLAP_SENTENCES = 2
BREAKPOINT_PERCENTILE = 70.0


# ============================================================================
# DOCUMENTS
# ============================================================================


def load_sample_documents() -> list[Document]:
	"""Load markdown documents from the texts directory."""
	configured_path = Path(TEXTS_DIR).expanduser()
	if not configured_path.exists() or not configured_path.is_dir():
		raise FileNotFoundError(
			f"Could not find texts directory: {configured_path}. "
			"Set TEXTS_DIR to a valid folder path."
		)

	markdown_files = sorted(configured_path.glob("*.md"))
	if not markdown_files:
		raise ValueError(f"No .md files found in texts directory: {configured_path}")

	documents: list[Document] = []
	for file_path in markdown_files:
		content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
		if not content:
			continue
		documents.append(
			Document(
				page_content=content,
				metadata={"source": file_path.name, "path": str(file_path)},
			)
		)

	return documents


# ============================================================================
# SEMANTIC CHUNKING
# ============================================================================


def split_into_sentences(text: str) -> list[str]:
	"""Split text into simple sentence-like units."""
	sentences = re.split(r"(?<=[.!?])\s+", text.strip())
	return [s.strip() for s in sentences if s and s.strip()]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
	"""Compute cosine similarity between two vectors."""
	dot = sum(a * b for a, b in zip(vec_a, vec_b))
	norm_a = sum(a * a for a in vec_a) ** 0.5
	norm_b = sum(b * b for b in vec_b) ** 0.5
	if norm_a == 0 or norm_b == 0:
		return 0.0
	return dot / (norm_a * norm_b)


def percentile(values: list[float], q: float) -> float:
	"""Compute percentile with linear interpolation."""
	if not values:
		return 0.0
	ordered = sorted(values)
	if len(ordered) == 1:
		return ordered[0]

	rank = (len(ordered) - 1) * (q / 100.0)
	lower = int(rank)
	upper = min(lower + 1, len(ordered) - 1)
	weight = rank - lower
	return ordered[lower] * (1 - weight) + ordered[upper] * weight


def semantic_chunk_documents(
	documents: list[Document],
	embeddings: OpenAIEmbeddings,
) -> list[Document]:
	"""
	Chunk documents by semantic boundaries.

	We split text into sentences, embed each sentence, and insert chunk breaks
	where adjacent sentence similarity drops below a percentile threshold.
	"""
	chunked_docs: list[Document] = []

	for doc in documents:
		sentences = split_into_sentences(doc.page_content)
		if not sentences:
			continue
		if len(sentences) == 1:
			chunked_docs.append(doc)
			continue

		sentence_vectors = embeddings.embed_documents(sentences)
		adjacent_similarities = [
			cosine_similarity(sentence_vectors[i], sentence_vectors[i + 1])
			for i in range(len(sentence_vectors) - 1)
		]
		breakpoint_threshold = percentile(adjacent_similarities, BREAKPOINT_PERCENTILE)

		break_after_index = {
			i
			for i, score in enumerate(adjacent_similarities)
			if score <= breakpoint_threshold
		}

		current: list[str] = []
		chunk_index = 0

		for i, sentence in enumerate(sentences):
			current.append(sentence)

			reached_max = len(current) >= MAX_SENTENCES_PER_CHUNK
			semantic_break = i in break_after_index and len(current) >= 2

			if reached_max or semantic_break:
				chunked_docs.append(
					Document(
						page_content=" ".join(current).strip(),
						metadata={
							**doc.metadata,
							"chunk_index": chunk_index,
							"chunk_method": "semantic",
						},
					)
				)
				chunk_index += 1

				overlap = current[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES > 0 else []
				current = overlap.copy()

		if current:
			chunked_docs.append(
				Document(
					page_content=" ".join(current).strip(),
					metadata={
						**doc.metadata,
						"chunk_index": chunk_index,
						"chunk_method": "semantic",
					},
				)
			)

	return chunked_docs


# ============================================================================
# VECTOR STORE
# ============================================================================


def initialize_vector_store(documents: list[Document], embeddings: OpenAIEmbeddings) -> Chroma:
	"""Create a Chroma collection and load semantic chunks."""
	vector_store = Chroma(
		collection_name=COLLECTION_NAME,
		embedding_function=embeddings,
		persist_directory=VECTOR_STORE_DIR,
	)

	existing_count = vector_store._collection.count()

	if existing_count > 0 and not FORCE_REINDEX:
		print(
			f"Using existing vector store with {existing_count} chunks. "
			"Skipping re-indexing."
		)
		return vector_store

	if FORCE_REINDEX and existing_count > 0:
		print("FORCE_REINDEX=true detected. Rebuilding vector store...")
		vector_store.delete_collection()
		vector_store = Chroma(
			collection_name=COLLECTION_NAME,
			embedding_function=embeddings,
			persist_directory=VECTOR_STORE_DIR,
		)

	if documents:
		for start in range(0, len(documents), BATCH_SIZE):
			batch = documents[start : start + BATCH_SIZE]
			vector_store.add_documents(batch)
		print(f"Added {len(documents)} chunks to vector store")

	return vector_store


# ============================================================================
# RETRIEVAL
# ============================================================================


def retrieve(vector_store: Chroma, query: str) -> list[Document]:
	"""
	Retrieve the top-k most relevant chunks for a query.

	Unlike agentic RAG, traditional RAG retrieves directly via
	similarity search — no agent or reasoning loop involved.

	Args:
		vector_store: ChromaDB instance
		query: User query string

	Returns:
		List of top-k relevant documents
	"""
	retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
	results = retriever.invoke(query)
	return results


# ============================================================================
# PIPELINE
# ============================================================================


def run_traditional_rag_pipeline(query: str) -> list[Document]:
	"""Run semantic chunking -> indexing -> retrieval."""
	if not os.getenv("OPENAI_API_KEY"):
		raise EnvironmentError("OPENAI_API_KEY is not set.")

	print("=" * 60)
	print("TRADITIONAL RAG PIPELINE - SEMANTIC CHUNKING")
	print("=" * 60)

	print("\n[1] Loading source documents...")
	docs = load_sample_documents()
	print(f"Loaded {len(docs)} documents")

	print("\n[2] Building semantic chunks...")
	embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
	semantic_chunks = semantic_chunk_documents(docs, embeddings)
	print(f"Created {len(semantic_chunks)} semantic chunks")

	print("\n[3] Initializing ChromaDB...")
	vector_store = initialize_vector_store(semantic_chunks, embeddings)
	print(f"Indexed {vector_store._collection.count()} chunks")

	print(f"\n[4] Retrieving top {TOP_K} chunks for query: '{query}'")
	print("-" * 60)

	results = retrieve(vector_store, query)

	print("\nRetrieved Chunks:")
	for i, doc in enumerate(results, 1):
		print(f"\n[Chunk {i}]")
		print(f"Content: {doc.page_content[:500]}...")
		if doc.metadata:
			print(f"Metadata: {doc.metadata}")

	return results


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
	test_query = "What is Amazon's year-over-year change in revenue from FY2016 to FY2017 (in units of percents and round to one decimal place)? Calculate what was asked by utilizing the line items clearly shown in the statement of income."
	run_traditional_rag_pipeline(test_query)