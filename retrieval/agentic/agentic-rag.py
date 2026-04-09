"""
Agentic RAG pipeline with agentic chunking.

Agentic chunking uses an LLM to choose topic-aware chunk boundaries,
stores chunks in ChromaDB, and answers questions via a ReAct-style agent.

References:
- ChromaDB: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
- ReAct agents: https://docs.langchain.com/oss/python/langchain/agents
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


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
LLM_MODEL = "gpt-4o-mini"
PROJECT_ROOT = Path(__file__).parents[2]
load_project_env(PROJECT_ROOT)
DEFAULT_TEXTS_DIR = PROJECT_ROOT / "texts"
TEXTS_DIR = os.getenv("TEXTS_DIR", str(DEFAULT_TEXTS_DIR))
DEFAULT_VECTOR_STORE_DIR = PROJECT_ROOT / "database_agentic"
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", str(DEFAULT_VECTOR_STORE_DIR))
COLLECTION_NAME = "agentic_bm25_chunks_collection"
BATCH_SIZE = 5000
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "false").lower() == "true"

MAX_SENTENCES_PER_CHUNK = 6
MIN_SENTENCES_PER_CHUNK = 2
OVERLAP_SENTENCES = 1
CHUNK_PLANNING_WINDOW_SENTENCES = 80
CHUNK_PLANNING_WINDOW_OVERLAP = 10
CHUNK_PLANNING_MAX_SENTENCE_CHARS = 500
TOP_K = 5


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
# AGENTIC CHUNKING
# ============================================================================


def split_into_sentences(text: str) -> list[str]:
	"""Split text into sentence-like units."""
	sentences = re.split(r"(?<=[.!?])\s+", text.strip())
	return [s.strip() for s in sentences if s and s.strip()]


def _normalize_breakpoints(raw_breakpoints: list[int], sentence_count: int) -> list[int]:
	"""Keep sorted valid sentence-end indexes for chunk boundaries."""
	if sentence_count <= 1:
		return []
	valid = sorted({idx for idx in raw_breakpoints if 0 <= idx < sentence_count - 1})
	return valid


def _fallback_breakpoints(sentence_count: int) -> list[int]:
	"""Deterministic fallback if the chunking agent output is invalid."""
	if sentence_count <= MAX_SENTENCES_PER_CHUNK:
		return []

	breaks: list[int] = []
	idx = MAX_SENTENCES_PER_CHUNK - 1
	while idx < sentence_count - 1:
		breaks.append(idx)
		idx += MAX_SENTENCES_PER_CHUNK
	return breaks


def _truncate_for_prompt(text: str, max_chars: int = CHUNK_PLANNING_MAX_SENTENCE_CHARS) -> str:
	"""Trim long sentence strings in prompts to control token usage."""
	if len(text) <= max_chars:
		return text
	return text[: max_chars - 3].rstrip() + "..."


def _choose_breakpoints_for_window(window_sentences: list[str], llm: ChatOpenAI) -> list[int]:
	"""Run one chunk-planning LLM call on a sentence window."""
	if len(window_sentences) <= MAX_SENTENCES_PER_CHUNK:
		return []

	numbered_sentences = "\n".join(
		f"{i}: {_truncate_for_prompt(sentence)}"
		for i, sentence in enumerate(window_sentences)
	)
	prompt = (
		"You are a chunking planner for a RAG system.\n"
		"Group adjacent sentences into coherent chunks by topic/idea shifts.\n"
		"Rules:\n"
		f"- Every chunk should have {MIN_SENTENCES_PER_CHUNK} to {MAX_SENTENCES_PER_CHUNK} sentences when possible.\n"
		"- Prefer boundaries where topic changes.\n"
		"- Return ONLY valid JSON with this exact schema:\n"
		"{\"break_after\": [int, int, ...]}\n"
		"- Do not include markdown or explanations.\n\n"
		"Sentences:\n"
		f"{numbered_sentences}"
	)

	response = llm.invoke(prompt)

	try:
		text = str(getattr(response, "content", "")).strip()
		parsed = json.loads(text)
		raw_breaks = parsed.get("break_after", [])
		if not isinstance(raw_breaks, list):
			return []
		int_breaks = [int(x) for x in raw_breaks]
		return _normalize_breakpoints(int_breaks, len(window_sentences))
	except (json.JSONDecodeError, TypeError, ValueError):
		return []


def choose_chunk_breakpoints_with_agent(sentences: list[str], llm: ChatOpenAI) -> list[int]:
	"""
	Ask an LLM to propose chunk boundaries.

	The model returns JSON with `break_after` indexes. Each index means:
	create a chunk boundary after that sentence index.
	"""
	if len(sentences) <= MAX_SENTENCES_PER_CHUNK:
		return []

	window_size = max(MAX_SENTENCES_PER_CHUNK + 1, CHUNK_PLANNING_WINDOW_SENTENCES)
	window_overlap = max(0, min(CHUNK_PLANNING_WINDOW_OVERLAP, window_size - 1))
	step = max(1, window_size - window_overlap)

	global_breaks: list[int] = []
	for start in range(0, len(sentences), step):
		end = min(start + window_size, len(sentences))
		window_sentences = sentences[start:end]

		local_breaks = _choose_breakpoints_for_window(window_sentences, llm)
		for local_break in local_breaks:
			global_break = start + local_break
			if global_break < len(sentences) - 1:
				global_breaks.append(global_break)

		if end >= len(sentences):
			break

	normalized_breaks = _normalize_breakpoints(global_breaks, len(sentences))
	return normalized_breaks if normalized_breaks else _fallback_breakpoints(len(sentences))


def enforce_chunk_size_rules(breaks: list[int], sentence_count: int) -> list[int]:
	"""Adjust boundaries to avoid tiny chunks and very large chunks."""
	if sentence_count <= 1:
		return []

	all_breaks = _normalize_breakpoints(breaks, sentence_count)
	if not all_breaks:
		return _fallback_breakpoints(sentence_count)

	filtered: list[int] = []
	start = 0
	for brk in all_breaks:
		chunk_len = brk - start + 1
		if chunk_len >= MIN_SENTENCES_PER_CHUNK:
			filtered.append(brk)
			start = brk + 1

	# Add extra boundaries if any resulting chunk is too large.
	with_caps: list[int] = []
	start = 0
	for brk in filtered + [sentence_count - 1]:
		while brk - start + 1 > MAX_SENTENCES_PER_CHUNK:
			cap_break = start + MAX_SENTENCES_PER_CHUNK - 1
			if cap_break < sentence_count - 1:
				with_caps.append(cap_break)
			start = cap_break + 1
		if brk < sentence_count - 1:
			with_caps.append(brk)
		start = brk + 1

	return _normalize_breakpoints(with_caps, sentence_count)


def build_chunks_from_breakpoints(
	sentences: list[str],
	breakpoints: list[int],
	metadata: dict[str, Any],
) -> list[Document]:
	"""Materialize chunk documents from sentence breakpoints."""
	chunks: list[Document] = []
	start = 0
	chunk_index = 0

	for brk in breakpoints + [len(sentences) - 1]:
		if brk < start:
			continue
		chunk_sentences = sentences[start : brk + 1]
		if not chunk_sentences:
			continue

		chunks.append(
			Document(
				page_content=" ".join(chunk_sentences).strip(),
				metadata={
					**metadata,
					"chunk_index": chunk_index,
					"chunk_method": "agentic",
				},
			)
		)
		chunk_index += 1

		if OVERLAP_SENTENCES > 0:
			start = max(brk + 1 - OVERLAP_SENTENCES, 0)
		else:
			start = brk + 1

		if start >= len(sentences):
			break

	return chunks


def agentic_chunk_documents(documents: list[Document], llm: ChatOpenAI) -> list[Document]:
	"""Chunk documents with LLM-selected semantic/topic boundaries."""
	chunked_docs: list[Document] = []

	for doc in documents:
		sentences = split_into_sentences(doc.page_content)
		if not sentences:
			continue
		if len(sentences) <= MAX_SENTENCES_PER_CHUNK:
			chunked_docs.append(
				Document(
					page_content=" ".join(sentences),
					metadata={**doc.metadata, "chunk_index": 0, "chunk_method": "agentic"},
				)
			)
			continue

		raw_breaks = choose_chunk_breakpoints_with_agent(sentences, llm)
		breaks = enforce_chunk_size_rules(raw_breaks, len(sentences))
		chunked_docs.extend(build_chunks_from_breakpoints(sentences, breaks, doc.metadata))

	return chunked_docs


# ============================================================================
# VECTOR STORE
# ============================================================================


def initialize_vector_store(documents: list[Document], embeddings: OpenAIEmbeddings) -> Chroma:
	"""Create a Chroma collection and load agentic chunks."""
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
# TOOLS + AGENT
# ============================================================================


def create_agent_tools(vector_store: Chroma) -> list:
	"""Create retrieval tools used by the answering ReAct agent."""

	@tool
	def search_knowledge_base(query: str, k: int = TOP_K) -> str:
		"""Search the agentic chunk index and return top matching chunks."""
		results = vector_store.similarity_search_with_score(query, k=k)
		if not results:
			return "No relevant chunks found."

		lines = ["Top agentic chunks:"]
		for i, (doc, score) in enumerate(results, start=1):
			lines.append(f"[{i}] score={score:.4f}")
			lines.append(f"source={doc.metadata.get('source', 'unknown')}")
			lines.append(doc.page_content)
			lines.append("-")
		return "\n".join(lines)

	@tool
	def get_chunk_count() -> str:
		"""Return number of chunks currently stored in Chroma."""
		return f"Chunks in vector store: {vector_store._collection.count()}"

	return [search_knowledge_base, get_chunk_count]


def create_rag_agent(vector_store: Chroma) -> Any:
	"""Build a ReAct-style answering agent with retrieval tools."""
	llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
	tools = create_agent_tools(vector_store)
	return create_agent(model=llm, tools=tools)


def extract_final_text(agent_result: dict[str, Any]) -> str:
	"""Extract assistant text from LangGraph agent output."""
	messages = agent_result.get("messages", [])
	if messages:
		final_message = messages[-1]
		content = getattr(final_message, "content", "")
		if isinstance(content, list):
			return "\n".join(str(part) for part in content)
		return str(content)
	return str(agent_result)


# ============================================================================
# PIPELINE
# ============================================================================


def run_agentic_rag_pipeline(query: str) -> str:
	"""Run agentic chunking -> indexing -> retrieval-based answering."""
	if not os.getenv("OPENAI_API_KEY"):
		raise EnvironmentError("OPENAI_API_KEY is not set.")

	print("=" * 60)
	print("AGENTIC RAG PIPELINE - AGENTIC CHUNKING")
	print("=" * 60)

	print("\n[1] Initializing ChromaDB...")
	embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
	vector_store = initialize_vector_store([], embeddings)
	existing_count = vector_store._collection.count()
	print(f"Found {existing_count} chunks in vector store")

	if existing_count == 0 or FORCE_REINDEX:
		print("\n[2] Loading source documents...")
		docs = load_sample_documents()
		print(f"Loaded {len(docs)} documents")

		print("\n[3] Building agentic chunks...")
		chunking_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
		agentic_chunks = agentic_chunk_documents(docs, chunking_llm)
		print(f"Created {len(agentic_chunks)} agentic chunks")

		print("\n[4] Indexing agentic chunks...")
		vector_store = initialize_vector_store(agentic_chunks, embeddings)
		print(f"Indexed {vector_store._collection.count()} chunks")
	else:
		print("Skipping document loading and re-indexing because an existing vector store is available.")

	print("\n[5] Creating ReAct answering agent...")
	agent = create_rag_agent(vector_store)

	print(f"\n[6] Query: {query}")
	result = agent.invoke({"messages": [HumanMessage(content=query)]})
	answer = extract_final_text(result)

	print("\nAnswer:")
	print(answer)
	return answer


if __name__ == "__main__":
	test_query = "What is Amazon's year-over-year change in revenue from FY2016 to FY2017 (in units of percents and round to one decimal place)? Calculate what was asked by utilizing the line items clearly shown in the statement of income."
	run_agentic_rag_pipeline(test_query)
