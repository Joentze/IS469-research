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
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent


# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
VECTOR_STORE_DIR = "./database"
COLLECTION_NAME = "agentic_chunks_collection"

MAX_SENTENCES_PER_CHUNK = 6
MIN_SENTENCES_PER_CHUNK = 2
OVERLAP_SENTENCES = 1


# ============================================================================
# DOCUMENTS
# ============================================================================


def load_sample_documents() -> list[Document]:
	"""Load sample data. Replace this with your dataset loader."""
	return [
		Document(
			page_content=(
				"Python is a high-level programming language. "
				"It supports procedural, object-oriented, and functional styles. "
				"Developers use Python for APIs, automation, and data analysis. "
				"Python has a rich ecosystem for machine learning. "
				"Libraries like pandas and NumPy are common in analytics workflows. "
				"The language is often chosen for rapid prototyping."
			),
			metadata={"source": "python_overview.txt", "topic": "programming"},
		),
		Document(
			page_content=(
				"Retrieval augmented generation combines retrieval and generation. "
				"A retriever fetches context from a vector store. "
				"The LLM answers using the retrieved evidence. "
				"This generally improves factual grounding. "
				"Chunking strategy strongly affects retrieval quality. "
				"Evaluation should include precision and citation quality."
			),
			metadata={"source": "rag_notes.txt", "topic": "rag"},
		),
		Document(
			page_content=(
				"ReAct agents reason in steps and call tools when needed. "
				"Tool calls can retrieve context, run code, or query APIs. "
				"After observing tool output, the agent updates its plan. "
				"This loop helps with multi-step tasks. "
				"RAG tools are a strong fit for grounded question answering. "
				"Good prompts make tool usage more consistent."
			),
			metadata={"source": "agent_notes.txt", "topic": "agents"},
		),
	]


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


def choose_chunk_breakpoints_with_agent(sentences: list[str], llm: ChatOpenAI) -> list[int]:
	"""
	Ask an LLM to propose chunk boundaries.

	The model returns JSON with `break_after` indexes. Each index means:
	create a chunk boundary after that sentence index.
	"""
	if len(sentences) <= MAX_SENTENCES_PER_CHUNK:
		return []

	numbered_sentences = "\n".join(
		[f"{i}: {sentence}" for i, sentence in enumerate(sentences)]
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

	try:
		response = llm.invoke(prompt)
		text = str(getattr(response, "content", "")).strip()
		parsed = json.loads(text)
		raw_breaks = parsed.get("break_after", [])
		if not isinstance(raw_breaks, list):
			return _fallback_breakpoints(len(sentences))
		int_breaks = [int(x) for x in raw_breaks]
		breaks = _normalize_breakpoints(int_breaks, len(sentences))
		return breaks if breaks is not None else _fallback_breakpoints(len(sentences))
	except Exception:
		return _fallback_breakpoints(len(sentences))


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

	# Rebuild collection each run to prevent duplicate chunks.
	try:
		vector_store.delete_collection()
		vector_store = Chroma(
			collection_name=COLLECTION_NAME,
			embedding_function=embeddings,
			persist_directory=VECTOR_STORE_DIR,
		)
	except Exception:
		pass

	if documents:
		vector_store.add_documents(documents)

	return vector_store


# ============================================================================
# TOOLS + AGENT
# ============================================================================


def create_agent_tools(vector_store: Chroma) -> list:
	"""Create retrieval tools used by the answering ReAct agent."""

	@tool
	def search_knowledge_base(query: str, k: int = 4) -> str:
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
	return create_react_agent(model=llm, tools=tools)


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

	print("\n[1] Loading source documents...")
	docs = load_sample_documents()
	print(f"Loaded {len(docs)} documents")

	print("\n[2] Building agentic chunks...")
	chunking_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
	agentic_chunks = agentic_chunk_documents(docs, chunking_llm)
	print(f"Created {len(agentic_chunks)} agentic chunks")

	print("\n[3] Initializing ChromaDB...")
	embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
	vector_store = initialize_vector_store(agentic_chunks, embeddings)
	print(f"Indexed {vector_store._collection.count()} chunks")

	print("\n[4] Creating ReAct answering agent...")
	agent = create_rag_agent(vector_store)

	print(f"\n[5] Query: {query}")
	result = agent.invoke({"messages": [HumanMessage(content=query)]})
	answer = extract_final_text(result)

	print("\nAnswer:")
	print(answer)
	return answer


if __name__ == "__main__":
	test_query = "Explain how agentic chunking improves an agentic RAG pipeline."
	run_agentic_rag_pipeline(test_query)
