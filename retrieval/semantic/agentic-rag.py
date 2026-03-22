"""
Agentic RAG pipeline with semantic chunking.

Semantic chunking groups adjacent sentences by embedding similarity,
stores chunks in ChromaDB, and uses a ReAct-style agent to answer queries
with retrieval tools.

References:
- ChromaDB: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
- ReAct agents: https://docs.langchain.com/oss/python/langchain/agents
"""

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
COLLECTION_NAME = "semantic_chunks_collection"

MAX_SENTENCES_PER_CHUNK = 5
OVERLAP_SENTENCES = 1
BREAKPOINT_PERCENTILE = 30.0


# ============================================================================
# DOCUMENTS
# ============================================================================


def load_sample_documents() -> list[Document]:
	"""Load sample data. Replace this with your dataset loader."""
	return [
		Document(
			page_content=(
				"Python is a high-level programming language. "
				"It is popular for data science and automation. "
				"The language emphasizes readability and fast iteration. "
				"Many teams use Python to build retrieval pipelines. "
				"Python also has mature AI tooling and integrations."
			),
			metadata={"source": "python_overview.txt", "topic": "programming"},
		),
		Document(
			page_content=(
				"Retrieval augmented generation combines retrieval and generation. "
				"The retriever finds relevant context from a vector store. "
				"The language model then answers using grounded evidence. "
				"This approach reduces hallucinations in many workflows. "
				"Evaluation still matters because retrieval quality varies by corpus."
			),
			metadata={"source": "rag_notes.txt", "topic": "rag"},
		),
		Document(
			page_content=(
				"Agentic systems can call tools during reasoning. "
				"A ReAct agent alternates between thinking and acting. "
				"Tool calls often include retrieval, calculators, or APIs. "
				"When retrieval is available, agents can answer with better context. "
				"This is useful for multi-step question answering."
			),
			metadata={"source": "agent_notes.txt", "topic": "agents"},
		),
	]


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

	# Rebuild collection each run so repeated runs do not duplicate chunks.
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
	"""Create retrieval tools used by the ReAct agent."""

	@tool
	def search_knowledge_base(query: str, k: int = 4) -> str:
		"""Search the semantic chunk index and return the top matching chunks."""
		results = vector_store.similarity_search_with_score(query, k=k)
		if not results:
			return "No relevant chunks found."

		lines = ["Top semantic chunks:"]
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
	"""Build a ReAct-style agent with semantic retrieval tools."""
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
	"""Run semantic chunking -> indexing -> agentic retrieval + answer generation."""
	if not os.getenv("OPENAI_API_KEY"):
		raise EnvironmentError("OPENAI_API_KEY is not set.")

	print("=" * 60)
	print("AGENTIC RAG PIPELINE - SEMANTIC CHUNKING")
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

	print("\n[4] Creating ReAct agent...")
	agent = create_rag_agent(vector_store)

	print(f"\n[5] Query: {query}")
	result = agent.invoke({"messages": [HumanMessage(content=query)]})
	answer = extract_final_text(result)

	print("\nAnswer:")
	print(answer)
	return answer


if __name__ == "__main__":
	test_query = "Explain how agentic RAG works with semantic chunking."
	run_agentic_rag_pipeline(test_query)
