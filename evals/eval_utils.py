"""Shared utilities for evaluation scripts."""

from __future__ import annotations

import csv
import json
import re
import threading
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Ensure retrieval/ is importable (safe to call multiple times)
# ---------------------------------------------------------------------------
import sys
PROJECT_ROOT = Path(__file__).parents[1]
RETRIEVAL_ROOT = PROJECT_ROOT / "retrieval"
if str(RETRIEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_ROOT))

from indexes.bm25 import BM25Index
from indexes.chroma import ChromaIndex
from retrievers.bm25 import BM25Retriever
from retrievers.chroma import ChromaRetriever
from generators.traditional import TraditionalGenerator
from query_transformers.hyde import HyDETransformer
from query_transformers.identity import IdentityTransformer

import evaluate as retrieval_eval

DB_DIR = retrieval_eval.DB_DIR
LLM_MODEL = retrieval_eval.LLM_MODEL


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def sanitize(text: str) -> str:
    """Remove null bytes and ASCII control characters."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text).strip()


def dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[tuple[str, Any, str]] = set()
    unique: list[Document] = []
    for doc in documents:
        key = (str(doc.metadata.get("source", "")), doc.metadata.get("chunk_index"), doc.page_content)
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def extract_sources(documents: list[Document]) -> list[str]:
    return [str(doc.metadata.get("source", "")) for doc in documents]


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def make_chroma_retriever(store: Chroma, query_transformer=None) -> ChromaRetriever:
    """Wrap an existing Chroma store in a ChromaRetriever without re-embedding."""
    idx = ChromaIndex(persist_directory=str(DB_DIR))
    idx._vector_store = store
    return ChromaRetriever(idx, query_transformer=query_transformer)


def run_traditional_rag(query: str, store: Chroma, k: int, llm_model: str = LLM_MODEL) -> dict[str, Any]:
    docs = make_chroma_retriever(store, IdentityTransformer()).retrieve(query, k=k)
    answer = TraditionalGenerator(llm_model=llm_model).generate(query, docs)
    return {"answer": answer, "retrieved_contexts": [d.page_content for d in docs], "retrieved_sources": extract_sources(docs)}


def run_bm25(query: str, index: BM25Index, k: int, llm_model: str = LLM_MODEL) -> dict[str, Any]:
    docs = BM25Retriever(index).retrieve(query, k=k)
    answer = TraditionalGenerator(llm_model=llm_model).generate(query, docs)
    return {"answer": answer, "retrieved_contexts": [d.page_content for d in docs], "retrieved_sources": extract_sources(docs)}


def run_hyde(query: str, store: Chroma, k: int, llm_model: str = LLM_MODEL) -> dict[str, Any]:
    docs = make_chroma_retriever(store, HyDETransformer()).retrieve(query, k=k)
    answer = TraditionalGenerator(llm_model=llm_model).generate(query, docs)
    return {"answer": answer, "retrieved_contexts": [d.page_content for d in docs], "retrieved_sources": extract_sources(docs)}


def build_agent_tools(
    vector_store: Chroma,
    captured_docs: list[Document],
    lock: threading.Lock,
    k: int,
) -> list[Any]:
    @tool
    def search_knowledge_base(search_query: str, top_k: int = k) -> str:
        """Search the shared Chroma collection for relevant financial evidence."""
        results = vector_store.similarity_search_with_score(search_query, k=top_k)
        if not results:
            return "No relevant chunks found."
        docs = [doc for doc, _ in results]
        with lock:
            captured_docs.extend(docs)
        lines = ["Top retrieved chunks:"]
        for idx, (doc, score) in enumerate(results, start=1):
            lines.append(f"[{idx}] score={score:.4f}")
            lines.append(f"source={doc.metadata.get('source', 'unknown')}")
            lines.append(doc.page_content[:1200])
            lines.append("-")
        return "\n".join(lines)

    @tool
    def get_chunk_count() -> str:
        """Return how many chunks exist in the current shared Chroma collection."""
        return f"Chunks in vector store: {vector_store._collection.count()}"

    return [search_knowledge_base, get_chunk_count]


def run_agentic_rag(query: str, store: Chroma, k: int, llm: ChatOpenAI) -> dict[str, Any]:
    captured_docs: list[Document] = []
    lock = threading.Lock()
    from langchain.agents import create_agent

    tools = build_agent_tools(store, captured_docs, lock, k)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a financial-document QA agent. "
            "Always search the knowledge base before answering. "
            "Use only retrieved evidence. Perform calculations carefully. "
            "If the answer is not supported by the retrieved text, reply with 'Not found in context.'"
        ),
    )
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    messages = result.get("messages", [])
    content = getattr(messages[-1], "content", "") if messages else str(result)
    if isinstance(content, list):
        content = "\n".join(str(p) for p in content)
    answer = str(content)

    documents = dedupe_documents(captured_docs)
    if not documents:
        documents = make_chroma_retriever(store, IdentityTransformer()).retrieve(query, k=k)

    return {
        "answer": answer,
        "retrieved_contexts": [doc.page_content for doc in documents],
        "retrieved_sources": extract_sources(documents),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in results:
        chunking = row.get("chunking")
        method = row.get("method")
        if not chunking or not method:
            continue
        key = (chunking, method)
        bucket = grouped.setdefault(key, {
            "chunking": chunking, "method": method,
            "label": row.get("label", f"{chunking}+{method}"),
            "rows": 0, "judge_correctness": 0.0, "judge_groundedness": 0.0,
            "judge_relevance": 0.0, "judge_pass": 0.0, "latency_seconds": 0.0,
        })
        bucket["rows"] += 1
        for metric in ("judge_correctness", "judge_groundedness", "judge_relevance", "judge_pass", "latency_seconds"):
            value = row.get(metric)
            if isinstance(value, (int, float)):
                bucket[metric] += float(value)

    summary: list[dict[str, Any]] = []
    for bucket in grouped.values():
        n = max(bucket["rows"], 1)
        summary.append({
            "chunking": bucket["chunking"], "method": bucket["method"],
            "label": bucket["label"], "rows": bucket["rows"],
            "avg_judge_correctness": round(bucket["judge_correctness"] / n, 4),
            "avg_judge_groundedness": round(bucket["judge_groundedness"] / n, 4),
            "avg_judge_relevance": round(bucket["judge_relevance"] / n, 4),
            "avg_judge_pass": round(bucket["judge_pass"] / n, 4),
            "avg_latency_seconds": round(bucket["latency_seconds"] / n, 4),
        })
    summary.sort(key=lambda s: (-s["avg_judge_pass"], s["label"]))
    return summary


SUMMARY_FIELDS = [
    "chunking", "method", "label", "rows",
    "avg_judge_correctness", "avg_judge_groundedness",
    "avg_judge_relevance", "avg_judge_pass", "avg_latency_seconds",
]


def save_judge_results(
    raw: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    results_dir: Path,
    prefix: str,
) -> tuple[Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_path = results_dir / f"{prefix}_raw_results.json"
    raw_path.write_text(json.dumps({"results": raw}, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_json_path = results_dir / f"{prefix}_summary.json"
    summary_json_path.write_text(json.dumps({"summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_csv_path = results_dir / f"{prefix}_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    return raw_path, summary_json_path, summary_csv_path
