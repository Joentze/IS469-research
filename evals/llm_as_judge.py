"""Run a LangSmith LLM-as-a-judge evaluation over one RAG pipeline.

This script turns the local ``evals/testset.json`` into a LangSmith dataset,
reuses the retrieval pipeline code from ``evals/evaluate.py``, and scores the
generated answers with a custom LLM judge.

Example:
    python evals/llm_as_judge.py --pipeline semantic+traditional
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client, traceable, wrappers
from openai import OpenAI
from pydantic import BaseModel, Field

import evaluate as retrieval_eval


EVAL_DIR = Path(__file__).parent
DEFAULT_TESTSET = EVAL_DIR / "testset.json"
DEFAULT_DATASET_NAME = "is469-rag-eval"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

JUDGE_CLIENT = wrappers.wrap_openai(OpenAI())


class JudgeResponse(BaseModel):
    """Structured response returned by the judge model."""

    correctness: int = Field(
        ge=0,
        le=1,
        description="1 if the answer matches the reference answer, else 0.",
    )
    groundedness: int = Field(
        ge=0,
        le=1,
        description="1 if the answer is supported by the retrieved context, else 0.",
    )
    relevance: int = Field(
        ge=0,
        le=1,
        description="1 if the answer addresses the user's question, else 0.",
    )
    reasoning: str = Field(
        description="A short explanation justifying the scores.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a LangSmith LLM-as-a-judge evaluation for one RAG pipeline."
    )
    parser.add_argument(
        "--testset",
        default=str(DEFAULT_TESTSET),
        help="Path to the local JSON test set.",
    )
    parser.add_argument(
        "--pipeline",
        default="fixed+traditional",
        help=(
            "Pipeline to evaluate. "
            f"Available: {[p.name for p in retrieval_eval.ALL_PIPELINES]}"
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="LangSmith dataset name to create or reuse.",
    )
    parser.add_argument(
        "--replace-dataset",
        action="store_true",
        help="Delete and recreate the LangSmith dataset before uploading examples.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Model used by the evaluator LLM.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=retrieval_eval.DEFAULT_K,
        help="Top-k chunks to retrieve for the pipeline target.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="LangSmith evaluation concurrency. Keep low because retrieval indexes are shared.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild chunk caches and vector stores before evaluation.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default=None,
        help="Optional LangSmith experiment name prefix.",
    )
    return parser.parse_args()


def validate_environment() -> None:
    retrieval_eval.load_env()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set. Check your .env file.")

    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        raise EnvironmentError(
            "LANGSMITH_API_KEY not set. Add it to your environment or .env file."
        )


def select_pipeline(name: str) -> retrieval_eval.PipelineConfig:
    for pipeline in retrieval_eval.ALL_PIPELINES:
        if pipeline.name == name:
            return pipeline
    available = ", ".join(p.name for p in retrieval_eval.ALL_PIPELINES)
    raise ValueError(f"Unknown pipeline '{name}'. Available: {available}")


def load_testset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Test set not found at '{path}'. Run 'python evals/generate_testset.py' first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Test set at '{path}' is empty or invalid.")
    return data


def make_langsmith_examples(testset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for item in testset:
        query = item.get("query", "")
        reference_answer = item.get("reference_answer", "")
        source = item.get("source", "")
        if not query:
            continue
        examples.append(
            {
                "inputs": {"query": query},
                "outputs": {
                    "reference_answer": reference_answer,
                    "source": source,
                },
                "metadata": {"source": source},
            }
        )
    if not examples:
        raise ValueError("No usable examples found in the test set.")
    return examples


def ensure_dataset(
    client: Client,
    dataset_name: str,
    examples: list[dict[str, Any]],
    replace_dataset: bool,
) -> str:
    if replace_dataset and client.has_dataset(dataset_name=dataset_name):
        client.delete_dataset(dataset_name=dataset_name)

    if client.has_dataset(dataset_name=dataset_name):
        return dataset_name

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="IS469 financial-document QA evaluation set.",
    )
    client.create_examples(dataset_id=dataset.id, examples=examples)
    return dataset.name


def retrieve_documents(
    query: str,
    pipeline: retrieval_eval.PipelineConfig,
    stores_by: dict[str, Any],
    bm25_by: dict[str, Any],
    embeddings: OpenAIEmbeddings,
    llm: ChatOpenAI,
    k: int,
) -> list[Any]:
    if pipeline.retrieval == "traditional":
        return retrieval_eval.retrieve_traditional(query, stores_by[pipeline.chunking], k)
    if pipeline.retrieval == "bm25":
        return retrieval_eval.retrieve_bm25(query, bm25_by[pipeline.chunking], k)
    if pipeline.retrieval == "hyde":
        return retrieval_eval.retrieve_hyde(
            query,
            stores_by[pipeline.chunking],
            embeddings,
            llm,
            k,
        )
    raise ValueError(f"Unknown retrieval method: {pipeline.retrieval}")


def build_target(
    pipeline: retrieval_eval.PipelineConfig,
    stores_by: dict[str, Any],
    bm25_by: dict[str, Any],
    embeddings: OpenAIEmbeddings,
    llm: ChatOpenAI,
    k: int,
):
    @traceable(name=f"rag_pipeline_{pipeline.name.replace('+', '_')}")
    def rag_app(inputs: dict[str, Any]) -> dict[str, Any]:
        query = inputs["query"]
        docs = retrieve_documents(query, pipeline, stores_by, bm25_by, embeddings, llm, k)
        contexts = [doc.page_content for doc in docs]
        sources = [str(doc.metadata.get("source", "")) for doc in docs]
        answer = retrieval_eval.generate_answer(query, contexts, llm)
        return {
            "answer": answer,
            "retrieved_contexts": contexts,
            "retrieved_sources": sources,
            "pipeline": pipeline.name,
        }

    return rag_app


def format_contexts(contexts: list[str], limit: int = 3, chars: int = 1200) -> str:
    trimmed = []
    for i, text in enumerate(contexts[:limit], start=1):
        snippet = retrieval_eval._sanitize(text)[:chars]
        trimmed.append(f"[Context {i}]\n{snippet}")
    return "\n\n".join(trimmed) if trimmed else "[No retrieved context]"


def build_judge(judge_model: str):
    def llm_judge(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        instructions = """
You are grading an answer produced by a financial-document QA system.

Score each field as either 0 or 1.
- correctness: 1 only if the answer matches the reference answer in meaning.
- groundedness: 1 only if the answer is supported by the retrieved context.
- relevance: 1 only if the answer actually addresses the user's question.

Be strict. If the answer is vague, unsupported, or materially incomplete, give 0.
Return JSON that matches the provided schema.
""".strip()

        message = (
            f"Question: {inputs.get('query', '')}\n\n"
            f"Reference answer: {reference_outputs.get('reference_answer', '')}\n\n"
            f"Model answer: {outputs.get('answer', '')}\n\n"
            f"Retrieved sources: {outputs.get('retrieved_sources', [])}\n\n"
            f"Retrieved context:\n{format_contexts(outputs.get('retrieved_contexts', []))}"
        )

        response = JUDGE_CLIENT.beta.chat.completions.parse(
            model=judge_model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": message},
            ],
            response_format=JudgeResponse,
        )
        parsed = response.choices[0].message.parsed
        overall = int(
            parsed.correctness == 1
            and parsed.groundedness == 1
            and parsed.relevance == 1
        )

        return [
            {"key": "judge_correctness", "score": parsed.correctness},
            {"key": "judge_groundedness", "score": parsed.groundedness},
            {"key": "judge_relevance", "score": parsed.relevance},
            {"key": "judge_pass", "score": overall},
            {"key": "judge_reasoning", "value": parsed.reasoning},
        ]

    return llm_judge


def main() -> None:
    args = parse_args()
    validate_environment()

    testset_path = Path(args.testset)
    testset = load_testset(testset_path)
    examples = make_langsmith_examples(testset)
    pipeline = select_pipeline(args.pipeline)

    client = Client()
    dataset_name = ensure_dataset(
        client=client,
        dataset_name=args.dataset_name,
        examples=examples,
        replace_dataset=args.replace_dataset,
    )

    embeddings = OpenAIEmbeddings(model=retrieval_eval.EMBEDDING_MODEL)
    llm = ChatOpenAI(model=retrieval_eval.LLM_MODEL, temperature=0)
    _, stores_by, bm25_by = retrieval_eval.build_indexes(
        [pipeline], embeddings, llm, args.force_rebuild
    )

    target = build_target(
        pipeline=pipeline,
        stores_by=stores_by,
        bm25_by=bm25_by,
        embeddings=embeddings,
        llm=llm,
        k=args.k,
    )
    judge = build_judge(args.judge_model)

    experiment_prefix = args.experiment_prefix or f"{pipeline.name} llm-judge"
    description = (
        f"LLM-as-a-judge evaluation for {pipeline.name} on {len(examples)} examples "
        f"with k={args.k}."
    )

    print(f"Dataset: {dataset_name}")
    print(f"Pipeline: {pipeline.name}")
    print(f"Examples: {len(examples)}")
    print(f"Judge model: {args.judge_model}")

    results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=[judge],
        experiment_prefix=experiment_prefix,
        description=description,
        max_concurrency=args.max_concurrency,
    )

    experiment_name = getattr(results, "experiment_name", None)
    if experiment_name:
        print(f"Experiment: {experiment_name}")
    print("LangSmith evaluation complete.")


if __name__ == "__main__":
    main()
