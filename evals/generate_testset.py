"""
Generate a test set of Q&A pairs from the financial document corpus.

For each sampled document, uses gpt-4o-mini to generate specific factual
questions that can only be answered by reading that document. The source
document is the ground truth for retrieval evaluation.

Usage:
    python evaluation/generate_testset.py [--n-docs 25] [--questions-per-doc 2]

Output:
    evaluation/testset.json  — list of {query, source, reference_answer}
"""

import argparse
import json
import os
import random
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).parents[1]
TEXTS_DIR = PROJECT_ROOT / "texts"
OUTPUT_FILE = Path(__file__).parent / "testset.json"

DEFAULT_N_DOCS = 25
DEFAULT_QUESTIONS_PER_DOC = 2
PREVIEW_CHARS = 4000  # chars of each doc sent to GPT for question generation


# ============================================================================
# ENVIRONMENT
# ============================================================================


def load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
    if "OPENAI_API_KEY" not in os.environ and "API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]


# ============================================================================
# QUESTION GENERATION
# ============================================================================


def generate_questions(client: OpenAI, doc_text: str, filename: str, n: int) -> list[dict]:
    """Ask GPT to generate n factual questions from the document."""
    preview = doc_text[:PREVIEW_CHARS]
    prompt = (
        f"You are building an evaluation set for a financial document retrieval system.\n\n"
        f"Document ({filename}):\n{preview}\n\n"
        f"Generate {n} specific factual questions about this document. Each question must:\n"
        f"- Require reading this exact document to answer correctly\n"
        f"- Ask about specific numbers, dates, company names, or named financial figures\n"
        f"- Be answerable with a short phrase or sentence from the document\n\n"
        f"Output a JSON array only, no explanation:\n"
        f'[{{"question": "...", "answer": "..."}}]'
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=600,
    )
    content = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        if content.startswith("json"):
            content = content[4:]

    return json.loads(content.strip())


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAG evaluation test set")
    parser.add_argument("--n-docs", type=int, default=DEFAULT_N_DOCS,
                        help="Number of documents to sample")
    parser.add_argument("--questions-per-doc", type=int, default=DEFAULT_QUESTIONS_PER_DOC,
                        help="Questions to generate per document")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set. Check your .env file.")

    client = OpenAI()

    md_files = sorted(TEXTS_DIR.glob("*.md"))
    if not md_files:
        raise ValueError(f"No .md files found in {TEXTS_DIR}")

    random.seed(args.seed)
    sampled = random.sample(md_files, min(args.n_docs, len(md_files)))
    print(f"Sampling {len(sampled)} documents from {len(md_files)} total")

    testset: list[dict] = []
    for i, filepath in enumerate(sampled):
        print(f"[{i + 1}/{len(sampled)}] {filepath.name}", end=" ... ", flush=True)
        content = filepath.read_text(encoding="utf-8", errors="ignore").strip()

        if len(content) < 200:
            print("skipped (too short)")
            continue

        try:
            qas = generate_questions(client, content, filepath.name, args.questions_per_doc)
            for qa in qas:
                testset.append({
                    "query": qa["question"],
                    "reference_answer": qa["answer"],
                    "source": filepath.name,
                })
            print(f"generated {len(qas)} questions")
        except Exception as e:
            print(f"error: {e}")

    OUTPUT_FILE.write_text(json.dumps(testset, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(testset)} test queries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
