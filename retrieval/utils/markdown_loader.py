"""Markdown document loader."""

from pathlib import Path
from typing import List

from langchain_core.documents import Document


def load_markdown_documents(texts_dir: Path) -> List[Document]:
    """
    Load markdown documents from a directory.

    Args:
        texts_dir: Directory containing .md files

    Returns:
        List of Document objects with page_content and metadata

    Raises:
        FileNotFoundError: If directory doesn't exist or contains no .md files
        ValueError: If all markdown files are empty
    """
    if not texts_dir.exists() or not texts_dir.is_dir():
        raise FileNotFoundError(f"Texts directory not found: {texts_dir}")

    markdown_files = sorted(texts_dir.glob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"No markdown files found in: {texts_dir}")

    documents: List[Document] = []
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

    if not documents:
        raise ValueError(f"All markdown files in {texts_dir} were empty")

    return documents
