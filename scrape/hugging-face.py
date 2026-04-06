"""
Extract doc_link column from PatronusAI/financebench HuggingFace dataset.
Download PDFs from those URLs into ./data.
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse, unquote
from urllib.request import Request, urlopen

from datasets import load_dataset

# ./data relative to the process working directory (e.g. repo root when you run from there)
DATA_DIR = Path("data")
CHUNK_SIZE = 256 * 1024
USER_AGENT = "IS469-research/0.1 (+https://huggingface.co/datasets/PatronusAI/financebench)"
# Per-request socket timeout (ms → s). Applies to connection and read operations.
TIMEOUT_S = 5000 / 1000.0


def get_doc_links() -> list[str]:
    """Load the FinanceBench dataset and return all unique doc_links."""
    print("Loading PatronusAI/financebench dataset...")
    dataset = load_dataset("PatronusAI/financebench", split="train")

    print(f"Dataset loaded: {len(dataset)} rows")
    print(f"Columns: {dataset.column_names}\n")

    doc_links = dataset["doc_link"]
    unique_links = sorted(set(doc_links))

    print(f"Total doc_link values : {len(doc_links)}")
    print(f"Unique doc_link values: {len(unique_links)}\n")

    return unique_links


def _path_for_url(url: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    parsed = urlparse(url)
    tail = unquote(Path(parsed.path).name)
    if not tail or not tail.lower().endswith(".pdf"):
        tail = f"{digest}.pdf"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in tail)
    return DATA_DIR / f"{digest}_{safe}"


def _download_one(url: str, skip_existing: bool) -> tuple[Path | None, str | None]:
    """
    Download one URL. Returns (path, None) on success or skip-existing;
    (None, url) on failure; (None, None) if url is empty (skipped).
    """
    url = str(url).strip()
    if not url:
        return None, None
    dest = _path_for_url(url)
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        print(f"skip (exists): {dest.name}")
        return dest, None
    print(f"downloading -> {dest.name}")
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=TIMEOUT_S) as resp:
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
    except (TimeoutError, OSError, URLError) as e:
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        print(f"FAILED ({type(e).__name__}): {dest.name} — {e}")
        return None, url
    return dest, None


def download_pdfs(
    urls: list[str],
    *,
    skip_existing: bool = True,
    max_workers: int | None = None,
) -> tuple[list[Path], list[str]]:
    """
    Download URLs concurrently into ./data.

    Returns (paths_ok, urls_failed). paths_ok includes skipped existing files.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = len(urls)
    workers = max_workers if max_workers is not None else min(32, max(1, n))
    ok: list[Path] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_download_one, u, skip_existing): u for u in urls}
        for fut in as_completed(futs):
            path, bad = fut.result()
            if path is not None:
                ok.append(path)
            if bad is not None:
                failed.append(bad)
    failed.sort()
    return ok, failed


if __name__ == "__main__":
    links = get_doc_links()
    print(links)
    paths, failed_urls = download_pdfs(links)
    print(f"Done. {len(paths)} file(s) in {DATA_DIR}")
    print(f"Failed URLs ({len(failed_urls)}):")
    for u in failed_urls:
        print(u)