"""Download and prepare the ExtractBench credit-agreement subset.

Files are listed with the GitHub git-trees API at a pinned commit and fetched from
raw.githubusercontent.com. Each file is verified against the git blob SHA-1 that the
tree listing reports, so a corrupted or altered download is rejected.
"""

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

from .config import (
    BENCH_COMMIT,
    BENCH_PAIRS_DIR,
    BENCH_REPO,
    BENCH_SCHEMA_PATH,
    DataLayout,
)

ATTRIBUTION = """\
Source: https://github.com/{repo} at commit {commit}
Paper:  ExtractBench: A Benchmark and Evaluation Methodology for Complex Structured
        Extraction (Ferguson et al., 2026), arXiv:2602.12247
Subset: {subset}

The benchmark repository is MIT licensed (LICENSE.extract-bench, Copyright (c) 2026
Contextual AI). The dataset README says only "See the repository LICENSE for terms".
The PDFs are credit agreements from public company filings; the MIT notice may not
cover rights in the underlying filings. Check before redistributing.
"""


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "credit-doc-lab"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def raw_url(path: str, commit: str = BENCH_COMMIT) -> str:
    return f"https://raw.githubusercontent.com/{BENCH_REPO}/{commit}/{urllib.parse.quote(path)}"


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def list_tree(commit: str = BENCH_COMMIT) -> dict[str, str]:
    """Return {path: blob_sha} for every file at the pinned commit."""
    url = f"https://api.github.com/repos/{BENCH_REPO}/git/trees/{commit}?recursive=1"
    tree = json.loads(_get(url))
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree listing was truncated; cannot enumerate dataset files")
    return {e["path"]: e["sha"] for e in tree["tree"] if e["type"] == "blob"}


def select_documents(tree: dict[str, str]) -> list[str]:
    """Doc IDs that have both a PDF and a gold file directly under pdf+gold/.

    Files under pdf+gold/_extra/ have no gold answers and are skipped.
    """
    prefix = BENCH_PAIRS_DIR + "/"
    pdfs, golds = set(), set()
    for path in tree:
        if not path.startswith(prefix) or "/" in path[len(prefix):]:
            continue
        name = path[len(prefix):]
        if name.endswith(".pdf"):
            pdfs.add(name[: -len(".pdf")])
        elif name.endswith(".gold.json"):
            golds.add(name[: -len(".gold.json")])
    return sorted(pdfs & golds)


def _fetch_verified(path: str, expected_blob: str, dest: Path) -> dict:
    if dest.exists() and git_blob_sha1(dest.read_bytes()) == expected_blob:
        data = dest.read_bytes()
    else:
        data = _get(raw_url(path))
        if git_blob_sha1(data) != expected_blob:
            raise RuntimeError(f"Integrity check failed for {path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return {
        "source_path": path,
        "source_url": raw_url(path),
        "git_blob_sha1": expected_blob,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def prepare(layout: DataLayout, only: list[str] | None = None) -> dict:
    """Download the subset into `layout` and write a manifest. Idempotent."""
    tree = list_tree()
    doc_ids = select_documents(tree)
    if only:
        unknown = sorted(set(only) - set(doc_ids))
        if unknown:
            raise ValueError(f"Unknown document id(s): {unknown}. Available: {doc_ids}")
        doc_ids = [d for d in doc_ids if d in only]

    files: dict[str, dict] = {}
    files["schema"] = _fetch_verified(BENCH_SCHEMA_PATH, tree[BENCH_SCHEMA_PATH], layout.schema)
    files["license"] = _fetch_verified(
        "LICENSE", tree["LICENSE"], layout.root / "LICENSE.extract-bench"
    )
    documents = {}
    for doc_id in doc_ids:
        pdf_path = f"{BENCH_PAIRS_DIR}/{doc_id}.pdf"
        gold_path = f"{BENCH_PAIRS_DIR}/{doc_id}.gold.json"
        documents[doc_id] = {
            "pdf": _fetch_verified(pdf_path, tree[pdf_path], layout.pdf(doc_id)),
            "reference": _fetch_verified(
                gold_path, tree[gold_path], layout.reference_file(doc_id)
            ),
        }
    manifest = {
        "source_repo": f"https://github.com/{BENCH_REPO}",
        "source_commit": BENCH_COMMIT,
        "files": files,
        "documents": documents,
    }
    layout.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    (layout.root / "ATTRIBUTION.txt").write_text(
        ATTRIBUTION.format(repo=BENCH_REPO, commit=BENCH_COMMIT, subset=BENCH_PAIRS_DIR)
    )
    return manifest


def list_prepared(layout: DataLayout) -> list[str]:
    if not layout.pdfs.exists():
        return []
    return sorted(p.stem for p in layout.pdfs.glob("*.pdf"))
