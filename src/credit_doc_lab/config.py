"""Shared constants and path helpers."""

import os
from pathlib import Path

# Upstream benchmark pinned to an exact commit so downloads are reproducible.
BENCH_REPO = "ContextualAI/extract-bench"
BENCH_COMMIT = "ca817638852a23bb17ca9aaeb82ccdd29e9af9ca"
BENCH_SUBSET_DIR = "dataset/finance/credit_agreement"
BENCH_SCHEMA_PATH = f"{BENCH_SUBSET_DIR}/credit_agreement-schema.json"
BENCH_PAIRS_DIR = f"{BENCH_SUBSET_DIR}/pdf+gold"

DEFAULT_DATA_DIR = Path("data/credit_agreement")
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_SPLITS_FILE = Path("splits/credit_agreement.json")

# Model name comes from the official google-genai README examples; it is not
# checked against the live model list. Override with --model or the env var.
FALLBACK_MODEL = "gemini-3.8-flash"

PROMPT_VERSION = "v1"
# Conservative cap for inline PDF bytes. Larger files would need the Files API,
# which this milestone does not implement.
MAX_INLINE_PDF_BYTES = 15 * 1024 * 1024


def default_model() -> str:
    return os.environ.get("CREDIT_DOC_LAB_MODEL") or FALLBACK_MODEL


class DataLayout:
    """Where prepared data lives. Reference answers are kept in their own directory."""

    def __init__(self, root: Path = DEFAULT_DATA_DIR):
        self.root = Path(root)
        self.schema = self.root / "schema.json"
        self.pdfs = self.root / "pdfs"
        self.reference = self.root / "reference"
        self.manifest = self.root / "manifest.json"

    def pdf(self, doc_id: str) -> Path:
        return self.pdfs / f"{doc_id}.pdf"

    def reference_file(self, doc_id: str) -> Path:
        return self.reference / f"{doc_id}.gold.json"
