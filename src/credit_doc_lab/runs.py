"""Persisting runs, evaluating saved predictions, and splits."""

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import DataLayout
from .models import EvaluationResult, OutputValidation, RunRecord
from .schema import load_schema, validate_output
from .scoring import score_prediction

RUN_FILE = "run.json"


def save_run(record: RunRecord, runs_dir: Path) -> Path:
    run_dir = Path(runs_dir) / record.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / RUN_FILE).write_text(record.model_dump_json(indent=2) + "\n")
    return run_dir


def load_run(run_dir: Path) -> RunRecord:
    return RunRecord.model_validate_json((Path(run_dir) / RUN_FILE).read_text())


def find_runs(runs_dir: Path) -> list[Path]:
    return sorted(p.parent for p in Path(runs_dir).glob(f"*/{RUN_FILE}"))


def _load_reference(layout: DataLayout, doc_id: str) -> tuple[dict, str]:
    path = layout.reference_file(doc_id)
    if not path.exists():
        raise FileNotFoundError(f"No reference answers for {doc_id!r} at {path}. Run `prepare` first.")
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def evaluate_run(
    run_dir: Path,
    layout: DataLayout,
    *,
    scorer: str = "baseline",
    judge_model: str | None = None,
) -> EvaluationResult:
    """Evaluate a saved run against reference answers. Never calls the extraction model."""
    record = load_run(run_dir)
    schema = load_schema(layout.schema)
    return evaluate_prediction(
        schema, layout, record.document.doc_id, record.parsed_prediction,
        prediction_status=record.status, output_validation=record.validation,
        run_id=record.run_id, scorer=scorer, judge_model=judge_model, out_dir=Path(run_dir),
    )


def evaluate_prediction(
    schema: dict,
    layout: DataLayout,
    doc_id: str,
    prediction: Any,
    *,
    prediction_status: str,
    output_validation: OutputValidation | None = None,
    run_id: str | None = None,
    scorer: str = "baseline",
    judge_model: str | None = None,
    out_dir: Path | None = None,
) -> EvaluationResult:
    gold, ref_sha = _load_reference(layout, doc_id)
    kwargs = dict(
        doc_id=doc_id, reference_sha256=ref_sha, prediction_status=prediction_status,
        output_validation=output_validation, run_id=run_id,
    )
    if scorer == "baseline":
        result = score_prediction(schema, gold, prediction, **kwargs)
    elif scorer == "official":
        from .official import score_official

        result = score_official(schema, gold, prediction, judge_model=judge_model, **kwargs)
    else:
        raise ValueError(f"Unknown scorer: {scorer}")
    if out_dir is not None:
        (out_dir / f"evaluation_{scorer}.json").write_text(result.model_dump_json(indent=2) + "\n")
    return result


def validation_for_prediction(prediction: Any, schema: dict) -> OutputValidation:
    errors = validate_output(prediction, schema)
    return OutputValidation(status="schema_violations" if errors else "valid", errors=errors)


def make_split(doc_ids: list[str], holdout: int, seed: int) -> dict:
    """Deterministic dev/holdout split: order by hash(seed, id), first `holdout` reserved."""
    if not 0 < holdout < len(doc_ids):
        raise ValueError(f"--holdout must be between 1 and {len(doc_ids) - 1}")
    ranked = sorted(doc_ids, key=lambda d: hashlib.sha256(f"{seed}:{d}".encode()).hexdigest())
    return {
        "seed": seed,
        "dev": sorted(ranked[holdout:]),
        "holdout": sorted(ranked[:holdout]),
    }
