"""Adapter around the official ExtractBench evaluator (optional dependency).

Install with `pip install -e ".[official]"`. Its `string_semantic` and `array_llm`
metrics call an LLM judge through LiteLLM, so scoring needs network access and a judge
model/credentials; the judge choice affects scores. Deterministic metrics run locally.
"""

import asyncio
import dataclasses
import os
from datetime import datetime, timezone
from typing import Any

from .models import EvaluationResult, EvaluationSummary, FieldResult, OutputValidation

DEFAULT_JUDGE_MODEL = "gemini/gemini-2.5-flash"  # AI Studio route to the benchmark's default judge family
DESCRIPTION = (
    "Official ExtractBench scorer (extract-bench package), with an LLM judge for "
    "semantic/array fields."
)


class OfficialScorerUnavailable(RuntimeError):
    pass


def _absent(v: Any) -> bool:
    return v is None or v == []


def _outcome(passed: bool, gold: Any, pred: Any) -> str:
    if passed:
        return "correct"
    if _absent(pred) and not _absent(gold):
        return "missing"
    if _absent(gold) and not _absent(pred):
        return "unexpected"
    return "incorrect"


def score_official(
    schema: dict,
    gold: dict,
    prediction: Any,
    *,
    doc_id: str,
    reference_sha256: str,
    prediction_status: str,
    output_validation: OutputValidation | None = None,
    run_id: str | None = None,
    judge_model: str | None = None,
) -> EvaluationResult:
    # Must be set before extract_bench is first imported (read at import time).
    os.environ["DEFAULT_LLM_MODEL"] = judge_model or DEFAULT_JUDGE_MODEL
    try:
        from extract_bench import ReportBuilder, ReportConfig
    except ImportError as exc:
        raise OfficialScorerUnavailable(
            'extract-bench is not installed. Run: pip install -e ".[official]"'
        ) from exc

    if not isinstance(prediction, dict):
        raise ValueError(
            f"No usable prediction for {doc_id} ({prediction_status}); "
            "the official scorer needs a JSON object. Use the baseline scorer to record it as invalid."
        )
    report = asyncio.run(ReportBuilder(ReportConfig()).build_async(schema, gold, prediction))
    # The official evaluator records judge failures as score 0 with reason "error"
    # instead of raising; scoring those would silently deflate results.
    failed = [o.normalized_path or o.path for o in report.field_outcomes if o.reason == "error"]
    if failed:
        raise RuntimeError(
            f"LLM judge failed for {len(failed)} field(s) (e.g. {failed[0]}) using model "
            f"{os.environ['DEFAULT_LLM_MODEL']!r}. Check GEMINI_API_KEY / --judge-model; "
            "no evaluation was saved."
        )
    fields = [
        FieldResult(
            path=o.normalized_path or o.path,
            metric=o.metric_id,
            outcome=_outcome(o.passed, o.gold_value, o.extracted_value),
            score=float(o.score),
            passed=bool(o.passed),
            gold=o.gold_value,
            predicted=o.extracted_value,
            note=o.reason,
            details={
                "reasoning": o.reasoning,
                "array_breakdown": dataclasses.asdict(o.array_breakdown)
                if o.array_breakdown else None,
            },
        )
        for o in report.field_outcomes
    ]
    counts = {k: 0 for k in ("correct", "incorrect", "missing", "unexpected", "invalid")}
    for f in fields:
        counts[f.outcome] += 1
    return EvaluationResult(
        scorer="official",
        scorer_description=DESCRIPTION,
        doc_id=doc_id,
        run_id=run_id,
        reference_sha256=reference_sha256,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        prediction_status=prediction_status,
        output_validation=output_validation,
        summary=EvaluationSummary(
            n_fields=len(fields),
            outcome_counts=counts,
            field_score=round(report.field_score, 4),
            pass_rate=round(report.overall_pass_rate, 4),
        ),
        fields=fields,
        official_extra={
            "judge_model": os.environ["DEFAULT_LLM_MODEL"],
            "overall_score_item_weighted": round(report.overall_score, 4),
        },
    )
