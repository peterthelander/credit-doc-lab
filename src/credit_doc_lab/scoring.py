"""Baseline scorer: deterministic, offline, no LLM judge.

This is NOT the official ExtractBench scorer. The official scorer uses LLM judges for
`string_semantic` and `array_llm` fields; this baseline approximates those with string
similarity so evaluation works offline and reproducibly. Known limitations:

* Paraphrases that mean the same thing are scored as wrong (under-credit).
* Near-verbatim text with a meaningful difference can pass (over-credit); short strings
  containing digits (dates, amounts) must match exactly to reduce this.
* Array items are matched greedily, one-to-one, by string similarity.
* Scores are not comparable with published ExtractBench numbers.

Use `--scorer official` for the benchmark's own method.
"""

import math
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from .models import EvaluationResult, EvaluationSummary, FieldResult, OutputValidation

SCORER_DESCRIPTION = (
    "Baseline scorer (offline, string-similarity approximation of the benchmark's "
    "metrics; not the official LLM-judged ExtractBench scorer)."
)
FUZZY_THRESHOLD = 0.8  # same default as the official string_fuzzy metric
NUMBER_TOLERANCE = 1e-6

_MISSING = object()
_DEFAULT_METRIC = {
    "string": "string_semantic",
    "number": "number_tolerance",
    "integer": "integer_exact",
    "boolean": "boolean_exact",
    "array": "array_llm",
}


def _is_absent(value: Any, node_type: str | None) -> bool:
    return value is _MISSING or value is None or (node_type == "array" and value == [])


def _node_type(node: dict) -> str | None:
    t = node.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t is None and "anyOf" in node:
        for branch in node["anyOf"]:
            bt = _node_type(branch)
            if bt and bt != "null":
                return bt
    return t if t != "null" else None


def _branch(node: dict, kind: str) -> dict:
    """The sub-schema for `kind`, looking through anyOf-with-null."""
    if node.get("type") == kind:
        return node
    for b in node.get("anyOf", []):
        if b.get("type") == kind:
            return b
    return node


def _metric_id(node: dict, node_type: str | None) -> str:
    cfg = node.get("evaluation_config")
    if isinstance(cfg, str):
        return cfg
    if isinstance(cfg, dict) and cfg.get("metrics"):
        return cfg["metrics"][0]["metric_id"]
    return _DEFAULT_METRIC.get(node_type or "", "string_semantic")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip(" \t.,;:")


def _similar(gold: str, pred: str) -> float:
    g, p = _norm(gold), _norm(pred)
    if g == p:
        return 1.0
    if re.search(r"\d", g) and len(g) <= 40:
        return 0.0  # dates/amounts/short identifiers: exact only
    return SequenceMatcher(None, g, p, autojunk=False).ratio()


def _type_ok(value: Any, node_type: str | None) -> bool:
    if node_type == "string":
        return isinstance(value, str)
    if node_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if node_type == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer()
        )
    if node_type == "boolean":
        return isinstance(value, bool)
    if node_type == "array":
        return isinstance(value, list)
    if node_type == "object":
        return isinstance(value, dict)
    return True


def _score_array(gold: list, pred: list) -> tuple[float, bool, dict]:
    unused = list(range(len(pred)))
    matched, missed = [], []
    for g in gold:
        best, best_sim = None, 0.0
        for i in unused:
            sim = (
                _similar(g, pred[i])
                if isinstance(g, str) and isinstance(pred[i], str)
                else float(g == pred[i])
            )
            if sim > best_sim:
                best, best_sim = i, sim
        if best is not None and best_sim >= FUZZY_THRESHOLD:
            unused.remove(best)
            matched.append(g)
        else:
            missed.append(g)
    spurious = [pred[i] for i in unused]
    recall = len(matched) / len(gold)
    precision = len(matched) / (len(matched) + len(spurious)) if matched or spurious else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    details = {
        "matched": len(matched),
        "missed_gold": missed,
        "spurious_pred": spurious,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    return recall, not missed and not spurious, details  # score is recall, as in the official scorer


def _compare(metric: str, gold: Any, pred: Any) -> tuple[float, bool, str | None, dict | None]:
    if metric == "string_exact":
        ok = gold == pred
        return float(ok), ok, None, None
    if metric == "string_case_insensitive":
        ok = gold.casefold() == pred.casefold()
        return float(ok), ok, None, None
    if metric in ("string_fuzzy", "string_semantic", "string_llm"):
        sim = _similar(gold, pred)
        note = None if metric == "string_fuzzy" else f"{metric} approximated by string similarity"
        return round(sim, 4), sim >= FUZZY_THRESHOLD, note, None
    if metric in ("number_exact", "number_tolerance"):
        tol = 0.0 if metric == "number_exact" else NUMBER_TOLERANCE
        ok = math.isclose(gold, pred, rel_tol=tol, abs_tol=tol)
        return float(ok), ok, None, None
    if metric in ("integer_exact", "boolean_exact"):
        ok = gold == pred
        return float(ok), ok, None, None
    if metric in ("array_llm", "array"):
        score, ok, details = _score_array(gold, pred)
        return round(score, 4), ok, "array_llm approximated by greedy similarity matching", details
    raise ValueError(f"Unsupported metric for baseline scorer: {metric}")


def _score_leaf(path: str, node: dict, gold: Any, pred: Any) -> FieldResult | None:
    node_type = _node_type(node)
    metric = _metric_id(node, node_type)
    if metric == "skip":
        return None
    g_absent, p_absent = _is_absent(gold, node_type), _is_absent(pred, node_type)
    g_val = None if gold is _MISSING else gold
    p_val = None if pred is _MISSING else pred

    def result(outcome, score, passed, note=None, details=None):
        return FieldResult(
            path=path, metric=metric, outcome=outcome, score=score, passed=passed,
            gold=g_val, predicted=p_val, note=note, details=details,
        )

    if g_absent and p_absent:
        return result("correct", 1.0, True, "both absent")
    if p_absent:
        return result("missing", 0.0, False)
    if g_absent:
        return result("unexpected", 0.0, False)
    if not _type_ok(pred, node_type):
        return result("invalid", 0.0, False, f"expected {node_type}, got {type(pred).__name__}")
    if node_type == "array":
        items = _branch(node, "array").get("items", {})
        item_type = items.get("type")
        if item_type and not all(_type_ok(i, item_type) for i in pred):
            return result("invalid", 0.0, False, f"array items must be {item_type}")
    score, passed, note, details = _compare(metric, gold, pred)
    return result("correct" if passed else "incorrect", score, passed, note, details)


def _walk(node: dict, path: str, gold: Any, pred: Any, out: list[FieldResult]) -> None:
    node_type = _node_type(node)
    if node_type == "object":
        obj = _branch(node, "object")
        props = obj.get("properties", {})
        if pred is not _MISSING and pred is not None and not isinstance(pred, dict):
            out.append(FieldResult(
                path=path or "<root>", metric="object", outcome="invalid", score=0.0,
                passed=False, predicted=pred, note=f"expected object, got {type(pred).__name__}",
            ))
            return
        g = gold if isinstance(gold, dict) else {}
        p = pred if isinstance(pred, dict) else {}
        for key, child in props.items():
            _walk(child, f"{path}.{key}" if path else key, g.get(key, _MISSING), p.get(key, _MISSING), out)
        for key in p.keys() - props.keys():
            out.append(FieldResult(
                path=f"{path}.{key}" if path else key, metric="none", outcome="unexpected",
                score=0.0, passed=False, predicted=p[key], note="key not in schema",
            ))
        return
    field = _score_leaf(path, node, gold, pred)
    if field is not None:
        out.append(field)


def score_prediction(
    schema: dict,
    gold: dict,
    prediction: Any,
    *,
    doc_id: str,
    reference_sha256: str,
    prediction_status: str,
    output_validation: OutputValidation | None = None,
    run_id: str | None = None,
) -> EvaluationResult:
    """Score one prediction. A missing/unparseable prediction marks every gold field."""
    fields: list[FieldResult] = []
    if prediction is None or not isinstance(prediction, dict):
        pred_for_walk: Any = _MISSING
        _walk(schema, "", gold, pred_for_walk, fields)
        # No usable output: no field earns credit, not even "both absent".
        fields = [
            f.model_copy(update={
                "outcome": "invalid", "score": 0.0, "passed": False,
                "note": f"no usable output ({prediction_status})",
            })
            for f in fields
        ]
    else:
        _walk(schema, "", gold, prediction, fields)

    counts = {o: 0 for o in ("correct", "incorrect", "missing", "unexpected", "invalid")}
    for f in fields:
        counts[f.outcome] += 1
    n = len(fields)
    summary = EvaluationSummary(
        n_fields=n,
        outcome_counts=counts,
        field_score=round(sum(f.score for f in fields) / n, 4) if n else 0.0,
        pass_rate=round(sum(f.passed for f in fields) / n, 4) if n else 0.0,
    )
    return EvaluationResult(
        scorer="baseline",
        scorer_description=SCORER_DESCRIPTION,
        doc_id=doc_id,
        run_id=run_id,
        reference_sha256=reference_sha256,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        prediction_status=prediction_status,
        output_validation=output_validation,
        summary=summary,
        fields=fields,
    )
