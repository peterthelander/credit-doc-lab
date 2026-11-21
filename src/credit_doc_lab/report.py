"""Experiment reporting: a Markdown summary of saved runs and their evaluations."""

from pathlib import Path

from .models import EvaluationResult
from .runs import find_runs, load_run


def _tokens(usage: dict | None) -> str:
    """Token counts exactly as the API reported them; absent keys are omitted."""
    keys = (("in", "prompt_token_count"), ("out", "candidates_token_count"),
            ("thinking", "thoughts_token_count"), ("total", "total_token_count"))
    parts = [f"{label}={usage[k]}" for label, k in keys if usage and usage.get(k) is not None]
    return " ".join(parts) or "n/a"


def build_report(runs_dir: Path, scorer: str = "baseline", details: bool = False) -> str:
    rows, groups = [], {}
    for run_dir in find_runs(runs_dir):
        rec = load_run(run_dir)
        eval_path = run_dir / f"evaluation_{scorer}.json"
        ev = EvaluationResult.model_validate_json(eval_path.read_text()) if eval_path.exists() else None
        rows.append((rec, ev))
        if ev:
            groups.setdefault((rec.model, rec.prompt_version), []).append(ev)

    lines = [f"# Run report (scorer: {scorer})", ""]
    if scorer == "baseline":
        lines += [
            "> Scores come from the offline **baseline scorer** (string-similarity approximation),",
            "> not the official ExtractBench scorer; do not compare them with published numbers.",
            "",
        ]
    lines += [
        "| run | document | model | status | sec | tokens | fields ok | correct | incorrect | missing | unexpected | invalid |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rec, ev in rows:
        if ev:
            c = ev.summary.outcome_counts
            tail = (f"{ev.summary.pass_rate:.1%}", c["correct"], c["incorrect"], c["missing"], c["unexpected"], c["invalid"])
        else:
            tail = ("not evaluated",) + ("",) * 5
        lines.append(
            f"| {rec.run_id} | {rec.document.doc_id} | {rec.model} | {rec.status} | "
            f"{rec.elapsed_seconds} | {_tokens(rec.token_usage)} | " + " | ".join(map(str, tail)) + " |"
        )
    if groups:
        lines += ["", "## Mean per model/prompt (macro-average over documents)", "",
                  "| model | prompt | docs | mean field pass rate | mean field score |", "|---|---|---|---|---|"]
        for (model, prompt), evs in groups.items():
            n = len(evs)
            lines.append(
                f"| {model} | {prompt} | {n} | {sum(e.summary.pass_rate for e in evs) / n:.1%} | "
                f"{sum(e.summary.field_score for e in evs) / n:.3f} |"
            )
        lines += ["", "Small sample: no confidence intervals are computed; differences between "
                  "runs on a handful of documents may be noise."]
    if details:
        for rec, ev in rows:
            if not ev:
                continue
            bad = [f for f in ev.fields if f.outcome != "correct"]
            lines += ["", f"## {rec.document.doc_id} ({rec.run_id}): {len(bad)} non-correct field(s)"]
            for f in bad:
                lines.append(f"- `{f.path}` [{f.outcome}] gold={_short(f.gold)} predicted={_short(f.predicted)}")
    return "\n".join(lines) + "\n"


def _short(v, n: int = 80) -> str:
    s = repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."
