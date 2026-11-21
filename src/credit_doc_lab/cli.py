"""Command-line interface: prepare, split, extract, batch, evaluate, report."""

import argparse
import json
import sys
import threading
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

from . import config as cfg
from .config import DataLayout

PROGRESS_INTERVAL_SECONDS = 15.0


def _progress(doc_id: str, message: str) -> None:
    print(f"{doc_id}: {message}", file=sys.stderr, flush=True)


@contextmanager
def _show_elapsed(doc_id: str) -> Iterator[None]:
    stop = threading.Event()
    started = time.monotonic()

    def report_elapsed() -> None:
        while not stop.wait(PROGRESS_INTERVAL_SECONDS):
            elapsed = time.monotonic() - started
            _progress(doc_id, f"Still working ({elapsed:.0f}s elapsed); Gemini does not report server-side progress.")

    thread = threading.Thread(target=report_elapsed, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def _layout(args) -> DataLayout:
    return DataLayout(args.data_dir)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=cfg.DEFAULT_DATA_DIR, help="prepared data directory")


def _add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help=f"Gemini model name (default: $CREDIT_DOC_LAB_MODEL or {cfg.FALLBACK_MODEL})")
    p.add_argument("--runs-dir", type=Path, default=cfg.DEFAULT_RUNS_DIR)
    p.add_argument("--temperature", type=float, default=None, help="default: model default")
    p.add_argument("--max-output-tokens", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=3, help="retries for transient API errors (default 3)")
    p.add_argument("--timeout", type=float, default=300.0, help="per-request timeout in seconds")


def _run_config(args):
    from .models import RunConfig

    return RunConfig(
        temperature=args.temperature, max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries, timeout_seconds=args.timeout,
    )


def _extract_one(pdf: Path, schema: dict, schema_path: str, doc_id: str, args, client) -> int:
    from .extract import extract_document
    from .runs import save_run

    with _show_elapsed(doc_id):
        record = extract_document(
            pdf, schema, schema_path=schema_path, model=args.model or cfg.default_model(),
            doc_id=doc_id, config=_run_config(args), client=client,
            progress=lambda message: _progress(doc_id, message),
        )
        _progress(doc_id, f"Saving run (status: {record.status}).")
        run_dir = save_run(record, args.runs_dir)
    extra = f" ({record.error})" if record.error else ""
    print(f"{doc_id}: {record.status} in {record.elapsed_seconds}s -> {run_dir}{extra}")
    return 0 if record.status == "ok" else 1


def cmd_prepare(args) -> int:
    from .dataset import prepare

    manifest = prepare(_layout(args), args.only)
    n = len(manifest["documents"])
    print(f"Prepared {n} document(s) into {args.data_dir} from {manifest['source_repo']} @ {manifest['source_commit'][:10]}")
    print("See", args.data_dir / "ATTRIBUTION.txt")
    return 0


def cmd_split(args) -> int:
    from .dataset import list_prepared
    from .runs import make_split

    ids = list_prepared(_layout(args))
    if not ids:
        sys.exit("No prepared documents. Run `credit-doc-lab prepare` first.")
    if args.splits_file.exists() and not args.force:
        sys.exit(f"{args.splits_file} exists; pass --force to overwrite (this changes the holdout!).")
    split = make_split(ids, args.holdout, args.seed)
    args.splits_file.parent.mkdir(parents=True, exist_ok=True)
    args.splits_file.write_text(json.dumps(split, indent=2) + "\n")
    print(f"dev={len(split['dev'])} holdout={len(split['holdout'])} -> {args.splits_file}")
    return 0


def cmd_extract(args) -> int:
    from .extract import make_client
    from .schema import load_schema

    if args.pdf:
        if not args.schema:
            sys.exit("--schema is required with --pdf")
        pdf, schema_path, doc_id = args.pdf, args.schema, args.pdf.stem
    else:
        layout = _layout(args)
        pdf, schema_path, doc_id = layout.pdf(args.doc), layout.schema, args.doc
    _progress(doc_id, f"Loading schema {schema_path} and preparing Gemini client.")
    schema = load_schema(schema_path)
    client = make_client(args.timeout)
    return _extract_one(pdf, schema, str(schema_path), doc_id, args, client)


def cmd_batch(args) -> int:
    from .dataset import list_prepared
    from .extract import make_client
    from .schema import load_schema

    layout = _layout(args)
    available = list_prepared(layout)
    if args.doc:
        ids = args.doc
    elif args.split == "all":
        ids = available
    else:
        if not args.splits_file.exists():
            sys.exit(f"{args.splits_file} not found; run `credit-doc-lab split` first.")
        ids = json.loads(args.splits_file.read_text())[args.split]
    if args.split == "holdout" and not args.doc and not args.allow_holdout:
        sys.exit("Refusing to run the reserved holdout without --allow-holdout. Tune on dev first.")
    unknown = [d for d in ids if d not in available]
    if unknown:
        sys.exit(f"Not prepared: {unknown}. Run `credit-doc-lab prepare`.")
    if args.limit:
        ids = ids[: args.limit]
    _progress("batch", f"Loading schema and preparing Gemini client for {len(ids)} document(s).")
    schema = load_schema(layout.schema)
    client = make_client(args.timeout)
    failures = sum(_extract_one(layout.pdf(d), schema, str(layout.schema), d, args, client) for d in ids)
    print(f"Batch done: {len(ids) - failures}/{len(ids)} ok")
    return 1 if failures else 0


def cmd_evaluate(args) -> int:
    from .runs import evaluate_prediction, evaluate_run, find_runs, validation_for_prediction
    from .schema import load_schema

    layout = _layout(args)
    results = []
    if args.prediction:
        if not args.doc:
            sys.exit("--doc is required with --prediction")
        schema = load_schema(layout.schema)
        pred = json.loads(args.prediction.read_text())
        results.append(evaluate_prediction(
            schema, layout, args.doc, pred, prediction_status="saved_prediction",
            output_validation=validation_for_prediction(pred, schema),
            scorer=args.scorer, judge_model=args.judge_model,
        ))
    else:
        run_dirs = args.runs or find_runs(args.runs_dir)
        if not run_dirs:
            sys.exit("No runs found.")
        for rd in run_dirs:
            results.append(evaluate_run(rd, layout, scorer=args.scorer, judge_model=args.judge_model))
    for index, r in enumerate(results):
        if index:
            print()
        c = r.summary.outcome_counts
        print(f"Document: {r.doc_id}")
        if r.run_id:
            print(f"Run:      {r.run_id}")
        print(f"Scorer:   {r.scorer}")
        print(f"Results:  {r.summary.pass_rate:.1%} passed | mean field score: {r.summary.field_score:.3f}")
        print("Counts:   " + " | ".join(f"{k}: {v}" for k, v in c.items()))
        print("\nFields:")
        for f in r.fields:
            print(f"  {f.path} [{f.outcome}]")
            if args.details:
                for label, value in (("Reference", f.gold), ("Predicted", f.predicted)):
                    text = json.dumps(value, ensure_ascii=False)
                    print(f"    {label}:")
                    print(textwrap.fill(text, width=88, initial_indent="      ",
                                        subsequent_indent="      ", replace_whitespace=False))
    if args.scorer == "baseline":
        print("\nNote: baseline scorer is an offline approximation, not the official ExtractBench scorer.")
    return 0


def cmd_report(args) -> int:
    from .report import build_report

    text = build_report(args.runs_dir, args.scorer, args.details)
    if args.output:
        args.output.write_text(text)
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credit-doc-lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="download the ExtractBench credit-agreement subset")
    _add_common(p)
    p.add_argument("--only", nargs="+", metavar="DOC_ID", help="download only these documents")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("split", help="write a deterministic dev/holdout split")
    _add_common(p)
    p.add_argument("--holdout", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits-file", type=Path, default=cfg.DEFAULT_SPLITS_FILE)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("extract", help="extract one document with Gemini (needs an API key)")
    _add_common(p)
    _add_run_options(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--doc", help="prepared document id")
    g.add_argument("--pdf", type=Path, help="path to any PDF (requires --schema)")
    p.add_argument("--schema", type=Path, help="JSON schema path (with --pdf)")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("batch", help="extract several prepared documents")
    _add_common(p)
    _add_run_options(p)
    p.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    p.add_argument("--splits-file", type=Path, default=cfg.DEFAULT_SPLITS_FILE)
    p.add_argument("--doc", nargs="+", help="explicit document ids (overrides --split)")
    p.add_argument("--limit", type=int, help="only the first N documents")
    p.add_argument("--allow-holdout", action="store_true")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("evaluate", help="score saved predictions (never calls the extraction model)")
    _add_common(p)
    p.add_argument("runs", nargs="*", type=Path, help="run directories (default: all under --runs-dir)")
    p.add_argument("--runs-dir", type=Path, default=cfg.DEFAULT_RUNS_DIR)
    p.add_argument("--prediction", type=Path, help="a bare prediction JSON file (needs --doc)")
    p.add_argument("--doc", help="document id for --prediction")
    p.add_argument("--scorer", choices=["baseline", "official"], default="baseline")
    p.add_argument("--judge-model", default=None, help="LiteLLM judge model for the official scorer")
    p.add_argument("--details", action="store_true", help="show reference and predicted values for all fields")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("report", help="Markdown summary of saved runs")
    p.add_argument("--runs-dir", type=Path, default=cfg.DEFAULT_RUNS_DIR)
    p.add_argument("--scorer", choices=["baseline", "official"], default="baseline")
    p.add_argument("--details", action="store_true")
    p.add_argument("--output", type=Path)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - friendly CLI errors
        if isinstance(exc, (FileNotFoundError, ValueError, RuntimeError)):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
