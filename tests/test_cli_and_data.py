import builtins
import json
import threading

import pytest

from credit_doc_lab import cli
from credit_doc_lab.config import DataLayout
from credit_doc_lab.dataset import git_blob_sha1, select_documents
from credit_doc_lab.models import RunConfig
from credit_doc_lab.extract import extract_document
from credit_doc_lab.report import build_report
from credit_doc_lab.runs import make_split, save_run

from conftest import SYNTHETIC_GOLD, SYNTHETIC_SCHEMA, FakeClient, make_response

P = "dataset/finance/credit_agreement/pdf+gold"


def test_git_blob_sha1_known_value():
    assert git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_select_documents_skips_extra_and_unpaired():
    tree = {f"{P}/a.pdf": "1", f"{P}/a.gold.json": "2", f"{P}/b.pdf": "3",
            f"{P}/_extra/c.pdf": "4", f"{P}/_extra/c.gold.json": "5", "other/x.pdf": "6"}
    assert select_documents(tree) == ["a"]


def test_split_is_deterministic_and_disjoint():
    ids = [f"doc{i}" for i in range(10)]
    a, b = make_split(ids, 3, 0), make_split(ids, 3, 0)
    assert a == b and len(a["holdout"]) == 3
    assert not set(a["dev"]) & set(a["holdout"]) and set(a["dev"]) | set(a["holdout"]) == set(ids)
    assert make_split(ids, 3, 1)["holdout"] != a["holdout"]


def test_help_needs_no_api_key(monkeypatch, capsys):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0


def test_cli_preserves_unexpected_import_error(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "extract" and level == 1:
            raise AssertionError("error handler should not import extract")
        return original_import(name, globals, locals, fromlist, level)

    def fail(args):
        raise ModuleNotFoundError("missing dependency", name="missing_dependency")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(cli, "cmd_report", fail)
    with pytest.raises(ModuleNotFoundError, match="missing dependency") as exc:
        cli.main(["report"])
    assert exc.value.name == "missing_dependency"


def test_cli_reports_expected_errors(monkeypatch, capsys):
    def fail(args):
        raise FileNotFoundError("missing file")

    monkeypatch.setattr(cli, "cmd_report", fail)
    assert cli.main(["report"]) == 2
    assert "error: missing file" in capsys.readouterr().err


def test_extract_accepts_google_api_key(tmp_path, monkeypatch, pdf):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(SYNTHETIC_SCHEMA))
    received = {}

    def fake_client(**kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr("credit_doc_lab.extract.genai.Client", fake_client)
    monkeypatch.setattr(cli, "_extract_one", lambda *args: 0)
    assert cli.main(["extract", "--pdf", str(pdf), "--schema", str(schema_path)]) == 0
    assert received["api_key"] == "test-google-key"


def test_extract_progress_heartbeat_and_saved_run(tmp_path, monkeypatch, pdf, capsys):
    import credit_doc_lab.extract as extract

    heartbeat = threading.Event()
    original_progress = cli._progress

    def observe_progress(doc_id, message):
        original_progress(doc_id, message)
        if "Still working" in message:
            heartbeat.set()

    class SlowModels:
        def generate_content(self, **kwargs):
            assert heartbeat.wait(2), "heartbeat did not appear during request"
            return make_response(json.dumps(SYNTHETIC_GOLD))

    class SlowClient:
        models = SlowModels()

    monkeypatch.setattr(cli, "_progress", observe_progress)
    monkeypatch.setattr(cli, "PROGRESS_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(extract, "make_client", lambda timeout: SlowClient())
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(SYNTHETIC_SCHEMA))
    runs = tmp_path / "runs"
    assert cli.main(["extract", "--pdf", str(pdf), "--schema", str(schema_path),
                     "--runs-dir", str(runs)]) == 0
    output = capsys.readouterr()
    assert "Starting Gemini request (attempt 1/4)" in output.err
    assert "Still working" in output.err
    assert "Gemini does not report server-side progress" in output.err
    assert "Response received; parsing and validating output" in output.err
    assert "Saving run (status: ok)" in output.err
    assert "synthetic_doc: ok" in output.out
    assert len(list(runs.glob("*/run.json"))) == 1


def test_elapsed_reporter_stops_on_interrupt(monkeypatch):
    messages = []
    monkeypatch.setattr(cli, "_progress", lambda doc_id, message: messages.append(message))
    monkeypatch.setattr(cli, "PROGRESS_INTERVAL_SECONDS", 0.01)
    with pytest.raises(KeyboardInterrupt):
        with cli._show_elapsed("doc"):
            raise KeyboardInterrupt
    assert messages == []


def test_batch_reports_progress_per_document(tmp_path, monkeypatch, capsys):
    import credit_doc_lab.extract as extract

    layout = DataLayout(tmp_path / "data")
    layout.pdfs.mkdir(parents=True)
    layout.schema.write_text(json.dumps(SYNTHETIC_SCHEMA))
    for doc in ("first", "second"):
        layout.pdf(doc).write_bytes(b"%PDF-1.4 synthetic")
    monkeypatch.setattr(extract, "make_client", lambda timeout: FakeClient(
        make_response(json.dumps(SYNTHETIC_GOLD))
    ))
    assert cli.main(["batch", "--split", "all", "--data-dir", str(layout.root),
                     "--runs-dir", str(tmp_path / "runs")]) == 0
    output = capsys.readouterr()
    assert "batch: Loading schema" in output.err
    for doc in ("first", "second"):
        assert f"{doc}: Starting Gemini request" in output.err
        assert f"{doc}: Saving run (status: ok)" in output.err
    assert "Batch done: 2/2 ok" in output.out


def test_evaluate_and_report_saved_run_offline(tmp_path, monkeypatch, pdf, capsys):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    layout = DataLayout(tmp_path / "data")
    layout.reference.mkdir(parents=True)
    layout.schema.write_text(json.dumps(SYNTHETIC_SCHEMA))
    layout.reference_file("synthetic_doc").write_text(json.dumps(SYNTHETIC_GOLD))
    pred = {**SYNTHETIC_GOLD, "borrower": "Wrong"}
    rec = extract_document(pdf, SYNTHETIC_SCHEMA, schema_path=str(layout.schema), model="m",
                           config=RunConfig(), client=FakeClient(make_response(json.dumps(pred))),
                           sleep=lambda s: None)
    runs = tmp_path / "runs"
    run_dir = save_run(rec, runs)
    assert cli.main(["evaluate", "--data-dir", str(layout.root), "--runs-dir", str(runs), "--details"]) == 0
    output = capsys.readouterr().out
    assert "Document: synthetic_doc" in output
    assert "borrower [incorrect]\n    Reference:" in output
    assert "agreement_date [correct]\n    Reference:" in output
    assert '"Wrong"' in output
    assert "baseline scorer is an offline approximation" in output
    assert (run_dir / "evaluation_baseline.json").exists()
    assert cli.main(["evaluate", "--data-dir", str(layout.root), str(run_dir)]) == 0
    compact = capsys.readouterr().out
    assert "borrower [incorrect]" in compact
    assert "agreement_date [correct]" in compact
    assert "    Reference:" not in compact
    report = build_report(runs)
    assert "baseline scorer" in report and "tokens" in report and "n/a" in report


def test_evaluate_details_shows_all_outcomes_and_complete_values(tmp_path, capsys):
    layout = DataLayout(tmp_path / "data")
    layout.reference.mkdir(parents=True)
    layout.schema.write_text(json.dumps(SYNTHETIC_SCHEMA))
    gold = {**SYNTHETIC_GOLD, "borrower": "Reference " + "word " * 35 + "reference-end",
            "agreement_date": None}
    pred = {**gold, "borrower": "Prediction " + "other " * 35 + "prediction-end",
            "agreement_date": "2020-01-31", "lenders": None,
            "terms": {**gold["terms"], "secured": "invalid"}}
    layout.reference_file("synthetic_doc").write_text(json.dumps(gold))
    prediction = tmp_path / "prediction.json"
    prediction.write_text(json.dumps(pred))
    assert cli.main(["evaluate", "--data-dir", str(layout.root), "--prediction",
                     str(prediction), "--doc", "synthetic_doc", "--details"]) == 0
    output = capsys.readouterr().out
    assert "borrower [incorrect]" in output
    assert "agreement_date [unexpected]" in output
    assert "lenders [missing]" in output
    assert "terms.amount [correct]" in output
    assert "terms.secured [invalid]" in output
    assert "reference-end" in output and "prediction-end" in output
    assert "null" in output
    assert all(len(line) <= 88 for line in output.splitlines())
