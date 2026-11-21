import json

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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0


def test_evaluate_and_report_saved_run_offline(tmp_path, monkeypatch, pdf, capsys):
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
    assert "borrower [incorrect]" in capsys.readouterr().out
    assert (run_dir / "evaluation_baseline.json").exists()
    report = build_report(runs)
    assert "baseline scorer" in report and "tokens" in report and "n/a" in report
