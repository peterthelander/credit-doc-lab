"""Reference answers must never reach a model request."""

import ast
import inspect
import json
from pathlib import Path

from credit_doc_lab import extract, prompts, schema as schema_mod
from credit_doc_lab.config import DataLayout
from credit_doc_lab.models import RunConfig
from credit_doc_lab.runs import save_run

from conftest import SENTINEL, SYNTHETIC_GOLD, FakeClient, make_response

SRC = Path(extract.__file__).parent
FORBIDDEN = {"scoring", "official", "runs", "dataset", "report"}


def imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0] if node.level == 0 else (node.module or ""))
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
    return names


def test_extraction_modules_do_not_import_reference_or_scoring_code():
    for mod in (extract, prompts, schema_mod):
        assert not (imported_modules(Path(mod.__file__)) & FORBIDDEN), mod.__name__


def test_extraction_api_has_no_reference_parameter():
    params = set(inspect.signature(extract.extract_document).parameters)
    assert not {p for p in params if "gold" in p or "reference" in p}
    assert set(inspect.signature(extract.build_request).parameters) == {"pdf_bytes", "schema", "config"}


def test_gold_on_disk_never_appears_in_request_or_run_record(tmp_path, pdf, schema):
    layout = DataLayout(tmp_path / "data")
    layout.reference.mkdir(parents=True)
    layout.reference_file("synthetic_doc").write_text(json.dumps(SYNTHETIC_GOLD))
    # The model reply echoes gold (as if memorised); only the *request* is under test.
    client = FakeClient(make_response(json.dumps({"borrower": "x"})))
    rec = extract.extract_document(pdf, schema, schema_path="s", model="m", config=RunConfig(),
                                   client=client, sleep=lambda s: None)
    call = client.models.calls[0]
    request = repr(call["contents"]) + repr(call["config"])
    assert SENTINEL not in request
    assert "Alpha Bank" not in request and "2020-01-31" not in request
    assert SENTINEL not in save_run(rec, tmp_path / "runs").joinpath("run.json").read_text()
