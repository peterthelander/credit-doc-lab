import json

import pytest
from google.genai import errors, types

from credit_doc_lab.extract import DocumentTooLargeError, MissingApiKeyError, extract_document, make_client
from credit_doc_lab.models import RunConfig
from credit_doc_lab.runs import load_run, save_run

from conftest import SYNTHETIC_GOLD, FakeClient, make_response


def run(pdf, schema, client, **cfg):
    return extract_document(pdf, schema, schema_path="s.json", model="test-model",
                            config=RunConfig(**cfg), client=client, sleep=lambda s: None)


def test_success_records_metadata_and_usage(pdf, schema):
    usage = types.GenerateContentResponseUsageMetadata(prompt_token_count=10, candidates_token_count=5, total_token_count=15)
    client = FakeClient(make_response(json.dumps(SYNTHETIC_GOLD), usage))
    rec = run(pdf, schema, client)
    assert rec.status == "ok" and rec.validation.status == "valid"
    assert rec.parsed_prediction == SYNTHETIC_GOLD
    assert rec.token_usage == {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15}
    assert rec.model == "test-model" and rec.sdk_version and rec.prompt_version
    assert len(rec.document.sha256) == 64 and rec.elapsed_seconds >= 0
    call = client.models.calls[0]
    assert call["config"].response_mime_type == "application/json"
    assert "evaluation_config" not in json.dumps(call["config"].response_json_schema)


def test_missing_usage_is_not_invented(pdf, schema):
    rec = run(pdf, schema, FakeClient(make_response(json.dumps(SYNTHETIC_GOLD))))
    assert rec.token_usage is None


@pytest.mark.parametrize("text,status", [("not json", "invalid_json"), ('{"borrower": 5}', "schema_violations"), ("", "no_output")])
def test_invalid_outputs_are_recorded(pdf, schema, text, status):
    rec = run(pdf, schema, FakeClient(make_response(text)))
    assert rec.status == "invalid_output" and rec.validation.status == status
    assert rec.raw_response_text == text


def test_transient_errors_retry_then_succeed(pdf, schema):
    err = errors.ServerError(503, {"error": {"message": "busy"}})
    client = FakeClient(err, err, make_response(json.dumps(SYNTHETIC_GOLD)))
    rec = run(pdf, schema, client, max_retries=3)
    assert rec.status == "ok" and len(client.models.calls) == 3
    assert [a.attempt for a in rec.attempt_errors] == [1, 2]


def test_retries_are_bounded(pdf, schema):
    client = FakeClient(errors.ServerError(500, {"error": {"message": "boom"}}))
    rec = run(pdf, schema, client, max_retries=2)
    assert rec.status == "api_error" and len(client.models.calls) == 3
    assert "ServerError" in rec.error


def test_permanent_errors_do_not_retry(pdf, schema):
    client = FakeClient(errors.ClientError(403, {"error": {"message": "bad key"}}))
    rec = run(pdf, schema, client, max_retries=3)
    assert rec.status == "api_error" and len(client.models.calls) == 1


def test_rate_limit_is_retried(pdf, schema):
    client = FakeClient(errors.ClientError(429, {"error": {"message": "slow down"}}), make_response(json.dumps(SYNTHETIC_GOLD)))
    assert run(pdf, schema, client).status == "ok"


def test_oversized_pdf_rejected(tmp_path, schema, monkeypatch):
    monkeypatch.setattr("credit_doc_lab.extract.MAX_INLINE_PDF_BYTES", 10)
    big = tmp_path / "big.pdf"
    big.write_bytes(b"x" * 11)
    with pytest.raises(DocumentTooLargeError):
        run(big, schema, FakeClient(make_response("{}")))


def test_missing_api_key_message(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError, match="GEMINI_API_KEY"):
        make_client(10)


def test_run_roundtrip(pdf, schema, tmp_path):
    rec = run(pdf, schema, FakeClient(make_response(json.dumps(SYNTHETIC_GOLD))))
    d = save_run(rec, tmp_path / "runs")
    assert load_run(d) == rec
