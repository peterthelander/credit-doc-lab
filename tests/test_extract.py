import json

import httpx
import httpx2
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


def test_progress_reports_retry_and_validation(pdf, schema):
    err = errors.ServerError(503, {"error": {"message": "busy"}})
    client = FakeClient(err, make_response(json.dumps(SYNTHETIC_GOLD)))
    messages = []
    delays = []
    rec = extract_document(
        pdf, schema, schema_path="s.json", model="test-model",
        config=RunConfig(max_retries=1), client=client, sleep=delays.append,
        progress=messages.append,
    )
    assert rec.status == "ok"
    assert any("Reading PDF" in message for message in messages)
    assert any("model test-model" in message for message in messages)
    assert "Starting Gemini request (attempt 1/2)." in messages
    assert any("ServerError, HTTP 503" in message and "retrying in" in message for message in messages)
    assert len(delays) == 1 and 2 <= delays[0] <= 3
    assert "Starting Gemini request (attempt 2/2)." in messages
    assert "Response received; parsing and validating output." in messages


def test_progress_reports_final_failure(pdf, schema):
    messages = []
    rec = extract_document(
        pdf, schema, schema_path="s.json", model="test-model",
        config=RunConfig(max_retries=1),
        client=FakeClient(errors.ClientError(403, {"error": {"message": "bad key"}})),
        progress=messages.append,
    )
    assert rec.status == "api_error"
    assert any("ClientError" in message and "no more retries" in message for message in messages)
    assert not any("retrying in" in message for message in messages)


@pytest.mark.parametrize("transport", [httpx, httpx2])
def test_transport_errors_retry_then_succeed(pdf, schema, transport):
    client = FakeClient(transport.ConnectError("connection failed"), make_response(json.dumps(SYNTHETIC_GOLD)))
    rec = run(pdf, schema, client, max_retries=1)
    assert rec.status == "ok" and len(client.models.calls) == 2
    assert rec.attempt_errors[0].retryable


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


@pytest.mark.parametrize(
    "google_key,gemini_key,expected",
    [
        ("google-key", None, "google-key"),
        (None, "gemini-key", "gemini-key"),
        ("google-key", "gemini-key", "google-key"),
        ("", "gemini-key", "gemini-key"),
        ("google-key", "", "google-key"),
    ],
)
def test_make_client_api_key_selection(monkeypatch, google_key, gemini_key, expected):
    for name, value in (("GOOGLE_API_KEY", google_key), ("GEMINI_API_KEY", gemini_key)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    calls = []

    def fake_client(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("credit_doc_lab.extract.genai.Client", fake_client)
    make_client(10)
    assert calls[0]["api_key"] == expected
    assert calls[0]["http_options"].timeout == 10000


@pytest.mark.parametrize("value", [None, ""])
def test_missing_api_key_message(monkeypatch, value):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    if value is not None:
        monkeypatch.setenv("GOOGLE_API_KEY", value)
        monkeypatch.setenv("GEMINI_API_KEY", value)
    with pytest.raises(MissingApiKeyError, match="GOOGLE_API_KEY or GEMINI_API_KEY"):
        make_client(10)


def test_run_roundtrip(pdf, schema, tmp_path):
    rec = run(pdf, schema, FakeClient(make_response(json.dumps(SYNTHETIC_GOLD))))
    d = save_run(rec, tmp_path / "runs")
    assert load_run(d) == rec
