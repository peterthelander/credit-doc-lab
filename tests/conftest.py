"""Shared fixtures. Everything here is SYNTHETIC test data, not benchmark data."""

import json
from pathlib import Path

import pytest
from google.genai import types

# Mirrors the shape of the real schema (nullable anyOf, array, scorer-only keys).
SYNTHETIC_SCHEMA = {
    "type": "object",
    "properties": {
        "borrower": {"type": "string", "description": "Borrower", "evaluation_config": "string_fuzzy"},
        "agreement_date": {
            "description": "YYYY-MM-DD",
            "evaluation_config": "string_semantic",
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
        "lenders": {
            "type": "array", "items": {"type": "string"}, "evaluation_config": "array_llm",
        },
        "terms": {
            "type": "object",
            "properties": {
                "amount": {"anyOf": [{"type": "number"}, {"type": "null"}], "evaluation_config": "number_exact"},
                "secured": {"anyOf": [{"type": "boolean"}, {"type": "null"}], "evaluation_config": "boolean_exact"},
            },
        },
    },
}

SENTINEL = "ZZ-SYNTHETIC-REFERENCE-ANSWER-7f3a"
SYNTHETIC_GOLD = {
    "borrower": f"Acme {SENTINEL} Holdings",
    "agreement_date": "2020-01-31",
    "lenders": ["Alpha Bank", "Beta Bank", "Gamma Bank"],
    "terms": {"amount": 500000000, "secured": False},
}


class FakeModels:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        b = self.behaviours.pop(0) if len(self.behaviours) > 1 else self.behaviours[0]
        if isinstance(b, Exception):
            raise b
        return b


class FakeClient:
    def __init__(self, *behaviours):
        self.models = FakeModels(behaviours)


def make_response(text, usage=None, finish="STOP"):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(parts=[types.Part(text=text)], role="model"), finish_reason=finish)],
        usage_metadata=usage,
    )


@pytest.fixture
def schema():
    return json.loads(json.dumps(SYNTHETIC_SCHEMA))


@pytest.fixture
def pdf(tmp_path) -> Path:
    p = tmp_path / "synthetic_doc.pdf"
    p.write_bytes(b"%PDF-1.4\n% synthetic placeholder, not a real agreement\n")
    return p
