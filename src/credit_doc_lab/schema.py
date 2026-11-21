"""Extraction-schema helpers: loading, translation for Gemini, output validation."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

# Keys that exist only for the benchmark scorer. They are not part of JSON Schema,
# and they tell the model nothing about the document, so they are never sent.
SCORER_ONLY_KEYS = {"evaluation_config"}


def load_schema(path: Path) -> dict:
    schema = json.loads(Path(path).read_text())
    Draft202012Validator.check_schema(schema)
    return schema


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def to_gemini_schema(schema: Any) -> Any:
    """Return a copy of `schema` suitable for Gemini's `response_json_schema`.

    The only translation is removing scorer-only keys (`evaluation_config`); the rest
    (object/array/anyOf-with-null/description/required) is passed through unchanged.
    Limitation: nothing here checks the schema against Gemini's supported JSON Schema
    subset, so an unsupported keyword surfaces as an API error (HTTP 400), not here.
    """
    if isinstance(schema, dict):
        return {
            k: to_gemini_schema(v)
            for k, v in schema.items()
            if k not in SCORER_ONLY_KEYS
        }
    if isinstance(schema, list):
        return [to_gemini_schema(v) for v in schema]
    return copy.deepcopy(schema)


def validate_output(parsed: Any, schema: dict) -> list[str]:
    """Return human-readable schema violations ([] means valid)."""
    validator = Draft202012Validator(to_gemini_schema(schema))
    errors = sorted(validator.iter_errors(parsed), key=lambda e: list(map(str, e.path)))
    return [
        f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:200]}"
        for e in errors
    ]
