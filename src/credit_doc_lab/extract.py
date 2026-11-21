"""Gemini baseline extractor: PDF + schema in, validated JSON out.

This module must never see reference answers. `extract_document` takes only a PDF path
and a schema; there is no parameter or import through which gold data could arrive.
"""

import hashlib
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .config import MAX_INLINE_PDF_BYTES, PROMPT_VERSION
from .models import (
    AttemptError,
    DocumentInfo,
    OutputValidation,
    RunConfig,
    RunRecord,
    SchemaInfo,
)
from .prompts import SYSTEM_INSTRUCTION, USER_INSTRUCTION
from .schema import sha256_json, to_gemini_schema, validate_output


class MissingApiKeyError(RuntimeError):
    pass


class DocumentTooLargeError(ValueError):
    pass


def sdk_version() -> str:
    return metadata.version("google-genai")


def make_client(timeout_seconds: float) -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise MissingApiKeyError(
            "GEMINI_API_KEY is not set. Put it in .env (see .env.example) or export it."
        )
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
    )


def build_request(
    pdf_bytes: bytes, schema: dict, config: RunConfig
) -> tuple[list[Any], types.GenerateContentConfig]:
    """Everything sent to the model: PDF bytes, fixed instructions, translated schema."""
    contents = [
        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
        USER_INSTRUCTION,
    ]
    gen_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_json_schema=to_gemini_schema(schema),
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
    )
    return contents, gen_config


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return _status_code(exc) in (408, 429)
    return isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError))


def _call_with_retries(
    call: Callable[[], Any],
    max_retries: int,
    sleep: Callable[[float], None],
    attempt_errors: list[AttemptError],
) -> Any:
    """Run `call`, retrying transient failures at most `max_retries` times."""
    for attempt in range(1, max_retries + 2):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised if final
            retryable = is_retryable(exc)
            attempt_errors.append(
                AttemptError(
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    message=str(exc)[:1000],
                    status_code=_status_code(exc),
                    retryable=retryable,
                )
            )
            if not retryable or attempt > max_retries:
                raise
            sleep(min(30.0, 2.0 ** attempt) + random.uniform(0, 1))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in text)


def extract_document(
    pdf_path: Path,
    schema: dict,
    *,
    schema_path: str,
    model: str,
    doc_id: str | None = None,
    config: RunConfig | None = None,
    client: Any = None,
    sleep: Callable[[float], None] = time.sleep,
) -> RunRecord:
    """Extract from one PDF. API/validation failures are recorded, not raised."""
    config = config or RunConfig()
    pdf_path = Path(pdf_path)
    pdf_bytes = pdf_path.read_bytes()
    if len(pdf_bytes) > MAX_INLINE_PDF_BYTES:
        raise DocumentTooLargeError(
            f"{pdf_path.name} is {len(pdf_bytes)} bytes; inline PDF input is capped at "
            f"{MAX_INLINE_PDF_BYTES}. Files API upload is not implemented yet."
        )
    doc_id = doc_id or pdf_path.stem
    client = client or make_client(config.timeout_seconds)

    contents, gen_config = build_request(pdf_bytes, schema, config)
    started_at = _now()
    t0 = time.perf_counter()
    attempt_errors: list[AttemptError] = []
    response = None
    error: str | None = None
    try:
        response = _call_with_retries(
            lambda: client.models.generate_content(
                model=model, contents=contents, config=gen_config
            ),
            config.max_retries,
            sleep,
            attempt_errors,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {str(exc)[:1000]}"
    elapsed = time.perf_counter() - t0

    text = raw = usage = finish_reason = parsed = validation = None
    status = "api_error"
    if response is not None:
        raw = response.model_dump(mode="json", exclude_none=True)
        usage = raw.get("usage_metadata") or None
        candidates = raw.get("candidates") or []
        if candidates:
            finish_reason = candidates[0].get("finish_reason")
        try:
            text = response.text
        except Exception:  # noqa: BLE001 - SDK raises on some blocked responses
            text = None
        status = "invalid_output"
        if not text:
            validation = OutputValidation(
                status="no_output", errors=[f"empty response (finish_reason={finish_reason})"]
            )
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                validation = OutputValidation(status="invalid_json", errors=[str(exc)])
            else:
                errors = validate_output(parsed, schema)
                if errors:
                    validation = OutputValidation(status="schema_violations", errors=errors)
                else:
                    validation = OutputValidation(status="valid")
                    status = "ok"
        if status != "ok":
            error = "; ".join(validation.errors)[:1000]

    finished_at = _now()
    run_id = (
        f"{started_at[:19].replace('-', '').replace(':', '')}Z_{_safe(doc_id)}"
        f"_{_safe(model)}_{uuid.uuid4().hex[:6]}"
    )
    return RunRecord(
        run_id=run_id,
        status=status,
        document=DocumentInfo(
            doc_id=doc_id,
            path=str(pdf_path),
            sha256=hashlib.sha256(pdf_bytes).hexdigest(),
            bytes=len(pdf_bytes),
        ),
        schema_info=SchemaInfo(
            path=schema_path,
            sha256=sha256_json(schema),
            sent_sha256=sha256_json(to_gemini_schema(schema)),
        ),
        model=model,
        sdk_version=sdk_version(),
        prompt_version=PROMPT_VERSION,
        config=config,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=round(elapsed, 3),
        attempt_errors=attempt_errors,
        token_usage=usage,
        finish_reason=finish_reason,
        raw_response_text=text,
        raw_response=raw,
        parsed_prediction=parsed,
        validation=validation,
        error=error,
    )
