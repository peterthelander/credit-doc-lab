"""Pydantic models for run records and evaluation results."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    doc_id: str
    path: str
    sha256: str
    bytes: int


class SchemaInfo(BaseModel):
    path: str
    sha256: str  # canonical JSON of the schema as loaded
    sent_sha256: str  # canonical JSON of what was sent to Gemini (scorer keys removed)


class RunConfig(BaseModel):
    input_mode: str = "inline_pdf"
    temperature: float | None = None
    max_output_tokens: int | None = None
    max_retries: int = 3
    timeout_seconds: float = 300.0


class OutputValidation(BaseModel):
    status: Literal["valid", "no_output", "invalid_json", "schema_violations"]
    errors: list[str] = Field(default_factory=list)


class AttemptError(BaseModel):
    attempt: int
    error_type: str
    message: str
    status_code: int | None = None
    retryable: bool = False


class RunRecord(BaseModel):
    run_id: str
    # ok: valid output; invalid_output: response received but unusable; api_error: no response
    status: Literal["ok", "invalid_output", "api_error"]
    document: DocumentInfo
    schema_info: SchemaInfo
    model: str
    sdk_version: str
    prompt_version: str
    config: RunConfig
    started_at: str
    finished_at: str
    elapsed_seconds: float
    attempt_errors: list[AttemptError] = Field(default_factory=list)
    token_usage: dict[str, Any] | None = None  # exactly as reported by the API, else None
    finish_reason: str | None = None
    raw_response_text: str | None = None
    raw_response: dict[str, Any] | None = None
    parsed_prediction: Any = None
    validation: OutputValidation | None = None
    error: str | None = None


Outcome = Literal["correct", "incorrect", "missing", "unexpected", "invalid"]


class FieldResult(BaseModel):
    path: str
    metric: str
    outcome: Outcome
    score: float
    passed: bool
    gold: Any = None
    predicted: Any = None
    note: str | None = None
    details: dict[str, Any] | None = None


class EvaluationSummary(BaseModel):
    n_fields: int
    outcome_counts: dict[str, int]
    field_score: float  # flat mean of per-field scores
    pass_rate: float


class EvaluationResult(BaseModel):
    scorer: Literal["baseline", "official"]
    scorer_description: str
    doc_id: str
    run_id: str | None = None
    reference_sha256: str
    evaluated_at: str
    prediction_status: str  # ok / invalid_output / api_error / saved_prediction
    output_validation: OutputValidation | None = None
    summary: EvaluationSummary
    fields: list[FieldResult]
    official_extra: dict[str, Any] | None = None
