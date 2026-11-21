# credit-doc-lab

Personal research project: compare approaches for extracting structured loan terms from public
credit agreements. **Milestone 1** is a plain Gemini baseline (PDF + JSON schema in, JSON out) with
reproducible, offline-capable evaluation. A document-navigation agent comes later. This is a
prototype, not evidence of production reliability.

## Dataset: what was verified

Inspected [ContextualAI/extract-bench](https://github.com/ContextualAI/extract-bench) at commit
`ca81763…` (cloned and read locally):

| Need | Finding |
|---|---|
| Source PDFs | Yes: 10 credit agreements in `dataset/finance/credit_agreement/pdf+gold/` (~1–2 MB each). 4 more PDFs in `_extra/` have no reference answers and are skipped. |
| Schema | Yes: `credit_agreement-schema.json` (JSON Schema + per-field `evaluation_config`; 13 scored fields). |
| Reference answers | Yes: `*.gold.json`, described as human-validated. |
| Scoring | Yes: `extract_bench` Python package. Deterministic metrics, plus LLM-judged `string_semantic` / `array_llm` (via LiteLLM; default judge `vertex_ai/gemini-2.5-flash`). |
| License | Repo is MIT (© 2026 Contextual AI). The dataset README just says "See the repository LICENSE". PDFs come from public filings, so MIT may not cover rights in the underlying documents. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). |

Not verified: the paper (arXiv:2602.12247) could not be fetched from this environment (no route to
arxiv.org), so only the repo README/code were read. Quirks seen in the data: a few gold values are
the literal string `"NOT_FOUND"` (the official scorer treats it as ordinary text, and the schema
never asks for it, so a model returning `null` is marked wrong on those fields); long text fields
(`borrowing_request`, `use_of_proceeds`) hold verbatim, often messy contract text.

`prepare` downloads these via the GitHub trees API + `raw.githubusercontent.com` at the pinned
commit and verifies each file's git blob SHA-1 against the tree listing. Data goes to
`data/credit_agreement/` (git-ignored); `pdfs/` and `reference/` are separate directories.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -e ".[dev]"                               # use ".[dev,official]" for the official scorer
cp .env.example .env                                  # then put your key in GEMINI_API_KEY
```

`GEMINI_API_KEY` is read from the environment (or `.env`). It is needed only by `extract`/`batch`
(and by the official scorer's judge). `--help`, tests, `prepare`, `split`, baseline `evaluate`, and
`report` work without it.

## Usage

```bash
credit-doc-lab prepare                      # download the subset (~18 MB)
# splits/credit_agreement.json is already committed; `split` regenerates it

credit-doc-lab extract --doc adbe_credit_agreement_2000_08_09 --model <gemini-model>
credit-doc-lab evaluate runs/<run-dir> --details
credit-doc-lab batch --split dev --model <gemini-model>    # 6 dev documents
credit-doc-lab evaluate                                    # all runs, baseline scorer
credit-doc-lab report --details --output runs/report.md
```

Other inputs: `extract --pdf my.pdf --schema my-schema.json`; score a bare prediction file with
`evaluate --prediction pred.json --doc <id>`. Model: `--model`, else the `CREDIT_DOC_LAB_MODEL`
environment variable, else `gemini-3.8-flash` (the name used in the official google-genai README;
not checked against the live model list, so pass `--model` explicitly).

### How extraction works

* The PDF is sent inline as `application/pdf` bytes via the `google-genai` SDK
  (`Part.from_bytes`), with `response_mime_type="application/json"` and `response_json_schema`.
* **Schema translation** (explicit, in `schema.py`): only the scorer-only `evaluation_config` keys
  are removed; everything else is passed through. Nothing checks Gemini's supported-schema subset,
  so an unsupported keyword would show up as an API 400.
* The output is parsed and validated against the JSON Schema (`jsonschema`); run records and
  evaluation results are Pydantic models. Bad JSON, schema violations and empty replies are saved as
  `invalid_output` runs, not discarded.
* Retries: transient failures (HTTP 5xx, 408, 429, network errors) retry up to `--max-retries`
  (default 3) with backoff; other errors fail immediately. Every failed attempt is recorded.
* Limitations: PDFs over 15 MB are rejected (inline only; the Files API is not implemented).
  Page-count/size limits were **not** checked against Google's docs (ai.google.dev was unreachable
  here), so confirm them for large agreements. Only the SDK README/source were consulted.
* Reference answers never enter requests: `extract.py` has no reference parameter and imports no
  dataset/scoring code (enforced by `tests/test_reference_isolation.py`).

### Saved runs

Each run is `runs/<timestamp>_<doc>_<model>_<id>/run.json` with: document id/path/SHA-256, schema
hash (as loaded and as sent), model, SDK version, prompt version, config, start/end time, elapsed
seconds, token usage exactly as the API reported it (`null` if absent), finish reason, raw response
text and full response dump, parsed prediction, validation result, and errors. `evaluate` writes
`evaluation_<scorer>.json` into the same directory. Prices are not computed.

## Evaluation

Both scorers report per-field outcomes: **correct**, **incorrect**, **missing** (gold has a value,
prediction null/absent), **unexpected** (prediction has a value, gold is null/absent), and
**invalid** (wrong type, or no usable output, which scores every field 0).

* `--scorer baseline` (default): offline, deterministic, **not the official scorer**. It follows
  each field's `evaluation_config` but approximates the LLM-judged metrics with string similarity
  (threshold 0.8, as in the official `string_fuzzy`; short strings containing digits must match
  exactly; arrays are matched greedily and scored by recall). It under-credits paraphrases and can
  over-credit near-verbatim text; its numbers are not comparable with published ExtractBench results.
* `--scorer official`: wraps `extract_bench.ReportBuilder` (needs the `official` extra). It needs an
  LLM judge: default `gemini/gemini-2.5-flash` through LiteLLM using `GEMINI_API_KEY`
  (`--judge-model` to change). If any judge call fails the command errors instead of saving
  deflated scores. Only that failure path was exercised here; no judge call succeeded (no API key),
  so the adapter's conversion code and LiteLLM's use of `GEMINI_API_KEY` for `gemini/` models are
  untested end to end.

## Reserving documents for evaluation

`splits/credit_agreement.json` is a deterministic 6 dev / 4 holdout split (seed 0), committed so the
holdout cannot drift. Tune prompts, models and settings on **dev only**. Run the holdout once, at
the end, with the configuration frozen (`batch --split holdout --allow-holdout`), and report dev and
holdout separately. Do not use `split --force` after you have seen results. Any iteration driven by
holdout results turns it into a second dev set.

## Limitations of a small public benchmark

Ten documents (13 scored fields each) give very noisy estimates; differences of a few fields are not
meaningful, and no confidence intervals are computed. The documents are public and may appear in
model training data. Gold labels have quirks (above). There is one schema and one document type.
A cited quote or any other evidence being present would not prove a value is semantically correct;
this milestone has no citations anyway. Results here do not establish production reliability.

## Tests

```bash
pytest          # offline; model calls are mocked
```

Covers output validation, scoring outcomes, retry bounds, run records, offline evaluation/report,
split determinism, and reference isolation. `docs/example_report_SYNTHETIC.md` is a **synthetic**
format example (mocked model, made-up data), not a real result.
