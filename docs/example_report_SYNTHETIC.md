# SYNTHETIC EXAMPLE REPORT: NOT REAL RESULTS

Produced offline by `credit-doc-lab report --details` from a hand-written fixture: a made-up
"agreement", a made-up reference answer, and a **mocked** model reply (no Gemini call, no
ExtractBench document). It only shows the report format and says nothing about model accuracy.
Token usage shows `n/a` because the mock reported none; real runs record what the API returns.

# Run report (scorer: baseline)

> Scores come from the offline **baseline scorer** (string-similarity approximation),
> not the official ExtractBench scorer; do not compare them with published numbers.

| run | document | model | status | sec | tokens | fields ok | correct | incorrect | missing | unexpected | invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <run-id> | synthetic_doc | SYNTHETIC-MOCK-MODEL | ok | 0.0 | n/a | 20.0% | 1 | 3 | 1 | 0 | 0 |

## Mean per model/prompt (macro-average over documents)

| model | prompt | docs | mean field pass rate | mean field score |
|---|---|---|---|---|
| SYNTHETIC-MOCK-MODEL | v1 | 1 | 20.0% | 0.419 |

Small sample: no confidence intervals are computed; differences between runs on a handful of documents may be noise.

## synthetic_doc (<run-id>): 4 non-correct field(s)
- `borrower` [incorrect] gold='Acme ZZ-SYNTHETIC-REFERENCE-ANSWER-7f3a Holdings' predicted='Acme Holdings'
- `agreement_date` [missing] gold='2020-01-31' predicted=None
- `lenders` [incorrect] gold=['Alpha Bank', 'Beta Bank', 'Gamma Bank'] predicted=['Alpha Bank', 'Beta Bank', 'Delta Bank']
- `terms.secured` [incorrect] gold=False predicted=True
