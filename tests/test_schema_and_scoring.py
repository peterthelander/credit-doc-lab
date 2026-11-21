import json

from credit_doc_lab.schema import to_gemini_schema, validate_output
from credit_doc_lab.scoring import score_prediction

from conftest import SYNTHETIC_GOLD


def score(schema, pred, status="ok"):
    return score_prediction(schema, SYNTHETIC_GOLD, pred, doc_id="d", reference_sha256="x",
                            prediction_status=status)


def outcomes(result):
    return {f.path: f.outcome for f in result.fields}


def test_gemini_schema_drops_scorer_keys_only(schema):
    sent = to_gemini_schema(schema)
    assert "evaluation_config" not in json.dumps(sent)
    assert sent["properties"]["borrower"]["description"] == "Borrower"
    assert sent["properties"]["agreement_date"]["anyOf"][1] == {"type": "null"}
    assert "evaluation_config" in json.dumps(schema)  # original untouched


def test_validate_output(schema):
    assert validate_output(SYNTHETIC_GOLD, schema) == []
    bad = {**SYNTHETIC_GOLD, "borrower": 3}
    errors = validate_output(bad, schema)
    assert errors and errors[0].startswith("borrower")


def test_perfect_prediction_scores_fully(schema):
    r = score(schema, SYNTHETIC_GOLD)
    assert r.summary.pass_rate == 1.0 and set(outcomes(r).values()) == {"correct"}


def test_field_outcomes(schema):
    pred = {
        "borrower": "Totally Different Ltd",          # incorrect
        "agreement_date": None,                        # missing
        "lenders": ["Alpha Bank", "Beta Bank", "Extra Bank"],  # partial -> incorrect
        "terms": {"amount": "500000000", "secured": True},     # invalid / incorrect
    }
    r = score(schema, pred)
    o = outcomes(r)
    assert o["borrower"] == "incorrect"
    assert o["agreement_date"] == "missing"
    assert o["lenders"] == "incorrect"
    assert o["terms.amount"] == "invalid"
    assert o["terms.secured"] == "incorrect"
    lenders = next(f for f in r.fields if f.path == "lenders")
    assert lenders.details["matched"] == 2 and lenders.details["spurious_pred"] == ["Extra Bank"]
    assert lenders.score == round(2 / 3, 4)


def test_unexpected_answer_when_gold_is_null(schema):
    gold = {**SYNTHETIC_GOLD, "agreement_date": None}
    r = score_prediction(schema, gold, SYNTHETIC_GOLD, doc_id="d", reference_sha256="x", prediction_status="ok")
    assert outcomes(r)["agreement_date"] == "unexpected"


def test_near_miss_date_is_not_fuzzy_matched(schema):
    r = score(schema, {**SYNTHETIC_GOLD, "agreement_date": "2020-01-30"})
    assert outcomes(r)["agreement_date"] == "incorrect"


def test_unusable_output_marks_every_field_invalid(schema):
    r = score(schema, None, status="invalid_output")
    assert r.summary.pass_rate == 0.0
    assert set(outcomes(r).values()) == {"invalid"}


def test_extra_key_is_unexpected(schema):
    r = score(schema, {**SYNTHETIC_GOLD, "made_up": 1})
    assert outcomes(r)["made_up"] == "unexpected"
