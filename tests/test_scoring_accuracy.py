"""Tests for PubMedQA answer accuracy scoring (biomedqa.scoring.accuracy)."""

import math
from biomedqa.schema import QueryRecord, System
from biomedqa.scoring.accuracy import accuracy
from biomedqa.scoring.retrieval import wilson_interval


def _make_record(
    query_id: str,
    gold_final_decision: str | None,
    final_decision: str | None,
) -> QueryRecord:
    return QueryRecord(
        run_id="test_run",
        query_id=query_id,
        question="Test question?",
        system=System.JOINT,
        seed=0,
        gold_final_decision=gold_final_decision,
        final_decision=final_decision,
    )


def test_accuracy_hand_built_fixture():
    """Test accuracy calculation, Wilson interval, and confusion matrix on a hand-built fixture."""
    records = [
        _make_record("q1", "yes", "yes"),    # correct
        _make_record("q2", "yes", "yes"),    # correct
        _make_record("q3", "yes", "no"),     # wrong
        _make_record("q4", "no", "no"),      # correct
        _make_record("q5", "no", "no"),      # correct
        _make_record("q6", "no", "yes"),     # wrong
        _make_record("q7", "maybe", "maybe"),# correct
        _make_record("q8", "maybe", "yes"),  # wrong
        _make_record("q9", "yes", None),     # unparsed -> mapped to maybe in confusion
        _make_record("q10", "no", None),     # unparsed -> mapped to maybe in confusion
    ]

    res = accuracy(records)

    assert res["n_total"] == 10
    assert res["n_eval"] == 8
    assert res["n_unparsed"] == 2
    assert res["correct"] == 5

    # Point accuracy: 5 / 8 = 0.625
    assert res["accuracy"] == 0.625

    # Hand-checked Wilson interval for 5/8
    expected_point, expected_lo, expected_hi = wilson_interval(5, 8)
    assert abs(res["accuracy"] - expected_point) < 1e-9
    assert abs(res["wilson_lower"] - expected_lo) < 1e-9
    assert abs(res["wilson_upper"] - expected_hi) < 1e-9

    # Confusion matrix structure and sums
    conf = res["confusion"]
    assert conf["yes"]["yes"] == 2
    assert conf["yes"]["no"] == 1
    assert conf["yes"]["maybe"] == 1

    assert conf["no"]["yes"] == 1
    assert conf["no"]["no"] == 2
    assert conf["no"]["maybe"] == 1

    assert conf["maybe"]["yes"] == 1
    assert conf["maybe"]["no"] == 0
    assert conf["maybe"]["maybe"] == 1

    # 3x3 confusion matrix sums to total record count
    total_conf = sum(sum(conf[g].values()) for g in conf)
    assert total_conf == res["n_total"] == 10


def test_accuracy_unparsed_none_decision_handling():
    """Test that final_decision=None is excluded from accuracy calculation and mapped to maybe in confusion."""
    records = [
        _make_record("q1", "yes", "yes"),
        _make_record("q2", "no", None),
        _make_record("q3", "maybe", None),
    ]

    res = accuracy(records)

    assert res["n_total"] == 3
    assert res["n_eval"] == 1
    assert res["n_unparsed"] == 2
    assert res["correct"] == 1
    assert res["accuracy"] == 1.0

    conf = res["confusion"]
    assert conf["no"]["maybe"] == 1
    assert conf["maybe"]["maybe"] == 1
    assert sum(sum(conf[g].values()) for g in conf) == 3


def test_accuracy_empty_input():
    """Test accuracy behavior on empty input records."""
    res = accuracy([])

    assert res["n_total"] == 0
    assert res["n_eval"] == 0
    assert res["n_unparsed"] == 0
    assert res["correct"] == 0
    assert math.isnan(res["accuracy"])
    assert math.isnan(res["wilson_lower"])
    assert math.isnan(res["wilson_upper"])

    conf = res["confusion"]
    assert sum(sum(conf[g].values()) for g in conf) == 0


def test_accuracy_case_and_whitespace_insensitivity():
    """Test that decision strings are normalized (case-insensitive, trimmed)."""
    records = [
        _make_record("q1", " YES ", "yes"),
        _make_record("q2", "NO", " No "),
        _make_record("q3", "Maybe", "MAYBE"),
    ]

    res = accuracy(records)

    assert res["n_eval"] == 3
    assert res["correct"] == 3
    assert res["accuracy"] == 1.0


def test_accuracy_unresolvable_gold_excluded():
    """Test that a record whose gold label cannot be resolved is excluded from all counts and confusion."""
    records = [
        _make_record("q1", "yes", "yes"),
        _make_record("unknown_pubid_999999", None, "yes"),  # gold is None and pubid unresolvable
    ]

    res = accuracy(records)

    assert res["n_total"] == 2
    assert res["n_eval"] == 1
    assert res["n_unparsed"] == 0
    assert res["correct"] == 1
    assert res["accuracy"] == 1.0

    conf = res["confusion"]
    total_conf = sum(sum(conf[g].values()) for g in conf)
    assert total_conf == 1  # Only q1 is in confusion matrix
