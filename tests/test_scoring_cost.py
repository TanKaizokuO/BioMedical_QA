from __future__ import annotations

import math
import pytest

from biomedqa.schema import CostRecord
from biomedqa.scoring.cost import overhead_ratio, per_query_cost


def test_per_query_cost_empty():
    res = per_query_cost([])
    assert res == {}


def test_per_query_cost_single_record_range_columns():
    rec = CostRecord(
        run_id="run-1",
        query_id="q1",
        component="generate",
        backend="vllm:model",
        input_tokens=120,
        output_tokens=30,
        usd=0.005,
        wall_s=1.5,
    )
    res = per_query_cost([rec])
    assert "generate" in res
    gen = res["generate"]
    assert gen["n_queries"] == 1

    # Check that min == max == mean == median == 120 for single record
    in_tok = gen["input_tokens"]
    assert in_tok["mean"] == 120.0
    assert in_tok["median"] == 120
    assert in_tok["min"] == 120
    assert in_tok["max"] == 120
    assert in_tok["total"] == 120
    assert in_tok["n_valid"] == 1
    assert in_tok["n_missing"] == 0

    out_tok = gen["output_tokens"]
    assert out_tok["mean"] == 30.0
    assert out_tok["median"] == 30
    assert out_tok["min"] == 30
    assert out_tok["max"] == 30

    wall = gen["wall_s"]
    assert wall["mean"] == 1.5
    assert wall["median"] == 1.5
    assert wall["min"] == 1.5
    assert wall["max"] == 1.5


def test_per_system_aggregation_known_totals():
    recs = [
        # System 1: generate / q1 (2 stages)
        CostRecord("run-1", "q1", "generate", "vllm:model", 100, 20, None, 1.0),
        CostRecord("run-1", "q1", "generate", "vllm:model", 50, 10, None, 0.5),
        # System 1: generate / q2 (1 stage)
        CostRecord("run-1", "q2", "generate", "vllm:model", 300, 40, None, 2.0),
        # System 2: judge / q1 (1 stage)
        CostRecord("run-1", "q1", "judge", "anthropic:opus", 500, 15, 0.05, 1.2),
        # System 2: judge / q2 (1 stage)
        CostRecord("run-1", "q2", "judge", "anthropic:opus", 700, 25, 0.07, 1.8),
    ]

    res = per_query_cost(recs)
    assert set(res.keys()) == {"generate", "judge"}

    gen = res["generate"]
    assert gen["n_queries"] == 2
    # q1 total: input 150, output 30, wall 1.5
    # q2 total: input 300, output 40, wall 2.0
    assert gen["input_tokens"]["mean"] == 225.0
    assert gen["input_tokens"]["min"] == 150
    assert gen["input_tokens"]["max"] == 300
    assert gen["input_tokens"]["total"] == 450

    assert gen["output_tokens"]["mean"] == 35.0
    assert gen["output_tokens"]["min"] == 30
    assert gen["output_tokens"]["max"] == 40
    assert gen["output_tokens"]["total"] == 70

    assert gen["wall_s"]["mean"] == 1.75
    assert gen["wall_s"]["min"] == 1.5
    assert gen["wall_s"]["max"] == 2.0

    # Local model USD is None across all records
    assert gen["usd"]["mean"] is None
    assert gen["usd"]["n_missing"] == 2

    jdg = res["judge"]
    assert jdg["n_queries"] == 2
    assert jdg["usd"]["mean"] == pytest.approx(0.06)
    assert jdg["usd"]["min"] == pytest.approx(0.05)
    assert jdg["usd"]["max"] == pytest.approx(0.07)
    assert jdg["usd"]["total"] == pytest.approx(0.12)


def test_null_token_cost_row_propagates_as_missing():
    # Record 1 stage A has input_tokens=100, output_tokens=None (rejected call)
    # Record 1 stage B has input_tokens=50, output_tokens=20
    rec_rejected = CostRecord("run-1", "q1", "generate", "vllm:model", 100, None, None, 1.0)
    rec_ok = CostRecord("run-1", "q1", "generate", "vllm:model", 50, 20, None, 0.5)

    res = per_query_cost([rec_rejected, rec_ok])
    gen = res["generate"]

    # Input tokens were reported for both stages -> sum = 150
    assert gen["input_tokens"]["mean"] == 150.0
    assert gen["input_tokens"]["n_valid"] == 1
    assert gen["input_tokens"]["n_missing"] == 0

    # Output tokens had a None in stage A -> query total evaluates to None (missing)
    # Must NOT treat None as 0 (which would give 20 and report a cheap query)
    assert gen["output_tokens"]["mean"] is None
    assert gen["output_tokens"]["total"] is None
    assert gen["output_tokens"]["n_valid"] == 0
    assert gen["output_tokens"]["n_missing"] == 1


def test_overhead_ratio_against_zero_dollar_local_verifier():
    judge_recs = [
        CostRecord("run-1", "q1", "judge", "anthropic:opus", 500, 10, 0.05, 2.0),
        CostRecord("run-1", "q2", "judge", "anthropic:opus", 600, 12, 0.06, 2.5),
    ]

    # MiniCheck emits NO CostRecord at all (local compute, $0 USD)
    ov = overhead_ratio([], judge_recs)
    assert math.isinf(ov["usd_ratio"])
    assert ov["usd_ratio"] > 0
    assert ov["input_token_ratio"] is None
    assert ov["output_token_ratio"] is None
    assert ov["wall_s_ratio"] is None

    # Local model with 0.0 USD cost records
    local_recs = [
        CostRecord("run-1", "q1", "verify", "local:minicheck", None, None, 0.0, 0.1),
        CostRecord("run-1", "q2", "verify", "local:minicheck", None, None, 0.0, 0.1),
    ]
    ov_local = overhead_ratio(local_recs, judge_recs)
    assert math.isinf(ov_local["usd_ratio"])


def test_overhead_ratio_against_empty_judge_series():
    ours_recs = [
        CostRecord("run-1", "q1", "generate", "vllm:model", 200, 50, 0.01, 1.0),
    ]
    ov = overhead_ratio(ours_recs, [])
    assert ov["usd_ratio"] is None
    assert ov["input_token_ratio"] is None
    assert ov["output_token_ratio"] is None
    assert ov["wall_s_ratio"] is None
    assert ov["note"] == "empty judge series"


def test_overhead_ratio_both_positive_costs():
    ours = [
        CostRecord("run-1", "q1", "generate", "api:model", 100, 20, 0.005, 1.0),
        CostRecord("run-1", "q2", "generate", "api:model", 100, 20, 0.005, 1.0),
    ]
    judge = [
        CostRecord("run-1", "q1", "judge", "anthropic:opus", 500, 10, 0.05, 2.0),
        CostRecord("run-1", "q2", "judge", "anthropic:opus", 500, 10, 0.05, 2.0),
    ]

    ov = overhead_ratio(ours, judge)
    assert pytest.approx(ov["usd_ratio"]) == 10.0
    assert pytest.approx(ov["input_token_ratio"]) == 5.0
    assert pytest.approx(ov["output_token_ratio"]) == 0.5
    assert pytest.approx(ov["wall_s_ratio"]) == 2.0
