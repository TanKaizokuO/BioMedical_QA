"""Tests for G3 cost clause derivation and provenance (scripts/g3_report.py --costs)."""

from __future__ import annotations

import json
import pytest
import subprocess
import sys
from pathlib import Path

from biomedqa.schema import (
    CostRecord,
    SupportLabel,
    write_jsonl,
)
from test_g3_report import make_records_file

_REPO = Path(__file__).resolve().parents[1]


def run_driver(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(_REPO / "scripts/g3_report.py"), *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_derived_cost_ratio_crossing_threshold(tmp_path: Path) -> None:
    scores = [0.9] * 5 + [0.1] * 5
    labels = [SupportLabel.SUPPORTED] * 5 + [SupportLabel.NOT_SUPPORTED] * 5
    rec_path = make_records_file(tmp_path / "passing.records.jsonl", scores=scores, labels=labels)
    out_path = tmp_path / "passing_derived.json"

    # Synthetic CostRecords (judge $0.20 per query, ours $0.01 per query -> ratio 20.0x >= 10.0x)
    cost_records = [
        CostRecord(
            run_id="r1",
            query_id=f"q{i}",
            component="judge",
            backend="anthropic:claude-opus-5",
            usd=0.20,
            input_tokens=100,
            output_tokens=10,
            wall_s=1.5,
        )
        for i in range(5)
    ] + [
        CostRecord(
            run_id="r1",
            query_id=f"q{i}",
            component="verify",
            backend="vllm:minicheck",
            usd=0.01,
            input_tokens=100,
            output_tokens=5,
            wall_s=0.1,
        )
        for i in range(5)
    ]
    costs_path = tmp_path / "synthetic_derived_pass.costs.jsonl"
    write_jsonl(costs_path, cost_records)

    res = run_driver(["--records", str(rec_path), "--costs", str(costs_path), "--out", str(out_path)])
    assert res.returncode == 0
    assert "G3 PASSES: true" in res.stdout

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passes"] is True
    assert data["cost_passes"] is True
    assert data["cost_ratio"] == 20.0
    assert data["cost_source_type"] == "derived"
    assert data["cost_records_source"] == str(costs_path)
    assert data["cost_records_sha256"] is not None
    assert data["cost_provenance"]["n_judge_cost_records"] == 5
    assert data["cost_provenance"]["n_ours_cost_records"] == 5
    assert "judge_run_specification" in data


def test_derived_cost_ratio_below_threshold(tmp_path: Path) -> None:
    scores = [0.9] * 5 + [0.1] * 5
    labels = [SupportLabel.SUPPORTED] * 5 + [SupportLabel.NOT_SUPPORTED] * 5
    rec_path = make_records_file(tmp_path / "passing.records.jsonl", scores=scores, labels=labels)
    out_path = tmp_path / "failing_derived.json"

    # Synthetic CostRecords (judge $0.05 per query, ours $0.01 per query -> ratio 5.0x < 10.0x)
    cost_records = [
        CostRecord(
            run_id="r1",
            query_id=f"q{i}",
            component="judge",
            backend="anthropic:claude-opus-5",
            usd=0.05,
            input_tokens=100,
            output_tokens=10,
            wall_s=1.0,
        )
        for i in range(5)
    ] + [
        CostRecord(
            run_id="r1",
            query_id=f"q{i}",
            component="verify",
            backend="vllm:minicheck",
            usd=0.01,
            input_tokens=100,
            output_tokens=5,
            wall_s=0.1,
        )
        for i in range(5)
    ]
    costs_path = tmp_path / "synthetic_derived_fail.costs.jsonl"
    write_jsonl(costs_path, cost_records)

    res = run_driver(["--records", str(rec_path), "--costs", str(costs_path), "--out", str(out_path)])
    assert res.returncode == 0
    assert "G3 PASSES: false" in res.stdout

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passes"] is False
    assert data["cost_passes"] is False
    assert data["cost_ratio"] == 5.0
    assert "cost_ratio_below_threshold" in data["reason"]


def test_derived_cost_ratio_missing_judge_records(tmp_path: Path) -> None:
    scores = [0.9] * 5 + [0.1] * 5
    labels = [SupportLabel.SUPPORTED] * 5 + [SupportLabel.NOT_SUPPORTED] * 5
    rec_path = make_records_file(tmp_path / "passing.records.jsonl", scores=scores, labels=labels)
    out_path = tmp_path / "missing_judge.json"

    # Synthetic CostRecords with NO judge component
    cost_records = [
        CostRecord(
            run_id="r1",
            query_id=f"q{i}",
            component="verify",
            backend="vllm:minicheck",
            usd=0.0,
            input_tokens=100,
            output_tokens=5,
            wall_s=0.1,
        )
        for i in range(5)
    ]
    costs_path = tmp_path / "no_judge.costs.jsonl"
    write_jsonl(costs_path, cost_records)

    res = run_driver(["--records", str(rec_path), "--costs", str(costs_path), "--out", str(out_path)])
    assert res.returncode == 0
    assert "G3 PASSES: false" in res.stdout

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passes"] is False
    assert data["cost_passes"] is False
    assert data["cost_ratio"] is None
    assert "missing_judge_cost_evidence" in data["reason"]

def test_unpriced_verifier_with_positive_judge_fails(tmp_path: Path) -> None:
    scores = [0.9] * 5 + [0.1] * 5
    labels = [SupportLabel.SUPPORTED] * 5 + [SupportLabel.NOT_SUPPORTED] * 5
    rec_path = make_records_file(tmp_path / "passing.records.jsonl", scores=scores, labels=labels)
    out_path = tmp_path / "unpriced.json"

    # Judge has positive cost ($0.20), local verifier has $0.0 USD
    cost_records = [
        CostRecord(
            run_id="r1",
            query_id=f"q{i}",
            component="judge",
            backend="anthropic:claude-opus-5",
            usd=0.20,
            input_tokens=100,
            output_tokens=10,
            wall_s=1.5,
        )
        for i in range(5)
    ] + [
        CostRecord(
            run_id="r1",
            query_id=f"q{i}",
            component="verify",
            backend="vllm:minicheck",
            usd=0.0,
            input_tokens=100,
            output_tokens=5,
            wall_s=0.1,
        )
        for i in range(5)
    ]
    costs_path = tmp_path / "unpriced.costs.jsonl"
    write_jsonl(costs_path, cost_records)

    res = run_driver(["--records", str(rec_path), "--costs", str(costs_path), "--out", str(out_path)])
    assert res.returncode == 0
    assert "G3 PASSES: false" in res.stdout

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passes"] is False
    assert data["cost_passes"] is False
    assert "verifier_cost_unpriced" in data["reason"]


def test_operator_supplied_gpu_rate_prices_verifier(tmp_path: Path) -> None:
    scores = [0.9] * 5 + [0.1] * 5
    labels = [SupportLabel.SUPPORTED] * 5 + [SupportLabel.NOT_SUPPORTED] * 5
    rec_path = make_records_file(tmp_path / "passing.records.jsonl", scores=scores, labels=labels)
    out_path = tmp_path / "priced_by_rate.json"

    # Judge $0.20/query, local verifier usd=0.0, wall_s=0.1s
    cost_records = [
        CostRecord(
            run_id="r1",
            query_id=f"q{i}",
            component="judge",
            backend="anthropic:claude-opus-5",
            usd=0.20,
            input_tokens=100,
            output_tokens=10,
            wall_s=1.5,
        )
        for i in range(5)
    ] + [
        CostRecord(
            run_id="r1",
            query_id=f"q{i}",
            component="verify",
            backend="vllm:minicheck",
            usd=0.0,
            input_tokens=100,
            output_tokens=5,
            wall_s=0.1,
        )
        for i in range(5)
    ]
    costs_path = tmp_path / "rate_priced.costs.jsonl"
    write_jsonl(costs_path, cost_records)

    # 360.0 $/hr = 0.10 $/sec -> 0.1s wall_s = $0.01 per query -> ratio 20.0x
    res = run_driver([
        "--records", str(rec_path),
        "--costs", str(costs_path),
        "--verifier-gpu-hourly-rate", "360.0",
        "--out", str(out_path),
    ])
    assert res.returncode == 0
    assert "G3 PASSES: true" in res.stdout

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passes"] is True
    assert data["cost_passes"] is True
    assert data["cost_ratio"] == pytest.approx(20.0)
    assert data["cost_provenance"]["verifier_gpu_hourly_rate"] == 360.0

def test_rate_supplied_without_measured_timing_remains_unevaluable(tmp_path: Path) -> None:
    scores = [0.9] * 5 + [0.1] * 5
    labels = [SupportLabel.SUPPORTED] * 5 + [SupportLabel.NOT_SUPPORTED] * 5
    rec_path = make_records_file(tmp_path / "passing.records.jsonl", scores=scores, labels=labels)
    out_path = tmp_path / "rate_no_timing.json"

    # Judge has positive cost ($0.20), local verifier has wall_s=None (missing timing)
    cost_records = [
        CostRecord(
            run_id="r1",
            query_id=f"q{i}",
            component="judge",
            backend="anthropic:claude-opus-5",
            usd=0.20,
            input_tokens=100,
            output_tokens=10,
            wall_s=1.5,
        )
        for i in range(5)
    ] + [
        CostRecord(
            run_id="r1",
            query_id=f"q{i}",
            component="verify",
            backend="vllm:minicheck",
            usd=None,
            input_tokens=100,
            output_tokens=5,
            wall_s=None,
        )
        for i in range(5)
    ]
    costs_path = tmp_path / "no_timing.costs.jsonl"
    write_jsonl(costs_path, cost_records)

    # Rate supplied without timing => must remain unevaluable (verifier_cost_unpriced)
    res = run_driver([
        "--records", str(rec_path),
        "--costs", str(costs_path),
        "--verifier-gpu-hourly-rate", "360.0",
        "--out", str(out_path),
    ])
    assert res.returncode == 0
    assert "G3 PASSES: false" in res.stdout

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passes"] is False
    assert data["cost_passes"] is False
    assert data["cost_ratio"] is None
    assert "verifier_cost_unpriced" in data["reason"]

def test_mutually_exclusive_cost_flags(tmp_path: Path) -> None:
    rec_path = make_records_file(tmp_path / "dummy.records.jsonl")
    costs_path = tmp_path / "dummy.costs.jsonl"
    write_jsonl(costs_path, [])
    out_path = tmp_path / "out.json"

    res = run_driver([
        "--records", str(rec_path),
        "--cost-ratio", "15.0",
        "--costs", str(costs_path),
        "--out", str(out_path),
    ])
    assert res.returncode == 1
    assert "mutually exclusive" in res.stderr
