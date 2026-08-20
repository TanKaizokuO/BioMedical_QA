"""Tests for scripts/w9_stratified_parity_report.py (ADR-0009 §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from biomedqa.schema import Claim, CostRecord, QueryRecord, System, write_jsonl
from scripts.w9_stratified_parity_report import print_stratified_report


def _words(n: int) -> str:
    return " ".join(["w"] * n)


def test_stratified_parity_report_script_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The new script must read records + costs JSONL, handle output_tokens: null without TypeError,
    and print per-stratum results plus overall verdict."""
    prefix = tmp_path / "run_test"
    records_file = prefix.with_suffix(".records.jsonl")
    costs_file = prefix.with_suffix(".costs.jsonl")

    records = []
    costs = []

    for i in range(1, 6):
        qid = f"q{i}"
        records.append(
            QueryRecord(
                run_id="run_test",
                query_id=qid,
                question="Question?",
                system=System.JOINT,
                seed=0,
                claims=[Claim(claim_id="c1", text=_words(12))],
                completion_tokens=None,
            )
        )
        records.append(
            QueryRecord(
                run_id="run_test",
                query_id=qid,
                question="Question?",
                system=System.POST_HOC,
                seed=0,
                claims=[Claim(claim_id="c1", text=_words(13))],
                completion_tokens=None,
            )
        )

        # Include cost rows with output_tokens: None (null in json)
        costs.append(CostRecord(run_id="run_test", query_id=qid, component="generate", backend="vllm", output_tokens=None))
        costs.append(CostRecord(run_id="run_test", query_id=qid, component="generate", backend="vllm", output_tokens=100))
        costs.append(CostRecord(run_id="run_test", query_id=qid, component="generate", backend="vllm", output_tokens=None))
        costs.append(CostRecord(run_id="run_test", query_id=qid, component="generate", backend="vllm", output_tokens=50))

    write_jsonl(records_file, records)
    write_jsonl(costs_file, costs)

    print_stratified_report(prefix, max_tokens=3584, min_queries=5)
    captured = capsys.readouterr().out

    assert "ADR-0009 §5 W9 Stratified Robustness Check" in captured
    assert "OVERALL W9 STRATIFIED ROBUSTNESS VERDICT: PASS" in captured
    assert "simple" in captured
    assert "UNDERPOWERED" in captured  # compound stratum or 11+ claims stratum underpowered
