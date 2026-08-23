"""Tests for scripts/estimate_judge_cost.py.

Validates token accounting determinism, artifact JSON structure, unit count over real artifact (1257),
zero CostRecord emission, and graceful handling of zero-citation records.
"""

import json
from pathlib import Path
from unittest.mock import patch

from estimate_judge_cost import (
    count_tokens_char_approx,
    estimate_judge_cost,
)

REAL_SCORED_RECORDS = Path(
    "docs/harvest/generate_fp05_n100_guided_v4.verifier_scored.records.jsonl"
)


def test_token_accounting_deterministic() -> None:
    text = "You are grading whether a passage supports a claim.\n\nPASSAGE:\nTest passage\n\nCLAIM:\nTest claim"
    tokens1 = count_tokens_char_approx(text)
    tokens2 = count_tokens_char_approx(text)
    assert tokens1 == 24  # ceil(94 / 4.0) = 24

    res1 = estimate_judge_cost(REAL_SCORED_RECORDS)
    res2 = estimate_judge_cost(REAL_SCORED_RECORDS)
    assert res1["total_input_tokens"] == res2["total_input_tokens"]
    assert res1["usd_total"] == res2["usd_total"]


def test_artifact_structure_and_strict_json(tmp_path: Path) -> None:
    out_json = tmp_path / "estimate.json"
    dummy_records = tmp_path / "dummy.jsonl"
    dummy_records.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "claims": [
                    {
                        "text": "Claim text",
                        "citations": [{"quoted_text": "Passage text"}],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    art = estimate_judge_cost(dummy_records, out_json)
    assert out_json.exists()

    raw_content = out_json.read_text(encoding="utf-8")
    parsed = json.loads(raw_content)

    assert parsed["estimate_only"] is True
    assert parsed["is_measured_evidence"] is False
    assert parsed["n_units"] == 1
    assert parsed["tokenization_method"] == "char_approx_4.0_chars_per_token"
    assert isinstance(parsed["assumptions"], list)
    assert "docs/adr/0004-local-generator-frontier-judge.md:75" in parsed["price_source"]
    assert "sensitivity_range" in parsed
    assert "floor" in parsed["sensitivity_range"]
    assert "pessimistic_2x" in parsed["sensitivity_range"]


def test_real_artifact_unit_count_1257() -> None:
    assert REAL_SCORED_RECORDS.exists()
    art = estimate_judge_cost(REAL_SCORED_RECORDS)
    assert art["n_units"] == 1257
    assert art["total_input_tokens"] == 191172
    assert art["total_output_tokens_assumed"] == 2514
    assert round(art["usd_total"], 4) == 1.0187
    assert art["sensitivity_range"]["floor"]["usd_total"] == art["usd_total"]
    assert round(art["sensitivity_range"]["pessimistic_2x"]["usd_total"], 4) == 1.0816
    assert round(art["sensitivity_range"]["max_budget_8"]["usd_total"], 4) == 1.2073


def test_no_cost_record_emitted(tmp_path: Path) -> None:
    cost_records_created = []

    def mock_init(self, *args, **kwargs):
        cost_records_created.append(self)

    with patch("biomedqa.schema.CostRecord.__init__", side_effect=mock_init, autospec=True):
        art = estimate_judge_cost(REAL_SCORED_RECORDS, tmp_path / "est.json")

    assert len(cost_records_created) == 0
    assert not (tmp_path / "costs.jsonl").exists()
    assert not Path("costs.jsonl").exists()


def test_zero_citations_returns_zero_units(tmp_path: Path) -> None:
    zero_rec = tmp_path / "zero.jsonl"
    zero_rec.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "claims": [{"text": "Vanilla claim without citations", "citations": []}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    art = estimate_judge_cost(zero_rec, tmp_path / "zero_est.json")
    assert art["n_units"] == 0
    assert art["total_input_tokens"] == 0
    assert art["total_output_tokens_assumed"] == 0
    assert art["usd_input"] == 0.0
    assert art["usd_output"] == 0.0
    assert art["usd_total"] == 0.0
    assert art["per_unit_usd"] == 0.0
