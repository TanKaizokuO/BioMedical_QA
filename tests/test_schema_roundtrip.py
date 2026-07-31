"""The schema is frozen only if the round-trip is tested. Otherwise it is a convention.

These tests exist to fail loudly on the two mistakes that are irrecoverable rather than merely
annoying: losing a field on write, and binarizing a value that scoring needs continuous.
"""

from __future__ import annotations

import json

import pytest

from biomedqa.schema import (
    MAX_CITATIONS,
    Citation,
    Claim,
    CostRecord,
    Granularity,
    HumanLabel,
    QueryRecord,
    RetrievedPassage,
    SupportLabel,
    System,
    VerifierScore,
    query_record_from_dict,
    read_query_records,
    to_dict,
    write_jsonl,
)


def make_record() -> QueryRecord:
    """A record exercising every nested type, including the ones only annotation populates."""
    return QueryRecord(
        run_id="run-abc123",
        query_id="21645374",
        question="Does metformin reduce all-cause mortality in type 2 diabetes?",
        system=System.JOINT,
        seed=0,
        retrieved=[
            RetrievedPassage("21645374:0", 1, 12.5, "rerank", text="Metformin ..."),
            RetrievedPassage("18952324:2", 2, 9.1, "rerank"),
        ],
        gold_passage_ids=["21645374:0", "21645374:1"],
        claims=[
            Claim(
                claim_id="c0",
                text="Metformin reduces all-cause mortality in patients with type 2 diabetes.",
                citations=[Citation("21645374:0", 0, 9, quoted_text="Metformin")],
                granularity=Granularity.DECONTEXTUALIZED_ATOMIC,
                verifier_scores=[VerifierScore("minicheck", 0.83, latency_s=0.04)],
                human_labels=[HumanLabel("ann1", SupportLabel.PARTIAL, True, citation_index=0)],
                source_start=0,
                source_end=70,
            )
        ],
        raw_generation="Metformin reduces all-cause mortality ... [1]",
        final_decision="yes",
        gold_final_decision="yes",
        latency_s=1.23,
        prompt_tokens=812,
        completion_tokens=96,
    )


def test_roundtrip_is_lossless():
    original = make_record()
    restored = query_record_from_dict(to_dict(original))
    assert restored == original, "a field was lost or mutated crossing the serialisation boundary"


def test_roundtrip_survives_jsonl(tmp_path):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [make_record()])
    (restored,) = list(read_query_records(path))
    assert restored == make_record()


def test_enums_serialise_as_their_values():
    d = to_dict(make_record())
    assert d["system"] == "joint"
    assert d["claims"][0]["granularity"] == "decontextualized_atomic"
    assert d["claims"][0]["human_labels"][0]["support_label"] == "PARTIAL"
    json.dumps(d)  # must be plain-JSON serialisable, with no custom encoder


def test_unknown_field_raises_rather_than_being_dropped():
    d = to_dict(make_record())
    d["hit_at_5"] = True  # precisely the kind of precomputed value that must never appear
    with pytest.raises(TypeError):
        query_record_from_dict(d)


def test_verifier_score_stays_continuous():
    """A stored boolean fixes one operating point and discards the AUROC sweep irrecoverably."""
    score = to_dict(make_record())["claims"][0]["verifier_scores"][0]["score"]
    assert isinstance(score, float) and not isinstance(score, bool)
    assert 0.0 < score < 1.0


def test_support_label_is_four_way_on_disk():
    """The binary collapse is derived, never stored."""
    label = to_dict(make_record())["claims"][0]["human_labels"][0]["support_label"]
    assert label in {e.value for e in SupportLabel}
    assert SupportLabel(label).is_supporting is True  # PARTIAL collapses to supporting


def test_citation_span_is_validated_on_construction():
    with pytest.raises(ValueError):
        Citation("21645374:0", 10, 4)


class TestValidate:
    """`validate()` reports contract violations and never repairs them — the violation is a
    measurement, and G0 ranks candidate generators on exactly these."""

    def test_clean_record_has_no_problems(self):
        assert make_record().validate() == []

    def test_cap_violation_is_reported_but_citations_are_kept(self):
        record = make_record()
        record.claims[0].citations = [
            Citation("21645374:0", i, i + 1) for i in range(MAX_CITATIONS + 1)
        ]
        problems = record.validate()
        assert any("exceeds the cap" in p for p in problems)
        assert len(record.claims[0].citations) == MAX_CITATIONS + 1, "citations were silently dropped"

    def test_citation_outside_the_retrieved_set_is_reported(self):
        record = make_record()
        record.claims[0].citations = [Citation("99999999:0", 0, 5)]
        assert any("not in the retrieved set" in p for p in record.validate())

    def test_vanilla_baseline_may_not_carry_citations(self):
        record = make_record()
        record.system = System.VANILLA
        assert any("vanilla" in p for p in record.validate())

    def test_non_contiguous_ranks_are_reported(self):
        record = make_record()
        record.retrieved[1].rank = 5
        assert any("1-indexed contiguous" in p for p in record.validate())


def test_cost_record_carries_table_4_columns():
    """Table 4's caption was written in W0; these are the columns it promises."""
    cost = to_dict(CostRecord("run-abc123", "21645374", "generate", "vllm:model",
                              input_tokens=812, output_tokens=96, usd=0.0, wall_s=1.23))
    for column in ("input_tokens", "output_tokens", "usd", "wall_s"):
        assert column in cost
