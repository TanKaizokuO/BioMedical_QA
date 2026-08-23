"""Tests for verifier score population module and driver script."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from biomedqa.schema import (
    Citation,
    Claim,
    Granularity,
    HumanLabel,
    QueryRecord,
    RetrievedPassage,
    SupportLabel,
    System,
    VerifierScore,
    to_dict,
)
from biomedqa.scoring.verifier_scores import (
    DEFAULT_VERIFIER_NAME,
    populate_verifier_scores,
)
from scripts.populate_verifier_scores import main as script_main


def _make_sample_record(
    query_id: str = "q1",
    claim_id: str = "c1",
    passages: list[tuple[str, str]] | None = None,
    citations: list[tuple[str, int, int]] | None = None,
    human_labels: list[HumanLabel] | None = None,
) -> QueryRecord:
    if passages is None:
        passages = [("p1", "Passage text one for testing."), ("p2", "Passage text two for testing.")]
    if citations is None:
        citations = [("p1", 0, 16), ("p2", 0, 16)]

    retrieved = [
        RetrievedPassage(passage_id=pid, rank=i + 1, score=1.0 - 0.1 * i, retriever="bm25", text=ptext)
        for i, (pid, ptext) in enumerate(passages)
    ]
    cits = [Citation(passage_id=pid, char_start=cs, char_end=ce) for pid, cs, ce in citations]

    claim = Claim(
        claim_id=claim_id,
        text="Claim hypothesis text",
        citations=cits,
        granularity=Granularity.DECONTEXTUALIZED_ATOMIC,
        verifier_scores=[],
        human_labels=human_labels or [],
        source_start=10,
        source_end=50,
    )

    return QueryRecord(
        run_id="run1",
        query_id=query_id,
        question="Sample question?",
        system=System.JOINT,
        seed=42,
        retrieved=retrieved,
        claims=[claim],
    )


def test_cache_hit_populates_right_score_at_citation_index():
    record = _make_sample_record()
    # Premise for cit 0: "Passage text one"
    # Premise for cit 1: "Passage text two"
    hyp = "Claim hypothesis text"
    cache = {
        ("Passage text one", hyp): 0.88,
        ("Passage text two", hyp): 0.34,
    }

    populated, cov = populate_verifier_scores([record], cache)
    assert cov["n_records"] == 1
    assert cov["n_claims"] == 1
    assert cov["n_citations"] == 2
    assert cov["n_scored"] == 2
    assert cov["n_missing"] == 0
    assert cov["n_extra_citations"] == 1
    assert cov["coverage_rate"] == 1.0
    assert cov["n_citations"] == cov["n_scored"] - cov["n_extra_citations"] + cov["n_missing"] + cov["n_extra_citations"]
    scores = populated[0].claims[0].verifier_scores
    assert len(scores) == 2
    assert scores[0] == VerifierScore(name=DEFAULT_VERIFIER_NAME, score=0.88, latency_s=None)
    assert scores[1] == VerifierScore(name=DEFAULT_VERIFIER_NAME, score=0.34, latency_s=None)


def test_cache_miss_reported_not_imputed():
    record = _make_sample_record()
    hyp = "Claim hypothesis text"
    # Put only citation 0 in cache
    cache = {
        ("Passage text one", hyp): 0.75,
    }

    populated, cov = populate_verifier_scores([record], cache)
    assert cov["n_scored"] == 1
    assert cov["n_missing"] == 1
    assert cov["missing_pairs"] == [
        {"query_id": "q1", "claim_id": "c1", "citation_index": 1}
    ]

    scores = populated[0].claims[0].verifier_scores
    assert len(scores) == 1
    assert scores[0].score == 0.75
    # No imputed score (e.g. 0.0 or 0.5) added for citation 1
    assert len(scores) == 1
    # Original claim and record preserved
    assert populated[0].claims[0].claim_id == "c1"


def test_roundtrip_preserves_unrelated_fields():
    labels = [
        HumanLabel("ann1", SupportLabel.SUPPORTED, True, citation_index=0),
        HumanLabel("ann1", SupportLabel.NOT_SUPPORTED, True, citation_index=1),
    ]
    record = _make_sample_record(human_labels=labels)
    cache = {
        ("Passage text one", "Claim hypothesis text"): 0.9,
        ("Passage text two", "Claim hypothesis text"): 0.1,
    }

    populated, _ = populate_verifier_scores([record], cache)

    orig_dict = to_dict(record)
    pop_dict = to_dict(populated[0])

    # Remove verifier_scores from comparison
    orig_dict["claims"][0]["verifier_scores"] = []
    pop_dict["claims"][0]["verifier_scores"] = []

    assert orig_dict == pop_dict


def test_determinism_byte_identical():
    record = _make_sample_record()
    cache = {
        ("Passage text one", "Claim hypothesis text"): 0.95,
        ("Passage text two", "Claim hypothesis text"): 0.05,
    }

    pop1, cov1 = populate_verifier_scores([record], cache)
    pop2, cov2 = populate_verifier_scores([record], cache)

    assert [to_dict(r) for r in pop1] == [to_dict(r) for r in pop2]
    assert cov1 == cov2


def test_citation_index_alignment_with_human_labels():
    labels = [
        HumanLabel("ann1", SupportLabel.SUPPORTED, True, citation_index=0),
        HumanLabel("ann2", SupportLabel.PARTIAL, True, citation_index=1),
    ]
    record = _make_sample_record(human_labels=labels)
    cache = {
        ("Passage text one", "Claim hypothesis text"): 0.82,
        ("Passage text two", "Claim hypothesis text"): 0.45,
    }

    populated, _ = populate_verifier_scores([record], cache)
    claim = populated[0].claims[0]

    assert len(claim.verifier_scores) == 2
    assert len(claim.human_labels) == 2
    # verifier_scores[0] aligns with citation 0 (and human_labels[0].citation_index == 0)
    assert claim.human_labels[0].citation_index == 0
    assert claim.verifier_scores[0].score == 0.82
    assert claim.human_labels[1].citation_index == 1
    assert claim.verifier_scores[1].score == 0.45


def test_empty_claim_and_zero_citation_edge_cases():
    # Record with no claims
    empty_record = QueryRecord(
        run_id="run1", query_id="q_empty", question="Q?", system=System.JOINT, seed=0, claims=[]
    )
    pop_empty, cov_empty = populate_verifier_scores([empty_record], {})
    assert cov_empty["n_records"] == 1
    assert cov_empty["n_claims"] == 0
    assert cov_empty["n_citations"] == 0
    assert cov_empty["n_scored"] == 0
    assert cov_empty["n_missing"] == 0
    assert cov_empty["coverage_rate"] == 1.0

    # Claim with zero citations
    zero_cit_claim = Claim(claim_id="c_nocit", text="Uncited claim", citations=[])
    record_nocit = QueryRecord(
        run_id="run1", query_id="q_nocit", question="Q?", system=System.JOINT, seed=0, claims=[zero_cit_claim]
    )
    pop_nocit, cov_nocit = populate_verifier_scores([record_nocit], {})
    assert cov_nocit["n_claims"] == 1
    assert cov_nocit["n_citations"] == 0
    assert len(pop_nocit[0].claims[0].verifier_scores) == 0


def test_script_refuses_writing_into_source_path(monkeypatch, tmp_path):
    records_file = tmp_path / "test.records.jsonl"
    records_file.write_text("{}\n", encoding="utf-8")
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "populate_verifier_scores.py",
            "--records",
            str(records_file),
            "--cache",
            str(cache_file),
            "--out",
            str(records_file),
        ],
    )

    ret = script_main()
    assert ret == 1


class StubVerifier:
    name = DEFAULT_VERIFIER_NAME

    def score_pairs(self, pairs):
        return [VerifierScore(name=self.name, score=0.99, latency_s=0.042) for _ in pairs]


def test_verifier_fallback_when_allowed():
    record = _make_sample_record()
    cache = {}  # Empty cache

    verifier = StubVerifier()
    populated, cov = populate_verifier_scores(
        [record], cache, verifier=verifier, allow_verifier=True
    )

    assert cov["n_scored"] == 2
    assert cov["n_missing"] == 0
    scores = populated[0].claims[0].verifier_scores
    assert len(scores) == 2
    assert scores[0].score == 0.99
    assert scores[0].latency_s == 0.042
