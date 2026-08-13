"""ALCE citation precision / recall / F1 — `scoring/citation.py`.

φ is a stub the test controls, so the entailment decision is fixed and the arithmetic is the only
thing under test. Spans carry `quoted_text`, so `passages=None` exercises the fallback path and no
passage table is needed.
"""

from __future__ import annotations

import math

import pytest

from biomedqa.schema import Citation, Claim, QueryRecord, System
from biomedqa.scoring.citation import citation_f1, citation_precision, citation_recall


def _cite(quote: str) -> Citation:
    return Citation(passage_id="p", char_start=0, char_end=0, quoted_text=quote)


def _claim(*quotes: str, text: str = "dose and outcome") -> Claim:
    return Claim(claim_id="c", text=text, citations=[_cite(q) for q in quotes])


# φ that entails iff the premise mentions every required token — the union is what matters.
def _phi_needs(*tokens: str):
    return lambda premise, _hyp: all(t in premise for t in tokens)


# --- recall: union entailment ---------------------------------------------------------------------

def test_recall_is_one_only_when_the_union_entails():
    # Neither span alone carries both tokens; concat "dose outcome" does.
    claim = _claim("dose", "outcome")
    assert citation_recall(claim, _phi_needs("dose", "outcome")) == 1.0


def test_recall_is_zero_when_the_union_does_not_entail():
    claim = _claim("dose")
    assert citation_recall(claim, _phi_needs("dose", "outcome")) == 0.0


def test_recall_is_zero_for_an_uncited_claim():
    assert citation_recall(_claim(), _phi_needs("dose")) == 0.0


# --- precision: remove-it-and-see -----------------------------------------------------------------

def test_jointly_necessary_citations_are_both_relevant():
    # Each span fails alone; removing either breaks the union entailment, so neither is irrelevant.
    claim = _claim("dose", "outcome")
    assert citation_precision(claim, _phi_needs("dose", "outcome")) == pytest.approx(1.0)


def test_an_irrelevant_citation_drops_precision():
    # "full" entails alone; "junk" fails alone AND the rest ({"full"}) already suffice -> irrelevant.
    claim = _claim("full", "junk")
    assert citation_precision(claim, _phi_needs("full")) == pytest.approx(0.5)


def test_a_lone_non_entailing_citation_is_not_irrelevant():
    # Removing it leaves nothing that could suffice, so the frozen rule keeps it. Precision 1.0;
    # the recall term is what carries the penalty (see the corpus test).
    claim = _claim("junk")
    assert citation_precision(claim, _phi_needs("full")) == pytest.approx(1.0)


def test_precision_is_nan_for_an_uncited_claim():
    assert math.isnan(citation_precision(_claim(), _phi_needs("dose")))


# --- corpus-level F1 and the abstention denominator -----------------------------------------------

def _record(claims: list[Claim]) -> QueryRecord:
    return QueryRecord(run_id="r", query_id="q", question="?", system=System.JOINT, seed=0, claims=claims)


def test_corpus_f1_excludes_abstentions_from_recall_but_reports_both():
    answered = _claim("x", text="metformin reduces mortality")
    # Real abstention (no citation, states an incapacity about the source material) — ADR-0010.
    abstention = Claim(claim_id="c2", text="The passages do not address the dose.", citations=[])
    record = _record([answered, abstention])

    result = citation_f1([record], _phi_needs("x"))

    assert result["n_claims"] == 2
    assert result["n_answered"] == 1
    assert result["n_abstentions"] == 1
    assert result["n_citations"] == 1
    assert result["n_relevant_citations"] == 1
    # Headline recall is over answered claims only; recall_all_claims keeps the abstention in view.
    assert result["recall"] == pytest.approx(1.0)
    assert result["recall_all_claims"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["f1_all_claims"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)


def test_corpus_precision_is_micro_averaged_not_a_mean_of_ratios():
    # Claim A: 1 citation, relevant. Claim B: "full" relevant, "junk" irrelevant -> 1 of 2.
    # Micro precision = (1 + 1) / (1 + 2) = 2/3. A mean of per-claim ratios would give (1 + 0.5)/2 = 0.75.
    a = _claim("full", text="a")
    b = _claim("full", "junk", text="b")
    result = citation_f1([_record([a, b])], _phi_needs("full"))
    assert result["n_relevant_citations"] == 2
    assert result["n_citations"] == 3
    assert result["precision"] == pytest.approx(2 / 3)


def test_recall_is_zero_when_no_claim_is_entailed():
    claim = _claim("dose", text="a")
    result = citation_f1([_record([claim])], _phi_needs("absent-token"))
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
