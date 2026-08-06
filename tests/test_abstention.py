"""Abstention detection, and the validation ADR-0010 conditions itself on.

ADR-0010 keeps `SCHEMA_VERSION` at 1.0.0 on the argument that `Claim.text` + `Claim.citations` +
`raw_generation` already carry everything abstention detection needs. That argument is a bet, and
`test_g0_llama_abstention` / `test_g0_qwen_abstention` are how it gets paid: both assert against the
**real** G0 bake-off output, which contains the two structurally different abstention shapes that
motivated the decision. If either fails, the raw material is insufficient and the no-bump decision is
revisited **before** the gold set launches Sep 7 — not after.

The rest guards the expensive direction of being wrong. A false positive silently shrinks the
citation-recall denominator, which flatters whichever system produced it; a false negative merely
reproduces today's known bug on a single claim. So the substantive-negative cases below matter more
than the abstention cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biomedqa.schema import Citation, Claim, QueryRecord, System
from biomedqa.scoring.abstention import (
    ABSTENTION_RULE_VERSION,
    abstention_claims,
    abstention_rate,
    answered_claims,
    is_abstention,
    is_full_abstention,
)

# `docs/harvest/g0`, not `runs/g0`: `runs/` is gitignored, and these two records are the evidence
# ADR-0010 rests on. Under `runs/` the tests below would skip on every machine but the one that ran
# G0 — silently, which is the failure mode they exist to prevent.
G0_DIR = Path(__file__).resolve().parents[1] / "docs" / "harvest" / "g0"


def claim(text: str, *, cited: bool = False) -> Claim:
    citations = [Citation("21645374:0", 0, 10, quoted_text="0123456789")] if cited else []
    return Claim(claim_id="c1", text=text, citations=citations)


def record(*claims: Claim) -> QueryRecord:
    return QueryRecord(
        run_id="r", query_id="10781708", question="q?", system=System.JOINT, seed=0,
        claims=list(claims),
    )


# ---------------------------------------------------------------------------------------------
# The G0 validation. These are the tests ADR-0010 rests on.
# ---------------------------------------------------------------------------------------------


def g0_answer(model_fragment: str, pubid: str = "10781708") -> str:
    matches = sorted(G0_DIR.glob(f"bakeoff_{model_fragment}*.json"))
    if not matches:
        pytest.skip(f"no G0 bake-off record for {model_fragment!r} in {G0_DIR}")
    payload = json.loads(matches[0].read_text())
    for row in payload["records"]:
        if str(row["pubid"]) == pubid:
            return row["answer"]
    pytest.fail(f"pubid {pubid} absent from {matches[0].name}")


def test_g0_llama_abstention() -> None:
    """Llama emitted the abstention *as an uncited claim inside the list* — item 11 of 11.

    This is the shape the chosen generator (ADR-0004 / D1) actually produces, so it is the one the
    recall denominator depends on.
    """
    answer = g0_answer("hugging-quants")
    line = [ln for ln in answer.splitlines() if ln.strip().startswith("11.")]
    assert line, "the G0 Llama record no longer has an 11th claim; re-read ADR-0010 before editing"
    assert is_abstention(claim(line[0].split(".", 1)[1].strip()))


def test_g0_qwen_abstention() -> None:
    """Qwen emitted it as trailing prose *outside* the claim list, which the parser absorbed into a
    phantom sixth claim (`n_claims: 6` over five bullets).

    No field on `Claim` could have represented this, which is ADR-0010's second argument against
    `Claim.is_abstention`. The predicate must still catch it once the parser hands it over.
    """
    answer = g0_answer("qwen")
    trailing = answer.split("\n\n")[-1].strip()
    assert "cannot be directly answered" in trailing, "G0 Qwen record changed; re-read ADR-0010"
    assert is_abstention(claim(trailing))


def test_g0_llama_substantive_claims_are_not_abstentions() -> None:
    """The other ten claims must survive. This is the false-positive guard on real output."""
    answer = g0_answer("hugging-quants")
    substantive = [
        ln.split(".", 1)[1].strip()
        for ln in answer.splitlines()
        if ln.strip()[:2].rstrip(".").isdigit() and not ln.strip().startswith("11.")
    ]
    assert len(substantive) == 10
    for text in substantive:
        assert not is_abstention(claim(text, cited=True))


# ---------------------------------------------------------------------------------------------
# False positives — the expensive direction.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # A substantive negative finding. Says "does not reduce", names no source material.
        "Metformin does not reduce all-cause mortality in patients over 75.",
        # A substantive hedge about the world, not about the passages.
        "The mechanism by which metformin reduces mortality is not well understood.",
        # Negative result with a number — the exact shape C9's numerics stratum cares about.
        "The relative risk was not statistically significant at the 5% level.",
        # Names a source but asserts something about it rather than declining.
        "The passage reports an incidence of 0.6% during hospitalisation.",
    ],
)
def test_substantive_claims_are_never_abstentions(text: str) -> None:
    assert not is_abstention(claim(text))


def test_a_cited_claim_is_never_an_abstention() -> None:
    """Even with abstention-shaped wording. A claim that cites a passage is asserting something
    about it, and dropping it from the recall denominator would flatter the system that emitted it.
    """
    text = "The provided passages do not address long-term prophylaxis."
    assert is_abstention(claim(text))
    assert not is_abstention(claim(text, cited=True))


# ---------------------------------------------------------------------------------------------
# True positives, across the phrasings a generator actually reaches for.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The question of whether prophylaxis in all patients makes sense is not addressed by the provided passages.",
        "This question cannot be answered based on the provided passages.",
        "The question cannot be directly answered based on the provided passages.",
        "The retrieved documents do not discuss the effectiveness of prophylaxis.",
        "There is insufficient information in the provided context to determine the dose.",
        "The passages provide no information about mortality outcomes.",
    ],
)
def test_abstention_phrasings(text: str) -> None:
    assert is_abstention(claim(text))


# ---------------------------------------------------------------------------------------------
# The denominator rule — the actual fix to issue #9.
# ---------------------------------------------------------------------------------------------


def test_abstention_leaves_the_recall_denominator_but_not_the_record() -> None:
    """The bug in one assertion: ten good claims plus one abstention is a denominator of ten.

    Named in issue #9 — an abstaining answer must not score as non-compliant.
    """
    rec = record(
        *[Claim(claim_id=f"c{i}", text=f"Finding {i}.",
                citations=[Citation("p:0", 0, 4, quoted_text="abcd")]) for i in range(10)],
        Claim(claim_id="c10", text="The question is not addressed by the provided passages."),
    )
    assert len(rec.claims) == 11
    assert len(answered_claims(rec)) == 10
    assert len(abstention_claims(rec)) == 1
    assert rec.validate() == []          # nothing about abstention is a schema violation
    assert not is_full_abstention(rec)   # partial abstention is the observed behaviour


def test_full_abstention_is_distinguishable() -> None:
    rec = record(claim("The provided passages do not address this question."))
    assert is_full_abstention(rec)
    assert answered_claims(rec) == []


def test_empty_record_is_not_a_full_abstention() -> None:
    """A record with no claims is a parse failure or an empty generation, not a decision to decline.
    Conflating them would hide the first behind the second.
    """
    assert not is_full_abstention(record())


def test_abstention_rate_is_claim_level() -> None:
    rec = record(
        Claim(claim_id="a", text="Finding.", citations=[Citation("p:0", 0, 4, quoted_text="abcd")]),
        Claim(claim_id="b", text="The passages do not address the dose."),
    )
    assert abstention_rate([rec]) == 0.5
    assert abstention_rate([]) == 0.0


def test_rule_version_is_recorded_in_config() -> None:
    """A number must always state which rule produced it — the reason this lives in config.

    And the version must move with the patterns: a rule change that leaves the recorded version
    behind makes two incomparable sets of numbers look identical in the manifest.
    """
    from biomedqa.config import RunConfig

    assert RunConfig().scoring.abstention_rule_version == ABSTENTION_RULE_VERSION


def test_scoring_config_is_in_the_run_hash() -> None:
    """Re-scoring under a different rule must be visibly a different configuration."""
    from dataclasses import replace

    from biomedqa.config import RunConfig

    base = RunConfig()
    revised = replace(base, scoring=replace(base.scoring, abstention_rule_version="1.1.0"))
    assert base.hash() != revised.hash()
    # ...but not a different *index*: abstention is scored after retrieval, never during it.
    assert base.index_fingerprint() == revised.index_fingerprint()


def test_schema_version_did_not_move() -> None:
    """ADR-0010's headline: abstention added no schema field and no version bump.

    If this fails, someone added a field — read ADR-0010 before deciding they were wrong, and note
    that after Sep 7 a schema change costs the gold set.
    """
    from biomedqa.schema import SCHEMA_VERSION

    assert SCHEMA_VERSION == "1.0.0"
