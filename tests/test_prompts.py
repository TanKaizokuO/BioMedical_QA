"""The prompt contract: what the model is asked for, and what survives parsing."""

from __future__ import annotations

import pytest

from biomedqa.prompts import (
    CONTEXT_DEPTH,
    build_prompt,
    effort_is_matched,
    iteration_counts,
    locate_quote,
    parse_response,
    render_context,
)
from biomedqa.schema import MAX_CITATIONS, RetrievedPassage, System

PASSAGE_TEXT = "Metformin reduced HbA1c by 1.2% over 24 weeks. No severe hypoglycaemia occurred."
OTHER_TEXT = "Placebo showed no significant change in HbA1c."


def _passages(n=3):
    texts = [PASSAGE_TEXT, OTHER_TEXT, "A third abstract about dosing."]
    return [
        RetrievedPassage(
            passage_id=f"p{i}", rank=i, score=1.0 / i, retriever="rerank", text=texts[i - 1]
        )
        for i in range(1, n + 1)
    ]


def test_context_refuses_passages_with_no_text():
    """A passage rendered empty is one the model cannot cite, and the loss would be booked against
    the system instead of the harness. `_rerank` refuses identically."""
    ps = _passages(2)
    ps[1] = RetrievedPassage(passage_id="p2", rank=2, score=0.5, retriever="rerank", text=None)

    with pytest.raises(ValueError, match="context needs passage text"):
        render_context(ps)


def test_context_is_depth_limited_and_rank_ordered():
    ps = list(reversed(_passages(3)))

    rendered = render_context(ps, depth=2)

    assert rendered.index("[p1]") < rendered.index("[p2]")
    assert "[p3]" not in rendered


def test_context_depth_follows_the_gate():
    """ADR-0015 gates G1 at hit@10, so the drafted context is the one the gate certified."""
    assert CONTEXT_DEPTH == 10


def test_the_citation_cap_is_stated_identically_wherever_a_system_cites():
    """ADR-0005 / CONTEXT.md: an unequal cap makes C2's gap a budget artifact.

    The comparison is per *citing stage*, not per system. Joint cites while answering; post-hoc
    cites in its second pass; vanilla never cites, which is the one legitimate way to differ.
    """
    joint = build_prompt(System.JOINT, "Does metformin help?", _passages(), MAX_CITATIONS)
    post_hoc = build_prompt(
        System.POST_HOC,
        "Does metformin help?",
        _passages(),
        MAX_CITATIONS,
        stage="cite",
        answer="CLAIM 1: X.",
    )
    vanilla = build_prompt(System.VANILLA, "Does metformin help?", _passages(), MAX_CITATIONS)

    cap = f"at most {MAX_CITATIONS} passages per claim"
    assert cap in joint and cap in post_hoc
    assert f"up to {MAX_CITATIONS} CITE lines" in joint
    assert f"up to {MAX_CITATIONS} CITE lines" in post_hoc
    assert "CITE" not in vanilla


def test_the_format_example_spells_passage_ids_the_way_the_context_does():
    """W4 live smoke, 0/3 clean parses on both citing systems. The context block printed `[p1]`
    while the format example asked for a bare `passage_id`, so the model echoed the brackets it had
    been shown and every CITE line was rejected for an id the harness itself spelled two ways. The
    example and `render_context` have to agree, or the loss is booked against the system."""
    ps = _passages(1)

    assert render_context(ps).splitlines()[0] == "[p1]"

    joint = build_prompt(System.JOINT, "Q?", ps, MAX_CITATIONS)
    example = next(ln for ln in joint.splitlines() if ln.startswith("CITE:"))

    assert "[passage_id]" in example
    assert "passage_id ||" not in joint  # the unbracketed spelling that caused the mismatch


def test_cite_lines_attach_to_the_claim_above_them():
    """Two live smokes showed the 8B model numbering CITE lines 1..k within each claim. Read as a
    claim id, every claim's first citation landed on c1 — corrupting the claim-to-citation mapping
    C2 measures and inventing cap violations for claims that had cited once. Line order is the one
    signal the model got right, so line order is the grammar. A number, if written, is ignored."""
    raw = (
        "DECISION: yes\n"
        "CLAIM 1: Metformin reduced HbA1c in the trial population.\n"
        "CITE 1: [p1] || Metformin reduced HbA1c by 1.2%\n"
        "CLAIM 2: Placebo did not change HbA1c.\n"
        "CITE 1: [p2] || Placebo showed no significant change in HbA1c.\n"
    )

    out = parse_response(raw, _passages(), MAX_CITATIONS)

    assert out.errors == []
    assert [len(c.citations) for c in out.claims] == [1, 1]  # not [2, 0]
    assert out.claims[1].citations[0].passage_id == "p2"


def test_a_cite_before_any_claim_is_still_an_error():
    out = parse_response("DECISION: yes\nCITE: [p1] || Metformin\n", _passages(), MAX_CITATIONS)

    assert any("CITE line precedes any CLAIM" in e for e in out.errors)


def test_both_citing_stages_get_the_identical_attachment_rule():
    """A grammar rule only one citing system saw would move C2's gap by itself."""
    joint = build_prompt(System.JOINT, "Q?", _passages(), MAX_CITATIONS)
    post_hoc = build_prompt(
        System.POST_HOC, "Q?", _passages(), MAX_CITATIONS, stage="cite", answer="CLAIM 1: X."
    )

    rule = "A CITE line supports the CLAIM line directly above it."
    assert rule in joint and rule in post_hoc


def test_the_quote_rule_is_scoped_away_from_the_claim_rule():
    """Unscoped, "write each claim so it stands alone" read as advice about the whole reply, and
    the live smokes showed the model composing standalone *quotes* that the passage does not
    contain. Both citing stages must carry the scoping sentence; vanilla must carry neither."""
    joint = build_prompt(System.JOINT, "Q?", _passages(), MAX_CITATIONS)
    post_hoc = build_prompt(
        System.POST_HOC, "Q?", _passages(), MAX_CITATIONS, stage="cite", answer="CLAIM 1: X."
    )
    vanilla = build_prompt(System.VANILLA, "Q?", _passages(), MAX_CITATIONS)

    scope = "apply the CLAIM rules to it"  # unwrapped fragment; the rule text hard-wraps
    assert scope in joint and scope in post_hoc
    assert scope not in vanilla
    assert all("Write each CLAIM line so it stands alone" in p for p in (joint, post_hoc, vanilla))


def test_all_three_systems_are_asked_for_the_same_claim_unit():
    """ADR-0005's unit is the treatment-invariant part: a baseline whose claims are shaped
    differently is being compared on the wrong axis."""
    rendered = [
        build_prompt(s, "Q?", _passages(), MAX_CITATIONS) for s in System
    ]

    assert all("Resolve every pronoun" in p for p in rendered)
    assert all("states exactly one thing" in p for p in rendered)


def test_post_hoc_first_pass_never_mentions_citing():
    """A first pass that knows citations are coming is already doing joint grounding, and C2's gap
    would close for a reason unrelated to the systems."""
    answer_stage = build_prompt(
        System.POST_HOC, "Q?", _passages(), MAX_CITATIONS, stage="answer"
    )

    assert "CITE" not in answer_stage
    assert "quote" not in answer_stage.lower().split("Question:")[0]


def test_post_hoc_cite_stage_requires_the_first_pass_answer():
    with pytest.raises(ValueError, match="needs the answer"):
        build_prompt(System.POST_HOC, "Q?", _passages(), MAX_CITATIONS, stage="cite")


def test_locate_quote_is_exact_and_refuses_near_misses():
    """A fuzzy match would fabricate offsets for text the passage does not contain, and every
    downstream verifier reads those offsets as ground truth."""
    hit = locate_quote("reduced HbA1c by 1.2%", "p1", PASSAGE_TEXT)

    assert hit is not None
    assert PASSAGE_TEXT[hit.char_start : hit.char_end] == "reduced HbA1c by 1.2%"
    assert len(hit.quoted_text) == hit.char_end - hit.char_start  # what validate() checks
    assert locate_quote("reduced HbA1c by 1.3%", "p1", PASSAGE_TEXT) is None


def test_parse_recovers_claims_citations_and_decision():
    raw = (
        "DECISION: yes\n"
        "CLAIM 1: Metformin reduced HbA1c in the trial population.\n"
        "CITE 1: p1 || Metformin reduced HbA1c by 1.2% over 24 weeks.\n"
        "CLAIM 2: Placebo did not change HbA1c.\n"
        "CITE 2: p2 || Placebo showed no significant change in HbA1c.\n"
    )

    out = parse_response(raw, _passages(), MAX_CITATIONS)

    assert out.errors == []
    assert out.decision == "yes"
    assert [c.claim_id for c in out.claims] == ["c1", "c2"]
    assert out.claims[0].citations[0].char_start == 0


def test_a_quote_the_model_did_not_copy_is_an_error_not_a_repair():
    """G2 gates on >=95% valid parse; the gate is only real if failures are allowed to appear."""
    raw = "DECISION: no\nCLAIM 1: X.\nCITE 1: p1 || Metformin reduced HbA1c by 9.9%.\n"

    out = parse_response(raw, _passages(), MAX_CITATIONS)

    assert out.claims[0].citations == []
    assert any("not found verbatim" in e for e in out.errors)


def test_over_cap_citations_are_kept_so_the_violation_stays_visible():
    """Trimming the fourth citation would erase the evidence that a system ignored the cap the
    whole fairness argument rests on. `validate()` reports it; the parser must not hide it."""
    raw = (
        "DECISION: maybe\nCLAIM 1: X.\n"
        + "".join(f"CITE 1: p1 || {q}\n" for q in ("Metformin", "reduced", "HbA1c", "1.2%"))
    )

    out = parse_response(raw, _passages(), MAX_CITATIONS)

    assert len(out.claims[0].citations) == 4
    assert any("exceeds the cap" in e for e in out.errors)


def test_quotes_containing_the_separator_still_parse():
    ps = [
        RetrievedPassage(
            passage_id="p1", rank=1, score=1.0, retriever="rerank", text="Risk a || b was low."
        )
    ]
    raw = "DECISION: yes\nCLAIM 1: Risk was low.\nCITE 1: p1 || Risk a || b was low.\n"

    out = parse_response(raw, ps, MAX_CITATIONS)

    assert out.errors == []
    assert out.claims[0].citations[0].quoted_text == "Risk a || b was low."


def test_citing_a_passage_outside_the_context_is_an_error():
    raw = "DECISION: yes\nCLAIM 1: X.\nCITE 1: p99 || whatever\n"

    out = parse_response(raw, _passages(), MAX_CITATIONS)

    assert any("not in the context" in e for e in out.errors)


def test_a_bracketed_passage_id_parses_because_that_is_the_taught_spelling():
    """Brackets are the delimiter `render_context` prints, not part of the id. Stripping them
    reads the grammar; it does not repair a wrong answer, and a bad quote is still an error."""
    raw = "DECISION: yes\nCLAIM 1: X.\nCITE 1: [p1] || Metformin reduced HbA1c by 1.2%\n"

    out = parse_response(raw, _passages(), MAX_CITATIONS)

    assert out.errors == []
    assert out.claims[0].citations[0].passage_id == "p1"


def test_missing_decision_and_claims_are_both_reported():
    out = parse_response("I cannot answer this question.", _passages(), MAX_CITATIONS)

    assert "no DECISION line" in out.errors
    assert "no CLAIM lines" in out.errors


def test_joint_and_post_hoc_stay_on_equal_effort():
    """The equal-effort protocol, mechanised. If this fails, one system's prompt was iterated more
    than the other's: either spend the matching cycles or record the imbalance in the paper —
    do not delete the test."""
    counts = iteration_counts()

    assert effort_is_matched(), f"prompt-iteration budgets have drifted: {counts}"
