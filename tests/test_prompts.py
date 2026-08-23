"""The prompt contract: what the model is asked for, and what survives parsing."""

from __future__ import annotations

import json

import pytest

from biomedqa.prompts import (
    CONTEXT_DEPTH,
    MAX_CLAIM_WORDS,
    PARITY_ITERATION_LIMIT,
    PARITY_ITERATIONS,
    PARITY_LOOP_CLOSED,
    POST_HOC_ANSWER_TEMPLATE,
    PROMPT_ITERATIONS,
    RUNAWAY_CHAIN_MIN,
    build_citation_response_format,
    build_prompt,
    claim_stem,
    effort_is_matched,
    iteration_counts,
    locate_quote,
    parity_budget_remains,
    parity_iteration_count,
    parity_loop_is_open,
    parse_response,
    post_hoc_answer_template_digest,
    render_context,
    runaway_chains,
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

#: Verbatim from joint's `21074975` in `docs/harvest/parity_iter1b.records.jsonl` — claim 5 of 13,
#: 38 words, the longest claim in that query that is still a real claim. Claims 6..13 are the
#: degenerate tail: each re-emits its predecessor's full text plus one more `and the risk ...`
#: clause, at 59, 80, 101, 122, 143, 164, 206 and finally 731 words.
REAL_LONG_CLAIM = (
    "The risk of transition to psychosis is associated with the level of distress associated with "
    "attenuated psychotic symptoms, but the association is not significant, and the level of "
    "distress is not a useful criterion for enriching UHR samples."
)

#: The observed accumulation, reproduced by the clause that drove it.
RUNAWAY_CLAIM = REAL_LONG_CLAIM + (
    " and the risk of transition to psychosis is higher in individuals with higher levels of "
    "distress associated with attenuated psychotic symptoms," * 8
)


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


def test_locate_quote_normalizes_wrapping_quotes_and_whitespace_verbatim():
    """Quotes wrapped in quotes or carrying newline/space differences are recovered as verbatim passage spans."""
    hit_q = locate_quote('"reduced HbA1c by 1.2%"', "p1", PASSAGE_TEXT)
    assert hit_q is not None
    assert PASSAGE_TEXT[hit_q.char_start : hit_q.char_end] == "reduced HbA1c by 1.2%"

    passage_nl = "reduced HbA1c\nby 1.2%"
    hit_nl = locate_quote("reduced HbA1c by 1.2%", "p1", passage_nl)
    assert hit_nl is not None
    assert passage_nl[hit_nl.char_start : hit_nl.char_end] == "reduced HbA1c\nby 1.2%"


def test_locate_quote_recovers_a_case_drifted_quote_as_the_passages_own_text():
    """The 8B model lower-cases a quote's first letter when the span starts mid-sentence in its
    head. The span it names is real, so it is found — and `quoted_text` is what the passage says,
    never what the model typed, or the offsets and the text would disagree."""
    hit = locate_quote("reduced hba1c by 1.2%", "p1", PASSAGE_TEXT)

    assert hit is not None
    assert hit.quoted_text == "reduced HbA1c by 1.2%"
    assert PASSAGE_TEXT[hit.char_start : hit.char_end] == hit.quoted_text
    # Case tolerance is not licence to invent: a wrong number is still refused.
    assert locate_quote("reduced hba1c by 1.3%", "p1", PASSAGE_TEXT) is None


def test_a_recovered_quote_is_reported_rather_than_counted_clean():
    raw = "DECISION: yes\nCLAIM 1: X.\nCITE: p1 || reduced hba1c by 1.2%\n"

    out = parse_response(raw, _passages(), MAX_CITATIONS)

    assert out.errors == []
    assert any("matched only after normalising" in note for note in out.recovered)


def test_an_id_missing_its_chunk_index_is_read_when_only_one_chunk_could_be_meant():
    raw = "DECISION: yes\nCLAIM 1: X.\nCITE: [p1:] || Metformin reduced HbA1c by 1.2%\n"

    out = parse_response(raw, _passages(), MAX_CITATIONS)

    assert out.errors == []
    assert out.claims[0].citations[0].passage_id == "p1"
    assert any("read as 'p1'" in note for note in out.recovered)


def test_an_ambiguous_document_id_stays_an_error_because_guessing_misattributes_evidence():
    ps = [
        RetrievedPassage(passage_id="doc:0", rank=1, score=1.0, retriever="rerank", text="Alpha."),
        RetrievedPassage(passage_id="doc:1", rank=2, score=0.9, retriever="rerank", text="Beta."),
    ]
    raw = "DECISION: yes\nCLAIM 1: X.\nCITE: [doc:] || Alpha.\n"

    out = parse_response(raw, ps, MAX_CITATIONS)

    assert any("not in the context" in e for e in out.errors)
    assert out.recovered == []


def test_an_empty_claim_line_is_padding_and_is_not_a_claim():
    """A live probe caught the model filling a short reply with bare `CLAIM n:` lines up to the
    count it was given. Counting those as claims would let padding pass the positional match."""
    raw = "DECISION: yes\nCLAIM 1: X.\nCLAIM 2:\nCLAIM 3:\n"

    out = parse_response(raw, _passages(), MAX_CITATIONS)

    assert [c.claim_id for c in out.claims] == ["c1"]
    assert sum("is empty" in e for e in out.errors) == 2


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


def test_a_runaway_repetition_claim_is_a_parse_error():
    """Joint `21074975` emitted a 731-word non-terminating repetition loop and the parser accepted
    it **clean**: no error, one claim, and 0.0 citation recall because nothing entails it. 34 of
    joint's 719 claims exceed 30 words and every one of them scored 0/1. A generator failure that
    parses clean is charged to the system's grounding rather than to the decoder that produced it,
    which is exactly the misattribution ADR-0005 exists to prevent."""
    raw = f"DECISION: maybe\nCLAIM 1: {RUNAWAY_CLAIM}\n"

    out = parse_response(raw, _passages(), MAX_CITATIONS)

    assert any("exceeds the max claim length" in e for e in out.errors)


def test_an_oversized_claim_is_kept_so_the_failure_stays_countable():
    """Same rule as over-cap citations: errors are data. Dropping the claim would delete the
    evidence of non-termination from `records.jsonl` and quietly shrink G2's denominator, so the
    arm that degenerates would look like the arm that emitted fewer claims."""
    raw = f"DECISION: maybe\nCLAIM 1: {RUNAWAY_CLAIM}\nCITE 1: [p1] || Metformin\n"

    out = parse_response(raw, _passages(), MAX_CITATIONS)

    assert [c.claim_id for c in out.claims] == ["c1"]
    assert out.claims[0].text == RUNAWAY_CLAIM
    assert out.claims[0].citations[0].passage_id == "p1"


def test_the_guard_clears_the_longest_real_claim_in_the_run():
    """The threshold is a **pathology** detector, not a style rule, and this is the test that stops
    it being tightened into one. Claim-length p95 is 29 (joint), 29 (post-hoc) and 34 (vanilla)
    words in `parity_iter1b`, so a guard at 30 words would flag 4.73% / 3.06% / 9.43% of claims —
    it would fail G2's >=95% valid-parse bar on vanilla by itself, and it would move C2's gap by
    penalising the three arms at three different rates. At 50 it costs 2.78% / 0.24% / 0.25%: the
    asymmetry that remains is joint's actual degeneracy, which is the finding, not the instrument."""
    raw = f"DECISION: yes\nCLAIM 1: {REAL_LONG_CLAIM}\n"

    out = parse_response(raw, _passages(), MAX_CITATIONS)

    assert len(REAL_LONG_CLAIM.split()) == 38
    assert MAX_CLAIM_WORDS > 34, "must clear every arm's p95, or the guard scores prose length"
    assert out.errors == []


def test_the_boundary_is_inclusive_so_the_threshold_reads_as_written():
    """`max_claim_words` is the largest acceptable claim, not the smallest rejected one."""
    at = " ".join(["word"] * MAX_CLAIM_WORDS)
    over = " ".join(["word"] * (MAX_CLAIM_WORDS + 1))

    assert parse_response(f"DECISION: yes\nCLAIM 1: {at}\n", _passages(), MAX_CITATIONS).errors == []
    assert any(
        "exceeds the max claim length" in e
        for e in parse_response(
            f"DECISION: yes\nCLAIM 1: {over}\n", _passages(), MAX_CITATIONS
        ).errors
    )


def test_the_guard_is_one_number_shared_by_the_parser_and_the_scoring_config():
    """Parse errors are re-derived at scoring time from `raw_generation` (`generate.py`), so the
    guard is a **scoring** rule under ADR-0010 — revising it must re-score, never force a re-run.
    Two copies of the threshold would let a re-scored G2 disagree with the run log that produced
    it, and the disagreement would be invisible."""
    from biomedqa.config import ScoringConfig

    assert ScoringConfig().max_claim_words == MAX_CLAIM_WORDS


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


def test_parity_cycles_are_charged_to_neither_system():
    """ADR-0009 §7's third ledger line, mechanised.

    The silent failure: a parity cycle gets appended to `PROMPT_ITERATIONS[POST_HOC]` because that
    is where post-hoc prompt edits "obviously" go. Two things break at once and neither is visible
    at the call site. The paper reports the baseline as more engineered than it was — the opposite
    of the undercount §4 argues is the safe direction to be wrong in. And `effort_is_matched()`
    goes false, whose only in-bounds repair is a JOINT cycle that ADR-0009 §4 forbids for the
    loop's duration. Parity work is a fairness control, not method development; it is counted
    separately or not at all.
    """
    baseline = iteration_counts()
    system_totals = sum(len(v) for v in PROMPT_ITERATIONS.values())

    assert not any(
        it in v for it in PARITY_ITERATIONS for v in PROMPT_ITERATIONS.values()
    ), "a parity cycle is booked to a system ledger; ADR-0009 §7 charges it to neither"

    assert iteration_counts() == baseline
    assert sum(len(v) for v in PROMPT_ITERATIONS.values()) == system_totals
    assert effort_is_matched(), (
        "parity cycles must not disturb the joint/post-hoc balance: " f"{iteration_counts()}"
    )


def test_parity_iterations_are_numbered_from_one_without_gaps():
    """A ledger nobody can audit is a ledger that gets reconstructed from memory in October — the
    same reason `PROMPT_ITERATIONS` exists. A skipped `n` hides a cycle that was spent.
    """
    assert [it.n for it in PARITY_ITERATIONS] == list(range(1, len(PARITY_ITERATIONS) + 1))


def test_the_parity_loop_stops_at_a_hard_ten():
    """ADR-0009 §5: "A hard 10. Not '~10' — a bound written with a tilde grants exactly the
    permission it exists to deny." The counter is only a bound if something reads it.
    """
    assert PARITY_ITERATION_LIMIT == 10
    assert parity_iteration_count() <= PARITY_ITERATION_LIMIT
    assert parity_budget_remains() == (parity_iteration_count() < PARITY_ITERATION_LIMIT)



def test_the_parity_loop_is_closed_on_the_run_it_says_it_is():
    """ADR-0009 §5's termination and §6's unblinding are the same event, so the record of it has to
    be checkable. The figures here are `docs/harvest/parity_iter1b.md`'s, recomputed from the
    artifacts by `tests/test_scoring_granularity.py` — this test is the ledger side of it."""
    closed = PARITY_LOOP_CLOSED
    assert closed is not None and not parity_loop_is_open()
    assert closed.run == "parity_iter1b"
    assert closed.iterations_used == parity_iteration_count() == 1
    assert (closed.joint_median_words_per_claim, closed.post_hoc_median_words_per_claim) == (15, 17)
    assert closed.gap == pytest.approx(2 / 15, abs=0.0001)
    assert closed.interval == pytest.approx((0.0, 1 / 7), abs=0.0001)
    assert closed.gap <= 0.15, "the loop may not be closed on a basis the gate failed"


def test_terminating_early_does_not_retract_the_w9_check():
    """§5's asymmetric rule is pre-registered in the paper's methods section, so it is not
    retractable because the iteration that closed the loop happened to pass. The residual is
    positive on every basis — post-hoc's claims are still the coarser ones — and that is the branch
    that makes the W9 stratified robustness check mandatory."""
    assert PARITY_LOOP_CLOSED is not None
    assert PARITY_LOOP_CLOSED.residual_favours_c2
    assert PARITY_LOOP_CLOSED.gap > 0


def test_the_post_hoc_template_is_frozen_at_the_terminating_run():
    """The §8 freeze, mechanised. After termination the post-hoc prompt is the artifact the first
    citation-F1 is computed from; editing it silently would make the reported gate verdict describe
    a prompt that no longer exists — and because the blind has lifted, any such edit is tuning with
    F1 known. If this fails, either revert the template or the loop is being reopened, which is a
    methods-section decision and not a code change."""
    assert PARITY_LOOP_CLOSED is not None
    assert post_hoc_answer_template_digest() == PARITY_LOOP_CLOSED.post_hoc_answer_template_sha256
    assert len(POST_HOC_ANSWER_TEMPLATE) > 0


def test_the_budget_left_over_is_not_spendable():
    """The loop closed at 1 of 10, so `parity_budget_remains()` is still True — it answers "is there
    an iteration left", not "may one be spent". Spending one now would tune post-hoc's prompt with
    citation-F1 known, which §6 forbids and which the freeze above is what actually prevents."""
    assert parity_budget_remains()
    assert parity_iteration_count() < PARITY_ITERATION_LIMIT
    assert not parity_loop_is_open(), (
        "budget remaining is not permission: termination is what governs, and it has happened"
    )


def _guided_schema(passages, claim_count=2, max_citations=MAX_CITATIONS):
    fmt = build_citation_response_format(passages, claim_count, max_citations)
    assert fmt is not None
    assert fmt["type"] == "json_schema"
    return fmt["json_schema"]["schema"]


def test_guided_citation_quotes_are_verbatim_by_construction():
    """The whole point of the constrained decode: every quote the decoder can reach is a span the
    passage really contains, so `locate_quote` cannot fail on a schema-legal reply. This is what
    took `quote_located_rate` from 0.74 to 1.00 on the A4000 — asking for verbatim copies had
    already been tried three prompt iterations deep."""
    passages = _passages(3)
    text_by_id = {p.passage_id: p.text for p in passages}
    branches = _guided_schema(passages)["properties"]["claims"]["items"]["properties"]["citations"][
        "items"
    ]["anyOf"]

    assert branches, "no citation branch was compiled from three non-empty passages"
    for branch in branches:
        pid = branch["properties"]["passage_id"]["const"]
        for quote in branch["properties"]["quote"]["enum"]:
            assert quote in text_by_id[pid], f"{quote!r} is not a span of {pid}"
            assert locate_quote(quote, pid, text_by_id[pid]) is not None


def test_guided_citation_pairs_each_quote_with_its_own_passage():
    """`const` id beside the `enum` of that passage's spans, one branch per passage. A flat schema
    with a shared quote enum would let the model file passage 2's sentence under passage 1's id —
    exactly the mis-attribution the line grammar had to be repaired for twice."""
    passages = _passages(2)
    branches = _guided_schema(passages)["properties"]["claims"]["items"]["properties"]["citations"][
        "items"
    ]["anyOf"]

    by_id = {b["properties"]["passage_id"]["const"]: set(b["properties"]["quote"]["enum"]) for b in branches}
    assert set(by_id) == {"p1", "p2"}
    assert not any(q in PASSAGE_TEXT for q in by_id["p2"])
    assert not any(q in OTHER_TEXT for q in by_id["p1"])


def test_guided_citation_schema_takes_the_cap_from_the_caller():
    """A schema with its own idea of the cap would enforce a different fairness contract than
    `QueryRecord.validate()` reports on."""
    schema = _guided_schema(_passages(2), claim_count=4, max_citations=2)
    claims = schema["properties"]["claims"]
    assert (claims["minItems"], claims["maxItems"]) == (4, 4)
    assert claims["items"]["properties"]["citations"]["maxItems"] == 2
    index = claims["items"]["properties"]["claim_index"]
    assert (index["minimum"], index["maximum"]) == (1, 4)


def test_guided_citation_schema_is_none_when_no_passage_yields_a_span():
    """`None` sends `cite_claims` back to the prose stage. The earlier draft padded the enum with
    `"N/A"` instead, which put a string the passage does not contain inside the one structure whose
    purpose is that this cannot happen — and made the model's only legal choice a quote-not-found
    error."""
    empty = [
        RetrievedPassage(passage_id="p1", rank=1, score=1.0, retriever="rerank", text=""),
        RetrievedPassage(passage_id="p2", rank=2, score=0.5, retriever="rerank", text="short"),
    ]
    assert build_citation_response_format(empty, 2, MAX_CITATIONS) is None


def test_fixed_count_citation_schema_output_unchanged_for_post_hoc():
    """Fixed-count schema output remains unchanged byte-for-byte when claim_count is an int."""
    fmt = build_citation_response_format(_passages(2), claim_count=2, max_citations=3)
    assert fmt is not None
    assert fmt["type"] == "json_schema"
    js = fmt["json_schema"]
    assert js["name"] == "recitation_response"
    schema = js["schema"]
    assert schema["required"] == ["claims"]
    claims = schema["properties"]["claims"]
    assert claims["minItems"] == 2
    assert claims["maxItems"] == 2
    assert claims["items"]["required"] == ["claim_index", "citations"]


def test_variable_length_joint_schema_accepts_plausible_reply():
    """Variable-length schema for joint arm covers decision + claims + citations and accepts plausible joint JSON."""
    passages = _passages(2)
    fmt = build_citation_response_format(passages, claim_count=None, max_citations=3, is_joint=True)
    assert fmt is not None
    assert fmt["type"] == "json_schema"
    js = fmt["json_schema"]
    assert js["name"] == "joint_response"
    schema = js["schema"]
    assert schema["required"] == ["decision", "claims"]
    claims = schema["properties"]["claims"]
    assert claims["minItems"] == 1
    assert claims["maxItems"] == 30
    assert claims["items"]["required"] == ["claim_index", "text", "citations"]

    reply = json.dumps({
        "decision": "yes",
        "claims": [
            {
                "claim_index": 1,
                "text": "Metformin reduces mortality in type 2 diabetes.",
                "citations": [{"passage_id": "p1", "quote": "No severe hypoglycaemia occurred."}],
            },
            {
                "claim_index": 2,
                "text": "Metformin improves glycaemic control.",
                "citations": [],
            },
        ],
    })
    parsed = parse_response(reply, passages, max_citations=3, require_decision=True)
    assert parsed.decision == "yes"
    assert len(parsed.claims) == 2
    assert parsed.claims[0].text == "Metformin reduces mortality in type 2 diabetes."
    assert len(parsed.claims[0].citations) == 1
    assert parsed.claims[1].text == "Metformin improves glycaemic control."
    assert parsed.errors == []

def test_json_reply_is_parsed_with_the_same_contract_as_the_line_grammar():
    passages = _passages(2)
    raw = json.dumps(
        {
            "claims": [
                {"claim_index": 1, "citations": [{"passage_id": "p1", "quote": "No severe hypoglycaemia occurred."}]},
                {"claim_index": 2, "citations": []},
            ]
        }
    )
    parsed = parse_response(raw, passages, MAX_CITATIONS, require_decision=False)

    assert parsed.errors == []
    assert [c.claim_id for c in parsed.claims] == ["c1", "c2"]
    assert [len(c.citations) for c in parsed.claims] == [1, 0]
    cit = parsed.claims[0].citations[0]
    assert PASSAGE_TEXT[cit.char_start : cit.char_end] == "No severe hypoglycaemia occurred."
    # The claim text is deliberately empty: `cite_claims` re-attaches the frozen claim it sent.
    assert all(c.text == "" for c in parsed.claims)


def test_json_reply_never_redirects_an_out_of_range_claim_index_to_c1():
    """Sending a stray citation to the first claim mis-attributes evidence and manufactures cap
    violations — the failure `PROMPT_ITERATIONS[JOINT]` n=3 was spent on. It is reported, not
    silently absorbed."""
    passages = _passages(2)
    raw = json.dumps(
        {
            "claims": [
                {"claim_index": 9, "citations": [{"passage_id": "p1", "quote": PASSAGE_TEXT}]},
                {"claim_index": 2, "citations": []},
            ]
        }
    )
    parsed = parse_response(raw, passages, MAX_CITATIONS, require_decision=False)

    assert parsed.claims[0].citations == []
    assert any("out of range" in e for e in parsed.errors)


def test_malformed_json_is_reported_as_json_not_as_a_missing_claim_line():
    """A truncated structured reply filed under `no CLAIM lines` would land in the wrong bucket of
    the error histogram `decompose_smoke.py` gates on."""
    parsed = parse_response('{"claims": [{"claim_index": 1,}', _passages(2), MAX_CITATIONS, require_decision=False)

    assert parsed.claims == []
    assert len(parsed.errors) == 1
    assert "malformed JSON" in parsed.errors[0]
    assert not any("CLAIM" in e for e in parsed.errors)

def test_json_reply_invalid_decision_string_returns_parse_failure():
    raw = json.dumps({"decision": "unknown", "claims": []})
    parsed = parse_response(raw, _passages(2), MAX_CITATIONS)

    assert parsed.decision is None
    assert parsed.claims == []
    assert any("decision 'unknown' is not one of" in e for e in parsed.errors)


def test_json_reply_missing_decision_returns_parse_failure():
    raw = json.dumps({"claims": []})
    parsed = parse_response(raw, _passages(2), MAX_CITATIONS, require_decision=True)

    assert parsed.decision is None
    assert parsed.claims == []
    assert any("no DECISION line" in e for e in parsed.errors)


def test_json_reply_non_string_decision_returns_parse_failure():
    raw = json.dumps({"decision": 123, "claims": []})
    parsed = parse_response(raw, _passages(2), MAX_CITATIONS)

    assert parsed.decision is None
    assert parsed.claims == []
    assert any("decision 123 is not one of" in e for e in parsed.errors)


def test_json_reply_truncated_repair_records_recovery_note():
    """Truncated guided-JSON reply repaired by appending a closing suffix records the repair in
    `recovered` without producing parse errors."""
    passages = _passages(2)
    raw_valid = json.dumps(
        {
            "decision": "yes",
            "claims": [
                {
                    "claim_index": 1,
                    "text": "Metformin reduces mortality in type 2 diabetes.",
                    "citations": [{"passage_id": "p1", "quote": "No severe hypoglycaemia occurred."}],
                }
            ],
        }
    )
    truncated = raw_valid[:-2]
    parsed = parse_response(truncated, passages, MAX_CITATIONS, require_decision=True)
    assert parsed.errors == []
    assert len(parsed.recovered) == 1
    assert "truncated" in parsed.recovered[0]
    assert "recovered by appending" in parsed.recovered[0]
    assert parsed.decision == "yes"
    assert len(parsed.claims) == 1


def test_json_reply_strictly_valid_has_empty_recovered():
    """Strictly valid guided-JSON reply returns an empty `recovered` list."""
    passages = _passages(2)
    raw = json.dumps(
        {
            "decision": "yes",
            "claims": [
                {
                    "claim_index": 1,
                    "text": "Metformin reduces mortality in type 2 diabetes.",
                    "citations": [{"passage_id": "p1", "quote": "No severe hypoglycaemia occurred."}],
                }
            ],
        }
    )
    parsed = parse_response(raw, passages, MAX_CITATIONS, require_decision=True)

    assert parsed.errors == []
    assert parsed.recovered == []
def test_claim_stem_normalises_text_and_strips_trailing_punctuation():
    raw1 = "  distress associated with attenuated psychotic symptoms.  \n"
    stem1 = claim_stem(raw1)
    assert stem1 == "distress associated with attenuated psychotic symptoms"

    raw2 = "distress associated with attenuated psychotic symptoms, but the association is not significant"
    stem2 = claim_stem(raw2)
    assert stem2.startswith(stem1)


def test_runaway_chains_boundary_check_rejects_word_continuation_and_accepts_punctuation():
    texts_word_cont = ["the cat", "the cats sleep"]
    assert runaway_chains(texts_word_cont) == []

    texts_comma = ["the cat", "the cat, sleeps"]
    assert runaway_chains(texts_comma) == [(0, 2)]


def test_runaway_3_chain_in_parse_response_emits_error_naming_ids_and_count():
    passages = _passages(1)
    raw = (
        "DECISION: yes\n"
        "CLAIM 1: distress associated with attenuated psychotic symptoms.\n"
        "CITE 1: p1 || Metformin reduced HbA1c\n"
        "CLAIM 2: distress associated with attenuated psychotic symptoms, but the association is not significant.\n"
        "CITE 2: p1 || Metformin reduced HbA1c\n"
        "CLAIM 3: distress associated with attenuated psychotic symptoms, but the association is not significant, and distress is not a criterion.\n"
        "CITE 3: p1 || Metformin reduced HbA1c\n"
    )
    parsed = parse_response(raw, passages, MAX_CITATIONS)
    assert parsed.recovered == []
    assert len(parsed.errors) == 1
    err = parsed.errors[0]
    assert "c3: extends c1's claim text through 3 nested claims (non-terminating generation)" in err
    assert "exceeds the max claim length" not in err


def test_runaway_2_chain_in_parse_response_emits_recovered_and_no_errors():
    passages = _passages(1)
    raw = (
        "DECISION: yes\n"
        "CLAIM 1: distress associated with attenuated psychotic symptoms.\n"
        "CITE 1: p1 || Metformin reduced HbA1c\n"
        "CLAIM 2: distress associated with attenuated psychotic symptoms, but the association is not significant.\n"
        "CITE 2: p1 || Metformin reduced HbA1c\n"
    )
    parsed = parse_response(raw, passages, MAX_CITATIONS)
    assert parsed.errors == []
    assert len(parsed.recovered) == 1
    rec = parsed.recovered[0]
    assert "c2: extends c1's claim text ('distress associated with attenuated psychotic symptoms, but ')" in rec


def test_unrelated_claims_produce_neither_error_nor_recovered():
    passages = _passages(1)
    raw = (
        "DECISION: yes\n"
        "CLAIM 1: Metformin reduced HbA1c by 1.2% over 24 weeks.\n"
        "CITE 1: p1 || Metformin reduced HbA1c\n"
        "CLAIM 2: No severe hypoglycaemia occurred.\n"
        "CITE 2: p1 || No severe hypoglycaemia occurred.\n"
    )
    parsed = parse_response(raw, passages, MAX_CITATIONS)
    assert parsed.errors == []
    assert parsed.recovered == []


def test_runaway_chains_maximal_non_overlapping_and_adjacency_restricted():
    texts_maximal = [
        "claim alpha",
        "claim alpha, extended once",
        "claim alpha, extended once, extended twice",
        "claim beta",
        "claim beta, extended",
    ]
    chains = runaway_chains(texts_maximal, min_length=2)
    assert chains == [(0, 3), (3, 2)]

    texts_non_adj = [
        "claim alpha",
        "unrelated claim beta",
        "claim alpha, extended once",
    ]
    assert runaway_chains(texts_non_adj) == []