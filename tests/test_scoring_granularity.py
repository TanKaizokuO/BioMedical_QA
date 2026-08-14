"""The ADR-0009 parity gate: median words/claim, the two bases, and the per-stage truncation trap.

The gate had been computed ad hoc three times before this module existed, and the last of those left
three unreproducible figures in `PARITY_ITERATIONS[0]`'s rationale. So the
load-bearing test here is not a unit test at all — it is
`test_the_published_iteration_0_table_is_reproduced`, which recomputes every gate figure in
`docs/harvest/parity_iter0.md` from the committed artifacts. That file is the baseline every future
iteration is compared against; if this module disagrees with it, one of the two is wrong and the
loop's verdict is unauditable.

The unit tests around it each name a way the gate can be wrong while still returning a number.
"""

from __future__ import annotations

import inspect
import statistics
from pathlib import Path

import pytest

from biomedqa.schema import Claim, CostRecord, QueryRecord, System, read_jsonl, read_query_records
from biomedqa.scoring.granularity import (
    CALL_ORDER,
    PARITY_TOLERANCE,
    arm_granularity,
    compound_profile,
    markers_in,
    parity_gate,
    stage_output_tokens,
    truncated_queries,
    words_in_claim,
)

_HARVEST = Path(__file__).resolve().parents[1] / "docs" / "harvest"


def _record(query_id: str, system: System, claims: list[str], *, completion_tokens=None):
    return QueryRecord(
        run_id="t",
        query_id=query_id,
        question="q?",
        system=system,
        seed=0,
        claims=[Claim(claim_id=f"c{i}", text=t) for i, t in enumerate(claims, 1)],
        completion_tokens=completion_tokens,
    )


def _arms(query_id: str, joint: list[str], post_hoc: list[str]):
    return [_record(query_id, System.JOINT, joint), _record(query_id, System.POST_HOC, post_hoc)]


def _words(n: int) -> str:
    return " ".join(["w"] * n)


def _costs(query_id: str, joint: int, ph_answer: int, ph_cite: int, vanilla: int):
    """Four rows per query, in `CALL_ORDER` — the shape `costs.jsonl` actually has."""
    return [
        CostRecord(run_id="t", query_id=query_id, component="generate", backend="vllm:m",
                   output_tokens=n)
        for n in (joint, ph_answer, ph_cite, vanilla)
    ]


# ---------------------------------------------------------------------------------------------
# The pre-commitment. ADR-0009 §3: ±15%, fixed before any measurement, never revised.
# ---------------------------------------------------------------------------------------------


def test_the_tolerance_is_fifteen_percent() -> None:
    assert PARITY_TOLERANCE == 0.15


def test_the_tolerance_cannot_be_passed_in() -> None:
    """A tolerance that is an argument is a tolerance that gets widened on the iteration that would
    otherwise fail — which is exactly the steering §3's pre-commitment exists to prevent. It must
    not be reachable through the gate's own signature."""
    assert "tolerance" not in inspect.signature(parity_gate).parameters
    assert "tolerance" not in inspect.signature(arm_granularity).parameters


# ---------------------------------------------------------------------------------------------
# The verdict, and the §5 asymmetry that hangs off its sign.
# ---------------------------------------------------------------------------------------------


def test_a_gap_inside_the_tolerance_passes() -> None:
    gate = parity_gate(_arms("q1", [_words(20)] * 3, [_words(22)] * 3))
    assert gate.gap == pytest.approx(0.10)
    assert gate.passes


def test_coarser_post_hoc_fails_and_makes_the_w9_check_mandatory() -> None:
    """ADR-0009 §5's one-sided fallback. Post-hoc coarser means post-hoc is penalised per claim, so
    C2's gap can appear without joint grounding doing any work — the direction that flatters the
    hypothesis, and the one that must not be reported as a bare number."""
    gate = parity_gate(_arms("q1", [_words(16)] * 3, [_words(20)] * 3))
    assert gate.gap == pytest.approx(0.25)
    assert not gate.passes
    assert gate.favours_c2
    assert gate.requires_w9_robustness_check


def test_a_gap_running_against_c2_does_not_trigger_the_w9_check() -> None:
    """The asymmetry is the point: a failure in the other direction is noted and proceeded past. If
    this ever went symmetric, the pre-registered rule in the paper's methods section would be a
    description of something the code does not do."""
    gate = parity_gate(_arms("q1", [_words(20)] * 3, [_words(14)] * 3))
    assert not gate.passes
    assert not gate.favours_c2
    assert not gate.requires_w9_robustness_check


def test_the_gap_is_measured_against_joint() -> None:
    """The denominator is joint, not the mean of the arms — 16 vs 20 is +25%, not +22%. A different
    denominator silently moves every historical verdict relative to a fixed ±15%."""
    gate = parity_gate(_arms("q1", [_words(16)], [_words(20)]))
    assert gate.gap == pytest.approx((20 - 16) / 16)


def test_a_record_with_no_claims_still_counts_as_a_query() -> None:
    """A query the arm produced nothing parseable for is an outcome, not a missing data point.
    Dropping it would quietly flatter whichever arm fails to parse most often — post-hoc, whose cite
    stage is the one that truncates."""
    records = [
        _record("q1", System.JOINT, [_words(10)]),
        _record("q2", System.JOINT, []),
        _record("q1", System.POST_HOC, [_words(10)]),
    ]
    joint = arm_granularity(records, System.JOINT)
    assert joint.n_records == 2
    assert joint.n_claims == 1
    assert joint.median_claims_per_query == 0.5


def test_an_arm_with_no_claims_at_all_is_refused() -> None:
    """words/claim over zero claims is not 0, it is undefined — and a run where an arm parsed
    nothing is a broken run, not a passing gate."""
    records = [_record("q1", System.JOINT, []), _record("q1", System.POST_HOC, [_words(9)])]
    with pytest.raises(ValueError, match="no claims"):
        parity_gate(records)


def test_the_untruncated_basis_drops_only_the_named_queries() -> None:
    records = _arms("q1", [_words(10)], [_words(30)]) + _arms("q2", [_words(10)], [_words(10)])
    both = parity_gate(records, basis="all")
    kept = parity_gate(records, basis="untruncated", exclude={"q1"})
    assert both.post_hoc.n_records == 2
    assert kept.post_hoc.n_records == 1
    assert kept.post_hoc.median_words_per_claim == 10


# ---------------------------------------------------------------------------------------------
# The trap: `costs.jsonl` has no stage field, and post-hoc's record total is the sum of two calls.
# ---------------------------------------------------------------------------------------------


def test_stage_tokens_are_checked_against_the_records_own_totals() -> None:
    records = [
        _record("q1", System.JOINT, [_words(5)], completion_tokens=100),
        _record("q1", System.POST_HOC, [_words(5)], completion_tokens=300),
        _record("q1", System.VANILLA, [_words(5)], completion_tokens=400),
    ]
    stages = stage_output_tokens(records, _costs("q1", 100, 120, 180, 400))
    assert stages["q1"] == {"joint": 100, "post_hoc_answer": 120, "post_hoc_cite": 180,
                            "vanilla": 400}


def test_cost_rows_in_the_wrong_order_are_refused() -> None:
    """The whole reason this function exists. `costs.jsonl` carries no system and no stage — only
    position — so a change in call order would silently book post-hoc's cite stage to vanilla and
    every truncation count downstream would be wrong with no error anywhere."""
    records = [
        _record("q1", System.JOINT, [_words(5)], completion_tokens=100),
        _record("q1", System.POST_HOC, [_words(5)], completion_tokens=300),
        _record("q1", System.VANILLA, [_words(5)], completion_tokens=400),
    ]
    shuffled = _costs("q1", 400, 120, 180, 100)  # vanilla first, joint last
    with pytest.raises(ValueError, match="order"):
        stage_output_tokens(records, shuffled)


def test_a_query_with_the_wrong_number_of_calls_is_refused() -> None:
    """Three systems, four calls. If generation ever emits a different number, position stops
    meaning what `CALL_ORDER` says it means."""
    records = [_record("q1", System.JOINT, [_words(5)], completion_tokens=100)]
    costs = _costs("q1", 100, 120, 180, 400)[:3]
    with pytest.raises(ValueError, match="expected 4"):
        stage_output_tokens(records, costs)


def test_post_hoc_truncation_is_per_call_not_per_record() -> None:
    """`completion_tokens` on a post-hoc record is the **sum of both stages**, so comparing it to
    `max_tokens` invents truncation that never happened. Here the sum is 300 against a cap of 256
    while neither call came close — the record is untruncated, and treating it otherwise would
    delete healthy post-hoc records from the untruncated basis and move the gate."""
    records = [
        _record("q1", System.JOINT, [_words(5)], completion_tokens=100),
        _record("q1", System.POST_HOC, [_words(5)], completion_tokens=300),
        _record("q1", System.VANILLA, [_words(5)], completion_tokens=100),
    ]
    trunc = truncated_queries(records, _costs("q1", 100, 150, 150, 100), max_tokens=256)
    assert trunc[System.POST_HOC.value] == set()


def test_a_truncated_answer_stage_disqualifies_the_post_hoc_record() -> None:
    """Claims are parsed from the *cite* stage, but the cite prompt embeds the stage-1 answer — so a
    truncated answer stage produces a shortened record too. Checking only the cite stage would let
    those through."""
    records = [
        _record("q1", System.JOINT, [_words(5)], completion_tokens=10),
        _record("q1", System.POST_HOC, [_words(5)], completion_tokens=356),
        _record("q1", System.VANILLA, [_words(5)], completion_tokens=10),
    ]
    trunc = truncated_queries(records, _costs("q1", 10, 256, 100, 10), max_tokens=256)
    assert trunc[System.POST_HOC.value] == {"q1"}
    assert trunc[System.JOINT.value] == set()


# ---------------------------------------------------------------------------------------------
# Compound markers — the pre-freeze proxy for "long atomic claim" vs "compound claim".
# ---------------------------------------------------------------------------------------------


def test_a_claim_with_no_marker_is_simple() -> None:
    assert markers_in("Metformin reduced HbA1c by 1.2% over 24 weeks.") == frozenset()


def test_each_marker_is_detected_independently() -> None:
    assert markers_in("A rose and B fell.") == {"and"}
    assert markers_in("A rose, which surprised nobody.") == {"subordinate"}
    assert markers_in("A rose, in trial one, by 2%.") == {"multi_comma"}


def test_a_single_comma_is_not_a_multi_comma_claim() -> None:
    """The boundary the rate is sensitive to: on `parity_iter0b` this marker is where post-hoc's
    excess is largest (13.6% vs 5.6%), so an off-by-one comma changes the iteration's diagnosis."""
    assert "multi_comma" not in markers_in("In trial one, A rose by 2%.")


def test_the_simple_claim_median_is_what_separates_verbosity_from_compounding() -> None:
    """Iteration 1's premise. If simple claims are equally long in both arms the gap is compounding
    and the lever is a splitting rule; if they are not, it is verbosity and the lever is a length
    target. Getting this backwards spends a bounded iteration on the wrong edit."""
    profile = compound_profile([_words(10), _words(20) + " and more", _words(30)])
    assert profile["n_claims"] == 3
    assert profile["n_simple_claims"] == 2
    assert profile["simple_claim_share"] == pytest.approx(2 / 3)
    assert profile["median_words_per_simple_claim"] == 20  # median(10, 30), not median of all three
    assert profile["marker_rate"]["and"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------------------------
# The baseline of record. Every figure below is quoted from `docs/harvest/parity_iter0.md`.
# ---------------------------------------------------------------------------------------------


def _load(prefix: str):
    records_path = _HARVEST / f"{prefix}.records.jsonl"
    costs_path = _HARVEST / f"{prefix}.costs.jsonl"
    if not records_path.exists() or not costs_path.exists():
        pytest.skip(f"{prefix} artifacts not present")
    records = list(read_query_records(records_path))
    costs = [CostRecord(**d) for d in read_jsonl(costs_path)]
    return records, costs


@pytest.mark.parametrize(
    "prefix,max_tokens,joint_all,post_hoc_all,joint_untrunc,post_hoc_untrunc,n_untrunc",
    [
        # `parity_iter0.md`, "The gate: FAIL, in every reading".
        ("parity_iter0", 1536, 15, 20, 14, 19, (89, 62)),
        ("parity_iter0b", 2560, 16, 20, 14, 20, (91, 84)),
    ],
)
def test_the_published_iteration_0_table_is_reproduced(
    prefix, max_tokens, joint_all, post_hoc_all, joint_untrunc, post_hoc_untrunc, n_untrunc
) -> None:
    """The audit. `docs/harvest/parity_iter0.md` is the baseline iteration 1 is judged against, and
    it was written from a computation that no longer exists. If this module cannot reproduce it, the
    gate's history is not comparable to its future."""
    records, costs = _load(prefix)

    every = parity_gate(records, basis="all")
    assert every.joint.median_words_per_claim == joint_all
    assert every.post_hoc.median_words_per_claim == post_hoc_all
    assert not every.passes and every.favours_c2

    trunc = truncated_queries(records, costs, max_tokens)
    kept = parity_gate(records, basis="untruncated",
                       exclude=trunc[System.JOINT.value] | trunc[System.POST_HOC.value])
    joint_only = arm_granularity(records, System.JOINT, exclude=trunc[System.JOINT.value])
    post_hoc_only = arm_granularity(records, System.POST_HOC,
                                    exclude=trunc[System.POST_HOC.value])
    assert (joint_only.n_records, post_hoc_only.n_records) == n_untrunc
    assert joint_only.median_words_per_claim == joint_untrunc
    assert post_hoc_only.median_words_per_claim == post_hoc_untrunc
    assert kept.basis == "untruncated"


def test_the_baseline_of_record_reads_exactly_twenty_five_percent() -> None:
    """`parity_iter0b`, all 100 records: joint 16 vs post-hoc 20, +25.0% against ±15%. This is the
    single number iteration 1 has to move, and the supporting quantities that show a median alone is
    not the whole picture."""
    records, _ = _load("parity_iter0b")
    gate = parity_gate(records, basis="all")

    assert gate.gap == pytest.approx(0.25)
    assert gate.requires_w9_robustness_check

    assert (gate.joint.n_claims, gate.post_hoc.n_claims) == (645, 895)
    assert gate.joint.mean_words_per_claim == pytest.approx(17.15, abs=0.005)
    assert gate.post_hoc.mean_words_per_claim == pytest.approx(21.35, abs=0.005)
    assert (gate.joint.p25_words_per_claim, gate.joint.p75_words_per_claim,
            gate.joint.p90_words_per_claim) == (12, 18, 22)
    assert (gate.post_hoc.p25_words_per_claim, gate.post_hoc.p75_words_per_claim,
            gate.post_hoc.p90_words_per_claim) == (16, 25, 31)
    assert gate.joint.median_claims_per_query == 5.0
    assert gate.post_hoc.median_claims_per_query == 8.0


def test_the_two_bases_still_disagree_by_eighteen_points() -> None:
    """Why the gate is always reported on both bases. Relieving the output cap moved the all-records
    gap 33% -> 25% but *widened* the untruncated-only gap to 42.9%, because dropping joint's own
    truncated records pulls joint's median down. A single number would have hidden that."""
    records, costs = _load("parity_iter0b")
    trunc = truncated_queries(records, costs, 2560)
    joint = arm_granularity(records, System.JOINT, exclude=trunc[System.JOINT.value])
    post_hoc = arm_granularity(records, System.POST_HOC, exclude=trunc[System.POST_HOC.value])
    gap = (post_hoc.median_words_per_claim - joint.median_words_per_claim) / joint.median_words_per_claim
    assert gap == pytest.approx(0.4286, abs=0.0001)


def test_the_published_at_cap_counts_are_reproduced() -> None:
    """`parity_iter0.md`'s per-stage table, which is what identified the cite stage as the binding
    constraint (38/100 at 1536, 16/100 at 2560) rather than the answer stage or joint."""
    records, costs = _load("parity_iter0b")
    stages = stage_output_tokens(records, costs)
    at_cap = {name: sum(1 for s in stages.values() if s[name] >= 2560) for name in CALL_ORDER}
    assert at_cap == {"joint": 9, "post_hoc_answer": 6, "post_hoc_cite": 16, "vanilla": 7}


def test_vanilla_sits_with_post_hoc_not_with_joint() -> None:
    """ADR-0010 excludes vanilla from the gate, but it is the untuned reference point, and it lands
    at 21 against post-hoc's 20 and joint's 16. **Joint's fine claims are the outlier** — which is
    what ADR-0009 §4's "joint's granularity is native" predicts, and it is the reason the loop tunes
    post-hoc rather than splitting the difference."""
    records, _ = _load("parity_iter0b")
    vanilla = arm_granularity(records, System.VANILLA)
    joint = arm_granularity(records, System.JOINT)
    post_hoc = arm_granularity(records, System.POST_HOC)
    assert vanilla.median_words_per_claim == 21
    assert abs(vanilla.median_words_per_claim - post_hoc.median_words_per_claim) < abs(
        vanilla.median_words_per_claim - joint.median_words_per_claim
    )


def test_the_compound_diagnostic_still_says_verbosity_not_compounding() -> None:
    """Iteration 1's premise, recomputed on markers that are now written down. Three of the four
    figures in `PARITY_ITERATIONS[0]`'s rationale do not reproduce (see the module docstring) — but
    the finding they were cited for does, and more strongly: and-coordination is level across the
    arms, while post-hoc's claims are longer even with no marker present at all.

    The silent failure this prevents is the opposite diagnosis. If the gap lived in *marked* claims
    it would be compounding, and iteration 1's length target would have spent a bounded iteration on
    the wrong edit."""
    records, _ = _load("parity_iter0b")
    profiles = {
        system: compound_profile([c for r in records if r.system is system for c in r.claims])
        for system in (System.JOINT, System.POST_HOC)
    }
    joint, post_hoc = profiles[System.JOINT], profiles[System.POST_HOC]

    # Verbosity: the gap survives with every compound marker excluded, at +28.6%.
    assert joint["median_words_per_simple_claim"] == 14
    assert post_hoc["median_words_per_simple_claim"] == 18

    # Not compounding by coordination — `_claim_rules()` splits "and" for all three systems.
    assert abs(post_hoc["marker_rate"]["and"] - joint["marker_rate"]["and"]) < 0.02

    # The one rationale figure that reproduces exactly, and where post-hoc's excess is largest.
    assert post_hoc["marker_rate"]["multi_comma"] == pytest.approx(0.136, abs=0.0005)
    assert joint["marker_rate"]["multi_comma"] == pytest.approx(0.056, abs=0.0005)



# ---------------------------------------------------------------------------------------------
# Iteration 1 — the first run where the two bases disagree. `docs/harvest/parity_iter1.md`.
# ---------------------------------------------------------------------------------------------


def test_the_joint_arm_did_not_move_between_iterations() -> None:
    """ADR-0009 §4 puts the joint prompt out of bounds for the loop's duration, and greedy decoding
    makes that checkable rather than merely asserted: joint's 100 generations must be byte-identical
    across iterations.

    The silent failure is a shared edit — to `_claim_rules()`, the context block, the parser — that
    moves both arms and makes the gap look closed by the loop when the loop did not close it. That
    would be an invisible violation of the one restriction the blind depends on.
    """
    before, _ = _load("parity_iter0b")
    after, _ = _load("parity_iter1")
    joint_before = {r.query_id: r.raw_generation for r in before if r.system is System.JOINT}
    joint_after = {r.query_id: r.raw_generation for r in after if r.system is System.JOINT}

    assert joint_before == joint_after


def test_iteration_1_passes_on_the_basis_of_record() -> None:
    """The verdict: joint 16 vs post-hoc 16, +0.0% against ±15%, on all 100 records."""
    records, _ = _load("parity_iter1")
    gate = parity_gate(records, basis="all")

    assert gate.joint.median_words_per_claim == 16
    assert gate.post_hoc.median_words_per_claim == 16
    assert gate.gap == pytest.approx(0.0)
    assert gate.passes
    assert not gate.requires_w9_robustness_check


def test_the_gated_quantity_is_insensitive_to_cite_stage_truncation() -> None:
    """The argument the iteration-1 verdict rests on, and the reason all-records is the sound basis
    once the two disagree.

    Truncation drops *trailing claims*; it does not shorten the claims that survive. So dropping the
    final — possibly mid-sentence — claim of every truncated post-hoc record must leave the median
    where it was. If this ever stops holding, the censoring is biasing the gate itself and the
    untruncated basis becomes the honest one again.
    """
    records, costs = _load("parity_iter1")
    truncated = truncated_queries(records, costs, 2560)[System.POST_HOC.value]
    assert len(truncated) == 26, "the run this argument was made about"

    post_hoc = [r for r in records if r.system is System.POST_HOC]
    as_recorded = [c for r in post_hoc for c in r.claims]
    tail_dropped = [
        c
        for r in post_hoc
        for c in (r.claims[:-1] if r.query_id in truncated and r.claims else r.claims)
    ]

    assert len(tail_dropped) < len(as_recorded)
    assert (
        statistics.median(words_in_claim(c.text) for c in tail_dropped)
        == statistics.median(words_in_claim(c.text) for c in as_recorded)
        == 16
    )


def test_iteration_1_answered_finer_rather_than_less() -> None:
    """The one reading that would make the pass worthless: words/claim falling because the model
    answers *less*, not *finer*. ADR-0009 §2 reports claims/query without gating it precisely so
    this is visible — a gate on words/claim alone cannot tell the two apart.
    """
    baseline, _ = _load("parity_iter0b")
    records, _ = _load("parity_iter1")
    before = arm_granularity(baseline, System.POST_HOC)
    after = arm_granularity(records, System.POST_HOC)

    assert after.median_words_per_claim < before.median_words_per_claim   # 20 -> 16
    assert after.median_claims_per_query > before.median_claims_per_query  # 8 -> 10
    assert after.n_claims > before.n_claims                               # 895 -> 1129


def test_the_untruncated_basis_still_fails_and_is_reported() -> None:
    """The disagreement is a finding, not a rounding error: +0.0% against +21.4% on the same run.
    It is asserted here so that a future change which quietly collapses the two bases into one
    number has to delete a test that says why both are printed."""
    records, costs = _load("parity_iter1")
    truncated = truncated_queries(records, costs, 2560)
    joint = arm_granularity(records, System.JOINT, exclude=truncated[System.JOINT.value])
    post_hoc = arm_granularity(records, System.POST_HOC,
                               exclude=truncated[System.POST_HOC.value])
    gap = ((post_hoc.median_words_per_claim - joint.median_words_per_claim)
           / joint.median_words_per_claim)

    assert (joint.n_records, post_hoc.n_records) == (91, 74)
    assert gap == pytest.approx(0.2143, abs=0.0001)
    assert gap > PARITY_TOLERANCE