"""Tests for verifier calibration metrics — `biomedqa.scoring.calibration`.

Covers AUROC, ECE (with bin counts), and threshold_sweep.
All expected values are hand-computed and documented per assertion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomedqa.schema import Citation, Claim, HumanLabel, QueryRecord, SupportLabel, System, VerifierScore
from biomedqa.scoring.calibration import (
    G3_AUROC_MIN,
    G3_COST_RATIO_MIN,
    auprc,
    auroc,
    bootstrap_ci,
    ece,
    gate_g3,
    join_scores_and_labels,
    threshold_sweep,
)

# --- AUROC tests ----------------------------------------------------------------------------------

def test_auroc_textbook_perfect_separation():
    # pos = [0.7, 0.9], neg = [0.2, 0.4] -> all pos > all neg -> AUROC = 1.0
    scores = [0.2, 0.4, 0.7, 0.9]
    labels = [False, False, True, True]
    assert auroc(scores, labels) == pytest.approx(1.0)


def test_auroc_textbook_imperfect_separation():
    # pos = [0.4, 0.8], neg = [0.1, 0.5]
    # 0.4 vs [0.1, 0.5] -> 1 win
    # 0.8 vs [0.1, 0.5] -> 2 wins
    # Total wins = 3 out of 4 pairs -> AUROC = 3/4 = 0.75
    scores = [0.1, 0.4, 0.5, 0.8]
    labels = [False, True, False, True]
    assert auroc(scores, labels) == pytest.approx(0.75)


def test_auroc_with_tied_scores():
    # pos = [0.5, 0.8], neg = [0.2, 0.5]
    # 0.5 vs [0.2, 0.5] -> 1 win (0.5>0.2) + 0.5 win (0.5==0.5) = 1.5 wins
    # 0.8 vs [0.2, 0.5] -> 2 wins (0.8>0.2, 0.8>0.5) = 2.0 wins
    # Total wins = 3.5 out of 4 pairs -> AUROC = 3.5 / 4 = 0.875
    scores = [0.2, 0.5, 0.5, 0.8]
    labels = [False, False, True, True]
    assert auroc(scores, labels) == pytest.approx(0.875)


def test_auroc_pure_tie_case():
    # pos = [0.5], neg = [0.5] -> 0.5 win -> AUROC = 0.5
    scores = [0.5, 0.5]
    labels = [True, False]
    assert auroc(scores, labels) == pytest.approx(0.5)


def test_auroc_heavily_skewed_vector():
    # 5 positives (5%) and 95 negatives (95%).
    # pos scores = [0.70, 0.75, 0.80, 0.85, 0.90]
    # 90 neg scores in [0.10, 0.55] (all < 0.70) -> 5 * 90 = 450 wins
    # 5 neg scores = [0.72, 0.77, 0.82, 0.87, 0.92]
    #   pos 0.70 vs 5 negs -> 0 wins
    #   pos 0.75 vs 5 negs -> 1 win (0.75 > 0.72)
    #   pos 0.80 vs 5 negs -> 2 wins (0.80 > 0.72, 0.77)
    #   pos 0.85 vs 5 negs -> 3 wins (0.85 > 0.72, 0.77, 0.82)
    #   pos 0.90 vs 5 negs -> 4 wins (0.90 > 0.72, 0.77, 0.82, 0.87)
    # Total wins = 450 + 10 = 460
    # Total pairs = 5 * 95 = 475
    # AUROC = 460 / 475 = 92 / 95 = 0.968421052631579
    pos_scores = [0.70, 0.75, 0.80, 0.85, 0.90]
    neg_scores = list(np.linspace(0.10, 0.55, 90)) + [0.72, 0.77, 0.82, 0.87, 0.92]

    scores = pos_scores + neg_scores
    labels = [True] * len(pos_scores) + [False] * len(neg_scores)

    expected = 460.0 / 475.0
    assert auroc(scores, labels) == pytest.approx(expected)


def test_auroc_degenerate_cases_raise_value_error():
    # All positive
    with pytest.raises(ValueError, match="all labels are positive"):
        auroc([0.1, 0.5, 0.9], [True, True, True])

    # All negative
    with pytest.raises(ValueError, match="all labels are negative"):
        auroc([0.1, 0.5, 0.9], [False, False, False])

    # Empty input
    with pytest.raises(ValueError, match="empty input"):
        auroc([], [])

    # Mismatched length
    with pytest.raises(ValueError, match="equal length"):
        auroc([0.5], [True, False])


# --- ECE tests ------------------------------------------------------------------------------------

def test_ece_with_empty_bin():
    # 4 items across 5 bins:
    # Bins: [0.0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
    # Scores: 0.05 (Bin 0), 0.25 (Bin 1), 0.45 (Bin 2), 0.85 (Bin 4)
    # Bin 3 (0.6, 0.8] is empty!
    scores = [0.05, 0.25, 0.45, 0.85]
    labels = [False, True, True, True]
    res = ece(scores, labels, bins=5)

    assert res["bin_counts"] == [1, 1, 1, 0, 1]
    assert res["bin_accuracies"] == [0.0, 1.0, 1.0, None, 1.0]
    assert res["bin_confidences"] == [0.05, 0.25, 0.45, None, 0.85]

    # Hand-computed gaps:
    # Bin 0: |0.0 - 0.05| = 0.05
    # Bin 1: |1.0 - 0.25| = 0.75
    # Bin 2: |1.0 - 0.45| = 0.55
    # Bin 3: count 0 -> 0.0
    # Bin 4: |1.0 - 0.85| = 0.15
    # ECE = (1/4) * (0.05 + 0.75 + 0.55 + 0.15) = 1.50 / 4 = 0.375
    assert res["ece"] == pytest.approx(0.375)


def test_ece_validation_raises():
    with pytest.raises(ValueError, match="empty input"):
        ece([], [])

    with pytest.raises(ValueError, match="equal length"):
        ece([0.5], [True, False])

    with pytest.raises(ValueError, match="positive integer"):
        ece([0.5], [True], bins=0)


# --- threshold_sweep tests ------------------------------------------------------------------------

def test_threshold_sweep_one_row_per_distinct_threshold():
    # scores has 3 distinct values: 0.2, 0.5, 0.8 (0.5 is repeated)
    scores = [0.2, 0.5, 0.5, 0.8]
    labels = [False, False, True, True]
    sweep = threshold_sweep(scores, labels)

    # Exactly 3 rows
    assert len(sweep) == 3
    assert [r["threshold"] for r in sweep] == [0.2, 0.5, 0.8]

    # Row 0: t = 0.2
    # preds: all True -> tp=2, fp=2, fn=0, tn=0
    # prec = 2/4 = 0.5, rec = 2/2 = 1.0, f1 = 2*(0.5*1.0)/(1.5) = 2/3
    r0 = sweep[0]
    assert (r0["tp"], r0["fp"], r0["fn"], r0["tn"]) == (2, 2, 0, 0)
    assert r0["precision"] == pytest.approx(0.5)
    assert r0["recall"] == pytest.approx(1.0)
    assert r0["f1"] == pytest.approx(2.0 / 3.0)

    # Row 1: t = 0.5
    # preds: [False, True, True, True] -> tp=2, fp=1, fn=0, tn=1
    # prec = 2/3, rec = 2/2 = 1.0, f1 = 2*(2/3)/(5/3) = 0.8
    r1 = sweep[1]
    assert (r1["tp"], r1["fp"], r1["fn"], r1["tn"]) == (2, 1, 0, 1)
    assert r1["precision"] == pytest.approx(2.0 / 3.0)
    assert r1["recall"] == pytest.approx(1.0)
    assert r1["f1"] == pytest.approx(0.8)

    # Row 2: t = 0.8
    # preds: [False, False, False, True] -> tp=1, fp=0, fn=1, tn=2
    # prec = 1.0, rec = 0.5, f1 = 2*(0.5)/1.5 = 2/3
    r2 = sweep[2]
    assert (r2["tp"], r2["fp"], r2["fn"], r2["tn"]) == (1, 0, 1, 2)
    assert r2["precision"] == pytest.approx(1.0)
    assert r2["recall"] == pytest.approx(0.5)
    assert r2["f1"] == pytest.approx(2.0 / 3.0)


def test_threshold_sweep_validation_raises():
    with pytest.raises(ValueError, match="empty input"):
        threshold_sweep([], [])

    with pytest.raises(ValueError, match="equal length"):
        threshold_sweep([0.5], [True, False])
# --- bootstrap_ci tests ---------------------------------------------------------------------------

def test_bootstrap_ci_clustered_wider_than_unclustered_under_correlation():
    # Pins ADR-0011 §2 ("clustering widens every interval").
    # Construct 20 questions with 10 claims each (200 claims total).
    # In 10 questions, all claims are 1.0; in 10 questions, all claims are 0.0.
    # Within-question claims are 100% correlated.
    units: list[float] = []
    clusters: list[str] = []
    for q_idx in range(20):
        q_id = f"q_{q_idx:02d}"
        val = 1.0 if q_idx < 10 else 0.0
        for _ in range(10):
            units.append(val)
            clusters.append(q_id)

    unclustered = bootstrap_ci(units, clusters=None, n_boot=2000, seed=42)
    clustered = bootstrap_ci(units, clusters=clusters, n_boot=2000, seed=42)

    # Clustered CI must be strictly wider than unclustered CI on correlated data
    assert clustered["width"] > unclustered["width"]
    assert clustered["n_clusters"] == 20
    assert unclustered["n_clusters"] == 200


def test_bootstrap_ci_resamples_complete_clusters():
    # Pins ADR-0011 §2 ("every unit of each drawn cluster comes along").
    # 3 questions ("q1", "q2", "q3"), each having 4 member units (12 units total).
    units = [(f"q{q}", i) for q in range(1, 4) for i in range(4)]
    clusters = [u[0] for u in units]

    observed_counts: list[dict[str, int]] = []

    def tracking_stat(resample: list[tuple[str, int]]) -> float:
        counts: dict[str, int] = {}
        for q_id, _ in resample:
            counts[q_id] = counts.get(q_id, 0) + 1
        observed_counts.append(counts)
        return 1.0

    bootstrap_ci(units, statistic=tracking_stat, clusters=clusters, n_boot=100, seed=123)
    assert len(observed_counts) == 101  # 1 point estimate + 100 bootstrap draws
    for counts in observed_counts:
        assert sum(counts.values()) == 12
        for q_id, count in counts.items():
            # Every observed question must appear in exact multiples of its 4 member units
            assert count % 4 == 0, f"Question {q_id} appeared {count} times, not a multiple of 4"


def test_bootstrap_ci_validation_mismatched_clusters_and_empty_units_raise():
    # Pins contract validation: mismatched cluster key count and empty units raise specific ValueError.
    with pytest.raises(
        ValueError, match=r"clusters has 4 keys for 5 units; one key per unit or None"
    ):
        bootstrap_ci([1, 2, 3, 4, 5], clusters=["q1", "q1", "q2", "q2"])

    with pytest.raises(
        ValueError, match=r"bootstrap_ci needs at least one unit; an empty CI is not a wide one"
    ):
        bootstrap_ci([])


def test_bootstrap_ci_determinism_and_seed_pin():
    # Pins seed determinism: identical seed -> bit-identical (lo, hi); different seed -> different interval.
    units = [0.1, 0.2, 0.4, 0.8, 0.9, 1.0]

    res_a1 = bootstrap_ci(units, n_boot=500, seed=20260804)
    res_a2 = bootstrap_ci(units, n_boot=500, seed=20260804)
    res_b = bootstrap_ci(units, n_boot=500, seed=99999999)

    # Identical seed produces bit-identical results
    assert res_a1["lower"] == res_a2["lower"]
    assert res_a1["upper"] == res_a2["upper"]
    assert res_a1["point"] == res_a2["point"]
    assert res_a1["width"] == res_a2["width"]

    # Different seed produces different bounds
    assert res_a1["lower"] != res_b["lower"] or res_a1["upper"] != res_b["upper"]

    # Deterministic fixed-seed exact regression pin for seed 20260804, n_boot 1000
    res_pin = bootstrap_ci(units, n_boot=1000, seed=20260804)
    assert res_pin["point"] == pytest.approx(0.5666666666666667)
    assert res_pin["lower"] == pytest.approx(0.2833333333333333)
    assert res_pin["upper"] == pytest.approx(0.8333333333333334)
    assert res_pin["width"] == pytest.approx(0.55)


def test_bootstrap_ci_resampling_unit_field_handling():
    # Pins ADR-0011 Consequences requirement for resampling_unit reporting.
    units = [1.0, 2.0, 3.0]
    clusters = ["q1", "q1", "q2"]

    # Unclustered defaults to "observation"
    unclustered = bootstrap_ci(units, clusters=None)
    assert unclustered["resampling_unit"] == "observation"

    # Clustered defaults to "question"
    clustered_default = bootstrap_ci(units, clusters=clusters)
    assert clustered_default["resampling_unit"] == "question"

    # Clustered accepts explicit custom cluster_unit (e.g. "query")
    clustered_custom = bootstrap_ci(units, clusters=clusters, cluster_unit="query")
    assert clustered_custom["resampling_unit"] == "query"


def test_bootstrap_ci_non_mean_statistic_receives_complete_resampled_dataset():
    # Pins contract: non-mean statistic receives complete resampled dataset, not pre-reduced scalar.
    units = [10.0, 20.0, 30.0, 40.0, 50.0]
    received_lengths: list[int] = []
    received_types: list[type] = []

    def custom_median(resample: Sequence[float]) -> float:
        received_lengths.append(len(resample))
        received_types.append(type(resample))
        sorted_vals = sorted(resample)
        n = len(sorted_vals)
        return float(sorted_vals[n // 2])

    res = bootstrap_ci(units, statistic=custom_median, n_boot=50, seed=7)

    # Point estimate and all 50 bootstrap draws must receive full list of length 5
    assert len(received_lengths) == 51  # 1 point estimate + 50 draws
    assert all(length == len(units) for length in received_lengths)
    assert all(t is list for t in received_types)
    assert res["point"] == 30.0

# --- AUPRC & Polarity Invariance tests -------------------------------------------------------------

def test_auprc_textbook():
    scores = [0.2, 0.4, 0.7, 0.9]
    labels = [False, False, True, True]
    assert auprc(scores, labels) == pytest.approx(1.0)


def test_auroc_polarity_invariance():
    # Pins polarity invariance: MiniCheck score is support probability, label is is_supporting.
    # Flipping both scores (1 - s) and labels (not l) leaves AUROC unchanged.
    scores = [0.1, 0.4, 0.5, 0.8]
    labels = [False, True, False, True]
    orig_auroc = auroc(scores, labels)

    flipped_scores = [1.0 - s for s in scores]
    flipped_labels = [not l for l in labels]
    flipped_auroc = auroc(flipped_scores, flipped_labels)

    assert orig_auroc == pytest.approx(0.75)
    assert flipped_auroc == pytest.approx(orig_auroc)


# --- Gate G3 tests --------------------------------------------------------------------------------

def test_gate_g3_auroc_boundary_below_0_75():
    # AUROC = 0.74 < 0.75 => auroc_passes False, passes False
    # pos = [0.4, 0.73], neg = [0.1, 0.5] -> pos 0.4 vs negs -> 1 win; pos 0.73 vs negs -> 2 wins
    # wait, pos = [0.39, 0.74], neg = [0.1, 0.4] -> 0.39 vs [0.1,0.4]=1 win; 0.74 vs [0.1,0.4]=2 wins -> 3/4 = 0.75
    # Let's construct AUROC = 0.70: pos=[0.3, 0.6], neg=[0.2, 0.5] -> 0.3 vs [0.2,0.5]=1 win; 0.6 vs [0.2,0.5]=2 wins -> 3/4=0.75
    # pos=[0.3, 0.45], neg=[0.2, 0.5] -> 0.3 vs [0.2,0.5]=1 win; 0.45 vs [0.2,0.5]=1 win -> 2/4 = 0.5
    # 3 pos, 3 neg: pos=[0.3, 0.5, 0.8], neg=[0.1, 0.4, 0.7]
    # 0.3 vs [0.1,0.4,0.7]=1; 0.5 vs [0.1,0.4,0.7]=2; 0.8 vs [0.1,0.4,0.7]=3 -> 6/9 = 0.6667
    # 5 pos, 5 neg: pos=[0.4, 0.5, 0.6, 0.8, 0.9], neg=[0.1, 0.2, 0.3, 0.7, 0.75]
    # 0.4: 3, 0.5: 3, 0.6: 3, 0.8: 5, 0.9: 5 -> total 19/25 = 0.76
    # pos=[0.4, 0.5, 0.55, 0.8, 0.9], neg=[0.1, 0.2, 0.3, 0.6, 0.75]
    # 0.4: 3, 0.5: 3, 0.55: 3, 0.8: 5, 0.9: 5 -> 19/25 = 0.76
    # pos=[0.4, 0.45, 0.55, 0.8, 0.9], neg=[0.1, 0.2, 0.3, 0.5, 0.75]
    # 0.4: 3, 0.45: 3, 0.55: 4, 0.8: 5, 0.9: 5 -> 20/25 = 0.80
    # pos=[0.4, 0.45, 0.46, 0.8, 0.9], neg=[0.1, 0.2, 0.3, 0.5, 0.75]
    # 0.4: 3, 0.45: 3, 0.46: 3, 0.8: 5, 0.9: 5 -> 19/25 = 0.76
    # pos=[0.4, 0.45, 0.46, 0.47, 0.9], neg=[0.1, 0.2, 0.3, 0.5, 0.75]
    # 0.4: 3, 0.45: 3, 0.46: 3, 0.47: 3, 0.9: 5 -> 17/25 = 0.68
    scores = [0.1, 0.2, 0.3, 0.5, 0.75, 0.4, 0.45, 0.46, 0.47, 0.9]
    labels = [False, False, False, False, False, True, True, True, True, True]
    res = gate_g3(scores, labels, cost_ratio=15.0)

    assert res["auroc"] == pytest.approx(0.68)
    assert res["auroc_passes"] is False
    assert res["passes"] is False
    assert "auroc_below_threshold" in res["reason"]


def test_gate_g3_auroc_boundary_exactly_0_75():
    # Pins exact boundary semantics (AUROC >= 0.75 per preregistration).
    scores = [0.1, 0.4, 0.5, 0.8]
    labels = [False, True, False, True]
    res = gate_g3(scores, labels, cost_ratio=10.0)

    assert res["auroc"] == pytest.approx(0.75)
    assert res["auroc_passes"] is True
    assert res["cost_passes"] is True
    assert res["passes"] is True
    assert res["reason"] == "pass"


def test_gate_g3_auroc_above_0_75_requires_cost_ratio():
    scores = [0.2, 0.4, 0.7, 0.9]
    labels = [False, False, True, True]
    # AUROC = 1.0 > 0.75
    # With satisfying cost_ratio=10.0 -> passes True
    res_pass = gate_g3(scores, labels, cost_ratio=10.0)
    assert res_pass["auroc_passes"] is True
    assert res_pass["cost_passes"] is True
    assert res_pass["passes"] is True

    # With unsatisfying cost_ratio=5.0 -> passes False
    res_fail = gate_g3(scores, labels, cost_ratio=5.0)
    assert res_fail["auroc_passes"] is True
    assert res_fail["cost_passes"] is False
    assert res_fail["passes"] is False
    assert "cost_ratio_below_threshold" in res_fail["reason"]


def test_gate_g3_missing_cost_ratio_fails():
    scores = [0.1, 0.4, 0.5, 0.8]
    labels = [False, True, False, True]
    res = gate_g3(scores, labels, cost_ratio=None)

    assert res["auroc_passes"] is True
    assert res["cost_passes"] is False
    assert res["passes"] is False
    assert "cost_ratio_missing" in res["reason"]


def test_gate_g3_degenerate_labels_and_empty_input():
    # All positive
    res_all_pos = gate_g3([0.1, 0.5, 0.9], [True, True, True], cost_ratio=10.0)
    assert math.isnan(res_all_pos["auroc"])
    assert res_all_pos["auroc_passes"] is False
    assert res_all_pos["passes"] is False
    assert "degenerate_labels" in res_all_pos["reason"]

    # All negative
    res_all_neg = gate_g3([0.1, 0.5, 0.9], [False, False, False], cost_ratio=10.0)
    assert math.isnan(res_all_neg["auroc"])
    assert res_all_neg["auroc_passes"] is False
    assert res_all_neg["passes"] is False
    assert "degenerate_labels" in res_all_neg["reason"]

    # Empty input
    res_empty = gate_g3([], [], cost_ratio=10.0)
    assert math.isnan(res_empty["auroc"])
    assert res_empty["auroc_passes"] is False
    assert res_empty["passes"] is False
    assert "empty_input" in res_empty["reason"]


def test_gate_g3_length_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="equal length"):
        gate_g3([0.1, 0.5], [True])


def test_gate_g3_clustered_evaluation_path():
    scores = [0.1, 0.4, 0.5, 0.8]
    labels = [False, True, False, True]
    clusters = ["q1", "q1", "q2", "q2"]

    res = gate_g3(scores, labels, clusters=clusters, cost_ratio=12.0, n_boot=200, seed=42)

    assert res["auroc_ci"] is not None
    assert res["auroc_ci"]["resampling_unit"] == "question"
    assert res["auroc_ci"]["n_clusters"] == 2
    assert res["n_clusters"] == 2


def test_gate_g3_determinism_and_auditable_verdict_structure():
    scores = [0.1, 0.4, 0.5, 0.8]
    labels = [False, True, False, True]
    clusters = ["q1", "q1", "q2", "q2"]

    res1 = gate_g3(scores, labels, clusters=clusters, cost_ratio=15.0, seed=20260804)
    res2 = gate_g3(scores, labels, clusters=clusters, cost_ratio=15.0, seed=20260804)

    # Determinism assertion: identical input produces byte-identical returned dict
    assert res1 == res2

    # Auditable verdict structure assertion: all required keys present
    expected_keys = {
        "auroc",
        "auroc_min",
        "auroc_passes",
        "auroc_ci",
        "n",
        "n_positive",
        "n_negative",
        "n_clusters",
        "n_no_majority",
        "no_majority_rate",
        "cost_ratio",
        "cost_ratio_min",
        "cost_passes",
        "passes",
        "reason",
    }
    assert set(res1.keys()) == expected_keys

    # Thresholds echoed assertion
    assert res1["auroc_min"] == G3_AUROC_MIN == 0.75
    assert res1["cost_ratio_min"] == G3_COST_RATIO_MIN == 10.0


# --- Score->Label Join tests ----------------------------------------------------------------------

def _make_record(
    query_id: str,
    claim_id: str,
    score: float,
    support_label: SupportLabel,
    annotator_id: str = "a1",
    verifier_name: str = "minicheck",
    citation_index: int | None = None,
) -> QueryRecord:
    return QueryRecord(
        run_id="run-test",
        query_id=query_id,
        question="?",
        system=System.JOINT,
        seed=0,
        claims=[
            Claim(
                claim_id=claim_id,
                text="Claim text",
                citations=[Citation(passage_id="p1", char_start=0, char_end=10)],
                verifier_scores=[VerifierScore(name=verifier_name, score=score)],
                human_labels=[
                    HumanLabel(
                        annotator_id=annotator_id,
                        support_label=support_label,
                        claim_validity=True,
                        citation_index=citation_index,
                    )
                ],
            )
        ],
    )


def test_join_scores_and_labels_happy_path():
    r1 = _make_record("q1", "c1", 0.8, SupportLabel.SUPPORTED)
    r2 = _make_record("q2", "c1", 0.2, SupportLabel.NOT_SUPPORTED)

    joined = join_scores_and_labels([r1, r2])
    assert len(joined) == 2
    assert (joined[0].score, joined[0].is_supporting, joined[0].question_id) == (0.8, True, "q1")
    assert (joined[1].score, joined[1].is_supporting, joined[1].question_id) == (0.2, False, "q2")


def test_join_scores_and_labels_rejects_duplicate_verifier_scores():
    r = _make_record("q1", "c1", 0.8, SupportLabel.SUPPORTED)
    r.claims[0].verifier_scores.append(VerifierScore(name="minicheck", score=0.9))

    with pytest.raises(ValueError, match="Duplicate verifier score"):
        join_scores_and_labels([r])


def test_join_scores_and_labels_rejects_duplicate_annotator_labels():
    r = _make_record("q1", "c1", 0.8, SupportLabel.SUPPORTED, annotator_id="a1")
    r.claims[0].human_labels.append(
        HumanLabel(annotator_id="a1", support_label=SupportLabel.PARTIAL, claim_validity=True)
    )

    with pytest.raises(ValueError, match="Duplicate annotator label"):
        join_scores_and_labels([r])


def test_join_scores_and_labels_detects_missing_scores_and_missing_annotations():
    # Missing verifier score
    r_no_score = _make_record("q1", "c1", 0.8, SupportLabel.SUPPORTED)
    r_no_score.claims[0].verifier_scores = []
    with pytest.raises(ValueError, match="Missing verifier score"):
        join_scores_and_labels([r_no_score])

    # Missing human label
    r_no_label = _make_record("q1", "c1", 0.8, SupportLabel.SUPPORTED)
    r_no_label.claims[0].human_labels = []
    with pytest.raises(ValueError, match="Missing human label"):
        join_scores_and_labels([r_no_label])


def test_join_scores_and_labels_rejects_invalid_labels():
    r = _make_record("q1", "c1", 0.8, SupportLabel.SUPPORTED)
    r.claims[0].human_labels[0].support_label = "INVALID_LABEL"  # type: ignore

    with pytest.raises(ValueError, match="Invalid label value"):
        join_scores_and_labels([r])


def test_join_scores_and_labels_multi_annotator_majority_and_primary_fallback():
    r = _make_record("q1", "c1", 0.8, SupportLabel.SUPPORTED, annotator_id="a1")
    r.claims[0].human_labels.extend(
        [
            HumanLabel(annotator_id="a2", support_label=SupportLabel.PARTIAL, claim_validity=True),
            HumanLabel(
                annotator_id="a3", support_label=SupportLabel.NOT_SUPPORTED, claim_validity=True
            ),
        ]
    )

    # Majority: SUPPORTED (True) + PARTIAL (True) vs NOT_SUPPORTED (False) -> 2 vs 1 -> True
    joined = join_scores_and_labels([r])
    assert joined[0].is_supporting is True

    # Tie case (1 True, 1 False):
    r_tie = _make_record("q1", "c1", 0.8, SupportLabel.SUPPORTED, annotator_id="a1")
    r_tie.claims[0].human_labels.append(
        HumanLabel(
            annotator_id="a2", support_label=SupportLabel.NOT_SUPPORTED, claim_validity=True
        )
    )

    # Tie without primary -> raises ValueError
    with pytest.raises(ValueError, match="Ambiguous multi-annotator ratings with no majority"):
        join_scores_and_labels([r_tie])

    # Tie with primary_annotator="a1" -> uses a1's rating (True)
    joined_prim = join_scores_and_labels([r_tie], primary_annotator="a1")


def test_join_scores_and_labels_no_majority_rate_and_preservation():
    # Unit 1: 2-1 majority unit (a1 SUPPORTED, a2 PARTIAL, a3 NOT_SUPPORTED -> 2 True vs 1 False)
    r1 = _make_record("q1", "c1", 0.8, SupportLabel.SUPPORTED, annotator_id="a1")
    r1.claims[0].human_labels.extend(
        [
            HumanLabel(annotator_id="a2", support_label=SupportLabel.PARTIAL, claim_validity=True),
            HumanLabel(
                annotator_id="a3", support_label=SupportLabel.NOT_SUPPORTED, claim_validity=True
            ),
        ]
    )

    # Unit 2: No-majority / tie unit (a1 SUPPORTED, a2 NOT_SUPPORTED -> 1 True vs 1 False)
    r2 = _make_record("q2", "c1", 0.3, SupportLabel.SUPPORTED, annotator_id="a1")
    r2.claims[0].human_labels.append(
        HumanLabel(
            annotator_id="a2", support_label=SupportLabel.NOT_SUPPORTED, claim_validity=True
        )
    )

    joined = join_scores_and_labels([r1, r2], primary_annotator="a1")

    # 1. Verification of units & primary tie-break
    assert len(joined) == 2
    assert joined[0].is_supporting is True  # 2-1 majority True
    assert joined[1].is_supporting is True  # Tie resolved by primary annotator a1 (SUPPORTED -> True)

    # 2. Count and rate values are exact
    assert joined.n_no_majority == 1
    assert joined.no_majority_rate == pytest.approx(0.5)

    # 3. Propagation into gate_g3 audit dict
    res_g3 = gate_g3(
        scores=[rec.score for rec in joined],
        labels=[rec.is_supporting for rec in joined],
        cost_ratio=10.0,
        n_no_majority=joined.n_no_majority,
        no_majority_rate=joined.no_majority_rate,
    )
    assert res_g3["n_no_majority"] == 1
    assert res_g3["no_majority_rate"] == pytest.approx(0.5)

    # 4. Raw per-annotator judgements preserved (ADR-0016 §3)
    assert len(r1.claims[0].human_labels) == 3
    assert r1.claims[0].human_labels[0].support_label == SupportLabel.SUPPORTED
    assert r1.claims[0].human_labels[1].support_label == SupportLabel.PARTIAL
    assert r1.claims[0].human_labels[2].support_label == SupportLabel.NOT_SUPPORTED

    assert len(r2.claims[0].human_labels) == 2
    assert r2.claims[0].human_labels[0].support_label == SupportLabel.SUPPORTED
    assert r2.claims[0].human_labels[1].support_label == SupportLabel.NOT_SUPPORTED
    assert len(joined[0].raw_labels) == 3
    assert len(joined[1].raw_labels) == 2
