"""Tests for verifier calibration metrics — `biomedqa.scoring.calibration`.

Covers AUROC, ECE (with bin counts), and threshold_sweep.
All expected values are hand-computed and documented per assertion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomedqa.scoring.calibration import auroc, bootstrap_ci, ece, threshold_sweep


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
