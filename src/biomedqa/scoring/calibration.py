"""Threshold sweep, AUROC, ECE, calibration bins — **Table 3**, and the G3 gate (AUROC ≥ 0.75).

This module implements `auroc`, `ece`, and `threshold_sweep`, promoted from
`notebooks/05_4_evaluation_auroc_calibration_ci.ipynb`. The notebook code uses simulated
score vectors. The risk is that score distributions on real skewed datasets can differ from
simulated distributions.

`bootstrap_ci` implements **question-clustered** resampling per ADR-0011 §2.

Every function here consumes **raw** `VerifierScore.score` values and a `SupportLabel` collapsed at
call time. Nothing upstream may binarize; that is the whole reason the sweep is possible.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the Receiver Operating Characteristic curve (AUROC).

    Handles ties in the score vector via Mann-Whitney U / pairwise rank sum semantics,
    where a tied pair receives 0.5 points.

    Raises ValueError on empty inputs, unequal input lengths, or degenerate label sets
    (all positive or all negative).
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores (len {len(scores)}) and labels (len {len(labels)}) must have equal length"
        )
    if not scores:
        raise ValueError("Cannot compute AUROC on empty input")

    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=bool)

    pos_scores = scores_arr[labels_arr]
    neg_scores = scores_arr[~labels_arr]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "Cannot compute AUROC when all labels are positive or all labels are negative"
        )

    sorted_neg = np.sort(neg_scores)
    left = np.searchsorted(sorted_neg, pos_scores, side="left")
    right = np.searchsorted(sorted_neg, pos_scores, side="right")

    wins = np.sum(left + 0.5 * (right - left))
    return float(wins / (n_pos * n_neg))


def ece(scores: Sequence[float], labels: Sequence[bool], bins: int = 10) -> dict[str, Any]:
    """Expected calibration error. Report the bin counts too — a low ECE over empty bins is noise.

    Returns a dict containing:
      - 'ece': float scalar expected calibration error
      - 'bin_counts': list[int] count of samples in each bin
      - 'bin_accuracies': list[float | None] accuracy per bin (None if empty)
      - 'bin_confidences': list[float | None] mean confidence per bin (None if empty)
      - 'bins': list[tuple[float, float]] (lo, hi) boundary per bin
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores (len {len(scores)}) and labels (len {len(labels)}) must have equal length"
        )
    if not scores:
        raise ValueError("Cannot compute ECE on empty input")
    if bins <= 0:
        raise ValueError("bins must be a positive integer")

    scores_arr = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    labels_arr = np.asarray(labels, dtype=bool)

    n_total = len(scores_arr)
    boundaries = np.linspace(0.0, 1.0, bins + 1)

    counts = [0] * bins
    bin_accuracies: list[float | None] = [None] * bins
    bin_confidences: list[float | None] = [None] * bins
    bin_ranges: list[tuple[float, float]] = []

    weighted_gap_sum = 0.0

    for j in range(bins):
        lo, hi = float(boundaries[j]), float(boundaries[j + 1])
        bin_ranges.append((lo, hi))

        if j == 0:
            mask = (scores_arr >= lo) & (scores_arr <= hi)
        else:
            mask = (scores_arr > lo) & (scores_arr <= hi)

        count = int(np.sum(mask))
        counts[j] = count

        if count > 0:
            bin_acc = float(np.mean(labels_arr[mask]))
            bin_conf = float(np.mean(scores_arr[mask]))
            bin_accuracies[j] = bin_acc
            bin_confidences[j] = bin_conf
            gap = abs(bin_acc - bin_conf)
            weighted_gap_sum += (count / n_total) * gap

    return {
        "ece": float(weighted_gap_sum),
        "bin_counts": counts,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bins": bin_ranges,
    }


def threshold_sweep(scores: Sequence[float], labels: Sequence[bool]) -> list[dict[str, Any]]:
    """P/R/F1 at every distinct threshold. The operating point is chosen on **dev**, once.

    Returns a list of dicts ordered by ascending threshold. Each dict contains:
      - 'threshold': float
      - 'precision': float
      - 'recall': float
      - 'f1': float
      - 'tp': int
      - 'fp': int
      - 'fn': int
      - 'tn': int
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores (len {len(scores)}) and labels (len {len(labels)}) must have equal length"
        )
    if not scores:
        raise ValueError("Cannot sweep thresholds on empty input")

    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=bool)

    distinct_thresholds = np.sort(np.unique(scores_arr))

    total_positives = int(np.sum(labels_arr))
    total_negatives = int(len(labels_arr) - total_positives)

    results = []
    for t in distinct_thresholds:
        preds = scores_arr >= t
        tp = int(np.sum(preds & labels_arr))
        fp = int(np.sum(preds & ~labels_arr))
        fn = total_positives - tp
        tn = total_negatives - fp

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / total_positives) if total_positives > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        results.append(
            {
                "threshold": float(t),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    return results


def bootstrap_ci(
    units: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float] | None = None,
    *,
    clusters: Sequence[Hashable] | None = None,
    cluster_unit: str = "question",
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260804,
) -> dict:
    """Percentile bootstrap CI, resampling **clusters** when `clusters` is given.

    `units` is the observation list and `statistic` maps a resampled list of units to one number,
    defaulting to the arithmetic mean — so a fraction is `bootstrap_ci(list_of_bools)`. The
    statistic takes the whole resample rather than being averaged over it, because the quantities
    this has to cover are not means: citation-F1 is the harmonic mean of **corpus-level** precision
    and recall (`CONTEXT.md`), and a CI for it cannot be assembled from per-claim numbers.

    **`clusters` is the ADR-0011 §2 parameter.** Pass one key per unit — the question id — and the
    resampling unit becomes the question: cluster keys are drawn with replacement and every unit of
    each drawn cluster comes along, so the correlation between claims of one answer is carried into
    the interval instead of being assumed away. Omitting it resamples units independently, which is
    the narrower and wrong interval for any claim-level quantity. Pairing is a property of
    `statistic` (compute the delta inside it), not of this parameter: pairing means the systems see
    the same question, clustering means the question is what gets drawn.

    The returned dict names `resampling_unit` because ADR-0011's Consequences require every caption
    that reports a CI to state it; a caller that forwards this dict cannot omit it by accident.
    """
    if not units:
        raise ValueError("bootstrap_ci needs at least one unit; an empty CI is not a wide one")
    if clusters is not None and len(clusters) != len(units):
        raise ValueError(
            f"clusters has {len(clusters)} keys for {len(units)} units; one key per unit or None"
        )
    stat = statistic if statistic is not None else _mean

    grouped: list[Sequence[Any]]
    if clusters is None:
        grouped = [[u] for u in units]
    else:
        by_key: dict[Hashable, list[Any]] = {}
        for key, unit in zip(clusters, units, strict=True):
            by_key.setdefault(key, []).append(unit)
        grouped = list(by_key.values())

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(n_boot, len(grouped)))
    replicates = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        resample = [u for i in draws[b] for u in grouped[i]]
        replicates[b] = stat(resample)

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(replicates, [alpha, 1.0 - alpha])
    return {
        "point": stat(units),
        "lower": float(lower),
        "upper": float(upper),
        "width": float(upper - lower),
        "n_units": len(units),
        "n_clusters": len(grouped),
        "resampling_unit": cluster_unit if clusters is not None else "observation",
        "n_boot": n_boot,
        "confidence": confidence,
        "seed": seed,
    }


def _mean(values: Sequence[Any]) -> float:
    return statistics.fmean(values)
