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

import math
import statistics
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..schema import QueryRecord, SupportLabel

#: Gate G3 thresholds (ROADMAP.md, research_roadmap.md §8)
G3_AUROC_MIN: float = 0.75
G3_COST_RATIO_MIN: float = 10.0



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
def auprc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the Precision-Recall curve (AUPRC / Average Precision).

    Raises ValueError on empty inputs, unequal input lengths, or degenerate label sets.
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores (len {len(scores)}) and labels (len {len(labels)}) must have equal length"
        )
    if not scores:
        raise ValueError("Cannot compute AUPRC on empty input")

    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=bool)

    n_pos = int(np.sum(labels_arr))
    n_neg = len(labels_arr) - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "Cannot compute AUPRC when all labels are positive or all labels are negative"
        )

    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels_arr, scores_arr))


def gate_g3(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    clusters: Sequence[Hashable] | None = None,
    cost_ratio: float | None = None,
    n_boot: int = 10_000,
    seed: int = 20260804,
    n_no_majority: int = 0,
    no_majority_rate: float | None = None,
) -> dict[str, Any]:
    """The G3 decision for cheap verifier evaluation (ROADMAP.md, research_roadmap.md §8).

    Passing requires **both** AUROC ≥ 0.75 for unsupported claim detection **and**
    per-claim cost reduction ≥ 10× vs the Opus judge baseline.

    `clusters` takes one question id per unit (ADR-0011 §2). The interval it produces is
    reported with its cluster count and never softens the verdict — `passes` is a function
    of point estimates alone.

    `cost_ratio` is an explicit input because cost evidence is benchmarked separately.
    When `cost_ratio` is None, `cost_passes` is False, `passes` is False, and `reason`
    names the missing evidence.

    `n_no_majority` and `no_majority_rate` report the count and rate of multi-annotator
    units where no strict majority existed (resolved by primary tie-break, ADR-0016).
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores (len {len(scores)}) and labels (len {len(labels)}) must have equal length"
        )

    n = len(scores)
    labels_arr = np.asarray(labels, dtype=bool) if n > 0 else np.array([], dtype=bool)
    n_pos = int(np.sum(labels_arr)) if n > 0 else 0
    n_neg = n - n_pos

    reasons: list[str] = []
    auroc_val: float = float("nan")

    if n == 0:
        reasons.append("empty_input")
    elif n_pos == 0 or n_neg == 0:
        reasons.append("degenerate_labels")
    else:
        auroc_val = auroc(scores, labels)

    auroc_passes = not math.isnan(auroc_val) and auroc_val >= G3_AUROC_MIN
    if not math.isnan(auroc_val) and auroc_val < G3_AUROC_MIN:
        reasons.append(f"auroc_below_threshold ({auroc_val:.4f} < {G3_AUROC_MIN})")

    cost_passes = (
        cost_ratio is not None
        and not math.isnan(cost_ratio)
        and cost_ratio >= G3_COST_RATIO_MIN
    )
    if cost_ratio is None:
        reasons.append("cost_ratio_missing")
    elif math.isnan(cost_ratio) or cost_ratio < G3_COST_RATIO_MIN:
        reasons.append(f"cost_ratio_below_threshold ({cost_ratio} < {G3_COST_RATIO_MIN})")

    ci: dict[str, Any] | None = None
    if clusters is not None:
        if len(clusters) != n:
            raise ValueError(
                f"clusters has {len(clusters)} keys for {n} units; one key per unit or None"
            )
        if n > 0 and n_pos > 0 and n_neg > 0:
            units = list(zip(scores, labels, strict=True))

            def _stat(pairs: Sequence[tuple[float, bool]]) -> float:
                s = [p[0] for p in pairs]
                l = [p[1] for p in pairs]
                try:
                    return auroc(s, l)
                except ValueError:
                    return float("nan")

            ci = bootstrap_ci(
                units,
                statistic=_stat,
                clusters=clusters,
                n_boot=n_boot,
                seed=seed,
            )

    if no_majority_rate is None:
        no_majority_rate = float(n_no_majority / n) if n > 0 else 0.0

    passes = auroc_passes and cost_passes
    reason_str = "; ".join(reasons) if reasons else "pass"
    n_clusters = len(set(clusters)) if clusters is not None else None

    return {
        "auroc": auroc_val,
        "auroc_min": G3_AUROC_MIN,
        "auroc_passes": auroc_passes,
        "auroc_ci": ci,
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_clusters": n_clusters,
        "n_no_majority": n_no_majority,
        "no_majority_rate": no_majority_rate,
        "cost_ratio": cost_ratio,
        "cost_ratio_min": G3_COST_RATIO_MIN,
        "cost_passes": cost_passes,
        "passes": passes,
        "reason": reason_str,
    }


@dataclass(slots=True)
class JoinedEvalRecord:
    """Joined verifier score and human label pair for one claim/citation unit."""

    score: float
    is_supporting: bool
    question_id: str
    claim_id: str
    citation_index: int | None = None
    raw_labels: tuple[HumanLabel, ...] = ()


class JoinedEvalList(list[JoinedEvalRecord]):
    """Joined evaluation records with multi-annotator diagnostic fields."""

    def __init__(
        self,
        records: Sequence[JoinedEvalRecord] = (),
        *,
        n_no_majority: int = 0,
        no_majority_rate: float = 0.0,
    ) -> None:
        super().__init__(records)
        self.n_no_majority: int = n_no_majority
        self.no_majority_rate: float = no_majority_rate


def join_scores_and_labels(
    records: Sequence[QueryRecord],
    *,
    verifier_name: str = "minicheck",
    citation_index: int | None = None,
    primary_annotator: str | None = None,
) -> JoinedEvalList:
    """Join Claim.verifier_scores and Claim.human_labels by (query_id, claim_id, citation_index).

    Rejects duplicate keys, ambiguous ratings with no majority, invalid labels, and missing data.
    Preserves question_id (QueryRecord.query_id) and binary collapse semantics.
    Surfaces n_no_majority count and no_majority_rate on the returned JoinedEvalList (ADR-0016).
    """
    joined: list[JoinedEvalRecord] = []
    n_no_majority = 0
    for record in records:
        qid = record.query_id
        for claim in record.claims:
            if not claim.citations:
                continue  # Uncited claims are not annotation units per ADR-0016

            matching_scores = [v for v in claim.verifier_scores if v.name == verifier_name]
            if not matching_scores:
                raise ValueError(
                    f"Missing verifier score {verifier_name!r} for query_id={qid!r}, claim_id={claim.claim_id!r}"
                )
            if len(matching_scores) > max(1, len(claim.citations)):
                raise ValueError(
                    f"Duplicate verifier score {verifier_name!r} for query_id={qid!r}, claim_id={claim.claim_id!r}"
                )

            target_indices = (
                range(len(claim.citations))
                if citation_index is None
                else [citation_index]
            )

            for idx in target_indices:
                if idx >= len(matching_scores):
                    raise ValueError(
                        f"Missing verifier score at citation_index={idx} for {verifier_name!r} on query_id={qid!r}, claim_id={claim.claim_id!r}"
                    )
                score = matching_scores[idx].score

                matching_labels = [
                    h for h in claim.human_labels if h.citation_index == idx or (h.citation_index is None and idx == 0)
                ]
                if not matching_labels:
                    raise ValueError(
                        f"Missing human label for query_id={qid!r}, claim_id={claim.claim_id!r}, citation_index={idx}"
                    )

                for hl in matching_labels:
                    if not isinstance(hl.support_label, SupportLabel):
                        raise ValueError(
                            f"Invalid label value {hl.support_label!r} for query_id={qid!r}, claim_id={claim.claim_id!r}"
                        )

                annotators = [hl.annotator_id for hl in matching_labels]
                if len(annotators) != len(set(annotators)):
                    raise ValueError(
                        f"Duplicate annotator label for query_id={qid!r}, claim_id={claim.claim_id!r}, citation_index={idx}"
                    )

                if len(matching_labels) == 1:
                    is_supp = matching_labels[0].support_label.is_supporting
                else:
                    pos = sum(1 for hl in matching_labels if hl.support_label.is_supporting)
                    neg = len(matching_labels) - pos
                    if pos > neg:
                        is_supp = True
                    elif neg > pos:
                        is_supp = False
                    else:
                        if primary_annotator is not None:
                            prim = [
                                hl for hl in matching_labels if hl.annotator_id == primary_annotator
                            ]
                            if len(prim) == 1:
                                is_supp = prim[0].support_label.is_supporting
                                n_no_majority += 1
                            else:
                                raise ValueError(
                                    f"No majority and primary annotator {primary_annotator!r} not found for query_id={qid!r}, claim_id={claim.claim_id!r}"
                                )
                        else:
                            raise ValueError(
                                f"Ambiguous multi-annotator ratings with no majority for query_id={qid!r}, claim_id={claim.claim_id!r}"
                            )

                joined.append(
                    JoinedEvalRecord(
                        score=score,
                        is_supporting=is_supp,
                        question_id=qid,
                        claim_id=claim.claim_id,
                        citation_index=idx,
                        raw_labels=tuple(matching_labels),
                    )
                )
    n_total = len(joined)
    no_majority_rate = float(n_no_majority / n_total) if n_total > 0 else 0.0
    return JoinedEvalList(
        joined,
        n_no_majority=n_no_majority,
        no_majority_rate=no_majority_rate,
    )


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
