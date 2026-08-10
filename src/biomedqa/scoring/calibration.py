"""Threshold sweep, AUROC, ECE, calibration bins — **Table 3**, and the G3 gate (AUROC ≥ 0.75).

**Mostly not yet implemented.** The sweep and its metrics are due W7 (Sep 14–20), promoted from
`notebooks/05_4_evaluation_auroc_calibration_ci.ipynb`, which is scale-free and promotes nearly
as-is — its risk is that it has only ever seen simulated score vectors, never a real skewed one.

`bootstrap_ci` is the exception and is implemented now: ADR-0011 §2 makes **question-clustered**
resampling a standing rule for every interval in the paper, and its Consequences put the signature
change with the W4 dry-run rather than with the rest of this module in W7. Every other function
here still raises.

Every function here consumes **raw** `VerifierScore.score` values and a `SupportLabel` collapsed at
call time. Nothing upstream may binarize; that is the whole reason the sweep is possible.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    raise NotImplementedError("W7")


def ece(scores: Sequence[float], labels: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error. Report the bin counts too — a low ECE over empty bins is noise."""
    raise NotImplementedError("W7")


def threshold_sweep(scores: Sequence[float], labels: Sequence[bool]) -> list[dict]:
    """P/R/F1 at every distinct threshold. The operating point is chosen on **dev**, once."""
    raise NotImplementedError("W7")


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
