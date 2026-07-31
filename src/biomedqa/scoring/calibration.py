"""Threshold sweep, AUROC, ECE, calibration bins — **Table 3**, and the G3 gate (AUROC ≥ 0.75).

**Not yet implemented.** Due W7 (Sep 14–20). Promoted from
`notebooks/05_4_evaluation_auroc_calibration_ci.ipynb`, which is scale-free and promotes nearly
as-is — its risk is that it has only ever seen simulated score vectors, never a real skewed one.

Every function here consumes **raw** `VerifierScore.score` values and a `SupportLabel` collapsed at
call time. Nothing upstream may binarize; that is the whole reason the sweep is possible.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    raise NotImplementedError("W7")


def ece(scores: Sequence[float], labels: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error. Report the bin counts too — a low ECE over empty bins is noise."""
    raise NotImplementedError("W7")


def threshold_sweep(scores: Sequence[float], labels: Sequence[bool]) -> list[dict]:
    """P/R/F1 at every distinct threshold. The operating point is chosen on **dev**, once."""
    raise NotImplementedError("W7")


def bootstrap_ci(values: Iterable[float], n_boot: int = 10_000, confidence: float = 0.95):
    raise NotImplementedError("W7")
