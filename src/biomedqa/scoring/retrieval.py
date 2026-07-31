"""Retrieval metrics — **Table 1**, and the G1 gate.

`gold_rank`, `hit_at_k`, and `wilson_interval` are implemented: they are what G1 is decided on, and
the failure mode they prevent (reporting a Wald interval, or fixing k at write time) is the kind
that is only discovered when a reviewer asks. `recall_at_k`, `mrr`, and `ndcg` follow in W3.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from ..schema import QueryRecord


def gold_rank(record: QueryRecord) -> int | None:
    """1-indexed rank of the best-ranked gold passage, or None if no gold passage was retrieved.

    Gold membership is a **set**: one abstract becomes many chunks, so this is the minimum rank over
    the gold set rather than a string equality (`docs/harvest/gold-passage-tracking.md`). Derived
    here, never stored — storing it at a fixed k is what would make G1's hit@10 fallback a re-run.
    """
    gold = set(record.gold_passage_ids)
    ranks = [p.rank for p in record.retrieved if p.passage_id in gold]
    return min(ranks) if ranks else None


def hit_at_k(records: Iterable[QueryRecord], k: int = 5) -> tuple[int, int]:
    """`(hits, n)` — the raw counts, so the caller can form both the proportion and its interval."""
    hits = n = 0
    for record in records:
        n += 1
        r = gold_rank(record)
        if r is not None and r <= k:
            hits += 1
    return hits, n


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """`(point, lower, upper)` — **Wilson**, not Wald.

    G1 gates on the Wilson *lower* bound, not the point estimate. Wald is wrong in exactly the
    regime this gate lives in: near p = 0.9 with n ≈ 100 it produces bounds that can exceed 1 and
    understates uncertainty (Lesson 5). Implemented rather than imported so the gate never depends
    on which statsmodels version is installed.
    """
    if n == 0:
        return (float("nan"),) * 3
    # Normal quantile for the two-sided interval; 1.959964 at 95%.
    z = _z_for(confidence)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _z_for(confidence: float) -> float:
    # Acklam-style inverse normal is overkill here; the project only ever reports 95% and 99%.
    table = {0.90: 1.644854, 0.95: 1.959964, 0.99: 2.575829}
    if confidence not in table:
        raise ValueError(f"confidence must be one of {sorted(table)}, got {confidence}")
    return table[confidence]


def recall_at_k(records: Iterable[QueryRecord], k: int = 5) -> float:
    """Fraction of an instance's gold passages retrieved by rank k, averaged over instances."""
    raise NotImplementedError("W3")


def mrr(records: Iterable[QueryRecord]) -> float:
    raise NotImplementedError("W3")


def ndcg(records: Iterable[QueryRecord], k: int = 10) -> float:
    raise NotImplementedError("W3")


def gate_g1(records: Sequence[QueryRecord], k: int = 5) -> dict:
    """The G1 decision, computed the way the gate is written (§4 Phase 1).

    Passing requires the point estimate ≥ 0.90 **and** the Wilson lower bound above 0.85. If it
    fails, the escalation ladder is in the roadmap — and it never includes tuning τ.
    """
    hits, n = hit_at_k(records, k)
    point, lower, upper = wilson_interval(hits, n)
    return {
        "k": k,
        "hits": hits,
        "n": n,
        "hit_at_k": point,
        "wilson_lower": lower,
        "wilson_upper": upper,
        "passes": point >= 0.90 and lower > 0.85,
    }
