"""Retrieval metrics — **Table 1**, and the G1 gate.

Every metric here is a function of two stored things: the full ranked list and the gold passage id
set. Nothing is precomputed at a fixed `k`, which is what let G1 move from k=5 to k=10 (ADR-0015)
as a re-score rather than a re-run.

Relevance is **binary and set-valued**. One gold abstract becomes many chunks, so the gold set has
2–7 members per dev query and a passage is relevant iff its id is in that set
(`docs/harvest/gold-passage-tracking.md`). Gold chunks the corpus never indexed stay in the
denominator of `recall_at_k`: a passage the pipeline cannot reach is a passage it did not retrieve,
and shrinking the denominator to the reachable subset would report a recall no user could observe.
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
    """Fraction of an instance's gold passages retrieved by rank k, averaged over instances.

    Macro-averaged: every query weighs the same regardless of how many chunks its abstract cut
    into, so a 7-chunk instance cannot dominate a 2-chunk one. Empty input is `nan`, matching
    `wilson_interval`.
    """
    total = 0.0
    n = 0
    for record in records:
        gold = set(record.gold_passage_ids)
        if not gold:
            raise ValueError(
                f"{record.query_id}: recall is undefined with no gold passages; dropping the "
                "query would silently change the denominator of a reported number"
            )
        found = {p.passage_id for p in record.retrieved if p.rank <= k and p.passage_id in gold}
        total += len(found) / len(gold)
        n += 1
    return total / n if n else float("nan")


def mrr(records: Iterable[QueryRecord]) -> float:
    """Mean reciprocal rank of the **first** gold chunk; a query with no gold retrieved scores 0."""
    total = 0.0
    n = 0
    for record in records:
        r = gold_rank(record)
        total += 1.0 / r if r is not None else 0.0
        n += 1
    return total / n if n else float("nan")


def ndcg(records: Iterable[QueryRecord], k: int = 10) -> float:
    """Binary-relevance nDCG@k, macro-averaged over queries.

    Gain is 1 for a gold chunk and 0 otherwise, discounted by `1 / log2(rank + 1)`. The ideal
    ranking puts `min(|gold|, k)` gold chunks first, so a query whose abstract cut into more chunks
    than `k` is not penalised for the ones that cannot fit.
    """
    total = 0.0
    n = 0
    for record in records:
        gold = set(record.gold_passage_ids)
        dcg = sum(
            1.0 / math.log2(p.rank + 1)
            for p in record.retrieved
            if p.rank <= k and p.passage_id in gold
        )
        ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
        total += dcg / ideal if ideal else 0.0
        n += 1
    return total / n if n else float("nan")


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
