"""PubMedQA yes/no/maybe accuracy — **secondary** (C6).

Reported second, always. The paper's claim is attribution quality (ADR-0002); accuracy exists to
show that attribution does not *cost* correctness, not to compete with SoTA PubMedQA systems. A
framing that leads with accuracy invites exactly the comparison this paper is positioned to avoid.

Gold decision resolution:
The function gets the gold decision from `QueryRecord.gold_final_decision` when present. If this
field is absent, the function resolves the decision from the frozen dev split using `load_instances()`
keyed on the PubMed ID prefix of `query_id`. A record with no resolvable gold decision is excluded
from every count, including the confusion matrix.

Unparsed generation handling:
Unparsed generations (`final_decision` is None or not a valid decision) are excluded from accuracy
and Wilson interval evaluation (`n_eval`). This prevents parse failures (measured separately in
Gate G2) from changing accuracy scores. In the 3x3 confusion matrix over ("yes", "no", "maybe"),
unparsed predictions are assigned to the "maybe" column as non-committal answers.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..data import load_instances
from ..schema import QueryRecord
from .retrieval import wilson_interval

VALID_DECISIONS = ("yes", "no", "maybe")

# Lazy cache for dev instance gold decisions if record.gold_final_decision is None
_DEV_GOLD_MAP: dict[str, str] | None = None


def _get_gold_map() -> dict[str, str]:
    global _DEV_GOLD_MAP
    if _DEV_GOLD_MAP is None:
        try:
            instances = load_instances()
            _DEV_GOLD_MAP = {str(inst.pubid): inst.final_decision.lower() for inst in instances}
        except (ImportError, RuntimeError, OSError, ValueError):
            # Degrade gracefully to an empty map if dataset loading fails (e.g. offline execution
            # without cached data), allowing records with stored gold_final_decision to score.
            _DEV_GOLD_MAP = {}
    return _DEV_GOLD_MAP


def _resolve_gold_decision(record: QueryRecord) -> str | None:
    if record.gold_final_decision is not None:
        g = record.gold_final_decision.strip().lower()
        if g in VALID_DECISIONS:
            return g
    pubid = record.query_id.split(":")[0]
    g = _get_gold_map().get(pubid)
    if g in VALID_DECISIONS:
        return g
    return None


def accuracy(records: Iterable[QueryRecord]) -> dict[str, Any]:
    """Accuracy with a Wilson interval, and the 3×3 confusion over yes/no/maybe.

    Returns:
        dict containing:
            - accuracy: float accuracy point estimate over parsed records (or nan if n_eval == 0)
            - wilson_lower: 95% Wilson CI lower bound
            - wilson_upper: 95% Wilson CI upper bound
            - correct: count of correct parsed predictions
            - n_eval: count of evaluated records (valid gold & valid parsed final_decision)
            - n_total: total number of input records
            - n_unparsed: count of records with unparsed / missing final_decision
            - confusion: 3x3 dict of dicts mapping gold -> predicted -> count over ("yes", "no", "maybe")
    """
    records_list = list(records)
    total = len(records_list)

    hits = 0
    n_eval = 0
    n_unparsed = 0

    confusion: dict[str, dict[str, int]] = {
        g: {p: 0 for p in VALID_DECISIONS} for g in VALID_DECISIONS
    }

    for record in records_list:
        gold = _resolve_gold_decision(record)
        if gold is None:
            continue

        pred_raw = (
            record.final_decision.strip().lower()
            if record.final_decision is not None
            else None
        )

        if pred_raw in VALID_DECISIONS:
            n_eval += 1
            if pred_raw == gold:
                hits += 1
            pred_conf = pred_raw
        else:
            n_unparsed += 1
            pred_conf = "maybe"  # Map unparsed generation to non-committal 'maybe'

        confusion[gold][pred_conf] += 1

    if n_eval > 0:
        acc_pt, lo, hi = wilson_interval(hits, n_eval)
    else:
        acc_pt = lo = hi = float("nan")

    return {
        "accuracy": acc_pt,
        "wilson_lower": lo,
        "wilson_upper": hi,
        "correct": hits,
        "n_eval": n_eval,
        "n_total": total,
        "n_unparsed": n_unparsed,
        "confusion": confusion,
    }
