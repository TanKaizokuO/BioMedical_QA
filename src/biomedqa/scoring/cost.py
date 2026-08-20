"""Cost and overhead — **Table 4**: input tokens / output tokens / $ per query / wall-clock (s).

Implemented in W7 (Sep 14–20). Functions `per_query_cost` and `overhead_ratio` compute the Table 4 metrics.

These four columns were fixed in W0, before any number existed, precisely so that `backends.py`
instruments for them in W2 rather than the gap being discovered in October. `per_query_cost` is the
function `paper/skeleton.md` names in Table 4's caption; if it stops existing, the caption is a lie
that fails loudly.

The measurement discipline is inherited: warm up and discard, vary the prompt so prefix caching
cannot serve a free second call, keep wall-clock separate from server-reported token counts, sample
peak memory on a thread, and **report the range, never only the mean**
(`docs/harvest/latency-benchmark-methodology.md`). The overhead run in W7 is done on a clean, idle
GPU over ≥5 runs — C5's claim is a ratio, and a noisy denominator moves it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any

from ..schema import CostRecord


def _get_val(rec: Any, field: str) -> Any:
    """Retrieve field value from dataclass or dict instance."""
    if hasattr(rec, field):
        return getattr(rec, field)
    if isinstance(rec, dict):
        return rec.get(field)
    return None


def _total_val(values: Iterable[float | int | None]) -> float | int | None:
    """Sum across stages, or `None` if any stage did not report.

    Follows the precedent set by `generate._total` (generate.py:323): a partial total would
    read as a cheap query in Table 4 rather than as missing instrumentation.
    """
    seen = list(values)
    if not seen:
        return None
    if any(v is None for v in seen):
        return None
    return sum(seen)  # type: ignore[arg-type]


def _summarize_metric(vals: list[float | int | None]) -> dict[str, Any]:
    """Calculate summary statistics (mean, median, min, max, total, and counts) for a metric."""
    n_queries = len(vals)
    valid_vals = [v for v in vals if v is not None]
    n_valid = len(valid_vals)
    n_missing = n_queries - n_valid

    if not valid_vals:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "total": None,
            "n_queries": n_queries,
            "n_valid": 0,
            "n_missing": n_missing,
        }

    tot = sum(valid_vals)
    avg = tot / n_valid
    med = statistics.median(valid_vals)
    mn = min(valid_vals)
    mx = max(valid_vals)

    return {
        "mean": avg,
        "median": med,
        "min": mn,
        "max": mx,
        "total": tot,
        "n_queries": n_queries,
        "n_valid": n_valid,
        "n_missing": n_missing,
    }


def per_query_cost(costs: Iterable[CostRecord]) -> dict:
    """Aggregate cost records to Table 4's row shape per system or component.

    Calculates per-query totals for input tokens, output tokens, USD cost, and wall-clock
    latency (s), returning mean, median, min, max, total, and valid/missing counts per metric.

    Null Token Handling:
        A cost record carrying `input_tokens: None` or `output_tokens: None` represents MISSING
        INSTRUMENTATION (for example, from a call rejection in `generate.py`), NOT zero tokens.
        Treating a null value as zero would misreport a rejected call as a cheap call. Following
        the precedent of `generate._total` (generate.py:323), if any stage cost record for a
        query has `None` for a token metric, the query-level total for that metric evaluates
        to `None` and is counted as missing rather than included as zero in statistics.
    """
    costs_list = list(costs)
    if not costs_list:
        return {}

    by_comp: dict[str, dict[str, list[Any]]] = {}
    unkeyed_counter = 0

    for c in costs_list:
        comp = _get_val(c, "component") or "overall"
        qid = _get_val(c, "query_id")
        if qid is None:
            qid = f"__unkeyed_{unkeyed_counter}__"
            unkeyed_counter += 1
        by_comp.setdefault(comp, {}).setdefault(qid, []).append(c)

    res: dict[str, dict[str, Any]] = {}
    for comp, q_map in by_comp.items():
        in_toks: list[float | int | None] = []
        out_toks: list[float | int | None] = []
        usds: list[float | int | None] = []
        walls: list[float | int | None] = []

        for _qid, q_records in q_map.items():
            in_toks.append(_total_val(_get_val(r, "input_tokens") for r in q_records))
            out_toks.append(_total_val(_get_val(r, "output_tokens") for r in q_records))

            usd_vals = [_get_val(r, "usd") for r in q_records]
            if all(v is None for v in usd_vals):
                q_usd = None
            else:
                q_usd = _total_val(usd_vals)
            usds.append(q_usd)

            walls.append(_total_val(_get_val(r, "wall_s") for r in q_records))

        res[comp] = {
            "input_tokens": _summarize_metric(in_toks),
            "output_tokens": _summarize_metric(out_toks),
            "usd": _summarize_metric(usds),
            "wall_s": _summarize_metric(walls),
            "n_queries": len(q_map),
        }

    return res


def overhead_ratio(ours: Iterable[CostRecord], judge: Iterable[CostRecord]) -> dict:
    """C5: how much cheaper the local verifier is than routing to the Opus 5 judge.

    Computes overhead ratios (judge / ours) for USD cost, input tokens, output tokens, and
    wall-clock latency.

    Zero-Dollar and Empty Baseline Handling:
        - MiniCheck local verifier emits NO `CostRecord` entries (local compute timed in
          `VerifierScore.latency_s`) and has zero USD API cost by construction. Dividing
          judge USD by zero would raise `ZeroDivisionError`. When `ours` has zero USD cost (or
          no cost records) and `judge` has positive USD cost, `usd_ratio` evaluates to
          `float("inf")` rather than raising `ZeroDivisionError`.
        - An empty `judge` series evaluates `usd_ratio` and token/latency ratios to `None`
          with an explicit note rather than raising an exception.
    """
    ours_list = list(ours)
    judge_list = list(judge)

    ours_summary = per_query_cost(ours_list)
    judge_summary = per_query_cost(judge_list)

    if not judge_list:
        return {
            "usd_ratio": None,
            "input_token_ratio": None,
            "output_token_ratio": None,
            "wall_s_ratio": None,
            "note": "empty judge series",
            "judge_stats": judge_summary,
            "ours_stats": ours_summary,
        }

    j_comp = next(iter(judge_summary.values())) if judge_summary else None
    o_comp = next(iter(ours_summary.values())) if ours_summary else None

    j_usd_mean = j_comp["usd"]["mean"] if j_comp else None
    o_usd_mean = o_comp["usd"]["mean"] if o_comp else None

    if not ours_list or o_usd_mean == 0.0 or (o_usd_mean is None and o_comp and o_comp["usd"]["n_valid"] == 0):
        if j_usd_mean is not None and j_usd_mean > 0:
            usd_ratio = float("inf")
        else:
            usd_ratio = None
    elif o_usd_mean is not None and o_usd_mean > 0 and j_usd_mean is not None:
        usd_ratio = j_usd_mean / o_usd_mean
    else:
        usd_ratio = None

    def _ratio(j_val: float | None, o_val: float | None) -> float | None:
        if j_val is None or o_val is None:
            return None
        if o_val == 0:
            return float("inf") if j_val > 0 else 1.0
        return j_val / o_val

    j_in = j_comp["input_tokens"]["mean"] if j_comp else None
    o_in = o_comp["input_tokens"]["mean"] if o_comp else None
    j_out = j_comp["output_tokens"]["mean"] if j_comp else None
    o_out = o_comp["output_tokens"]["mean"] if o_comp else None
    j_wall = j_comp["wall_s"]["mean"] if j_comp else None
    o_wall = o_comp["wall_s"]["mean"] if o_comp else None

    return {
        "usd_ratio": usd_ratio,
        "input_token_ratio": _ratio(j_in, o_in),
        "output_token_ratio": _ratio(j_out, o_out),
        "wall_s_ratio": _ratio(j_wall, o_wall),
        "judge_stats": judge_summary,
        "ours_stats": ours_summary,
    }
