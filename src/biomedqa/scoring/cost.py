"""Cost and overhead — **Table 4**: input tokens / output tokens / $ per query / wall-clock (s).

**Not yet implemented.** Due W7 (Sep 14–20), alongside the overhead measurement.

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

from collections.abc import Iterable

from ..schema import CostRecord


def per_query_cost(costs: Iterable[CostRecord]) -> dict:
    """Aggregate to Table 4's row shape, per system: tokens, USD, wall-clock, with ranges."""
    raise NotImplementedError("W7")


def overhead_ratio(ours: Iterable[CostRecord], judge: Iterable[CostRecord]) -> dict:
    """C5: how much cheaper the local verifier is than routing to the Opus 5 judge.

    If this lands under 10×, R4 applies — the headline is already attribution quality (ADR-0002),
    so it costs the subtitle, not the paper. Decide by Sep 20.
    """
    raise NotImplementedError("W7")
