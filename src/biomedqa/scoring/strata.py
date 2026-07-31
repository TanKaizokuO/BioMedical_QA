"""Stratified error analysis — **Table 5**: negation, numerics, scope/population.

**Not yet implemented.** Due W10 (Oct 5–11).

These three strata are where a cited span asserts the *opposite* of the claim rather than merely
failing to support it — which is why `CONTRADICTED` is a stored label rather than being collapsed
into `NOT_SUPPORTED` at annotation time. Collapsing would destroy this table before it was written,
and an annotator cannot be re-run (`CONTEXT.md`).

`notebooks/06_5_negation_numbers_scope.ipynb` contributes a taxonomy, not code — its strata are
hand-built on toy examples. The real analysis runs over gold-set claims.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..schema import QueryRecord

STRATA = ("negation", "numerics", "scope")


def stratify(records: Iterable[QueryRecord]) -> dict[str, list]:
    raise NotImplementedError("W10")


def error_rates_by_stratum(records: Iterable[QueryRecord]) -> dict[str, dict]:
    raise NotImplementedError("W10")
