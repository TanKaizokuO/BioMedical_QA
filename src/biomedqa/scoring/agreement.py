"""Inter-annotator agreement — **G4** (α ≥ 0.6 over the triple-labeled gold set).

Promoted from `notebooks/07_4_human_eval_agreement.ipynb` — with a correction, not just a port.

**The notebook simulates 3 labels; this project freezes 4** (`CONTEXT.md`). That is a correctness
bug to fix on promotion, not a scale assumption, because:

- **G4 gates on the binary collapse** `(SUPPORTED | PARTIAL) vs (NOT_SUPPORTED | CONTRADICTED)`,
  which is the quantity C4 consumes. Gate on what you measure with.
- **The 4-way ordinal α is reported alongside** as a secondary number, honestly, not instead.
- The stored label is always the 4-way one. `CONTRADICTED` is the payload of the biomedical
  failure-mode analysis, and an annotator cannot be re-run.

Three raters per unit, over whatever prefix of the shared question order all three completed
(ADR-0016). Two things that are *not* this module's job and must not leak into it: adjudicating
disagreements — α is computed on raw per-annotator labels, and any single gold label per claim is
chosen downstream — and the interval, which is a **question**-clustered bootstrap
(`calibration.bootstrap_ci`, ADR-0011 §2). Three labels on one claim are not three independent
units.

**What a unit is here is the caller's choice, deliberately.** These functions take a list of
per-unit label lists and nothing else, so the same code computes α over span judgements (the
quantity Table 3 compares the verifier against) and over the per-claim union judgements (the
quantity citation recall is scored on). `scripts/annotation_ingest.py` picks the populations;
picking one here would bury that decision in a formula.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from typing import Any

from ..schema import HumanLabel, SupportLabel
from .calibration import bootstrap_ci

#: The 4-way scale in **support order**, which is what makes the ordinal metric meaningful: the
#: binary collapse is then a single cut in this sequence rather than an arbitrary grouping of two
#: pairs. `CONTRADICTED` sits below `NOT_SUPPORTED` because the span asserting the opposite is
#: further from support than the span saying nothing at all.
ORDINAL_SCALE: tuple[SupportLabel, ...] = (
    SupportLabel.CONTRADICTED,
    SupportLabel.NOT_SUPPORTED,
    SupportLabel.PARTIAL,
    SupportLabel.SUPPORTED,
)

#: The gate, as `ROADMAP.md` writes it: a point estimate, and a volume of labelled claims.
#: ADR-0011 §3 dropped the CI trigger — the interval is reported, never used to soften the number.
G4_ALPHA_MIN = 0.6
G4_MIN_CLAIMS = 250

#: `delta2(n_c) -> (c, k) -> squared distance`. Ordinal distance depends on the marginals, so a
#: difference function cannot be a constant table; it is built once the coincidences are counted.
DeltaFactory = Callable[[Sequence[float]], Callable[[int, int], float]]


def _coincidence(units: Sequence[Sequence[int]], n_categories: int) -> list[list[float]]:
    """Krippendorff's coincidence matrix, weighted by `1 / (m - 1)` per unit.

    That weighting is the whole reason α tolerates units labelled a different number of times —
    a partial pass, a skipped claim, an annotator who stopped early. Units with a single coding
    are unpairable and contribute nothing: one label says nothing about agreement.
    """
    o = [[0.0] * n_categories for _ in range(n_categories)]
    for codes in units:
        m = len(codes)
        if m < 2:
            continue
        counts = Counter(codes)
        for c, n_c in counts.items():
            for k, n_k in counts.items():
                o[c][k] += (n_c * n_k - (n_c if c == k else 0)) / (m - 1)
    return o


def _alpha(units: Sequence[Sequence[int]], n_categories: int, delta2: DeltaFactory) -> float:
    """α = 1 − (n − 1) · Σ o_ck δ²_ck / Σ n_c n_k δ²_ck.

    Two degenerate results are conventions rather than measurements, and both are worth knowing
    about at the call site:

    * **No pairable unit** → `nan`. Refusing is right: zero would read as "no agreement".
    * **Every rater used one label** → `1.0`, because expected disagreement is zero. The corpus is
      *informationless*, not perfect, which is why `label_distribution()` is reported next to α
      everywhere in this repo.

    α can also go **negative**, on systematic disagreement. That is a real reading, not a bug.
    """
    o = _coincidence(units, n_categories)
    marginals = [sum(row) for row in o]
    n = sum(marginals)
    if n == 0:
        return float("nan")

    delta = delta2(marginals)
    observed = 0.0
    expected = 0.0
    for c in range(n_categories):
        for k in range(n_categories):
            if c == k:
                continue
            d = delta(c, k)
            observed += o[c][k] * d
            expected += marginals[c] * marginals[k] * d
    if expected == 0:
        return 1.0
    return 1.0 - (n - 1) * observed / expected


def _nominal(_marginals: Sequence[float]) -> Callable[[int, int], float]:
    """Every disagreement costs the same. Correct for the binary collapse — with two categories
    the nominal, ordinal and interval metrics all coincide — and wrong for the 4-way scale, where
    `SUPPORTED` vs `PARTIAL` is not the same mistake as `SUPPORTED` vs `CONTRADICTED`."""
    return lambda c, k: 0.0 if c == k else 1.0


def _ordinal(marginals: Sequence[float]) -> Callable[[int, int], float]:
    """Krippendorff's ordinal metric: `δ²_ck = (Σ_{g=c..k} n_g − (n_c + n_k) / 2)²`.

    Distance is measured in *observed labels crossed*, not in scale positions, so the cost of a
    disagreement depends on how densely the intervening categories were actually used. A rare
    middle category therefore does not manufacture distance, which is exactly the property this
    scale needs: `CONTRADICTED` is expected to be sparse.
    """

    def delta(c: int, k: int) -> float:
        lo, hi = (c, k) if c <= k else (k, c)
        total = sum(marginals[lo : hi + 1]) - (marginals[lo] + marginals[hi]) / 2.0
        return total * total

    return delta


def _codes(units: Sequence[Sequence[HumanLabel]], key: Callable[[HumanLabel], int]) -> list[list[int]]:
    return [[key(label) for label in unit] for unit in units]


def krippendorff_alpha_binary(labels: Sequence[Sequence[HumanLabel]]) -> float:
    """α over the binary collapse — **the G4 number**.

    `labels` is one inner sequence per annotation unit, holding that unit's raw per-annotator
    labels; ragged lengths are fine and are the normal case for a partial pass. The collapse is
    `SupportLabel.is_supporting`, derived here and never stored.
    """
    return _alpha(_codes(labels, lambda x: int(x.support_label.is_supporting)), 2, _nominal)


def krippendorff_alpha_ordinal(labels: Sequence[Sequence[HumanLabel]]) -> float:
    """α over the 4-way ordinal labels — reported as secondary.

    Uses the ordinal metric over `ORDINAL_SCALE`, not the nominal one: treating a
    `SUPPORTED`/`PARTIAL` split as the same disagreement as `SUPPORTED`/`CONTRADICTED` would
    understate agreement precisely where `CONTEXT.md` says the hard boundary lies.
    """
    index = {label: i for i, label in enumerate(ORDINAL_SCALE)}
    return _alpha(_codes(labels, lambda x: index[x.support_label]), len(ORDINAL_SCALE), _ordinal)


def label_distribution(labels: Sequence[Sequence[HumanLabel]]) -> dict[str, int]:
    """Raw label counts over every rating, in scale order.

    Reported wherever α is reported. α is a function of the label distribution as well as of the
    raters: a corpus that is 95% `SUPPORTED` cannot produce a high α however careful the
    annotators were, and a reader who sees only the number will misread that as bad annotators.
    """
    counts = Counter(label.support_label for unit in labels for label in unit)
    return {label.value: counts[label] for label in ORDINAL_SCALE}


def gate_g4(
    labels: Sequence[Sequence[HumanLabel]],
    *,
    n_claims: int,
    clusters: Sequence[Hashable] | None = None,
    n_boot: int = 10_000,
) -> dict:
    """The G4 decision, computed the way the gate is written (`ROADMAP.md`, ADR-0011 §3).

    Passing requires **both** ≥ 250 labelled claims and a point α ≥ 0.6 on the binary collapse.
    `n_claims` is the count of distinct claims in the gate's population and is passed in rather
    than derived, because `labels` is a span-level population: one claim contributes as many units
    as it has cited spans, and counting units here would pass the volume half of the gate on a
    third of the claims.

    `clusters` takes one **question** id per unit (ADR-0011 §2). The interval it produces is
    reported with its cluster count and never used to soften the verdict — `passes` is a function
    of the point estimate alone. Omitting `clusters` omits the interval rather than silently
    computing the narrower unclustered one, which would be the wrong number under a familiar name.
    """
    alpha = krippendorff_alpha_binary(labels)
    ci: dict[str, Any] | None = None
    if clusters is not None and labels:
        ci = bootstrap_ci(
            list(labels),
            krippendorff_alpha_binary,
            clusters=clusters,
            n_boot=n_boot,
        )
    return {
        "alpha_binary": alpha,
        "alpha_ordinal": krippendorff_alpha_ordinal(labels),
        "alpha_ci": ci,
        "n_units": len(labels),
        "n_claims": n_claims,
        "n_ratings": sum(len(unit) for unit in labels),
        "label_distribution": label_distribution(labels),
        "alpha_min": G4_ALPHA_MIN,
        "min_claims": G4_MIN_CLAIMS,
        "passes": (
            not math.isnan(alpha) and alpha >= G4_ALPHA_MIN and n_claims >= G4_MIN_CLAIMS
        ),
    }
