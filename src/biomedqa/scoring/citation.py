"""ALCE citation precision / recall / F1 — **Table 2**, and the G2 gate.

**Not yet implemented.** Due W5 (Aug 31 – Sep 6). Promoted from
`notebooks/03_2_citation_precision_recall.ipynb`, which is sound at any scale (pure functions over
labels) — but its φ is `cross-encoder/nli-deberta-v3-xsmall`, not MiniCheck.

Semantics are reused verbatim from ALCE and are frozen in `CONTEXT.md`:

    recall(c)   = 1 iff C ≠ ∅ ∧ φ(concat(C), c) = 1
    precision   = fraction of citations that are not *irrelevant*, where x is irrelevant iff
                  φ(x, c) = 0 ∧ φ(concat(C \\ {x}), c) = 1
    F1          = harmonic mean of corpus-level precision and recall

**F1 is the reported number**, because recall alone is gamed by citing everything — which is also
what the ≤3 cap defends against, and why the cap must be identical across all three systems.
Jointly necessary citations are legitimate: the remove-it-and-see rule already handles a claim whose
dose comes from one span and whose outcome comes from another.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..schema import Claim, QueryRecord


def citation_recall(claim: Claim, phi) -> float:
    raise NotImplementedError("W5 — union entailment; see module docstring")


def citation_precision(claim: Claim, phi) -> float:
    raise NotImplementedError("W5 — remove-it-and-see; see module docstring")


def citation_f1(records: Iterable[QueryRecord], phi) -> dict:
    """Corpus-level P/R/F1. Corpus-level, not the mean of per-claim F1s — they differ."""
    raise NotImplementedError("W5")
