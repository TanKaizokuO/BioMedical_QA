"""Inter-annotator agreement — **G4** (α ≥ 0.6 over the triple-labeled gold set).

**Not yet implemented.** Due W8 (Sep 21–27). Promoted from
`notebooks/07_4_human_eval_agreement.ipynb` — with a correction, not just a port.

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
"""

from __future__ import annotations

from collections.abc import Sequence

from ..schema import HumanLabel


def krippendorff_alpha_binary(labels: Sequence[Sequence[HumanLabel]]) -> float:
    """α over the binary collapse — the G4 number."""
    raise NotImplementedError("W8")


def krippendorff_alpha_ordinal(labels: Sequence[Sequence[HumanLabel]]) -> float:
    """α over the 4-way ordinal labels — reported as secondary."""
    raise NotImplementedError("W8")
