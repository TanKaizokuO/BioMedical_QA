"""Inter-annotator agreement — **G4** (α ≥ 0.6 on the overlap subset).

**Not yet implemented.** Due W8 (Sep 21–27). Promoted from
`notebooks/07_4_human_eval_agreement.ipynb` — with a correction, not just a port.

**The notebook simulates 3 labels; this project freezes 4** (`CONTEXT.md`). That is a correctness
bug to fix on promotion, not a scale assumption, because:

- **G4 gates on the binary collapse** `(SUPPORTED | PARTIAL) vs (NOT_SUPPORTED | CONTRADICTED)`,
  which is the quantity C4 consumes. Gate on what you measure with.
- **The 4-way ordinal α is reported alongside** as a secondary number, honestly, not instead.
- The stored label is always the 4-way one. `CONTRADICTED` is the payload of the biomedical
  failure-mode analysis, and an annotator cannot be re-run.
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
