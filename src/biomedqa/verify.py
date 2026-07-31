"""MiniCheck-Flan-T5-Large (+ AlignScore) and the Opus 5 judge baseline — **Table 3**.

**Not yet implemented.** Due W6–W7 (Sep 7–20), gated by G3 (verifier AUROC ≥ 0.75 for
unsupported-claim detection).

- **Raw scores only.** `VerifierScore.score` is continuous and is never thresholded on write; the
  threshold sweep, AUROC, ECE, and calibration bins all live in `scoring/`. A stored boolean fixes
  one operating point and discards the sweep irrecoverably.
- **The verifier and the judge share an identical API**, so C5's cost comparison comes from the same
  call path rather than two implementations that differ in ways the table cannot see.
- **Degradation on biomedical text is expected, not exceptional** (R7): MiniCheck is
  ANLI/synthetic-trained. Report the degradation first, then mitigate — the MedNLI fine-tune is the
  mitigation, which is why the PhysioNet application goes out in W0 rather than W7.
- φ in `notebooks/03_2` and `06_5` is `cross-encoder/nli-deberta-v3-xsmall`, not MiniCheck. The swap
  is real work, not a rename.
"""

from __future__ import annotations

from .config import VerifierConfig
from .schema import Claim, VerifierScore


def verify(claim: Claim, premise: str, config: VerifierConfig) -> VerifierScore:
    """Score one (premise = cited span, hypothesis = claim) pair. Continuous, never binarized."""
    raise NotImplementedError("W6 — see module docstring; store the raw score")
