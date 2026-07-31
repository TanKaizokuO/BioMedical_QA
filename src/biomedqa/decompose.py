"""Decontextualized atomic claims — **Table 2** (granularity is a config knob, i.e. the C7 rows).

**Not yet implemented.** Due W5 (Aug 31 – Sep 6).

- The unit is frozen in `CONTEXT.md` and ADR-0005: **atomic** (one assertion) and
  **decontextualized** (every pronoun, definite description, and implicit subject resolved). A bare
  atomic claim with an unresolved pronoun is unverifiable in isolation — the verifier takes
  `premise = cited span, hypothesis = claim`, and a non-self-contained hypothesis makes "does the
  span entail it?" indeterminate. An annotator instructed to use no outside knowledge is equally
  stuck.
- `sentence` and bare `atomic` remain available as `Granularity` settings, but **only as ablation
  rows** — the headline configuration is decontextualized atomic.
- Every claim records its char span in the raw generation (`Claim.source_start/source_end`).
  Decomposition quality is an upstream confound on every headline number; `claim_validity` converts
  it into a reportable decomposition-error rate, and the offsets keep it auditable.
- `notebooks/04_3_decompose_then_verify.ipynb` promotes here, but it works on toy claims and does
  not implement decontextualization — which is the hard part.
"""

from __future__ import annotations

from .config import GenerationConfig
from .schema import Claim


def decompose(answer: str, config: GenerationConfig) -> list[Claim]:
    """Split a generated answer into claims at `config.granularity`."""
    raise NotImplementedError("W5 — see module docstring for the frozen unit definition")
