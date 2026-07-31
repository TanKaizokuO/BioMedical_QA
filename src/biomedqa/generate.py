"""Joint claim-grounded generation, plus both baselines behind one API — **Table 2**.

**Not yet implemented.** Due W4 (Aug 24–30), gated by G2 (joint beats post-hoc on citation-F1 by a
margin whose CI excludes zero).

The three systems in `schema.System` share everything that is not the thing being compared:

- **same retriever, same generator, same citation cap.** An unequal cap makes C2's gap an artifact
  of citation budget rather than of joint grounding (`CONTEXT.md`).
- **matched, reported prompt-iteration budget.** The pre-emptive answer to "your post-hoc baseline
  is a straw man" is that both got the same effort and the budget is in the paper (§1). Start
  logging it in W3, not when someone asks.
- `vanilla` carries no citations by definition — `QueryRecord.validate()` enforces that one, since
  a vanilla run that emits citations is a bug in the harness, not an interesting result.
"""

from __future__ import annotations

from .config import GenerationConfig
from .schema import QueryRecord, System


def generate(question: str, passages: list, system: System, config: GenerationConfig) -> QueryRecord:
    """Produce one record. The raw generation is always stored verbatim before parsing."""
    raise NotImplementedError("W4 — see module docstring for the settled constraints")
