"""vLLM (local 8B AWQ on the A4000) | Anthropic (Opus 5) — **Table 4**.

**Not yet implemented.** Due W2 (~½ day), because the W8 code-freeze decision — which backend runs
the frozen test runs — is a config flag *only if this adapter exists* (ADR-0004).

- **vLLM is reached over HTTP, not imported.** It pins torch exactly, and adding it to this
  project's dependencies backtracks the resolver to vllm 0.2.5 and drags pydantic to 1.10.x
  workspace-wide (see the note in `pyproject.toml`). It runs on the box in its own environment; the
  generator backend is a network boundary. `scripts/g0_generator_bakeoff.py` already speaks this
  protocol and is the working reference.
- **Only the local backend is seedable.** The Claude API rejects `temperature`/`top_p`/`top_k` with
  a 400, so the ≥3-seed plan is implementable locally and nowhere else (ADR-0004). This is why
  development iteration is local and the frontier model is the judge, not the generator.
- Every call emits a `CostRecord`. Table 4's columns were fixed in W0 so that instrumentation
  happens here rather than being discovered missing in October.
"""

from __future__ import annotations

from .config import GenerationConfig
from .schema import CostRecord


def complete(prompt: str, config: GenerationConfig) -> tuple[str, CostRecord]:
    """One completion, plus its cost row. Backend chosen by `config.backend`."""
    raise NotImplementedError("W2 — see module docstring; scripts/g0_generator_bakeoff.py is the reference")
