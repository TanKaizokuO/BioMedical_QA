"""Seed loop, cost log, run manifest, config diff.

**Not yet implemented.** Due W2 (Aug 10–16). Promoted from
`notebooks/08_6_reproducible_eval_harness.ipynb`, whose config hashing is already the right
primitive (`config.canonical_hash`) — the notebook's in-memory record list is what must change, to
streamed `records.jsonl`, since 2M-corpus runs will not fit its shape.

One run directory, `runs/<run_id>/`:

    manifest.json    config hash, index fingerprint, split hash, git sha, model ids, timestamps
    records.jsonl    one QueryRecord per (question, system, seed)
    costs.jsonl      one CostRecord per billable or timed unit of work

**G5 is the reason this exists**: every cell of Tables 1–5 must be populated from a run manifest,
with CIs. A number whose manifest cannot be produced does not go in the paper.
"""

from __future__ import annotations

from pathlib import Path

from .config import RunConfig


def run_manifest(config: RunConfig, run_dir: Path) -> dict:
    """Write `manifest.json`: config hash, index fingerprint, split hash, git sha, timestamps."""
    raise NotImplementedError("W2 — see module docstring")
