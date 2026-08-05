"""BM25 | MedCPT dense | RRF fusion | cross-encoder rerank — **Table 1**.

**Not yet implemented.** Due W2–W3 (Aug 10–23), gated by G1 (hit@5 ≥ 0.90, Wilson lower bound
above 0.85, at a documented `(chunker, τ)`).

Settled decisions:

- **`bm25s`, not `rank_bm25`** — the latter is borderline at 2M (§3). Pyserini is the fallback if
  needed; Java 21 is present.
- **MedCPT is asymmetric**: `MedCPT-Query-Encoder` for queries, `MedCPT-Article-Encoder` for
  passages. Using one for both is a silent quality loss, not an error.
- **Passages carry no titles, gold or distractor** (`corpus.py`, `chunk.py`). MedCPT's article
  encoder is trained on the *(title, abstract)* pair, so this puts it off-distribution — uniformly,
  which is the point. **The empty title segment needs one convention, applied to every passage:**
  `tok("", abstract)` or single-segment `tok(abstract)`. Undecided; pick it by measuring dev hit@5
  both ways at W2, and record it in the index fingerprint — it is part of the index's identity.
  **Whatever is chosen, the title slot never receives the question.** `scripts/g0_medcpt_throughput.py`
  passes `row["question"]` there as a throughput stand-in; copying that here would index the query.
- The cascade is cheap-and-high-recall first (BM25 + dense → RRF over a ~100 pool), then expensive
  and high-precision (cross-encoder over that pool). Each stage is ablatable via `RetrievalConfig`,
  because Table 1 *is* those ablations.
- **Never tune τ to pass the gate.** R2's escalation ladder ends at relaxing to hit@10 and saying so
  in the paper, reframed as conditional attribution — never at moving the threshold quietly.
- The retrieved list is stored whole (`schema.QueryRecord.retrieved`); hit@k is computed in
  `scoring/retrieval.py` at a k chosen then, not now.
"""

from __future__ import annotations

from .config import RetrievalConfig


def retrieve(query: str, config: RetrievalConfig) -> list:
    """Return the ranked passage list for one query, at the stages enabled in `config`."""
    raise NotImplementedError("W2 — see module docstring for the settled constraints")
