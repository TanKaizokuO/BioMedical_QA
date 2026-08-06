# Harvest from `RAG_Debate_Agent` — reference material, frozen 2026-07-31

Everything this project needed from the retired base pipeline, copied here so that
`~/Code/Research/RAG_Debate_Agent` never has to be opened again. That repo is **read-only history**
(ADR-0007). It may be archived, moved, or deleted without affecting this project.

**Source:** `~/Code/Research/RAG_Debate_Agent` @ `7c5b86f` (clean tree), fix commit `e936d30`
(2026-07-02).

## Two kinds of thing live here, and only one of them is a harvest

`docs/*` is gitignored with `!docs/adr/` and `!docs/harvest/` as the only exceptions, so this
directory is one of exactly two places under `docs/` where a file can be **tracked**. That makes it
the home for two unrelated things, and the rules below govern only the first:

1. **The harvest itself** — the frozen reference material from `RAG_Debate_Agent`, listed under
   *Contents*. Rules 1–3 are about this.
2. **[`g0/`](g0/) — this project's own G0 run records** (moved here 2026-08-06 from `runs/g0/`,
   which is gitignored). **These are measurements, they are ours, and rule 2 does not apply to
   them.** They are here because `tests/test_abstention.py` reads them and ADR-0010's validation
   rests on them, so a gitignored directory was the wrong home: the tests skip silently when the
   files are absent, which on any machine but the one that ran G0 is exactly the wrong outcome.
   `scripts/g0_generator_bakeoff.py` now writes here directly.

## Rules for the harvest

1. **Nothing here is imported.** No module under `src/biomedqa/` may import from the base repo, and
   no file here is on the Python path. This is documentation.
2. **Nothing in the harvest is a measurement.** No number produced by the base pipeline enters this
   repo, a run manifest, or the paper. `pubmedqa_baseline_v2` is cancelled (ADR-0007). *(This is
   about the base pipeline's numbers. `g0/` holds our own — see above.)*
3. **Read the caveat before reusing.** Each item below states what survives the move to the 2M
   corpus and what does not. The base repo's design assumption — corpus *is* the gold set — is
   exactly the assumption ADR-0003 rejects, and it is baked into most of its code.

## Contents

| File | What it is |
|---|---|
| [`e936d30-rag_baseline.patch`](e936d30-rag_baseline.patch) | The verbatim `rag_baseline.py` diff from the fix commit. Kept whole so the fix's reasoning survives in the author's own comments. |
| [`pubmedqa-loading.md`](pubmedqa-loading.md) | Dataset loading + gold-context extraction, distilled for `src/biomedqa/data.py`. |
| [`gold-passage-tracking.md`](gold-passage-tracking.md) | The `e936d30` gold-tracking fields, and how they change under the 2M corpus and the least-processed-value rule. |
| [`latency-benchmark-methodology.md`](latency-benchmark-methodology.md) | The `benchmark.py` measurement protocol, for G0's generator bake-off. Methodology only — the Ollama transport does not carry over. |

## Deliberately not harvested

Recorded so the question is not reopened:

- **The retriever** — ChromaDB + `all-MiniLM-L6-v2`, dense-only, top-5. No BM25, no RRF, no
  reranker. Occupies no row of Table 1 (D2).
- **The generator path** — Ollama HTTP against local CPU `qwen2.5:7b`, ~88 s/query. Replaced by
  local 8B AWQ on the A4000 behind `backends.py` (ADR-0004).
- **The index** — `chroma_db/`, and the `pubmedqa_baseline_v2` collection that was never built.
- **The agent-slice structure** (`agents/`, Generator/Critic/Verifier debate) — a different paper.
- **`outputs/slice2_baseline_fixed.json`** and everything under `outputs/`, `logs/` — measurements
  of a configuration this project does not run.

## The one lesson worth more than the code

`rag_baseline.py` guarded re-indexing with:

```python
if current_count == INDEX_SIZE:
    print("Idempotency check passed (count == INDEX_SIZE). Skipping indexing.")
    return collection
```

The broken `pqa_artificial` collection *also* held exactly 1,000 documents. So the check passed, the
re-index was skipped, and the wrong index was silently preserved — that is the mechanism by which
the original bug survived long enough to reach a results file. The author saw it and renamed the
collection to `_v2` rather than trusting the guard; the comment in the patch says so.

**In this repo:** index freshness is decided by a hash of `(corpus_id, chunker_config, encoder_id)`
recorded in the run manifest (`config.py`). Cardinality is never evidence that an index is the index
you meant.
