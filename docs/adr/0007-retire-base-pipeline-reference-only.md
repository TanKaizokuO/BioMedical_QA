# ADR-0007 — `RAG_Debate_Agent` is retired as the project foundation; reference-only

**Status:** Accepted · **Date:** 2026-07-31 · **Supersedes the "Done: Slice 2 bug fixed" framing in**
`docs/project2_biomedical_attribution_rag_implementation_plan.md` §4

## Context

The implementation plan opened from an assumed asset: an *"existing RAG-over-PubMedQA pipeline,
Slice 2 bug fixed."* That pipeline lives in `~/Code/Research/RAG_Debate_Agent` (base commit
`7c5b86f`), not in this repo. `research_roadmap.md` §0 already established that its headline number
was never produced: the fix in `e936d30` — index `pqa_labeled`, not `pqa_artificial` — was written
into `rag_baseline.py` but never executed. The Chroma collection `pubmedqa_baseline_v2` does not
exist.

D2 (`research_roadmap.md` §2) already decided **rebuild, not port**, on architectural grounds. This
ADR closes the remaining question that D2 left implicit: *should the base pipeline still be re-run to
obtain a starting hit@5?*

## Decision

**No. The base pipeline is retired outright and is reference-only from 2026-07-31.**

Three things follow, and they are the operative content of this ADR:

1. **`pubmedqa_baseline_v2` is cancelled as a deliverable.** It is not a task, not a gate input, and
   not a number that appears in the paper. Nothing in this project blocks on it.
2. **No inherited retrieval measurement exists, and none is worth producing.** G1 begins from zero
   *by design*, not by neglect.
3. **`RAG_Debate_Agent` is read-only history.** No further commits are made there for this project;
   nothing in this repo imports from it. Everything this project needed has been copied into
   [`docs/harvest/`](../harvest/) and that directory — not the other repo — is the citable source.

## Why not just re-run it and get a number?

Because the number would be scientifically empty, and having it would be worse than not having it.

`rag_baseline.py` indexes exactly the 1,000 `pqa_labeled` abstracts and evaluates retrieval by asking
whether each question's *own* abstract comes back in the top 5. ADR-0003 already ruled that corpus
out: over 1,000 title-derived questions, hit@5 ≥ 0.90 is a lexical gimme, and — more seriously —
with one topically-relevant document retrievable, joint and post-hoc citation both score at ceiling
and **G2 returns null for setup reasons rather than scientific ones.**

So a re-run yields a high hit@5 that (a) measures a task the paper does not perform, (b) is not
comparable to any Table 1 cell, since Table 1 is defined over the 2M corpus per `(chunker, τ)`, and
(c) creates a standing invitation to quote "we started near 90%" in a paper where that sentence would
be false. The cost of the re-run is not the compute — it is ~88 s/query of CPU Ollama plus an
encode, then a number that has to be explained away every time it is seen again.

The architecture argument is D2's and stands unchanged: dense-only Chroma + `all-MiniLM-L6-v2`,
top-5, no BM25, no RRF, no reranker. There is no ablation row of Table 1 that this configuration
occupies.

## What was taken instead

Exactly the three items D2 sanctioned, plus one lesson. All are in
[`docs/harvest/`](../harvest/README.md); see that README for the full extraction and the caveats on
reuse.

| Harvested | Destination | Form |
|---|---|---|
| PubMedQA `pqa_labeled` load + gold-context extraction | `src/biomedqa/data.py` | Adapted — the gold *identity* survives, the gold-*as-corpus* design does not |
| Gold-passage tracking fields from `e936d30` (`gold_pubid`, `gold_retrieved`, `gold_rank`) | `src/biomedqa/schema.py` | Adapted — `gold_rank` is kept, `gold_retrieved` and any hit@5 are **not stored** (least-processed-value rule) |
| Latency-benchmark methodology from `benchmark.py` | G0 bake-off script | Methodology only, not code — the Ollama transport is replaced by vLLM |
| Index identity must be content-hashed, never count-checked | `src/biomedqa/config.py` run manifest | Lesson only — see below |

**The lesson, stated once so it is not re-learned:** `rag_baseline.py` guarded re-indexing with
`if collection.count() == INDEX_SIZE: skip`. Because the broken `pqa_artificial` collection also held
1,000 documents, that check silently preserved a wrong index — which is the mechanism by which the
original bug survived. In this repo, index freshness is decided by a **hash of (corpus id, chunker
config, encoder id)** recorded in the run manifest. A cardinality match is never sufficient evidence
that an index is the index you meant.

## Consequences

- `research_roadmap.md` §0 row 1 is reworded: the gate is not "un-attempted", it is *"begins from
  zero by design; no inherited measurement exists or is wanted."*
- Week 0's harvest obligation is discharged. The base repo can be left untouched indefinitely; if it
  is ever archived, moved, or deleted, this project is unaffected.
- **Risk accepted:** the project now has no end-to-end pipeline of any kind until `retrieve.py` and
  `generate.py` exist. That is the real state of the world as of 2026-07-31 and was already true —
  this ADR only stops it from being obscured by an asset that never worked.

## Related

ADR-0003 (retrieval corpus) supplies the corpus argument. `research_roadmap.md` §2 D2 supplies the
architecture argument. This ADR supplies the stop rule.
