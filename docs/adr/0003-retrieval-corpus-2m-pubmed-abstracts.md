# ADR-0003 — Retrieval corpus is ~2M PubMed abstracts, not the 1,000 gold contexts

**Status:** Accepted · **Date:** 2026-07-30 · **Decided in:** grilling session Q3

> A future reader will ask why the corpus is 2M and not full PubMed. This ADR exists for that
> reader — and for the more surprising question of why it is not 1,000.

## Context

PubMedQA `pqa_labeled` is 1,000 instances where each instance's context **is the abstract its
question was written from**. The base pipeline (`~/Code/Research/RAG_Debate_Agent`) indexed exactly
those 1,000 abstracts and retrieved over them.

That setup breaks this paper in two ways, one of them fatal:

1. **hit@5 ≥ 0.90 over 1,000 title-derived questions is a lexical gimme.** Table 1's retrieval
   cascade has nothing to buy — every stage looks equally good because the task is trivial.
2. **More seriously:** with effectively one topically-relevant document retrievable, joint and
   post-hoc citation both score near-ceiling and **the gap between them vanishes**. Gate **G2 would
   return null for setup reasons rather than scientific ones** — and it would be discovered in
   September, with five weeks left.

Citation *precision* only discriminates when plausible-but-wrong passages are available to cite.
A corpus without distractors cannot measure the thing this paper claims (ADR-0002).

## Decision

**Index ~2M PubMed abstracts:** all 1,000 gold contexts plus ~2M distractors.

Measured against the RTX A4000 (16 GB, exclusive — see ADR-0004): ~2 h MedCPT encode, ~3 GB fp16
embeddings, ~2.5 GB BM25 index, ~12 GB peak disk. Fits RAM comfortably.

## Consequences

- **`rank_bm25` (currently in `pyproject.toml`) is borderline at 2M — swap for `bm25s`.** Java 21 is
  present if Pyserini is preferred instead.
- A distractor-pool construction step is added to W0–W2.
- Encode-time figures above are **estimates, not measurements**. W0 benchmarks MedCPT throughput on
  1,000 abstracts to convert them before the full encode is committed.
- G1's hit@5 ≥ 0.90 becomes a real gate rather than a formality — it can now fail, which is the
  point.

## Alternatives rejected

- **The 1,000 gold contexts only** (the base repo's setup). Fatal for the reasons above.
- **Full MedRAG PubMed (23.9M).** Now feasible on the A4000 (~16–24 h encode, ~300 GB) but costs
  roughly a week of Phase 1 infrastructure work and very likely breaks G1 — hit@5 ≥ 0.90 at that
  scale is not realistically achievable. **Deferred to the 2027 journal extension** (ADR-0001).
