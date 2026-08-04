# ADR-0012 — The distractor pool is a uniform sample, monitored by a confusability probe

**Status:** Accepted · **Date:** 2026-08-04 · **Decided in:** grilling session (G0 follow-up)
**Completes** ADR-0003, which fixed the corpus **size** but never its **source or selection policy**

## Context

ADR-0003 decided the retrieval corpus is ~2M PubMed abstracts — all 1,000 PubMedQA gold contexts plus
~2M distractors — and gave the reason: with one topically-relevant document retrievable, joint and
post-hoc citation both score at ceiling, **the gap vanishes, and G2 returns null for setup reasons
rather than scientific ones.**

It never said **where the distractors come from or how they are chosen**, and that silence hides a
tension with opposite signs on the same knob:

| Gate | Wants |
|---|---|
| **G1** — hit@5 ≥ 0.90 | *easy* distractors |
| **G2** — citation precision must discriminate | *hard* distractors |

## Decision

### 1. Uniform random 2M from `MedRAG/pubmed`, seeded and reproducible

The sample is drawn uniformly at random, with the seed and the resulting ID list committed and
hashed into the run manifest via `RunConfig.index_fingerprint()`.

**Uniform is unarguable to a reviewer.** Hand-picked hard negatives look like a corpus engineered
around the gold set, and no amount of explanation recovers from that appearance in a paper whose
headline is a *fairness* comparison.

**Deduplicate the gold contexts against the sample on PMID.** PubMedQA contexts *are* PubMed
abstracts, so `MedRAG/pubmed` very likely already contains them; a naive union yields the same
abstract under two `passage_id`s and `gold_rank`/hit@5 silently miscount. This is the shape of the
staleness bug ADR-0007 exists to remember. **W2 blocker — it must land before the encode, not after.**

### 2. The hardness diagnostic is a retrieval-side entailment-confusability probe

**Replacing** the originally proposed LLM topic judge, which asked a non-Opus model whether ≥1 of the
non-gold top-5 was "on the same clinical topic," against a pre-committed ≥70% threshold.

That design had three defects:

1. **It would pass essentially always, and passing would mean nothing.** Over a 2M biomedical corpus
   with a hybrid retriever, a generous judge answers yes almost every time. A threshold
   pre-committed against a measurement with no established discriminating power buys false comfort.
2. **It measured the wrong thing.** A distractor damages citation precision only if it **plausibly
   entails a claim it should not**. Topical relatedness is a proxy for confusability, and a loose one.
3. **Its escalation contradicted §1.** "Escalate to injected hard negatives" is precisely the
   engineered-corpus appearance §1 rejects.

**The probe:** for each of ~100 dev questions, take the RRF-fused top-5 (no reranker until W3 —
re-confirm after), drop the gold, and score the question's **gold claims** against the **non-gold**
passages with the entailment model. Near-zero entailment across the pool means there is nothing
plausible to mis-cite and citation precision cannot discriminate, whatever a topic judge says.

**No threshold is pre-committed.** The first distribution is the first information anyone has about
this quantity; a threshold is set after seeing it, and that is honest here precisely because the
probe gates no tuning — it is an observation, not a target.

**Blind by construction.** The probe runs over retrieval output before generation exists, so it
cannot leak anything about citation-F1 and does not touch ADR-0009 §6.

**Cost, stated:** it pulls MiniCheck-Flan-T5-Large forward from Phase 3 (W5–W7) into **W2–W3** —
770M, local, `verify.py` already stubbed, call it ~½ day. W2 already holds the chunker sweep,
`bm25s`, RRF and the 2M encode.

### 3. If the probe says the pool is too easy

Escalate by **sampling more densely from the gold questions' own MeSH terms — uniformly within that
stratum, seeded, and declared in the paper** — never by hand-picking passages. The distinction is the
whole of §1: a declared stratified sample is a described procedure; a hand-picked set is a corpus
built around the answer.

## Consequences

- **Two early-warning systems for ADR-0003's fatal scenario were nearly lost at once.** The topic
  judge is replaced here; an early citation-F1 read was declined in ADR-0009 §6. Had both gone, the
  failure ADR-0003 was written to prevent would have been **unmonitored until the gate it kills**.
  The probe is the one instrument that remains, which is why it is aimed at the mechanism rather than
  a proxy.
- **`RetrievalConfig.corpus_id`** (`"pubmed-2m-v1"`) is the sample's identity; the seed and ID list
  are committed alongside it.
- **W2 gains the dedup step and ~½ day of pulled-forward verifier work**, and W2 is not a light week.
- **The probe's first distribution is a reportable number** — it belongs in the paper's setup section
  as evidence the corpus can measure what the paper claims, not only in a run log.

## Alternatives rejected

- **Hand-picked or mined hard negatives.** Fatal to the fairness framing; see §1.
- **The LLM topic judge with a pre-committed ≥70% threshold.** Three defects, above.
- **Demoting the topic judge to a monitored signal without replacing it.** Keeps a measurement that
  does not measure the mechanism, and leaves ADR-0003's failure mode effectively unwatched.
- **RRF score margin (gold vs rank-2) as the confusability proxy.** Free, computed from data W2
  already produces, and needs no verifier — but it conflates retrieval confidence with semantic
  confusability, which is the distinction that matters.
- **No monitoring; let G2 be the test.** The bet is that a uniform 8.4% sample of PubMed inevitably
  contains plausible distractors. Probably true, and unrecoverable if false.
