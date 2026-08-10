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

#### Result, 2026-08-10 — the pool is confusable, but only the tail says so

Run over the 2.16M index, dev split, RRF top-5 with gold dropped: **427 non-gold passages across 100
questions**, each scored as the max over the question's gold sentences (mean 8.8 of them).
`docs/harvest/confusability_probe.json`.

**The retrieved-side distribution alone was not interpretable, and the first reading of it was wrong.**
Mean 0.4245, median 0.3802, p90 0.7376, and 62/100 questions carrying a non-gold passage at ≥0.5
looks like a confusable pool. It is also consistent with MiniCheck simply saying yes a lot, and with
a max over ~9 sentences inflating any base rate — the same "passes always, means nothing" defect this
section rejected the topic judge for. So the run was repeated against a **paired uniform-random
control**: the same gold sentences, the same *number* of passages per question, drawn uniformly from
the 2M corpus at seed 12345 (`--random-control`, `docs/harvest/confusability_probe_control.json`).
Pairing is what makes it readable — the max-over-sentences inflation is identical on both arms and
cancels in the contrast.

| entailment ≥ | retrieved distractors | uniform-random passages | ratio | questions with ≥1 (ret / rand) |
|---|---|---|---|---|
| 0.3 | 62.1% | **67.7%** | **0.92** | 87 / 93 |
| 0.4 | 47.1% | 39.1% | 1.20 | 81 / 77 |
| 0.5 | 34.0% | 15.9% | 2.13 | 62 / 42 |
| 0.6 | 24.8% | 5.9% | 4.24 | 50 / 19 |
| 0.7 | **14.5%** | **2.1%** | **6.89** | 35 / 8 |
| 0.8 | 1.9% | 0.2% | 8.00 | 7 / 1 |

**MiniCheck's base rate is enormous, and it swallows the bulk of the distribution.** A uniformly
random PubMed abstract scores a median 0.3613 against a gold claim and clears 0.3 more than two thirds
of the time. **At 0.3 the retrieved distractors do worse than chance.** Any statement of the form
"62% of our distractors are entailment-confusable" is therefore not a finding; it is the model's floor.

**Retrieval's contribution is real and lives entirely in the upper tail**, where the ratio climbs
monotonically to ~7× at 0.7. Paired per question, the retrieved arm has the higher mean on **63 of
100** questions (sign test **p = 0.012**), with a mean per-question delta of only +0.050 — small,
because the bulk does not move and the tail is what does.

**The threshold §2 deferred, now set: τ_confusable = 0.7.** Not 0.5, MiniCheck's nominal operating
point, where a 15.9% random base rate still contaminates a third of what gets counted. At 0.7 the
random rate is 2.1%, so the measurement is mostly signal, and **35 of 100 dev questions carry at least
one distractor that plausibly entails a gold claim** against 8 by chance. That is the reportable
number for the setup section, and it replaces the mean, which is nearly chance.

Setting it after seeing the distribution is what §2 licensed, and it stays honest for the reason §2
gave: **the probe gates nothing.** No corpus, retriever or τ is tuned against it, and nothing here is
permitted to move a G-gate.

**§3 is not triggered.** Its escalation fires on a pool with nothing plausible to mis-cite; a 7×
enrichment over chance at τ_confusable is the opposite. No MeSH-stratified redraw. Citation precision
has something to discriminate, which is what ADR-0003's fatal scenario needed watched.

**Still open, and cheap to close at W3:** the top-5 here is pre-reranker, as §2 requires. The W3
cross-encoder changes which distractors survive, so `research_roadmap.md` W3's "re-confirm after"
means re-running **both arms** — a control-free re-run would restate the uninterpretable number.

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
