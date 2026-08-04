# ADR-0009 — Granularity parity is a measured diagnostic, not a fifth equal-effort condition

**Status:** Accepted · **Date:** 2026-08-04 · **Decided in:** grilling session (G0 follow-up)
**Refines** ADR-0002's equal-effort protocol · **Constrained by** ADR-0005 (the attribution unit)

## Context

C2's headline number is citation-F1, computed over **claims**. Joint generation emits claims
natively; the post-hoc baseline produces prose that `decompose.py` then splits. **Two different
mechanisms produce the unit the metric is denominated in.**

Coarser claims are harder to entail per claim, so if post-hoc's claims are systematically coarser,
post-hoc is systematically penalised — and **C2's gap appears without joint grounding doing any
work.** The bias points *toward* the hypothesis, which is the direction that must never go
unmeasured.

**What the evidence does and does not show.** G0 measured **9.2 claims/query (Llama) vs 3.8
(Qwen)** — same prompt, same passages. That is a divergence between two *models*, and D1 then fixed
the generator, so joint and post-hoc both run
`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`. **Mechanism-driven divergence has never been
measured.** It may be large; it may be 5%. This ADR builds the instrument that will tell us, and
deliberately does not assume the answer.

## Decision

### 1. Parity is a diagnostic, listed separately from the four enforced conditions

ADR-0002's protocol holds four conditions that are true **by construction** — you configure them and
they hold, and `schema.validate()` even checks the third:

| Condition | How it holds |
|---|---|
| same retriever / passages / *k* | set |
| same generator backend | set |
| same 3-citation cap + prompt token budget (±10%) | set, and checked in code |
| matched prompt-iteration budget | counted |

Granularity parity is **not** like these. It is an empirical outcome of prompt tuning: you aim at it
and then discover whether you hit it. Listing it as a fifth condition invites the reader to assume
all five hold, and converts a near-miss into a **disclosed failure of our own fairness protocol** —
*"by the authors' own criterion the comparison is unfair"* — which is a worse position than not
having named it, and strictly worse than naming it honestly.

**It is therefore reported as a separately-labelled measured fairness diagnostic:** four conditions
enforced, one quantity measured and disclosed whatever it says.

### 2. The gated quantity is median words/claim; claims/query is reported

Median words/claim drives per-claim entailment difficulty, which is the mechanism of the bias.
`claims/query` tracks answer length and is reported, not gated.

> **Open, deferred to W4 (see "Known weaknesses").** With total answer length already constrained to
> ±10% by the third enforced condition, words/claim and claims/query are near-mechanically linked.
> The choice is re-examined against the first real measurement.

### 3. Tolerance ±15%, dev only, pre-committed now and unmeasured

**±15% is fixed before any measurement exists, and it is not revised afterwards.** A tolerance chosen
after seeing the divergence is not a pre-commitment — it would be set to a number already known to be
reachable, possibly already reached, making the loop a no-op. Full blinding (§6) protects against
steering on F1; nothing but pre-commitment protects against steering on the tolerance itself.

**The tolerance does not need to be achievable.** Missing it is survivable by design — see §5.

### 4. Only the post-hoc decomposer is tuned

The parity loop may edit **the post-hoc decomposer prompt only**. The joint prompt is out of bounds
for the loop's duration.

Joint's granularity is *native* — the model emits claims directly — so the only knob that moves it is
the joint prompt. A loop with both arms in scope drifts into tuning the method itself and booking it
to a line charged to nobody, which is precisely what §7's ledger treatment claims to prevent. One
direction also makes §6's blinding meaningful: a blind loop free to touch both arms is blind about
*outcomes* but unconstrained about *actions*.

With one direction fixed, the effort demonstrably went into the **baseline**, so charging it to
neither system makes the reported baseline effort an *undercount* — the safe direction to be wrong
in, and the one that strengthens the answer to objection 7.

### 5. Exactly 10 iterations, or Aug 30, whichever comes first

**A hard 10.** Not "~10" — a bound written with a tilde grants exactly the permission it exists to
deny, and pairing a soft counter with "never tune until it passes" makes the prohibition decorative.

**A hard calendar drop-dead of Aug 30 (end of W4).** Hard-10 bounds the *work*; nothing otherwise
bounds the *calendar*, and ten iterations of prompt tuning can absorb an unbounded number of days.
The loop stops on Aug 30 whether or not parity is achieved.

*The original "frozen before the first W4 run (Aug 24)" was unimplementable.* `research_roadmap.md`
§5 builds joint generation **and** both baselines in W4 — the loop compares systems that W4 is
creating. More fundamentally, under §6 the parity freeze and the first citation-F1 computation are
**the same event**, so there was never a separate freeze to schedule; what was needed was a
termination deadline that leaves G2 runway.

**One-sided fallback on the residual gap** (observable while blind — it is the *granularity* gap in
words/claim, not the F1 gap):

- residual gap **favouring C2** (post-hoc coarser) → the **stratified robustness check becomes
  mandatory**, scheduled in **W9**
- residual gap **running against C2** → note it and proceed

The asymmetry is deliberate: demand more scrutiny when the residual bias points toward the
hypothesis, less when it points away. **It is pre-registered in the paper's methods section**, not
only here — asymmetric scrutiny disclosed in advance reads as rigour; the same rule disclosed
afterwards reads as post hoc.

### 6. The loop is fully blind

**Citation-F1 is not computed on any split, in any form, until the loop terminates.** No burn slice,
no mid-loop checkpoint, no correlated proxy.

**The accepted cost, stated plainly:** first citation-F1 lands ≈ **Aug 31**; G2 is **Sep 6**. If C2 is
null, R5 and Phase 2's contingency must fire inside a six-day window. **R5's trigger is therefore
pre-armed** rather than improvised. A single pre-committed unblinding on a disjoint burn slice was
considered and rejected — it would have bought roughly a week of warning at the cost of 20 dev
questions and a carve at the Aug 7 split freeze.

### 7. Parity tuning is a third disclosed ledger line, charged to neither system

A fairness-control cost, not method development — sound because §4 confines the tuning to the
baseline.

### 8. Decomposer/granularity freeze Sep 3; guidelines in two passes

**Sep 3** is a named, dated artifact three days before G2, protecting the gold set that launches
Sep 7 — ADR-0005 establishes that changing granularity after W6 orphans it.

Annotation guidelines are written in two passes:

- **from Aug 31** — unit-independent rules: no-outside-knowledge, the SUPPORTED/PARTIAL boundary,
  hedging, numerics, jointly-necessary citations
- **Sep 3–6** — worked examples only, built from frozen decomposer output

## Consequences

- **ADR-0002's protocol is restated as "four enforced conditions plus one measured diagnostic"** in
  `research_roadmap.md` §4 Phase 2 and in the paper's setup section.
- **W9 gains the stratified robustness check** — on the evidence so far (joint emits finer claims
  natively) the triggering branch is the *likely* one, not the unlikely one.
- **The paper's methods section gains the pre-registered asymmetric rule.**
- **The six-day first-F1-to-G2 window is a schedule fact**, not a contingency. R5's response is
  decided before Aug 31.
- **The compound-claim safety net fires late.** §2 delegates compound claims to `claim_validity`,
  which is annotated from W6 — *after* the Sep 3 freeze. Parity on words/claim cannot distinguish
  one long atomic claim from one compound claim of equal length, and nothing catches that before the
  freeze.

## Known weaknesses

Recorded rather than resolved, because a future reader will find them anyway:

1. **The motivating measurement is of the wrong contrast** (models, not mechanisms) — see Context.
2. **words/claim may be near-equivalent to claims/query** under the ±10% length condition, making §2's
   claimed independence weaker than stated. Re-examined in W4.
3. **±15% has no empirical basis.** It is a pre-committed yardstick, chosen for the property of being
   fixed in advance rather than for being calibrated.

## Alternatives rejected

- **A fifth enforced condition** (as originally drafted). Creates the "failed your own criterion"
  attack surface for a quantity that cannot be enforced by construction.
- **Tolerance derived from the first measurement.** Not a pre-commitment; steerable in exactly the
  way §3 exists to prevent.
- **Both arms tunable, joint-side iterations charged to joint's ledger line.** More faithful
  accounting, but it re-couples the loop to the method and weakens §6.
- **One pre-committed unblinding on a 20-question burn slice.** Buys ≈ a week of early warning on the
  highest-risk claim in the project; rejected in favour of an unbroken blind, accepting the six-day
  window.
- **Doing nothing.** The bias is real and points toward the hypothesis. Unnamed is the one outcome
  worse than named.
