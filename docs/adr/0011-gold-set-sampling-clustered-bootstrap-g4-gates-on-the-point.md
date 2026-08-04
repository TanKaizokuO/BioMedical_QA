# ADR-0011 — Gold-set sampling, clustered bootstrap, and what G4 gates on

**Status:** Accepted · **Date:** 2026-08-04 · **Decided in:** grilling session (G0 follow-up)
**Refines** ADR-0006 (annotation protocol) · **Constrains** every CI in the paper

## Context

ADR-0006 sized the gold set at 250–400 claims with a ~75-claim overlap subset, but never said **how
the claims are drawn**. The default reading — take every claim of enough answers — interacts badly
with the generator D1 chose: at **9.2 claims/query**, a 250-claim budget is the claims of roughly
**27 questions**.

That matters because claims within a question are **not independent**. They share the question, the
retrieved passages, the answer, and the topic. Any statistic that treats them as independent units
returns confidence intervals that are too narrow, and 27 questions is a thin base for a paper whose
evaluation carries the entire contribution (ADR-0006).

## Decision

### 1. Sample ~4 claims per question across ~62 questions

ADR-0006's **250-claim budget and ~3 h annotator ask are unchanged**; the claims are drawn from ~2.3×
as many questions. The overlap subset is sampled the same way — **~75 claims across ~19 questions**
instead of ~8.

This is a precondition for §2, not a neighbour of it: a cluster bootstrap over **8** clusters is close
to meaningless, and over 19 it is merely wide.

> **Open, and load-bearing (see "Known weaknesses").** ADR-0006's ~3 h estimate was derived for ~75
> claims; it was **not** re-derived for 19 questions' worth of reading rather than 8. Per-question
> context-switching — a new question, a new passage set — is real annotator load. **The estimate is
> re-derived before the two annotators confirm** (issue #7), not before the Aug 20 checkpoint.
> Revising the ask upward after someone has accepted is the worst available outcome.

### 2. Every bootstrap cluster-resamples **questions**, not claims

A standing rule, added to `research_roadmap.md` §8, applying to **every** interval in the paper —
G2's paired bootstrap, G4's α, and every CI in Tables 2–5.

**This is a change, not a restatement.** Nothing in the repo clusters today:

| | |
|---|---|
| `research_roadmap.md` G2 | *"a margin exceeding the **paired-bootstrap** CI"* |
| `research_roadmap.md` §4 Phase 5 | *"**paired** bootstrap for the rest"* |
| `research_roadmap.md` §8 rule 4 | *"**Pair** every comparison by question"* |
| `scoring/calibration.py:30` | `bootstrap_ci(values, n_boot, confidence)` — flat |

**Pairing is not clustering.** Pairing means joint and post-hoc are compared on the same question.
Clustering means the *resampling unit* is the question. Citation-F1 is defined in `CONTEXT.md` as the
harmonic mean of **corpus-level** precision and recall — micro-averaged over claims — so a "paired
bootstrap" over that quantity most naturally resamples claims, which is the error.

**The cost is explicit: clustering widens every interval, including the one G2 gates on.** At ~9.2
claims/question, correlated claims currently inflate the effective *n* by up to an order of
magnitude. **G2's threshold is unchanged**, so this raises the bar on the headline gate four weeks
before it is measured. That cost is paid knowingly, because CIs that are too narrow are a defect a
reviewer will find and we cannot.

**W4 dry-run:** on the first real end-to-end records, compute **both** the clustered and unclustered
interval and record the width difference. The point is to see it in August rather than at the gate.

### 3. G4 gates on the **point** estimate: α ≥ 0.6

The clustered CI is **always reported, with its cluster count stated**, and is never used to soften
the number.

**The 0.4 lower-bound trigger is dropped.** It was internally inconsistent: it declared the CI too
unstable to gate on and then used that same CI as a hard trigger one sentence later. If the interval
is trustworthy enough to force a re-pilot, it is trustworthy enough to gate; if it is too unstable to
gate, it is too unstable to trigger.

§2 makes this worse rather than better. Clustering over ~19 questions widens the α interval
substantially, so a point α of 0.65 with a clustered CI of [0.38, 0.85] would **pass the gate and trip
the re-pilot simultaneously** — two decisions interacting to produce a trigger that fires at
perfectly acceptable agreement.

It was also **duplicating a contingency that already exists**. `research_roadmap.md` §4 Phase 4:

> *If α < 0.6:* the guidelines are ambiguous, not the annotators. Revise guidelines, re-annotate the
> overlap. Report the final α whatever it is.

That contingency triggers on the point estimate, prescribes the right response, and is now the
**sole** re-pilot trigger.

### 4. The G1/G4 asymmetry is defended in writing, not left to be noticed

G1 gates on **point ≥ 0.90 and Wilson lower > 0.85**. G4 gates on the point alone. Stated plainly so
the difference reads as a reason rather than a convenience:

> G1's gate is a **proportion over 100 independent dev questions**, which supports a Wilson bound.
> G4's is **Krippendorff's α over ~19 question-clusters**, where the bootstrap interval is wide enough
> that gating on it would make the gate a coin flip rather than a standard. We gate on the point and
> report the interval with its cluster count, so a reader can see exactly why it is wide.

## Consequences

- **`scoring/calibration.py`'s `bootstrap_ci` grows a clustering parameter** — it currently takes a
  flat sequence of values and cannot express the question grouping. Due W7, but the signature change
  lands with the W4 dry-run.
- **G2 is harder to pass than it was yesterday**, on purpose. R5's response is unchanged.
- **Every table caption naming a CI must name the resampling unit**, or the number is unreadable.
- **α's reported interval carries its cluster count** (~19), not only its bounds.
- **W9 pressure.** A re-pilot triggered at G4 (Sep 27) lands in W9 — which now also holds ADR-0009's
  stratified robustness check *and* its original role as the absorber for Phase 3–5 slippage. **Three
  claims on the only slack week in the schedule.** Unresolved, and named here so it is not
  rediscovered in September.

## Known weaknesses

1. **The annotator-hours estimate is unverified** at 19 overlap questions — see §1. This is the item
   most likely to be wrong, and it is on the critical path (R3b).
2. **~19 clusters is few.** Clustering is correct and the interval it produces is honest, but it will
   be wide, and no amount of correctness narrows it. Widening the overlap subset would — at the cost
   of annotator hours we are already unsure about.
3. **Sampling 4 of ~9 claims per answer forecloses per-answer gold statistics.** A per-answer
   supported-fraction cannot be computed from the gold set. C3's hallucination rate runs off the
   verifier over all claims, so this is believed harmless — **confirm before W6.**

## Alternatives rejected

- **All claims of ~27 questions.** Maximises claims per question, minimises question diversity, and
  leaves the overlap subset at ~8 clusters — too few for §2 to mean anything.
- **Clustered for G4 only, paired-as-before for G2.** Defensible on the grounds that G2's unit
  genuinely is the claim, but it puts two resampling policies in one paper, which reads as
  convenience.
- **Keep the 0.4 CI trigger.** Safest-looking on paper; most likely to consume W9 for no gain.
- **Widen the overlap subset so the α interval narrows.** Statistically the real fix to §3, and the
  one to reach for *if* annotator capacity turns out larger than ADR-0006 assumed.
