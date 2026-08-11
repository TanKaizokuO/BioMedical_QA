# ADR-0016 — All three annotators label the full gold set

**Status:** Accepted · **Date:** 2026-08-11 · **Decided in:** annotation-capacity upgrade
**Supersedes** ADR-0013 §1 (the 3 h ceiling) and §2 (the overlap subset) · **Generalises** ADR-0013 §3
**Refines** ADR-0006 and ADR-0011 · **Constrains** G4's cluster count and the W5 annotation UI

## Context

ADR-0006 split the work by role: the primary labels everything, annotators 2 and 3 label an overlap
subset, and α is estimated on the overlap. ADR-0011 spread the 250 claims over ~62 questions, which
made the overlap ~19 questions. ADR-0013 found the ~3 h ask had never been costed for that question
count, inverted the design so that **time** was fixed and the subset sized to it, and landed on
2 claims × ~19 questions.

Every one of those decisions was downstream of one constraint: **two volunteers, ~3 h each.**

That constraint has changed. All three annotators have agreed to label the complete set. The
question is no longer how to buy the most reliability per annotator-hour — it is what to do with
roughly ten times the annotator capacity the design was built around.

## Decision

**All three annotators independently label the full ≥250-claim gold set.** There is no overlap
subset, because the overlap *is* the set.

```
Annotator 1 (primary)  → all ~250 claims  (4 claims × ~62 questions)
Annotator 2            → the same ~250 claims
Annotator 3            → the same ~250 claims
```

≈750 annotator-level claim evaluations; per annotator, ~250 union judgements plus one span
judgement per (claim, cited span) pair.

Unchanged from ADR-0006 and `CONTEXT.md`, and not reopened here: three annotators, two deliberately
non-expert, the no-outside-knowledge rule, span-grounded judgement of attribution rather than truth,
the four-way `SUPPORTED`/`PARTIAL`/`NOT_SUPPORTED`/`CONTRADICTED` label set, `claim_validity`, the
per-claim union judgement, never collapsing at write time, and real Krippendorff's α.

### 1. What it costs, stated before it is spent

ADR-0013's cost model, unchanged, applied to 62 questions × 4 claims per annotator
(`(62 × (30 + distinct×100) + 250 × (60·c + 10)) × 1.15`, distinct = 2.03 passages at 4
claims/question):

| citations/claim | h / annotator | h total |
|---|---|---|
| 1.01 (ADR-0013's G0 measurement) | 10.3 | 30.8 |
| 1.13 (run 4, post_hoc, parsed) | 10.8 | 32.5 |
| 1.50 (run 4, joint, parsed) | 12.6 | 37.8 |
| 1.87 (run 4, post_hoc, emitted) | 14.4 | 43.1 |
| 2.30 (run 4, joint, emitted) | 16.4 | 49.3 |

**10–16 h per annotator**, against the 3 h the previous design was sized to. ADR-0013 §4's 25%
expertise discount applies to the primary only; annotators 2 and 3 are non-expert **by design**, so
they get the undiscounted number. This is the priced cost of the upgrade and it is accepted, not
mitigated.

### 2. The shared randomized order makes this a strict superset of ADR-0013

ADR-0013 §3 guaranteed stoppability for two annotators. It now applies to all three and becomes the
load-bearing property of the design:

> **All three annotators work the same seeded randomized question order, question by question.** Any
> common prefix across the three is a complete, unbiased random subsample of the gold set, whatever
> point any of them stops at.

The consequence is why this upgrade carries no methodological risk. If capacity turns out smaller
than agreed, the triple-labeled common prefix shrinks — and at 3 h each it shrinks to roughly the
~19 questions ADR-0013's design bought deliberately. **The floor of the new design is the ceiling of
the old one.** Nothing is staked on the expanded commitment being met in full.

This is void if the three orders differ, or if a question is left half-labeled. It is a **W5
requirement on `data.py` and the annotation UI**, not a courtesy, and the UI must record per-question
completion so a partial pass is separable from a half-finished question.

### 3. G4, restated

> **≥250 claims labeled, and point Krippendorff's α ≥ 0.6 on the binary collapse**
> (`SUPPORTED|PARTIAL` vs `NOT_SUPPORTED|CONTRADICTED`), computed over the **triple-labeled common
> prefix** — the full set when all three finish, which is the target.

Reported alongside, none of them able to soften the point estimate: the 4-way ordinal α; the
clustered bootstrap CI **with its cluster count**; the no-majority rate; the human ceiling; the
decomposition-error rate from `claim_validity`; and per-annotator label distributions, which
full-set coverage makes informative for the first time.

The bootstrap still **resamples questions, not claims** (ADR-0011 §2). Three labels per claim do not
make claims independent — they share a question, a passage set, an answer and a topic. What the
upgrade buys is clusters: ~62 instead of ~19, so the interval narrows by roughly $\sqrt{62/19}
\approx 1.8\times$. That is the real gain, and it is a gain in **precision**, not in accuracy.

Disagreements are **never adjudicated before α is computed**. Raw per-annotator judgements are
preserved; `HumanLabel.annotator_id` already carries them. Adjudication, if it happens at all, is a
separate downstream artifact.

### 4. Blinding becomes load-bearing

Under ADR-0006 the primary labeled everything and the overlap estimated agreement between two
non-experts and the author. Now the author is one of three raters on identical units, and α is
computed across all three. The author knows the hypothesis and, for any claim, can often infer which
system produced it.

**The annotation UI presents claims stripped of system, model and run identity, in a seeded shuffle
that interleaves systems**; the mapping from annotation unit to `(system, run_id, query_id)` lives
outside the annotation file and is joined at scoring time. Annotators see the question, the claim,
and the cited spans — nothing else. No annotator sees another's judgements, or any aggregate over
them, before finishing.

### 5. The pilot is unchanged

10 claims × 3 annotators in W6, testing the **guidelines** — the `SUPPORTED`/`PARTIAL` boundary
first, plus jointly-necessary citations and a sampled abstention claim. Its claims cannot count
toward the main pass if the guidelines are revised afterwards. Guidelines are finalised before the
main pass opens.

## Consequences

- **The primary's pass is no longer the only complete one**, so the gold label for a claim is a
  choice among three rather than a single annotator's judgement. That choice is a scoring-time
  decision (majority, or the primary's label, with the no-majority rate reported either way) and it
  is made **after** α, never before.
- **`data.py` owes one seeded order shared by all three annotators**, not an overlap ordering. Due
  W5.
- **The annotation UI owes blinding and per-question completion.** A static HTML form writing JSONL
  still suffices; this is not a licence to build a tool.
- **`scoring/agreement.py` is unchanged in shape** — `krippendorff_alpha_binary` and
  `krippendorff_alpha_ordinal` over `Sequence[Sequence[HumanLabel]]` already accept three raters per
  unit. Only the docstring's "overlap subset" wording was stale.
- **R3 (gold annotation slips) becomes more likely, not less.** 30–49 annotator-hours must land in
  Sep 7–27. §2 is what makes the downside graceful, but the tripwire is now explicit: **if the
  triple-labeled common prefix is under ~19 questions on Sep 20**, the schedule is behind where the
  *superseded* design would have been, and G4 is at risk.
- **Nothing changes in `CONTEXT.md`.** The unit, the label set, the union judgement and the
  no-outside-knowledge rule are untouched — this ADR changes who labels how much, not what a label
  means.

## Known weaknesses

1. **α gains precision, not accuracy.** If the guidelines are ambiguous at the `SUPPORTED`/`PARTIAL`
   boundary, 62 clusters measure that ambiguity more precisely than 19 did. The pilot is still the
   only instrument that can catch it in time, and it is still 10 claims.
2. **Fatigue drift is now correlated across annotators.** Ten-plus hours of one judgement task will
   drift, and a *shared* question order means all three drift on the same units in the same
   direction — which inflates α rather than depressing it. §2's shared order is load-bearing for
   stoppability, so the answer is measurement, not redesign: the UI records per-question timestamps,
   and α is reported for the first and second halves of the order separately.
3. **Non-experts are charged the primary's per-claim rate.** ADR-0013's model was calibrated on the
   primary's own design; 10–16 h may itself be optimistic for a non-expert reading unfamiliar
   biomedical prose. The direction of that error is not favourable.
4. **250 claims are still ~62 clusters.** Tripling raters does not add units. Anyone reading "750
   judgements" as an effective sample size will be wrong, which is why the cluster count is reported
   next to every interval.
5. **The expanded commitment is not yet evidenced by delivered work.** It is an agreement, and
   agreements to 10–16 h of unpaid careful reading are the kind that erode. §2 is the entire
   insurance policy.

## Alternatives rejected

- **Keep the overlap-only design.** Declines free capacity that materially narrows G4's interval,
  for no methodological gain.
- **Independent random orders per annotator.** More robust to correlated fatigue drift (weakness 2),
  and fatal to §2: with different orders an early stop leaves ragged coverage, and the intersection
  stops being a clean random subsample of questions. Stoppability is worth more than drift
  resistance, because only one of the two threatens the gate.
- **Spend the extra capacity on more claims instead of more raters** (e.g. 500 claims, overlap-only).
  A bigger gold set with a weaker reliability estimate. C4 is an agreement claim; the raters are
  where the capacity belongs.
- **Adjudicate disagreements, then report α on the adjudicated set.** Reports the agreement of a
  process rather than of people, and inflates it. Adjudication stays downstream of α.
- **Amend ADR-0013 in place.** No ADR in this repo is edited after acceptance. ADR-0013's §1 and §2
  remain correct for the constraint they were written under, and this header records the change.
