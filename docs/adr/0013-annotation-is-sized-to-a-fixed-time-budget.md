# ADR-0013 — Annotation is sized to a fixed time budget, not a fixed claim count

**Status:** Accepted · **Date:** 2026-08-05 · **Decided in:** grilling session (annotator-hours re-derivation)
**Refines** ADR-0006 (annotation protocol) and ADR-0011 §1 · **Constrains** G4's cluster count

## Context

ADR-0011 §1 left one item open in writing: ADR-0006's **~3 h** annotator ask was derived for ~75
claims and **never re-derived** for the ~19 overlap *questions* the new sampling implies. Its Known
weakness 1 called it *"the item most likely to be wrong,"* on the critical path (R3b). This ADR
closes it.

**The re-derivation found two errors of opposite sign, which nearly cancelled.**

`research_roadmap.md` recorded the original basis verbatim: *"75 overlap claims ≈ 150–225 pair
judgements plus 75 union judgements. This is why the ask is ~3 h, not ~1–2 h."* So the **pair count
was costed**; the **question count never was**.

| | assumed | measured | source |
|---|---|---|---|
| citations/claim | 2–3 | **1.01** | 92 claims in `runs/g0/bakeoff_…llama…json` |
| overlap questions | 8 (implicit) | **19** | ADR-0011 §1 |

The pair count was overstated ~3×; the question count understated 2.4×. G0's prompt did not suppress
multi-citation — `prompt_template` offers `[n] or [n][m]` at `max_citations: 3` — though its passages
were sections of a **single** abstract, which is not the retrieval regime annotators will see.

### The measurement that drives this ADR

Distinct cited passages per question, resampled 4,000× per question from the G0 answers:

| claims sampled/question | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| distinct cited passages | **1.52** | 1.82 | **2.03** | 2.18 |

**Per-question cost is sublinear in claims; per-claim cost is linear.** Reading a question's passage
set is a near-fixed toll; judging a claim is not. Everything below follows from that asymmetry.

### Model and assumptions

Assumptions are stated because unstated ones are how the original estimate went wrong.

| | |
|---|---|
| first-pass comprehension, unfamiliar biomedical prose | **120 wpm** |
| re-reading a span already seen | **200 wpm** |
| claim text | **150 wpm** |
| 4-way `support_label` + `claim_validity` + recording | **36 s** |
| question + topic orientation | **30 s** |
| fatigue/breaks | **+15%** |
| first read of the guidelines | **+30 min**, one-off |

Measured inputs: claim 14.4 words (G0); full abstract 200 words, sectioned chunk 60 words, question
12.9 words (`pqa_labeled`, n=1000).

```
per claim      = 5.8 s (read) + 18.0 s (span) + 36.0 s (judge) = 60 s × cit/claim, + union
per question   = 30 s + (distinct passages × 200 w ÷ 120 wpm)
```

**Calibration check.** The same model applied to ADR-0006's own design (8 questions, all ~9.4 claims,
abstract-level chunks) returns **2.9 h** — it reproduces the number it was built to test. Applied to
ADR-0011's 4×19 design it returns **3.6 h**. The design change costs **+0.7 h**.

## Decision

### 1. The annotator ask is a **ceiling**, not an estimate: 3 h total

The largest input — citations/claim — cannot be measured in the regime that matters until the
**decomposer freezes Sep 3** (W5), after any plausible acceptance date. A point estimate is therefore
a promise we cannot keep: at 2.0 citations/claim the 4×19 design costs **5.9 h**.

So the fixed quantity is **time**, and the subset is sized to it. The ask can never be revised upward,
because the design absorbs the overrun instead of the annotator.

**3 h total, in two sittings:**

| | |
|---|---|
| pilot sitting (W6) | **~1.0 h** — 0.5 h first read of the guidelines + 10 practice claims across ~5 questions |
| main pass (W7–W8) | **~2.0 h** |

**The W6 pilot was never inside ADR-0006's ~3 h.** `research_roadmap.md` §5 W6 schedules a 10-claim,
3-annotator pilot; ADR-0006's table costs *"overlap only (~75)"* and nothing else. Neither was the
first read of the guidelines. This is an uncosted session, not a mis-estimate, and it is why the
total is stated as two sittings rather than one number.

### 2. The overlap subset is **2 claims per question**; questions are the protected dimension

When the budget binds, claims-per-question is cut and question count is preserved. This is the plan,
not a fallback — at equal cost it is a better design than ADR-0011's 4×19.

Main pass of 2.0 h = 7,200 s ÷ 1.15 = **6,261 s effective**; block = 182 s + 2 × per-claim:

| citations/claim | questions | claims |
|---|---|---|
| **1.01** (measured) | **19** | 38 |
| 1.5 | 15 | 30 |
| **2.0** (pessimistic tail) | **12** | 24 |

The likely case lands on the ~19 clusters ADR-0011 §2 assumed. Cutting the *other* dimension would
give 12 clusters in the likely case and worse in the tail — inside the range ADR-0011 §1 calls
*"close to meaningless."*

**This answers ADR-0011's Known weakness 2**, which wanted a wider overlap *"at the cost of annotator
hours we are already unsure about."* It costs no hours. Those hours were being spent re-reading the
same abstract for a question's third and fourth claim.

The **primary** keeps 4 claims/question. The asymmetry is not an inconsistency: the primary's budget
is claims (G4 requires ≥250), so more questions means more work — 2×125 costs 12.9 h against 4×62's
10.3 h. Fewer claims/question helps only when the budget is time.

### 3. The work is stoppable at any point, with no loss

Every completed question is a usable cluster; there is no all-or-nothing. This is guaranteed by
design, not by goodwill:

> **Annotators 2 and 3 work the same randomized question order, question by question.** Any common
> prefix is then a complete, unbiased random subsample, whatever point either of them stops at.

**This is a W5 requirement on the annotation UI and on `data.py`'s overlap ordering**, not a
courtesy — the guarantee is void if the two orders differ or if a question is left half-labeled.

### 4. The primary's ask is **~8–14 h**, not ADR-0006's 4–10 h, and it starts in W6

Same model, 62 questions × 4 claims: 62 × 233 s + 250 × 71 s, ×1.15 = **10.3 h**; **8.2 h** with a 25%
expertise discount; **13.7 h** at 2.0 citations/claim. **The stated 4 h lower bound is unreachable**
under any assumption — it implies 55 s per claim including all passage reading.

`research_roadmap.md` §5 put annotation "in progress" in W7 and "completes" in W8 — the same week as
code freeze, the backend decision, and seed-1 runs, with G4 on Sep 27, and W9 already triple-booked
(ADR-0011). **The primary pass moves to W6**, alongside the pilot. Phase 4 already licenses this:
*"start the moment Phase 2 produces stable outputs,"* and the decomposer freezes Sep 3.

## Consequences

- **G4's reported cluster count is no longer fixed at ~19** — it is whatever the budget bought, and it
  is reported as measured, with α's interval alongside it (ADR-0011 §3, §4 unchanged).
- **`data.py` gains the overlap ordering**, seeded and shared across annotators 2 and 3. Due W5.
- **The annotation UI must record per-question completion**, so a partial set is separable from a
  half-finished question. A static HTML form writing JSONL still suffices.
- **The pilot's 10 claims cannot be counted toward the main pass.** ADR-0006's contingency revises the
  guidelines on poor pilot α, which invalidates anything labeled under the old ones.
- **Nothing changes in `CONTEXT.md`.** The annotation record, the 4-way label, the union judgement and
  the ≤3 citation cap are untouched.
- **Issue #7 is an internal record.** The repo is private; the two annotators cannot read it. The
  ask reached them on another channel, and corrections go there.

## Known weaknesses

1. **The 12-cluster tail is bad.** If citations/claim reaches 2.0, the main pass buys 12 clusters —
   inside the range ADR-0011 §1 disparages. It is the priced cost of §1, accepted because two
   annotators lost before Sep 7 are not recoverable and ADR-0006's fallback is explicitly weaker.
   The tail assumes double the citations/claim G0 measured.
2. **1.01 citations/claim is measured in the wrong regime.** G0's passages were sections of one
   abstract; real retrieval gives five chunks from five documents plus distractors. Direction of the
   error is genuinely unknown — atomic claims argue for 1, heterogeneous passages argue for more.
   **Re-measure on the first end-to-end records (W4) and re-run §2's table**, before the W5 UI work.
3. **The chunker is undecided** (Phase 1). Abstract-level (200 w) vs section-level (60 w) chunks swing
   the per-question component by ~0.85 h at the 4×19 design. §2's table assumes abstract-level, the
   conservative case.
4. **Context-switching is charged at 30 s/question.** At 2 claims/question that fixed cost is
   amortised over half as much work, so if real switching is worse the §2 advantage shrinks. An
   annotator answering two questions per abstract may also find it more tedious than one answering
   four — a retention risk the arithmetic cannot see.
5. **38 claims is fewer than ADR-0006's 75.** α's point estimate is noisier, not only its interval.
   ADR-0006 already conceded *"on ~75 units it is not a point estimate."*

## Alternatives rejected

- **Send a revised point estimate (~4 h).** Recommended first, then abandoned: it swaps one number
  that may be wrong for another that may be wrong, and a *second* upward revision is worse than the
  first. Moot in any case once both annotators accepted — ADR-0011 §1 forbids revising upward then.
- **Send an honest range, "3–5 h."** Truthful; people anchor on the top of a range when deciding
  whether to volunteer, and a 67%-wide range reads as disorganised.
- **Cut questions, hold 4 claims/question.** Keeps the overlap identically shaped to the primary set —
  one less thing to explain in the paper. Costs clusters exactly when they are scarcest.
- **Say nothing and let 3 h stand.** The pilot would then be a surprise session in September, to
  annotators who had already raised concerns about the commitment.
- **Recruit a third and fourth annotator at ~1.5 h each.** Krippendorff's α tolerates variable raters
  per unit, so splitting the overlap across two pairs is legitimate. Costs recruiting lead time that
  does not exist before Sep 7. **This is the lever to reach for if capacity turns out larger.**
- **Amend ADR-0011 §1 in place.** No ADR in this repo has been edited after acceptance; every
  cross-reference lives in the newer ADR's header. §1's stale open note is the provenance of this
  document and is deliberately left standing.
