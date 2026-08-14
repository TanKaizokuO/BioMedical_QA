# The first citation-F1 — ADR-0009 §6's unblinding read, 2026-08-14

The parity loop closed on `parity_iter1b` (see `parity_iter1b.md` and ADR-0009's *Termination*), so
citation-F1 is computable for the first time. This is the **R5 early-warning read**, seventeen days
ahead of §6's ≈Aug 31 estimate.

```
uv run python scripts/first_citation_f1.py docs/harvest/parity_iter1b --n-boot 2000 --max-tokens 3584
```

Artifact: `parity_iter1b.citation_f1.json`. Intervals resample **questions** (ADR-0011 §2,
`calibration.bootstrap_ci`, 2000 draws); the delta is paired — both arms are scored on the drawn
questions inside the statistic.

## Read the caveats before the number

1. **φ is not MiniCheck.** `verify.py` is W6 (Sep 7–20) and raises today. φ here is
   `cross-encoder/nli-deberta-v3-xsmall` at `argmax == entailment` — φ in `notebooks/03_2` and `06_5`,
   named in `verify.py` as the thing MiniCheck replaces. 4,904 (premise, hypothesis) pairs, CPU.
2. **The records are a smoke run**, not a sample: 100 dev questions at `--max-tokens 3584`, whose own
   summary says *"not a gate run and not a sample"*. Vanilla is excluded (ADR-0010).
3. **The levels are not interpretable; the contrast is.** R7 predicted an ANLI-trained φ degrades on
   biomedical text and it does — recall 0.15–0.22 against precision 0.87–0.90. φ under-entails, which
   deflates recall directly and *inflates* precision through the "a lone non-entailing citation is not
   irrelevant" rule (`CONTEXT.md`). **Nothing here is a G2 number.**

## The number

| system | precision | recall | **F1** | 95% CI (question-clustered) | claims | citations |
|---|---|---|---|---|---|---|
| joint | 0.902 | 0.154 | **0.264** | [0.205, 0.331] | 719 | 1061 (957 not irrelevant) |
| post_hoc | 0.866 | 0.215 | **0.345** | [0.286, 0.403] | 1242 | 1807 (1564 not irrelevant) |

**joint − post_hoc F1 = −0.081, 95% [−0.157, +0.005] on 100 paired questions.**

**C2's direction is not established by this read, and the point estimate runs against it.** The
hypothesis is that joint generation grounds better than citing after the fact; here the baseline is
ahead by 8 F1 points, with an interval that only just includes zero. This is what §6's pre-armed R5
trigger exists for — arriving early rather than on Aug 31, which is the one piece of good news in it.

## Three checks, before anyone attributes this to the granularity residual

The obvious objection is that the parity loop left post-hoc's claims **coarser** (+13.3% median
words/claim) and coarser claims are harder to entail per claim — so the residual should penalise
post-hoc, not favour it. It does not explain the result, and it points the wrong way.

### Per-claim recall by claim length

| band (words) | joint n | joint recall | post_hoc n | post_hoc recall |
|---|---|---|---|---|
| 0–10 | 101 | 0.317 | 105 | 0.371 |
| 11–15 | 266 | 0.169 | 426 | 0.178 |
| 16–20 | 255 | 0.090 | 365 | **0.189** |
| 21–30 | 63 | 0.175 | 307 | 0.248 |
| 31+ | 34 | **0.000** | 38 | 0.184 |

**Post-hoc is ahead in every band**, so the contrast survives conditioning on the quantity ADR-0009
gates. The gap is widest at 16–20 words — where both medians sit — and joint entails **nothing** above
30 words, which is the runaway-claim pathology `parity_iter1b.md` recorded, not a granularity effect.

### The basis that excludes joint's runaway records

Dropping the 22 questions where either arm hit the output cap (the symmetric basis from
`parity_iter1b.md`):

| basis | joint | post_hoc | delta |
|---|---|---|---|
| all 100 questions | 0.264 | 0.345 | −0.081 |
| **78 untruncated questions** | **0.303** | **0.376** | **−0.073** |

Both arms improve and the gap barely moves. **Joint's deficit is not the degenerate records.**

### Citation budget and abstention

| | joint | post_hoc |
|---|---|---|
| citations per claim (mean) | 1.48 | 1.45 |
| uncited claims | 192 / 719 = **26.7%** | 384 / 1242 = **30.9%** |
| abstentions (ADR-0010, derived) | 0 | 1 |

Neither arm is buying recall with citations — the ≤3 cap is doing its job and the two arms sit on the
same budget. Post-hoc leaves a *larger* share of claims uncited, and an uncited claim scores recall 0,
so this too runs against post-hoc while post-hoc still wins.

## The leading alternative explanation, named and not yet tested

**Post-hoc quotes longer spans: median 23 words per citation against joint's 19** (mean 24.1 vs 21.0).
A longer premise is more likely to entail under a sentence-pair NLI model, so part of the −0.081 may
be φ's length sensitivity rather than grounding quality. This is the specific thing MiniCheck is
supposed to fix — it is built for document-level premises — so it is the first quantity to re-read at
W6, and the reason the contrast is *not* being called a result yet.

Two lesser ones, recorded: joint's φ pairs include 34 claims over 30 words that entail nothing, and
the abstention rule fires once in 1,961 claims, so `recall` and `recall_all_claims` are the same number
on this run and ADR-0010's second denominator carries no information here.

## What this changes

- **R5's trigger is live as an early warning.** §6 accepted a six-day window between the first F1 and
  G2 (Sep 6); the loop terminating at iteration 1 turned that into **twenty-three days**. The
  contingency does not have to be improvised.
- **Nothing about the parity loop is reopened.** The template is frozen at the terminating run
  (`prompts.PARITY_LOOP_CLOSED.post_hoc_answer_template_sha256`, checked in `tests/test_prompts.py`).
  Tuning post-hoc now would be tuning with F1 known, which is what §6 was protecting.
- **The W9 stratified robustness check keeps its mandate** and gains a preview: the length-band table
  above is the same conditioning it will do properly, and it says the contrast is not granularity.
- **The joint runaway-claim defect is now load-bearing**, not cosmetic: joint entails 0 of 34 claims
  over 30 words. Fixing the splitter and the non-terminating generation is W5/W6 work that moves a
  reported number, so it happens **before** the G2 read.
