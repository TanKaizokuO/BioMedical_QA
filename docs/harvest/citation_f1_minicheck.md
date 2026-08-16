# The citation-F1 re-read with the real verifier — MiniCheck, 2026-08-17

`verify.py` landed early (ADR-0020, commit `f447d27`), so the read
`first_citation_f1.md` deferred to W6 is available now. **This is the same read, not a new one.**

```
uv run python scripts/first_citation_f1.py docs/harvest/parity_iter1b \
  --phi minicheck --threshold 0.5 --batch-size 16 --n-boot 2000 --max-tokens 3584
```

Artifact: `parity_iter1b.citation_f1.minicheck.json`, beside the 2026-08-14
`parity_iter1b.citation_f1.json`.

**The records are unchanged.** Same `parity_iter1b` smoke run, same 100 dev questions, same
`--max-tokens 3584`, vanilla still excluded by ADR-0010. No generation was re-run, no prompt moved,
no template was touched — the parity loop stays frozen at its terminating run. **φ is the only thing
that moved**, from `cross-encoder/nli-deberta-v3-xsmall` at `argmax == entailment` to
`lytang/MiniCheck-Flan-T5-Large` at support probability ≥ 0.5, invoked the way the reference does
(ADR-0020: `"predict: {document}</s>{claim}"`, one decoder step, softmax over ids 3 and 209).
4,904 (premise, hypothesis) pairs, the same 4,904 as before.

## Read the caveats before the number

1. **Still a smoke run, still not a gate run.** 100 dev questions at `--max-tokens 3584`, whose own
   summary says *"not a gate run and not a sample"*. Swapping φ does not turn a smoke run into a
   sample. **Nothing here is a G2 number**, for the same reason as on Aug 14.
2. **τ = 0.5 is MiniCheck's own binarisation, not a tuned value.** It is the natural cut on a support
   probability, chosen before the numbers were seen and not fitted to them. The score distribution
   makes clear how much rides on it: over 4,904 pairs, mean 0.2873, **median 0.0553**, and the mass
   at or above 0.1 / 0.3 / 0.5 / 0.7 / 0.9 is 0.4074 / 0.2991 / **0.2667** / 0.2353 / 0.1827. The
   distribution is decisive rather than graded — most pairs sit near zero, and most of what clears
   0.1 goes on to clear 0.9 — so 0.5 is a cheap place to stand, but it is a *place*, and §5 below is
   what that costs.
3. **R7 is not retired by this swap.** R7 predicted a general-domain φ degrades on biomedical text.
   MiniCheck is a general-domain fact-checker, and the same under-entailing shape is still here —
   **recall 0.277–0.284 against precision 0.846–0.872** — one notch milder than the old φ's
   0.15–0.22 against 0.87–0.90, and in the same direction. Under the "a lone non-entailing citation
   is not irrelevant" rule (`CONTEXT.md`), that still deflates recall directly and inflates
   precision. The levels remain uninterpretable; the contrast is what is being read.

## The number

| system | precision | recall | **F1** | 95% CI (question-clustered) | claims | citations |
|---|---|---|---|---|---|---|
| joint | 0.872 | 0.284 | **0.428** | [0.316, 0.540] | 719 (0 abstentions) | 1061 (925 not irrelevant) |
| post_hoc | 0.846 | 0.277 | **0.418** | [0.357, 0.472] | 1242 (1241 answered, 1 abstention) | 1807 (1528 not irrelevant) |

**joint − post_hoc F1 = +0.011, 95% [−0.117, +0.137] on 100 paired questions** (2000 draws,
resampling questions, delta computed inside the statistic so both arms are scored on the drawn
questions). Post-hoc's ADR-0010 second denominator changes almost nothing: `recall_all_claims`
0.277 gives F1 0.417 against 0.418, because the abstention rule fires once in 1,961 claims.

## What changed is φ, and only φ

| quantity | φ = nli-deberta-v3-xsmall (Aug 14) | φ = MiniCheck (Aug 17) |
|---|---|---|
| joint precision / recall / **F1** | 0.902 / 0.154 / **0.264** | 0.872 / 0.284 / **0.428** |
| joint 95% CI | [0.205, 0.331] | [0.316, 0.540] |
| post_hoc precision / recall / **F1** | 0.866 / 0.215 / **0.345** | 0.846 / 0.277 / **0.418** |
| post_hoc 95% CI | [0.286, 0.403] | [0.357, 0.472] |
| **joint − post_hoc** | **−0.081** [−0.157, +0.005] | **+0.011** [−0.117, +0.137] |
| untruncated 78-question delta | −0.073 | +0.012 |

**The −0.081 became +0.011. The sign flipped, and the records did not change.** Both arms gained
recall (joint +0.130, post-hoc +0.062) at a small precision cost, and joint gained roughly twice as
much as post-hoc, which is the whole of the flip.

**What that licenses.** `first_citation_f1.md` named exactly one leading alternative explanation and
declined to call the contrast a result until it was tested: *post-hoc quotes longer spans — median 23
words per citation against joint's 19 — a longer premise is more likely to entail under a
sentence-pair NLI model, so part of the −0.081 may be φ's length sensitivity rather than grounding
quality; MiniCheck is built for document-level premises and is the specific thing that should fix
it.* That hypothesis is now supported by evidence rather than merely named: replacing the
sentence-pair φ with a document-level one removed the deficit, so **the placeholder φ was carrying
it.** The pre-registered prediction and the measurement agree, which is the strongest form this
evidence can take on unchanged records.

**What that does not license.** **C2's direction is still not established.** The interval
[−0.117, +0.137] straddles zero, and it is **wider** than the one it replaces — 0.254 against 0.162.
The read did not move a negative point estimate to a positive one *with the ambiguity resolved*; it
moved a nearly-significant negative to a firmly ambiguous positive. Aug 14's honest statement was
"C2's direction is not established by this read, and the point estimate runs against it." Today's is
"C2's direction is not established by this read, and the point estimate no longer runs against it."
Only the second clause changed.

## The threshold sweep is why this is not being called a result

All five rows come from **one scoring pass** — MiniCheck emits a probability per pair, so every τ is a
re-thresholding of the same 4,904 scores. The sweep costs nothing and hides nothing; it is reported
because it was free, and it would have been reported whichever way it came out.

| τ | joint F1 | post_hoc F1 | delta |
|---|---|---|---|
| 0.1 | 0.526 | 0.513 | **+0.013** |
| 0.3 | 0.437 | 0.444 | **−0.007** |
| 0.5 | **0.428** | **0.418** | **+0.011** |
| 0.7 | 0.352 | 0.388 | **−0.036** |
| 0.9 | 0.287 | 0.329 | **−0.041** |

Point estimates only, no bootstrap. **The sign is +, −, +, −, − across the sweep, and post-hoc leads
at τ ≥ 0.7** — by more, in fact, than joint leads anywhere. The headline +0.011 sits at the one
uncontested-looking point of a sequence that changes sign three times, and it is smaller than the
largest excursion against it. Nobody should read the +0.011 as a direction while that is true.

Two things this does not mean. It does not mean τ = 0.5 was chosen to favour joint: 0.1 favours it
more, and 0.5 is φ's own cut. And it does not mean the sweep should be searched for the best row —
**G3 (Sep 20) is what picks an operating point**, on verifier AUROC against an unsupported-claim
label, which is a criterion external to C2's contrast. Choosing τ here from these five F1 values
would be choosing a verifier threshold with the contrast it decides already known.

## Per-claim recall by claim length

| band (words) | joint n | joint R (MiniCheck) | post_hoc n | post_hoc R (MiniCheck) | joint R (old φ) | post_hoc R (old φ) |
|---|---|---|---|---|---|---|
| 0–10 | 101 | 0.446 | 105 | 0.467 | 0.317 | 0.371 |
| 11–15 | 266 | **0.361** | 426 | 0.232 | 0.169 | 0.178 |
| 16–20 | 255 | 0.157 | 365 | **0.279** | 0.090 | 0.189 |
| 21–30 | 63 | **0.317** | 307 | 0.270 | 0.175 | 0.248 |
| 31+ | 34 | 0.088 | 38 | **0.289** | 0.000 | 0.184 |

Under the old φ, **post-hoc led in every band** — which is what made the granularity objection
answerable and the length-sensitivity hypothesis merely a hypothesis. Under MiniCheck the table is
mixed: **joint leads at 11–15 (0.361 against 0.232) and at 21–30 (0.317 against 0.270)**, and loses
badly at 16–20 (0.157 against 0.279) and 31+ (0.088 against 0.289). A uniform ordering became a
split one, so the aggregate +0.011 is not a level shift applied evenly; it is joint winning the two
bands where its claim mass sits and still losing the long tail.

**Joint's 31+ band is still its worst, and it is still joint's defect, not φ's.** 0.088 against
post-hoc's 0.289, on 34 claims — the old φ read 0.000 on the same 34. MiniCheck can now find support
for three of them where the sentence-pair φ found none, which is a φ improvement; the three-to-one gap
against post-hoc on the same band is not. This is the runaway-claim pathology `parity_iter1b.md`
recorded with a query id (`21074975`, a single 731-word "claim" from a non-terminating `and …, and …`
loop, plus a splitter that does not split it). **It survives the φ swap**, which is the evidence that
it was never a measurement artifact. Fixing the splitter and the non-terminating generation still
moves a reported number, and still has to happen before the G2 read.

## The untruncated basis

Dropping the 22 questions where either arm hit the 3584-token output cap (the symmetric same-queries
basis from `parity_iter1b.md`):

| basis | joint | post_hoc | delta |
|---|---|---|---|
| all 100 questions | 0.428 | 0.418 | +0.011 |
| **78 untruncated questions** | **0.462** | **0.450** | **+0.012** |

**Both arms move up and the gap barely moves** — the same behaviour the old read showed
(0.264 → 0.303 and 0.345 → 0.376, delta −0.081 → −0.073). Truncation costs both arms a little F1 and
costs the contrast nothing measurable, under either φ. The contrast is not the degenerate records,
in either direction.

## What this changes

- **Gate G2 (Sep 6) now needs an interval that excludes zero, and does not have one.** ROADMAP §1's
  pass condition is joint beating post-hoc on citation-F1 by a margin whose paired-bootstrap CI
  excludes zero, with ≥95% valid claim parses (ADR-0019 Option H: `quote_located_rate ≥ 0.95` and
  `claim_parse_rate ≥ 0.95`). The φ blocker is gone; the width blocker is not. [INFERENCE] On this
  evidence the binding constraint on G2 is now question count and joint's long-claim tail, not the
  verifier.
- **The G2 read must be computed on a real gate run, not on this smoke run.** Every number above
  is 100 dev questions of a smoke run whose own summary disclaims being a sample. Re-reading the
  smoke run again with a better φ does not make it a gate basis, and re-reading it a third time would
  be re-reading records whose F1 is now known.
- **R5's early-warning contingency stays live, with its trigger condition changed.** Aug 14 armed it
  on a point estimate running against C2. That is no longer the situation; the situation is a
  straddling interval and a sign that flips inside the threshold sweep. The contingency is now for
  width, not for direction.
- **The verifier is no longer the open question in this read; the operating point is.** τ = 0.5 is
  defensible and untuned, and the sweep shows the contrast's sign depends on it. That makes G3
  (Sep 20) a prerequisite for interpreting G2's margin rather than a downstream gate, and it is
  fourteen days after G2.
- **Nothing about the parity loop is reopened.** The template remains frozen at the terminating run
  (`prompts.PARITY_LOOP_CLOSED.post_hoc_answer_template_sha256`, checked in `tests/test_prompts.py`).
  The read moved in post-hoc's disfavour and the prompt does not get to move in response.
- **The W9 stratified robustness check keeps its mandate**, and its preview got more interesting, not
  less: the length-band ordering is no longer uniform, so the conditioning W9 does properly is now
  the thing that decides whether the aggregate contrast is a grounding effect or a claim-length
  composition effect.
