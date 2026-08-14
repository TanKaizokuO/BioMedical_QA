# ADR-0009 parity loop — iteration 0 (measurement only)

`scripts/generate_smoke.py`, **100 dev questions × 3 systems**, live vLLM on the A4000 serving
`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, depth 10, `temperature=0.0`, `seed=0`.

Measured twice, at two output caps. **`parity_iter0b` is the baseline of record**; `parity_iter0`
is kept because the two together are what show the gate verdict is not a truncation artifact.

| run | `max_tokens` | `--max-model-len` | artifacts | commit |
|---|---|---|---|---|
| `parity_iter0` | 1536 | 8192 | `parity_iter0.{summary,records,costs}` | `da32cfd` |
| **`parity_iter0b`** | **2560** | **12288** | `parity_iter0b.{summary,records,costs}` | `e22dacf` |

**Iteration 0 is a measurement, not a tuning cycle.** No prompt was edited, so `PROMPT_ITERATIONS`
is unchanged and `effort_is_matched()` stays 4/4. Neither run consumed a loop iteration: raising the
output cap is a shared run-config change applied to all three systems, not a post-hoc prompt edit,
so it falls outside ADR-0009 §4's one-directional tuning restriction. **0 of 10 iterations used.**

**Blind held.** Nothing here is citation-F1 or a proxy for it (§6). words/claim is counted from
`CLAIM` line text only and is independent of whether any `CITE` line validates.

## The gate: FAIL, in every reading

Words/claim is `len(text.split())` over each parsed claim. Gate is §2/§3: median words/claim, joint
vs post-hoc, tolerance ±15%.

| basis | joint | post_hoc | gap | gate |
|---|---|---|---|---|
| **`iter0b`, all 100 records** | **16** | **20** | **+25.0%** | **FAIL** |
| `iter0b`, untruncated only (n=91 / 84) | 14 | 20 | +42.9% | fail |
| `iter0`, all 100 records | 15 | 20 | +33.3% | fail |
| `iter0`, untruncated only (n=89 / 62) | 14 | 19 | +35.7% | fail |

**Post-hoc claims are coarser, by 25–43% depending on basis — every reading is well outside ±15%.**
Under §5 this is the residual gap *favouring C2*, so **the W9 stratified robustness check becomes
mandatory**. ADR-0009's Consequences called that the likely branch; it is now the measured one.

Supporting quantities on `iter0b` (claims/query is reported, not gated, per §2):

| | joint | post_hoc | vanilla |
|---|---|---|---|
| median words/claim | 16 | 20 | 21 |
| mean words/claim | 17.15 | 21.35 | — |
| words/claim p25 / p75 / p90 | 12 / 18 / 22 | 16 / 25 / 31 | — |
| median claims/query | 5.0 | 8.0 | 8.0 |
| total claims parsed | 645 | 895 | 1283 |

`vanilla` is excluded from the gate per ADR-0010, recorded because it is the untuned reference point
— and it sits with post-hoc, not with joint. Joint's fine claims are the outlier, which is what §4's
"joint's granularity is native" predicts.

### The n=3 smoke overstated the gap

The provisional read off the Aug 10 smoke was joint ~9.5 / post-hoc ~15, a ~58% gap. At n=100 it is
16 / 20, **25%**. Same direction, less than half the magnitude. Three questions in file order were
not a sample; this is the first gateable number.

## Known weakness #2 is resolved: words/claim is *not* redundant with claims/query

§2 deferred to W4 the question of whether median words/claim is near-mechanically linked to
claims/query "with total answer length already constrained to ±10%". **The premise is wrong and the
two quantities are not linked.**

The ±10% enforced condition (ADR-0002) is on the **prompt** token budget. Nothing constrains output
length. So post-hoc is free to be — and is — both *longer per claim* and *more claims*: on `iter0b`
the words/claim gap is +25% while the claims/query gap is +60%. Had the two been mechanically
linked they would move together, and across both runs they do not.

**Keep median words/claim as the gated quantity** — §2's choice stands, now for a measured reason
rather than an assumed one. No ADR amendment is needed; §2's "re-examined against the first real
measurement" is discharged by this file.

## Why the cap was raised, and why the gate verdict does not depend on it

`max_tokens` is a **per-call** cap, and a post-hoc record's `completion_tokens` is the **sum of its
two stages** — that field cannot be compared against the cap. Per-stage output tokens, recovered
from `costs.jsonl` (four calls per query, in order joint / post_hoc answer / post_hoc cite /
vanilla), calls at the cap:

| call | `iter0` @1536 | `iter0b` @2560 |
|---|---|---|
| joint | 11/100 | 9/100 |
| post_hoc **answer** | 6/100 | 6/100 |
| post_hoc **cite** | **38/100** | **16/100** |
| vanilla | 7/100 | 7/100 |

**The cite stage is the one that matters.** `generate.py:134` parses the post-hoc record from
`parsed_from` — the *cite* stage's output, not the answer stage's — so cite-stage truncation drops
trailing claims off a post-hoc record. At 1536 that hit 38% of post-hoc records; at 2560, 16%.

Relieving it moved the headline gap 33% → 25%, so truncation was inflating it. But the correction is
bounded and does not point one way: on the untruncated-only basis the gap *widens* with the larger
cap (35.7% → 42.9%), because dropping joint's own truncated records pulls joint's median down. **No
reading of either run comes within 10 points of the ±15% tolerance.** Further headroom is not worth
another GPU run; `iter0b` is the baseline.

### Sizing note, recorded because the first attempt failed

The first re-measure attempt kept `--max-model-len 8192` and set `--max-tokens 2560`, sized off the
largest cite-stage prompt in `iter0` (5,601 tokens). It died on a bare `400 Bad Request` two thirds
of the way in, with no partial artifacts. **The cite prompt embeds the stage-1 answer**, so its
length is a function of answer length — raising the output cap lengthens the answers and moves the
very number the cap was sized against. `--max-model-len` had to go to 12288.

`backends.py` does not check `prompt_tokens + max_tokens` against the served `max_model_len` before
posting, so this surfaces as an unattributed HTTP 400 mid-run. Real robustness gap; W5 cleanup, not
touched during the loop.

## Where the loop stands

- Iterations used at the time of writing: **0 of 10**. Drop-dead **Aug 30**. (Iteration 1 was
  drafted 2026-08-14 against this baseline; the count lives in `PARITY_ITERATIONS`, not here.)
- Ledger: joint 4, post_hoc 4, matched — and parity cycles never touch it (ADR-0009 §7).
- Baseline of record: `parity_iter0b`, **joint 16 / post_hoc 20 median words/claim, +25.0%**.
- Iteration 1 edits **`POST_HOC_ANSWER_TEMPLATE` only** (§4 as amended 2026-08-13), aimed at finer
  claims. `_claim_rules()` and the joint prompt stay out of bounds; `decompose.py` is not involved.
- Every post-hoc cycle costs a matched joint cycle to keep `effort_is_matched()` true.
- Re-measurement runs use `--max-tokens 2560` against a 12288 server, or the comparison to this
  baseline is not like-for-like.
