# ADR-0009 parity loop — iteration 0 (measurement only)

`scripts/generate_smoke.py`, **100 dev questions × 3 systems**, live vLLM on the A4000 serving
`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, depth 10, `temperature=0.0`,
`max_tokens=1536`, `seed=0`. Finished 2026-08-13T19:13Z. Artifacts:
`parity_iter0.{summary.json,records.jsonl,costs.jsonl}` (commit `da32cfd`).

**Iteration 0 is a measurement, not a tuning cycle.** No prompt was edited, so `PROMPT_ITERATIONS`
is unchanged and `effort_is_matched()` stays 4/4. The first ledger entry lands with iteration 1.

**Blind held.** Nothing here is citation-F1 or a proxy for it (ADR-0009 §6). words/claim is counted
from `CLAIM` line text only and is independent of whether any `CITE` line validates.

## The gate

Words/claim is `len(text.split())` over each parsed claim. Gate is ADR-0009 §2/§3: median
words/claim, joint vs post-hoc, tolerance ±15%.

| quantity | joint | post_hoc | gap | gate |
|---|---|---|---|---|
| **pooled median words/claim** | **15.0** | **20.0** | **+33.3%** | **FAIL** |
| median of per-query medians | 14.75 | 19.5 | +32.2% | fail |
| median claims/query (reported, not gated) | 4.5 | 8.0 | +77.8% | — |
| total claims parsed | 549 | 778 | | |

**Post-hoc claims are coarser**, by more than double the tolerance. Under §5 this is the residual
gap *favouring C2*, so **the W9 stratified robustness check becomes mandatory** — the branch ADR-0009
already flagged as the likely one. That obligation is now measured, not predicted.

`vanilla` is 21.0 pooled median words/claim, 8.0 claims/query — excluded from the gate per ADR-0010,
recorded because it is the untuned reference point and sits with post-hoc, not with joint.

### The n=3 smoke overstated the gap

The provisional read off the Aug 10 smoke was joint ~9.5 / post-hoc ~15, a ~58% gap. At n=100 it is
15.0 / 20.0, **33%**. Same direction, much smaller magnitude. Three questions in file order were not
a sample; this is the first gateable number.

## Known weakness #2 is resolved: words/claim is *not* redundant with claims/query

ADR-0009 §2 deferred to W4 the question of whether median words/claim is near-mechanically linked to
claims/query "with total answer length already constrained to ±10%". **The premise is wrong and the
two quantities are not linked.**

The ±10% enforced condition (ADR-0002) is on the **prompt** token budget. Nothing constrains output
length. So post-hoc is free to be — and is — both *longer per claim* and *more claims*:

| | joint | post_hoc |
|---|---|---|
| median claim-words per query | 62 | 159 |
| median completion tokens | 638 | 1528 |

The gaps diverge sharply (+33% words/claim vs +78% claims/query). Had they been mechanically
linked they would move together. **Keep median words/claim as the gated quantity** — §2's choice
stands, now for a measured reason rather than an assumed one. No ADR amendment is needed; §2's
"re-examined against the first real measurement" is discharged by this file.

## Measurement caveat: post-hoc's cite stage is output-truncated on 38% of the run

`max_tokens=1536` is a **per-call** cap, and a post-hoc record's `completion_tokens` is the **sum of
its two stages** — so that field cannot be compared against 1536. Per-stage output tokens, recovered
from `parity_iter0.costs.jsonl` (four calls per query, in order joint / post_hoc answer / post_hoc
cite / vanilla):

| call | at cap | p50 output tokens |
|---|---|---|
| joint | 11/100 | 638 |
| post_hoc **answer** | 6/100 | 246 |
| post_hoc **cite** | **38/100** | 1274 |
| vanilla | 7/100 | 256 |

**The cite stage is the one that matters here.** `generate.py:134` parses the post-hoc record from
`parsed_from` — the *cite* stage's output, not the answer stage's — so it is cite-stage truncation
that drops trailing claims off a post-hoc record. Truncated generations end mid-line; one tail stops
at `…the findings generally demonstrate that`. This censors the **tail** of each answer, depressing
**claims/query** and leaving at most one partial claim per affected record.

**The gate reading survives it.** Restricting to records whose parsed-from call did not hit the cap:

| | joint (n=89) | post_hoc (n=62) | gap |
|---|---|---|---|
| pooled median words/claim | 14 | 19 | +35.7% |

The gap widens slightly rather than closing, and stays far outside ±15% either way. **Iteration 0's
FAIL is not an artifact of truncation.** claims/query, by contrast, is a censored number and should
be read as a lower bound for post-hoc.

Raising `max_tokens` is a shared run-config change applied to all three systems, not a post-hoc
prompt edit, so it is outside §4's one-directional tuning restriction and costs no ledger cycle.

**Ceiling.** The runbook serves with `--max-model-len 8192`. The largest observed cite-stage prompt
is 5,601 tokens, so `--max-tokens 2560` is the most that fits every observed prompt without
restarting vLLM on a different serving config.

## Where the loop stands

- Iterations used: **0 of 10**. Drop-dead **Aug 30**.
- Ledger: joint 4, post_hoc 4, matched.
- Next tuning cycle edits **`POST_HOC_ANSWER_TEMPLATE` only** (§4 as amended 2026-08-13), aimed at
  finer claims. `_claim_rules()` and the joint prompt stay out of bounds; `decompose.py` is not
  involved.
- Every post-hoc cycle costs a matched joint cycle to keep `effort_is_matched()` true.
