# ADR-0009 parity loop — iteration 1b: the same prompt, re-measured

`scripts/generate_smoke.py`, **100 dev questions × 3 systems**, live vLLM on the A4000 serving
`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, depth 10, `temperature=0.0`, `seed=0`,
**`--max-tokens 3584` against a 14336 server**. Every other config field is identical to
`parity_iter1`'s, and **no prompt changed**: this is `PARITY_ITERATIONS[0]`'s post-hoc template,
measured again with more output headroom.

**It charges no loop iteration.** A cap raised for all three arms at once is shared run config, not a
post-hoc prompt edit — the same reading `parity_iter0` → `parity_iter0b` was taken under, recorded in
`parity_iter0.md`. The ledger stays at **1 of 10**.

**Blind held.** No citation-F1, on any split, in any form. Every gate number below comes from
`scoring/granularity.py`:

```
uv run python scripts/parity_report.py docs/harvest/parity_iter1b --max-tokens 3584
```

## Why it was run, and whether it worked

`parity_iter1` passed on all records (+0.0%) and failed untruncated-only (+21.4%), because the
finer-claims edit made post-hoc's cite stage — the stage its claims are parsed from — hit the cap on
26 of 100 records. The re-measure was to relieve that censoring and see whether the disagreement was
the artifact it was argued to be.

It was. **Cite-stage truncation 26 → 16 of 100, and the bases now agree:**

| basis | joint | post_hoc | gap | gate |
|---|---|---|---|---|
| all 100 records | 15 | 17 | **+13.3%** | PASS |
| untruncated per arm (n=92 / 84) | 14 | 16 | **+14.3%** | PASS (was +21.4% FAIL) |
| untruncated, same 78 queries both arms | 15 | 16 | **+6.7%** | PASS |

The third basis is new, and it is the one the iteration-1 argument implied but did not compute:
dropping the 22 queries where *any* arm hit the cap, from *both* arms. It is symmetric, so it cannot
be a selection effect in one arm, and it is the tightest reading of the three. All three pass. The
baseline of record fails all three (+25.0% / +42.9% / +37.9%).

## The finding that decides how the loop should end

**The same post-hoc prompt read +0.0% here and +13.3% there.** Nothing about the prompt differs
between `parity_iter1` and `parity_iter1b`; only the shared cap does.

That is not a contradiction, it is the gate's resolution. The medians are 14–17 words, so **one word
is ~6.7%** and the ±15% tolerance is **two words wide**. The gated statistic is an integer median: it
cannot take a value between 15 and 16, and the difference between "+0.0%" and "+13.3%" is one word on
one arm.

So the point estimate is reported with a query-level bootstrap from now on (`gap_bootstrap_ci`,
resampling **queries**, not claims — claims from one generation are not independent draws):

| run | basis | point | 95% interval | reading |
|---|---|---|---|---|
| `parity_iter0b` | all records | +25.0% | **[+18.8%, +40.0%]** | outside ±15% throughout |
| `parity_iter1` | all records | +0.0% | [+0.0%, +13.3%] | inside throughout |
| `parity_iter1b` | all records | +13.3% | **[+0.0%, +14.3%]** | inside throughout |
| `parity_iter1b` | same-queries | +6.7% | [+6.7%, +21.4%] | **straddles** |

Two things follow, and they point the same way:

1. **The movement is real.** The baseline interval and the candidate interval do not overlap
   (+18.8% floor against a +14.3% ceiling). Iteration 1's edit closed a gap that was there.
2. **The residual is not resolvable.** It is one grid step wide, and one basis's interval reaches
   +21.4% — which is joint 14 against post-hoc 17, i.e. the same claim shapes read on 78 queries
   instead of 100. Another loop iteration would be tuning a prompt against that.

## The control moved, and this is the second time the server has done this

`parity_iter1`'s joint generations were byte-identical to `parity_iter0b`'s, which is what an
out-of-bounds prompt is supposed to look like. **Here 23 of joint's 100 differ** — same prompt, same
seed, `temperature=0.0`, only the cap and `--max-model-len` changed. Post-hoc: 53/100 identical to
`parity_iter1` (and 0/100 to `parity_iter0b`, which is the different prompt). Vanilla: 81/100.

This is the mechanism `parity_iter1.md` recorded on vanilla, now visible on joint too: greedy
decoding is deterministic given identical numerics, and changing `--max-model-len` changes KV-cache
block layout and prefill batching, so the numerics are not identical. **Byte-identity is a control
check only across runs whose server config matches.**

Across a config change the control has to be checked differently, so it was:

| joint, all records | 2560 | 3584 |
|---|---|---|
| median words/claim, **all 100** | 16 | **15** |
| median words/claim, **the 77 byte-identical** | **15** | **15** |
| claims on those 77 | **357** | **357** |
| median claims/query, all 100 | 5.0 | **4.0** |
| median claims/query, the 77 | **4** | **4** |

On the records that did not move, joint's claim lists are *identical* — same words, same count. Both
joint figures that appear to have shifted are unchanged on the invariant subset and flipped by which
records contribute, which is the same knife-edge the section above describes. **Joint did not get
finer; the pool tipped.** Had this not held, the +13.3% would have been unreadable — a control that
drifts is not a control.

## Coverage: still answering finer, not less

| | joint | post_hoc `iter0b` | `iter1` | `iter1b` |
|---|---|---|---|---|
| median claims/query | 4.0 | 8.0 | 10.0 | **10.0** |
| total claims parsed | 719 | 895 | 1129 | **1242** |
| median words/claim | 15 | 20 | 16 | **17** |

Post-hoc parses **1242 claims against joint's 719** and holds 10 claims/query. The pass is not the
model saying less.

## Compound profile, and where post-hoc is still coarser

`COMPOUND_MARKERS`, all records:

| | joint | post_hoc |
|---|---|---|
| simple-claim share | 62.3% | 61.4% |
| median words, simple claims | 14 | 15 |
| "and" | 36.7% | 34.2% |
| subordinate clause | 0.3% | **3.2%** |
| 2+ commas | 6.3% | **10.8%** |

Same shape as `parity_iter1`: the two markers the iteration targeted are down from the baseline
(4.8% → 3.2%, 13.6% → 10.8%) and remain post-hoc's residual excess. The simple-claim median is 15
against joint's 14 — the pre-freeze check that the pass is not a compound claim wearing a short
claim's word count (`claim_validity` is W6, after the Sep 3 freeze).

The residual is also visible in the quantiles, and it is honest to name it: post-hoc's p75 is 21
against joint's 18 (**+16.7%**, outside the tolerance if the gate had been written on p75 rather than
the median) while p25 is 13 against 12 and p90 is 25 against 22. Post-hoc's distribution is *tighter
and more central*; joint's is wider at both ends. On the **mean** post-hoc is now **shorter** than
joint — 17.53 against 18.92, −7.3% — for the reason in the next section.

## A defect found on the way, and deferred on purpose

Joint's mean words/claim rose 17.15 → 18.92 while its median fell. The cause is not granularity:

- **3.1% of joint's claims exceed 40 words** (post-hoc: 0.5%), and joint's p99 is 122 words.
- Query **21074975** yields a single "claim" of **731 words** — a degenerate `and …, and …`
  repetition loop that runs until the cap. The same record produced a 164-word claim at 2560. **Its
  length scales with the output cap**, which is the signature of a generation that never terminates,
  not of a coarse claim.
- Both joint records carrying a >100-word claim hit the cap. The symmetric basis excludes them, which
  is part of why it is the tightest reading.

Two consequences, neither of them this loop's business: `_claim_rules()` splits on "and" and did not
split this, so the splitter has a hole; and a 731-word claim will be passed to citation scoring as
one unit the moment the blind lifts. **Both are out of bounds under §4 and belong to W5/W6** —
recorded here with the query id so the fix is not rediscovered from a confusing F1 number.

Vanilla, for the record: median unchanged at **21**, mean 22.24 → 21.59, but total claims 1221 →
1622, concentrated in a handful of records with the same runaway shape (its top 5 records hold 38% of
its claims). Vanilla is excluded from the gate by ADR-0010; it is the untuned reference point, and it
is still the coarsest arm.

## Where the loop stands

- **Iterations used: 1 of 10** (`PARITY_ITERATIONS`) — this run charges none. Drop-dead **Aug 30**.
  System ledger untouched at joint 4 / post_hoc 4 (§7).
- **Gate: PASS on all three bases** (+13.3% / +14.3% / +6.7%), with the all-records interval
  [+0.0%, +14.3%] inside ±15% throughout. The baseline of record failed all three.
- **The residual favours C2 on every basis, so ADR-0009 §5's W9 stratified robustness check stays
  mandatory.** It was triggered at iteration 0 and a passing iteration does not retract a
  pre-registered check — that retraction is exactly the post-hoc steering §3 and §6 exist to prevent.
- **Recommendation: terminate the loop on this run.** The gap is closed as far as the gate can
  resolve, the residual is one grid step, and the loop's own measurements show that a further
  iteration would be fitting run-to-run noise (same prompt, +0.0% and +13.3%). Terminating unblinds
  citation-F1 for the first time (§6).
