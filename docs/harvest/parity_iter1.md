# ADR-0009 parity loop — iteration 1

`scripts/generate_smoke.py`, **100 dev questions × 3 systems**, live vLLM on the A4000 serving
`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, depth 10, `temperature=0.0`, `seed=0`,
`--max-tokens 2560` against a 12288 server — identical run config to `parity_iter0b`, the baseline of
record.

The one change is `POST_HOC_ANSWER_TEMPLATE` (`PARITY_ITERATIONS[0]`): a length target, trailing
qualifier clauses promoted to their own claims, and a ban on study-framing preamble. Nothing about
citing (§4; `tests/test_prompts.py::test_post_hoc_first_pass_never_mentions_citing`).

**Blind held.** No citation-F1, on any split, in any form. Every number here is computed by
`scoring/granularity.py` and reproducible with:

```
uv run python scripts/parity_report.py docs/harvest/parity_iter1 --max-tokens 2560
```

## The comparison is like-for-like, and this is checked rather than assumed

**Joint's 100 generations are byte-identical to `parity_iter0b`'s.** Its prompt was out of bounds and
the sampler is greedy, so this is what "the joint arm did not move" looks like when it is verified
instead of asserted — and it means every difference below belongs to the post-hoc edit.

**Vanilla's do not: 19 of 100 differ**, on an untouched prompt at `temperature=0.0`. See *A
reproducibility fact* below. It does not affect the gate — vanilla is excluded by ADR-0010 and its
median is unchanged at 21 — but it is recorded because it contradicts a determinism assumption the
paper would otherwise make silently.

## The gate: PASS on the basis of record, and the two bases disagree for the first time

| basis | joint | post_hoc | gap | gate |
|---|---|---|---|---|
| **all 100 records** | **16** | **16** | **+0.0%** | **PASS** |
| untruncated only (n=91 / 74) | 14 | 17 | +21.4% | fail |

At iteration 0 every basis failed, so the choice of basis never had to be adjudicated. It does now,
and the answer is **all records** — for a reason that is specific to what changed, not for the
reason that it passes.

### Why untruncated-only is the biased basis here, having been the robustness check at iteration 0

The edit made post-hoc write **more, shorter claims**. More claims means longer cite-stage output,
and longer output means the 2560 cap is hit more often: post-hoc cite truncation went **16 → 26 of
100**. So truncation is now a *consequence of the treatment*.

Conditioning on "this record was not truncated" is therefore conditioning on a post-treatment
variable that the treatment causes. It selects against exactly the records that demonstrate the
effect: the 26 truncated post-hoc records carry a median **14.5 claims/query** against 10.0 for the
rest, and contribute **460 of post-hoc's 1129 claims — 41% of the evidence, from 26% of the
records.** Dropping them raises post-hoc's median from 16 to 17 by removing its finest output, not by
removing an artifact.

**The defect the untruncated basis was built to catch is measured, and it is nil.** Truncation drops
*trailing claims*; it does not shorten the claims that survive. Dropping the final — possibly
mid-sentence — claim of every one of the 26 truncated records:

| | post_hoc claims | median words/claim |
|---|---|---|
| as recorded | 1129 | **16.0** |
| final claim of each truncated record dropped | 1103 | **16.0** |

Unchanged. Words/claim is a *per-claim* statistic, so censoring the tail of a claim list barely
touches it — which is why at iteration 0 both bases agreed on the FAIL, and why the all-records
reading is the sound one now that they disagree.

**This is not the same argument as `iter0` → `iter0b`.** There the concern was truncation *inflating*
a gap and the fix was more headroom, which moved the headline 33% → 25%. Here the gated quantity is
shown to be insensitive to the censoring, and the disagreement is a selection effect in the other
basis.

## The pass is not "answering less" — the failure mode the gate cannot see

This was the one reading that would have made a pass worthless: words/claim falling because the model
answers *less* rather than *finer*.

| | joint | post_hoc `iter0b` | post_hoc `iter1` |
|---|---|---|---|
| median claims/query | 5.0 | 8.0 | **10.0** |
| total claims parsed | 645 | 895 | **1129** |
| median words/claim | 16 | 20 | **16** |

**Claims/query went up, not down**, and post-hoc parsed 234 more claims than at the baseline. The
answers got finer-grained; they did not get shorter. Note also that words/claim fell 25% while
claims/query rose 25% — the two moved *independently and in opposite directions*, which is a second,
stronger confirmation of what `parity_iter0.md` settled about ADR-0009 known-weakness #2.

## The mechanism moved the way the rationale predicted

Compound profile, all records, `COMPOUND_MARKERS` from `scoring/granularity.py`:

| | joint | post_hoc `iter0b` | post_hoc `iter1` |
|---|---|---|---|
| simple-claim share | 65.4% | 58.3% | **63.9%** |
| median words, simple claims only | 14 | 18 | **15** |
| "and" | 33.5% | 35.0% | 33.7% |
| subordinate clause | 0.3% | 4.8% | **1.9%** |
| 2+ commas | 5.6% | 13.6% | **9.3%** |

Every marker moved toward joint, and the two the iteration explicitly targeted moved most:
subordinate clauses 4.8% → 1.9% (the qualifier-splitting rule) and the simple-claim median 18 → 15
(the length target). The "and" rate stayed level, as expected — `_claim_rules()` splits on it for all
three systems, so it was never the lever.

The simple-claim median is ADR-0009's only pre-freeze proxy for "one long atomic claim vs one
compound claim of equal length" (Consequences; `claim_validity` lands in W6, after the Sep 3 freeze).
At 15 against joint's 14 it says the pass is not a compound claim wearing a short claim's word count.

## What this does not settle

- **§5's W9 stratified robustness check stays mandatory.** It was triggered by the baseline of
  record at iteration 0, and a pre-registered asymmetric check is not retractable because a later
  measurement came out better — retracting it on this evidence is precisely the post-hoc steering §3
  and §6 exist to prevent. The all-records residual gap is 0.0% and the untruncated one favours C2;
  the cheap, disclosed option is to keep the check.
- **Post-hoc cite truncation is now worse (26/100)** and is a live confound on *claims/query*, which
  §2 reports rather than gates. The gated quantity is insensitive to it (above); the reported one is
  not, and 26 of post-hoc's claim lists are incomplete.
- **Joint remains the outlier.** Vanilla is at 21 and untuned. Post-hoc has been moved from vanilla's
  granularity to joint's *by prompt instruction*, which is the finding
  `POST_HOC_ANSWER_TEMPLATE`-was-byte-identical-to-`VANILLA_TEMPLATE` predicted, and it belongs in
  the paper's setup section.

## A reproducibility fact, recorded because it is load-bearing elsewhere

**`temperature=0.0` and `seed=0` did not make this server reproducible.** Vanilla's prompt was not
touched and 19 of its 100 generations still differ from `parity_iter0b`'s.

Joint's did not differ, and the asymmetry names the mechanism: generation issues four calls per
query in the order joint / post-hoc answer / post-hoc cite / vanilla. **Joint is issued before any
post-hoc call, vanilla after both** — so only vanilla sees server-side state (KV-cache block reuse,
prefix caching, prefill batching) that the post-hoc edit changed. Greedy decoding is deterministic
given identical numerics, and the numerics are not identical when the surrounding request stream
differs.

Consequences: run-to-run *equality* of an untouched arm cannot be assumed, only checked; and any
claim that the pipeline is reproducible from `(model, seed, temperature)` needs the server's request
history added to it, or a caveat. The effect here is small — vanilla's median is unchanged at 21 and
its mean moved 22.42 → 22.24 — but it is not zero, and it would be invisible on any summary
statistic. W5 cleanup, with `backends.py`'s missing `prompt_tokens + max_tokens` check.

## Where the loop stands

- **Iterations used: 1 of 10** (`PARITY_ITERATIONS`). Drop-dead **Aug 30**. System ledger untouched
  at joint 4 / post_hoc 4 (§7).
- **Gate on the basis of record: PASS, +0.0%** (joint 16, post-hoc 16).
- Baseline of record for any further comparison is still `parity_iter0b`; this run is the candidate.
- A re-measure at a larger `--max-tokens` would collapse the basis disagreement by relieving the
  cite-stage censoring, and — per `parity_iter0.md`, since raising the cap is a shared run-config
  change applied to all three systems rather than a post-hoc prompt edit — **costs no loop
  iteration.** Sizing, from this run's own cost rows: worst cite prompt 6,692 tokens, and the cite
  prompt grows with the answer, so a 3584 cap wants `--max-model-len 14336`.
