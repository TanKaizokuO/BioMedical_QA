# `frequency_penalty` sweep — repetition loop vs. format collapse (`f9c147b`)

`scripts/decompose_smoke.py --frequency-penalty {0.1,0.3,0.5,1.0} --max-tokens 4096 --n 15`, live
vLLM on the A4000 serving `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`. Same 15 dev
`post_hoc` questions at every point (`load_post_hoc`'s file-order slice), so the four rows are
directly comparable. Artifacts: `decompose_sweep_fp{01,03,05,10}.{summary.json,records.jsonl,costs.jsonl,manifest.json}`.

**Not a gate run.** n=15 sizes a sanity sweep, not G2's rate. What this answers: does
`frequency_penalty` alone close the repetition loop `docs/harvest/decompose_smoke.summary.json`
found (fp=0.0, `atomic` 83/98 queries with duplicate claims)?

## Result

| `frequency_penalty` | atomic claims | atomic dup | atomic q-not-found | atomic divergence | decon claims | decon dup | decon q-not-found | clean_decompose (both rows) |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 580 | 319 (55%) | 139 | 0.88 | 223 | 115 (52%) | 42 | 0.00 |
| 0.3 | 244 | 45 (18%)  | 51  | 0.82 | 70  | 2 (3%)    | 23 | 0.00 |
| 0.5 | 161 | 13 (8%)   | 27  | 0.80 | 38  | 0 (0%)    | 8  | atomic 0.07, decon 0.00 |
| 1.0 | 83  | 4 (5%)    | 10  | 0.67 | 21  | 0 (0%)    | 5  | 0.00 |

`sentence` is unaffected at every point (no model call): 177 claims, `clean_decompose_rate` 1.0,
`clean_cite_rate` 1.0, 0 duplicates — the control holds, confirming the sweep only touches the two
model-driven rows.

## Reading it

**The repetition loop is a real `frequency_penalty` response, and it saturates around 0.3–0.5.**
Duplicate-claim count falls monotonically and steeply from fp=0.1 to fp=0.5 (atomic: 319 → 45 → 13;
decon: 115 → 2 → 0) — `decon` is functionally solved by fp=0.3. Going to fp=1.0 buys atomic only 13
→ 4, a shrinking return, while total claim count keeps collapsing (580 → 83) because fewer sentences
get decomposed at all.

**`clean_decompose_rate` never recovers, at any tested value.** This is the finding that matters:
fixing the repetition loop does not fix parse rate, because a second, independent defect dominates
once the loop is suppressed — the model drifts off the `CLAIM <n> FROM <sentence>` grammar entirely
under penalty pressure (`CLAIM7FROM4`, `CLAIM S7.1 FROM S7`, `CLAIM 7 FROM (6)`) and stops partway
through longer answers, leaving trailing sentences with no claim (`sentences [4, 5, ...] produced no
claim — the answer was dropped, not decomposed`). At fp=1.0 this format-collapse mode is the
*majority* of `atomic`'s remaining errors (83 claims, only 4 duplicates — the rest are grammar
misses and dropped sentences), and it is not addressable by tuning `frequency_penalty` further: a
penalty strong enough to suppress duplicate emission of `CLAIM`/`FROM` is by construction a penalty
on emitting `CLAIM`/`FROM` at all.

**`atomic` divergence stays high throughout (0.67–0.88).** Expected and separate from both defects
above — it is `atomic` doing decontextualization work `unit_rules(ATOMIC)` withholds instruction
for (ADR-0018 §2's validity condition), not something a decoding knob should move much. The modest
downward drift with fp is more likely fewer total claims changing the denominator than a
grounding improvement.

**Monotonicity (`sentence ≥ atomic ≥ decontextualized_atomic`) is not stable at n=15.** Passes at
fp=0.1, 0.5, 1.0; fails at fp=0.3 (atomic median 13.0 < decon median 14.0). At this sample size a
single long or short answer moves the median — not evidence against the ordering, just evidence the
n=15 sanity slice is too small to check it.

## Decision

**No single `frequency_penalty` value clears G2's ≥95% clean-parse bar; none was expected to.** The
sweep answers the question it was built to answer — repetition-loop duplication is a real,
independently fixable decoding defect, resolved by fp≈0.3–0.5 — and surfaces the one it wasn't: a
grammar-adherence defect on long multi-sentence answers that the repetition-loop fix exposes rather
than causes. Conflating the two would have shipped a "fix" that traded one clean-parse blocker for
another of similar size.

**Recommended `frequency_penalty` for the next stage: 0.5.** It reaches decon's plateau (0 dup),
gets atomic to its best `clean_decompose_rate` in this sweep (0.07, vs 0.00 everywhere else), and
sits well short of 1.0's total-claims collapse. This is a decoding-knob recommendation only — it
does not by itself clear the ≥95% bar and should not be reported as doing so.

**The grammar-adherence defect needs a prompt or chunking fix, not a decoding knob**, before a
clean-parse baseline is measurable. Two candidates, neither attempted here (out of scope for a
sweep script): (a) cap the number of source sentences per `decompose()` call and batch long answers,
since every observed drop starts partway through answers with >20 sentences; (b) tighten
`unit_rules`'s grammar instruction with a negative example of the `CLAIM<n>FROM<k>`-without-spaces
drift observed at fp≥0.5.

## What this does not license

Fifteen questions at four points cannot set G2's rate, cannot confirm fp=0.5 generalizes to the full
`parity_iter1b` slice, and cannot rule out the grammar-adherence defect being answer-length-specific
rather than penalty-specific — that requires a run stratified by source sentence count, not run here.
