# ADR-0015: G1 is gated at hit@10, and the one chunker that passed at k=5 is disqualified

- **Status:** Accepted
- **Date:** 2026-08-10
- **Supersedes:** nothing. **Amends:** the gate table's G1 row (ROADMAP), which read k=5.
- **Depends on:** ADR-0014 §2 (no property every gold shares and no distractor has),
  ADR-0012 §2 (confusability), ADR-0009 (parity is measured, not assumed).

## Context

G1 was executed on 2026-08-10, thirteen days early, against Table 1 row 4 — the full cascade,
BM25 + dense + RRF + cross-encoder rerank — on the 100-question dev split.

    k=5   hits 86/100  point 0.8600  Wilson lower 0.7786   FAILS
    k=10  hits 94/100  point 0.9400  Wilson lower 0.8752   passes

G1 requires point >= 0.90 **and** Wilson lower > 0.85. At k=5 it fails on both clauses.

R2's documented ladder for a failing G1 ends at relaxing k and saying so in the paper, with the
chunker sweep as the last rung before that. The sweep proper is seven 2M-row index builds at ~2 h
of A4000 each. `chunker_pool_eval.py` answered it inside Table 1's own recorded row-4 pool instead,
licensed by a property of the reranker: a cross-encoder score is a function of the `(query,
passage)` pair alone, with no corpus statistics, so the `abstract` arm is row 4 *recomputed* rather
than estimated. Every arm is an upper bound — the reranker sees every chunk of 100 pooled abstracts
rather than a 100-deep chunk pool, and rivals outside the pool cannot compete — so an arm below
0.90 **cannot** pass for real and its build is refused on evidence.

Before that script was run, `g1_miss_analysis.json` registered a prediction: no arm reaches 0.90,
falsified if any does, **in which case the full build for that arm is owed**.

`docs/harvest/chunker_pool_eval.json` (harness check passed: abstract arm 0.8600, gold promoted in
0 queries and demoted in 0, against 1,466 sibling candidates the audit found re-chunking adds):

| chunker | hit@5 (UB) | Wilson lo | gold chunks/q | distractor chunks/q |
|---|---|---|---|---|
| abstract | 0.8600 | 0.7786 | 0.97 | 114.09 |
| **section** | **0.9400** | **0.8752** | **3.21** | **114.09** |
| sentence_window_3_1 | 0.8500 | 0.7672 | 6.97 | 742.05 |
| sentence_window_3_3 | 0.8900 | 0.8137 | 3.33 | 344.11 |
| **sentence_window_5_2** | **0.8700** | 0.7902 | **3.21** | **352.03** |
| fixed_width_512 | 0.8700 | 0.7902 | 2.86 | 321.25 |
| fixed_width_1024 | 0.8600 | 0.7786 | 1.68 | 183.72 |

**The prediction is falsified as written.** `section` reached 0.94 and, on its face, owes a build.

## Decision

### 1. `section` is ineligible, and the build it earned is refused on that ground

`section` is not a corpus chunking strategy. It is a gold-only transformation.

`encode_corpus.py:159-173` builds gold with `chunk_instance` — real PubMedQA section spans — and
MedRAG rows with `chunk_text(sections=None)`, because the corpus carries no section labels.
`chunk_text` degrades `"section"` to `"abstract"` when `sections is None`. Verified exactly, not
approximately, on all 100 dev abstracts: **the distractor path produces byte-identical chunks under
`section` and `abstract`, 0/100 texts differ**, while the gold path produces 330 chunks against
100. Under symmetric treatment `section` *is* `abstract`.

The measured arms say the same thing. `section` leaves distractor chunks/query at **114.09 —
identical in all 100/100 queries** to the `abstract` arm — and moves only gold, 0.97 → 3.21
chunks. Its entire +0.08 is one abstract being cut finer than the 114 it is ranked against.

The data carries its own matched control, and it is exact. `sentence_window_5_2` cuts gold into
**3.21** chunks/query — the identical granularity `section` gives it, to two decimals — but cuts
distractors as well, 352.03 against 114.09. It reads **0.8700**, not 0.9400.

Same gold granularity, 0.07 apart. The only variable between them is whether the other 114
abstracts received the same treatment. `sentence_window_3_3` (gold 3.33, distractors 344.11, reads
0.8900) says it again at a slightly different granularity. The gain is not from chunking gold; it
is from not chunking anything else.

That is "the one property every gold passage shares and no distractor has", which ADR-0014 §2
rejects as a systematic signal sitting in exactly the space hit@5 is measured in. `chunk.py`'s
module docstring already generalises §2 past titles — *"how the text was cut is such a property"* —
and both statements predate this sweep. The rule is not being invented to survive a result.

**A build would not repair this.** It would reproduce the asymmetry faithfully, because that is how
`encode_corpus.py` builds the index, and return a number that is real for this pipeline and still
not a reading of retrieval. The registered debt is discharged by ineligibility, not by the number.

This is the failure mode ADR-0014 named as fatal: G1 looking excellent for the wrong reason.

### 2. No chunker rescues G1 at k=5

Every eligible arm is at or below **0.89** as an upper bound (`sentence_window_3_3`), and a real
build reports no more than its bound. All seven builds are refused: `docs/harvest/chunker_arm_eligibility.json`,
`builds_owed: []`.

### 3. G1 is gated at hit@10 for the workshop submission, and the paper says so

    G1: hit@10 >= 0.90, Wilson lower > 0.85   —   0.9400 / 0.8752, passes.

The relaxation is reported in the paper as a relaxation, with the k=5 reading printed beside it.
It is the last rung of R2's ladder and every rung above it has now been spent.

**The threshold does not move.** 0.90 and 0.85 stand; only k changes, once, on the record. Tuning
either number to fit 0.86 is prohibited and remains so.

### 4. The leak check runs automatically, on every arm, forever

`gold_cut_asymmetry()` in `chunker_pool_eval.py` compares each configuration's gold path against
its own distractor path and flags any arm that cuts gold differently. It is CPU-only and exact.
`chunker_pool_eval.py` records it per arm and prints `GOLD-ONLY CUT` in the summary table;
`chunker_arm_eligibility.py` turns it into the build verdict.

A future reader does not have to remember ADR-0014 §2 while looking at a number that flatters the
gate. The artifact says it.

## Consequences

- **hit@5 = 0.86 is reported, not hidden.** It is the honest cascade number and it appears in
  Table 1 and in the G1 row beside the relaxed gate.
- **The generation stage receives a 10-passage context, not 5**, wherever G1's k is the binding
  constraint. Downstream prompt budgets assume that.
- **`section` stays in `SWEEP`** and stays flagged. Deleting it would lose the finding; the sweep's
  value now includes demonstrating the leak, which is a paper-worthy negative result about
  chunker evaluation on PubMedQA — gold carries structure the corpus does not, so any structure-
  aware chunker scores its own privilege.
- **A registered prediction was falsified and is recorded as falsified.** The prediction's terms
  were met by an arm those terms did not anticipate. Recording the miss, and the pre-existing rule
  that disposed of it, is the point of registering it at all.
- **~14 h of A4000 is not spent.** Seven builds refused on a bound plus an exact CPU check.

## Alternatives rejected

- **Pay the registered debt and build `section`.** ~2 h to produce a faithful measurement of a
  disqualified signal. The debt was for an arm that *rescues retrieval*; `section` rescues nothing
  and cannot, because under symmetric cutting it is `abstract`.
- **Strip section labels from gold so the sweep is symmetric.** That is a different index
  (`chunk_instance` loses ADR-0005's citation offsets) and a change to the gold representation made
  in response to a gate reading. Refused on ADR-0009's logic: diagnostics do not get to edit the
  thing they diagnose.
- **Keep k=5 and report G1 as failed.** Defensible, and the fallback if a reviewer rejects the
  relaxation. Not chosen because 0.94/0.8752 at k=10 supports the attribution claims the paper
  actually makes, which read a passage set rather than a single top hit.
- **Tune the reranker's pool depth or τ until k=5 passes.** Prohibited outright. The gate stops
  being a measurement the moment it is fitted.
