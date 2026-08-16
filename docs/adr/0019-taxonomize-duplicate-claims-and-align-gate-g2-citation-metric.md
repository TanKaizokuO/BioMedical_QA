# ADR-0019 — Taxonomize duplicate claims and align Gate G2 citation metric

**Status:** Accepted · **Date:** 2026-08-16 · **Decided in:** C7 decomposition-quality gap resolution session
**Refines** ADR-0005 (the attribution unit), ADR-0018 (claim validity conditions) · **Constrained by** ADR-0009 §8 (Sep 3 freeze)

## Context

Closing the C7 decomposition-quality gap before the Sep 3 2026 freeze (ADR-0009 §8) and Sep 6
Gate G2 requires achieving `clean_decompose_rate >= 0.95` and `clean_cite_rate >= 0.95` on the
`atomic` and `decontextualized_atomic` rows of `scripts/decompose_smoke.py`.

Guided JSON decoding (`c614589`) eliminated citation parsing errors: `quote_located_rate`,
`claim_parse_rate`, and `clean_cite_rate` all reached 1.0 on a live Llama-3.1-8B sanity run (n=3).
Following this fix, the single remaining failure mode impacting `clean_decompose_rate` is
**duplicate claim text**.

An empirical audit of duplicate claims across the pre-`35db468` baseline (n=100) and the current
guided generator (n=3) revealed two distinct duplicate populations, exposing flaws in treating
all duplicate claim text as non-terminating generation loops. Concurrently, evaluating citation
quality highlighted an asymmetry between all-or-nothing per-query metrics and ROADMAP §1's per-claim
specification for Gate G2.

This ADR records the two decisions that resolve these metric artifacts while preserving strict,
un-inflated quality standards.

## Decision

### 1. An exact-duplicate claim is classified by its cause, not treated uniformly as a non-terminating generation

Exact-duplicate claim text within a query and granularity cell is categorized into a three-part
taxonomy based on underlying cause:

1. **Same-reply repeat (intra-sentence redundancy):** Duplicate claim text produced within a single
   sentence decomposition call (e.g., Case A: `10757151:atomic`, claims c3 and c4 from source span
   `614–728`: *"Intraischemic preconditioning (IIP) without reperfusion before the index ischemia
   still provides cardioprotection"*). Claims form a mathematical set; the second copy carries no
   new information. It is collapsed to a single claim and recorded in `Decomposition.recovered`.
2. **Cross-sentence repeat (inter-sentence paraphrase / converse canonicalisation):** Duplicate
   claim text originating from distinct source sentences (e.g., Case B: `10490564:decontextualized_atomic`,
   claims c5 and c7 from spans `1555–1712` and `2331–2492`). Both claims are retained because their
   provenance spans differ, and the occurrence is recorded in `Decomposition.recovered`.
3. **Multiplicity ≥ 3 (unconstrained repetition loop):** Claim text repeating 3 or more times
   *within one sentence's reply*, or spanning 3 or more *distinct source sentences*, is classified
   as a non-terminating generation error and appended to `Decomposition.errors`. The two counters
   are separate because they detect different loops: one decoder call failing to halt, versus a
   text that keeps reappearing as the decomposer walks the answer.

#### Empirical basis for the ×3 threshold

The pre-`35db468` baseline (n=100, `docs/harvest/decompose_smoke.records.jsonl`) exhibited two
distinct populations across claim multiplicities:

- `sentence`: `{1: 1082, 2: 40, 3: 3, 4: 3, 5: 1, 9: 2, 10: 2, 20: 1}`
- `atomic`: `{1: 1105, 2: 157, 3: 22, 4: 5, 5: 2, 6: 4, 7: 2, 8: 2, 10: 1, 23: 1}`
- `decontextualized_atomic`: `{1: 244, 2: 20, 3: 2, 4: 1, 6: 1, 18: 1}`

A large ×2 mode was accompanied by a sparse, long tail (×9, ×10, ×18, ×20, ×23) representing genuine
unconstrained repetition loops. An unconstrained loop does not halt after emitting exactly one
extra copy; thus, multiplicity ≥ 3 empirically separates true generation loops from set redundancy
or converse sentence canonicalisation. Under the current guided generator (n=3), every duplicate is
multiplicity 2, with zero cells exhibiting multiplicity ≥ 3.

#### Case B in full: converse-sentence canonicalisation

Case B (`10490564:decontextualized_atomic`, c5 and c7) illustrates how the previous rule penalized
correct model behaviour. The two source sentences read:

- Span (1555, 1712): *"The presence of right ventricular involvement in inferior wall acute
  myocardial infarction is associated with a marked hypotensive response to nitroglycerin."*
- Span (2331, 2492): *"A marked hypotensive response to nitroglycerin in patients with inferior wall
  acute myocardial infarction suggests the presence of right ventricular involvement."*

These two source sentences are converses of one proposition. The `decontextualized_atomic`
decomposer **correctly** canonicalised both sentences to the same claim text—which is precisely the
intended function of decontextualisation. Charging this as `"repeats c5's claim text verbatim across
sentences N and M (non-terminating generation)"` created a metric false positive where superior
decontextualisation degraded the row's clean score.

#### Evolution of the standard and anti-inflation invariants

The previous rule—treating any cross-sentence verbatim repeat as a non-terminating generation error—was
defensible when written, as it targeted the ×20 and ×23 loops seen in the pre-`35db468` baseline.
What changed is the empirical evidence following guided generation fixes, not our quality standard.

To ensure deduplication and reclassification never quietly inflate `clean_decompose_rate`, two
invariants are enforced:

- Every collapsed same-reply repeat and every reclassified cross-sentence repeat MUST land in
  `Decomposition.recovered`, surfaced in summary metrics as `decompose_recovered_count` and
  `decompose_recovered_kinds`.
- `duplicate_claim_count` MUST continue to count every duplicate occurrence regardless of whether
  it was routed to `errors` or `recovered`.

### 2. Gate G2's citation bar is read per claim, not per query ("Option H")

In `scripts/decompose_smoke.py`, `clean_cite_rate` evaluates citation fidelity as an all-or-nothing
metric per query: a single drifted quote out of sixty citation lines fails the entire query.
Consequently, `clean_cite_rate` conflates answer length with citation fidelity.

ROADMAP §1 explicitly specifies Gate G2's citation quality bar on a per-claim basis: *"≥95% valid
claim parse"*.

Under Decision 2 ("Option H"):

- `quote_located_rate` and `claim_parse_rate`—already computed on their own per-claim denominators in
  `scripts/decompose_smoke.py`—are established as the official reported Gate G2 figures for
  citation quality.
- `clean_cite_rate` is retained as a strictly stronger secondary diagnostic reported in summary
  tables, but it is not a gating threshold for Gate G2.

This reporting alignment is adopted **before** reading the full n=100 smoke test run, ensuring it
is a pre-committed structural definition rather than a post-hoc adjustment fitted to empirical
results. Live n=3 sanity evidence (`quote_located_rate` = 1.0, `claim_parse_rate` = 1.0,
`clean_cite_rate` = 1.0, 0 unlocated quotes) demonstrates that both metrics agree when citation
quality is high.

## Consequences

- **C7 Post-Mortem Interpretation:** A high `clean_decompose_rate` certifies structural well-formedness
  and freedom from non-terminating loops. However, post-mortem evaluations must inspect non-fatal
  drift in `decompose_recovered_count` and `cite_recovered_count` alongside clean rates; clean rates
  may never be interpreted in isolation as proof of zero textual drift.
- **Baseline Freeze Rule:** `docs/harvest/decompose_smoke.*` must not be committed as an official
  baseline while rates remain below 0.95.
- **Sep 3 Freeze Constraint:** Per ADR-0009 §8, decomposer prompt structures and pipeline logic
  remain strictly frozen on Sep 3 2026 to protect the gold set annotation launch on Sep 7. All
  taxonomy implementations and metric configurations must be finalized prior to this date.

## Alternatives rejected

- **Widening quote matching via fuzzy/LCS string matching.** Rejected because approximate matching
  masks hallucinated quotes and corrupts exact substring provenance invariants established in
  ADR-0018 §1.
- **Uniformly ignoring all duplicate claims.** Rejected because multiplicity ≥ 3 represents genuine
  model failure (repetition loops) that must be caught and penalized.
- **Gating Gate G2 on `clean_cite_rate`.** Rejected because all-or-nothing per-query gating creates
  an artificial penalty proportional to answer length, contradicting ROADMAP §1's per-claim
  specification.
