# ADR-0010 — Abstention is derived at scoring time; the schema does not change

**Status:** Accepted · **Date:** 2026-08-04 · **Decided in:** grilling session (G0 follow-up)
**Closes the design half of** issue #9 · **Governed by** `schema.py`'s least-processed-value rule

## Context

The G0 bake-off surfaced a scorer bug: `score_compliance` counts a **correct abstention** as an
uncited claim. Left unfixed, Table 2 penalises a system for declining to answer and rewards one that
confabulates a citation — inverting the property this paper exists to measure.

The evidence is one question, `pubid 10781708`, and **the two models did not abstain in the same
shape.**

**Llama** (the chosen generator, ADR-0004 / D1) emitted the abstention as list item 11 — structurally
an ordinary claim, simply uncited:

> 10. The findings of the investigation were related to literature reports [2].
> **11. The question of whether prophylaxis in all patients makes sense is not addressed by the
> provided passages.**

**Qwen** did not emit a claim at all: five cited bullets, then a trailing prose paragraph *outside*
the list. The scorer reported `n_claims: 6, n_claims_cited: 5` — **the parser manufactured the sixth
claim out of that prose.**

Both abstentions were **partial**: substantive cited claims plus a statement of the gap. Full
abstention (zero substantive claims) has never been observed.

Any earlier framing that "both models abstained the same way" is wrong and should not be repeated —
Qwen's abstention was a parser artifact of the format break already noted in `research_roadmap.md`
§2.

## Decision

### 1. No `Claim.is_abstention` field. `SCHEMA_VERSION` stays `1.0.0`.

Abstention is **detected in `scoring/`**, as a versioned pure predicate over material the schema
already stores at 1.0.0: `Claim.text`, `Claim.citations`, and `QueryRecord.raw_generation`.

Three reasons, in order of weight:

1. **A stored boolean is a binarization.** A value derived by a rule and frozen into the record is
   exactly what the least-processed-value rule forbids. The distinction the field was meant to
   protect — abstention vs `claim_validity = false` — is best preserved by keeping *both* derivable
   from raw text, not by freezing one of them.
2. **A claim-level field cannot represent Qwen's shape at all.** Abstention can appear *outside* the
   claim structure, where no field on `Claim` can hold it. `raw_generation` captures it regardless.
   The raw material is strictly more expressive than the proposed field.
3. **The material is already sufficient for the generator we chose.** Llama's abstention is fully
   described by `Claim.text` plus `Claim.citations == []`.

**The detection rule is versioned in `RunConfig`** (`ScoringConfig.abstention_rule`), so a run always
records which rule produced its numbers, and re-scoring under a revised rule is a re-*score*, not a
re-*run*.

**Validation, due this week:** the predicate is checked against both `runs/g0/*.json` records — real
generator output containing both the list-item and trailing-prose forms. If it fails on either, the
raw material is insufficient and this decision is revisited **before** the gold set launches Sep 7.

*The honest cost of not bumping:* a schema change would have been free in August and expensive after
Sep 7. This ADR trades that insurance for the least-processed-value rule, and buys down the risk with
a validation run on n = 2 real records — which is two more than the alternative had.

### 2. Abstention claims are excluded from the **citation-recall denominator only**

Precision is unaffected **by construction**: an abstention claim carries zero citations, and
precision's denominator is the number of citations. Only recall's denominator — the number of claims
that ought to be cited — changes, from 11 to 10 on the Llama record above.

The rule is stated at exactly this precision because *"excluded from citation P/R denominators"* is
imprecise and would be implemented as touching both.

### 3. `abstention_rate` is a Table 2 column, and **both F1 numbers are always reported**

The naive fix **inverts the bug it fixes**: excluding abstentions from the recall denominator rewards
a system that abstains on hard claims. Today's scorer penalises correct abstention; the fix alone
would reward over-abstention. Both invert the property being measured.

Two structural protections, plus one reporting rule:

- **ADR-0009 §6** keeps the parity loop blind to F1, so prompts cannot be steered toward abstention
  during tuning.
- **`abstention_rate` is a Table 2 column** — but a column is a disclosure, not a guard.
- **Citation-F1 is reported on both denominators, always** — abstention-excluded as primary,
  abstention-included alongside it. **No threshold gates this.** Both are pure recomputations over
  identical stored records, so the cost is one extra call to a pure function, and there is no
  threshold to defend. A pre-committed number here would repeat the error of fixing a threshold
  against a quantity nobody has measured; a number chosen later would be chosen with the data in
  view.

### 4. No common-answered-set machinery

The earlier proposal — compute primary citation-F1 on the set of questions all three systems answered
— is **dropped**.

- **Vanilla carries no citations by construction** (`schema.validate()` enforces it), so conditioning
  a citation-F1 comparison on vanilla having answered shrinks the denominator for a system that
  contributes nothing to the metric.
- Full abstention is expected to be **rare**, so the record-level branch would almost never fire.
- With abstention claim-level, **the claim-level exclusion does all the real work.**

If a full abstention (zero substantive claims) does occur, the full-set number is reported as a
robustness check. The branch appears only if reality produces it.

### 5. `claim_validity` is not overloaded

An abstention is a well-formed, self-contained statement. Marking it invalid would corrupt ADR-0005's
decomposition-error rate, which is R10's only instrument.

## Consequences

- **`src/biomedqa/scoring/abstention.py`** holds the predicate; `scoring/citation.py` consumes it for
  the recall denominator. Both are pure functions over the schema.
- **`RunConfig` gains `ScoringConfig`**, carrying `abstention_rule` and its version. `CONFIG_VERSION`
  moves; `SCHEMA_VERSION` does not.
- **Issue #9 is larger than the scorer patch it describes** and its body is updated: the patch is
  §2 above, and §1/§3 are the design decisions around it.
- **Annotation guidelines (W5) must say what an annotator does with a sampled abstention claim.** The
  gold set samples ~4 claims per question (ADR-0011), so an abstention claim can be drawn.
- **The parser is a measurement instrument, not plumbing.** Qwen's phantom sixth claim shows the
  decomposer's handling of out-of-list prose changes the claim count and therefore every denominator.
  This is a `decompose.py` test case, not an edge case.

## Alternatives rejected

- **Add `Claim.is_abstention`, bump `SCHEMA_VERSION` to 1.1.0 now.** Cheap insurance in August against
  an expensive change in September — rejected because it stores a derived boolean, and because it
  cannot represent abstention that falls outside the claim list.
- **Instruct abstention explicitly in the prompt**, making the marker a raw observation rather than a
  detected one. Rejected: it would have to be added to all three systems for fairness, and it turns
  `abstention_rate` into a property of the prompt rather than a behaviour of the system. G0's
  abstentions were **spontaneous**, which is what makes them a measurement.
- **Overload `claim_validity`.** Corrupts the decomposition-error rate — see §5.
- **A pre-committed threshold on the joint-vs-post-hoc abstention gap.** Unmeasured quantity; see §3.
