# Decomposer & Granularity Freeze — declared 2026-08-23

**Status: FROZEN**, 8 days ahead of the ADR-0009 §8 Sep 3, 2026 planning date.

## Why now, not Sep 3

Sep 3 was a calendar buffer (three days before G2, four days before the Sep 7
annotation window), not itself a methodological condition. The condition ADR-0009
§8 and ADR-0005 actually impose is: **the decomposer's prompts and granularity
must not change once claim boundaries have been used to build the gold annotation
batch.** That condition is met today:

- `src/biomedqa/decompose.py`: last changed commit `2b00c69d9a5`, 2026-08-17
  ("generate, decompose: stop runaway claim four ways and set decoding penalty to
  0.5"). No change since.
- `src/biomedqa/prompts.py`: last changed commit `134ec7486704`, 2026-08-23
  ("fix(prompts): guard guided-JSON repair state and record recovery
  provenance"). Inspected: this touches only JSON-repair state initialization in
  `parse_response`, not `JOINT_JSON_TEMPLATE`, not `_claim_rules()`, not any
  claim-length or granularity guidance. It is the class of change HANDOFF.md:167
  pre-authorized as legitimate post-freeze ("a guided-decoding parse defect fix
  remains legitimate, but must not change claim-length guidance").
- The granularity parity loop (ADR-0009) closed 2026-08-14 at `parity_iter1b`.
- Gate G2 — which depends on the same decomposer output — **passed 2026-08-23**,
  two weeks ahead of its Sep 6 target, on `generate_fp05_n100_guided_v4`.
- The annotation batch actually shipped to annotators (`annotation/annotate_a1.html`,
  `_a2.html`, `_a3.html`, order_hash `42a52170009b`) was built 2026-08-23 from
  `docs/harvest/generate_fp05_n100_guided_v4.records.jsonl` — i.e. from this exact,
  already-stable decomposer output. There is no daylight between "the decomposer
  that produced the shipped batch" and "the decomposer as of this freeze."

Waiting until the calendar date Sep 3 would add no methodological guarantee not
already held today; it would only spend annotator time that doesn't need spending.

## What is frozen

| Component | Frozen commit | Date |
|---|---|---|
| `src/biomedqa/decompose.py` (decomposition prompt, granularity switch) | `2b00c69d9a503fba375a51d8747d3c0c3575f0c0` | 2026-08-17 |
| `src/biomedqa/prompts.py` (`_claim_rules()`, `JOINT_JSON_TEMPLATE`, shared claim grammar) | `134ec7486704cc1dd4eec5bfa456134d367e2f03` | 2026-08-23 (parse-repair fix only; no template/grammar edit) |

## Binding rule going forward (ADR-0009 §8, ADR-0005)

No edit to `decompose.py`'s prompt content, `prompts.py`'s `_claim_rules()` or
`JOINT_JSON_TEMPLATE`, or the `granularity` switch may land before Gate G4 closes.
Permitted: parser/repair bugfixes that do not alter claim-length or granularity
guidance (as above), scored and diffed against this table before merge. Any
prompt/grammar change requires discarding and re-annotating the affected claims
(ADR-0005) and is out of scope until after G4.

## Consequence for annotation start

This closes the "decomposer must be frozen before annotation opens" prerequisite.
It does not by itself open the pilot or main pass — see
`docs/harvest/w6_pilot_claims.md` for the pilot claim selection, and the
guideline-freeze step that follows pilot review (ADR-0006, ADR-0016).
