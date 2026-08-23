# W6 pilot claim selection — declared 2026-08-23

**Selection rule used:** the literal head of the shared seeded question order
already embedded in the shipped forms (`order_hash 42a52170009b`,
`ANNOTATION_SEED 20260907`) — not a hand-picked set from elsewhere in the batch.

## Why the literal prefix, not a curated set

ADR-0016 §2's fold-in property only holds if the pilot *is* the start of the
same pass: "any common completed prefix forms a statistically valid random
subsample," and pilot claims count toward the ≥250-claim gold set unless the
guidelines are revised afterward (ADR-0013, ADR-0016). Hand-picking claims from
elsewhere in the 100-question batch would (a) violate ANNOTATOR_GUIDE.md:27's
"work in the order the form gives you, do not skip ahead," and (b) break that
fold-in property, forcing the pilot claims to be thrown away even if the
guidelines survive review unchanged. No selection algorithm for the pilot is
specified anywhere else in the repo (ADR-0006, ADR-0013, ADR-0016 name the pilot
but not a sampling method), so "first N in the shared order" is the only
selection rule consistent with the mechanism the rest of the protocol already
commits to.

## Selected pilot set

**Question 1 of the shared order** (`order_index 0`, `question_uid
q_aaf5e7c879b4`, source `query_id 23735520`):

> "Can mental imagery functional magnetic resonance imaging predict recovery in
> patients with disorders of consciousness?"

11 claims (both systems pooled and blinded, per ADR-0016 §4):

| unit_id | claim text | boundary relevance |
|---|---|---|
| `u_1c07285867dc` | Mental imagery fMRI paradigms can detect residual awareness and covert command-following in patients thought entirely vegetative. | population/scope |
| `u_c064b37706e9` | Activation of ROIs in mental imagery fMRI is associated with recovery from minimally conscious state. | hedge ("associated with") vs causal wording |
| `u_557fbe26644a` | The specificity of mental imagery fMRI paradigms for predicting recovery from MCS is 40%. | numeric — exact figure claim |
| `u_b17dbadf1381` | fMRI can be used to identify patients with disorders of consciousness who show potential for recovery. | general/umbrella claim |
| `u_8ebe800c3bd7` | Mental imagery fMRI can identify patients with disorders of consciousness who show potential for recovery. | near-duplicate of above, different system |
| `u_97ecdf89f6fd` | In patients with vegetative state (VS), activation in ROIs during mental imagery fMRI is associated with recovery. | population term: VS specifically |
| `u_49e8f6e15c65` | The sensitivity of mental imagery fMRI paradigms for predicting recovery from MCS is 85%. | numeric — exact figure claim |
| `u_39d9116951ea` | The sensitivity and specificity of mental imagery fMRI for predicting recovery are high. | strength-of-language ("high" vs the exact 40%/85% figures above) — direct SUPPORTED-vs-PARTIAL drill |
| `u_15b8a4aca33c` | In patients with MCS, activation in ROIs during mental imagery fMRI is associated with emergence from MCS. | population term: MCS specifically (contrast with VS claim above) |
| `u_29e3b98df422` | VS patients who show significant BOLD signal activation in mental imagery fMRI can recover to at least MCS. | scoped/conditional claim |
| `u_5c0480d6243c` | VS patients who fail to show significant BOLD signal activation in mental imagery fMRI do not recover from VS. | negated claim — CONTRADICTED-boundary candidate |

## Why this question exercises the load-bearing boundary

Without any cherry-picking beyond taking the literal first question, this set
already contains:

- Two independent **numeric-figure vs vague-strength** pairs (`u_557fbe26644a`
  "40%" / `u_49e8f6e15c65` "85%" vs `u_39d9116951ea` "are high") — this is
  exactly ANNOTATOR_GUIDE.md's Numbers trap and Strength-of-language trap
  co-located on the same underlying finding, the single best stress test of the
  SUPPORTED/PARTIAL boundary in the batch.
- A **population/scope minimal pair** (VS-specific vs MCS-specific claims about
  the same ROI-activation finding), exactly the Population trap.
- A **negated claim** (`u_5c0480d6243c`) that is the best available candidate in
  this batch for exercising `CONTRADICTED`, which the guideline currently
  documents with only one synthetic (non-live-decomposer) example.
- Two **near-duplicate claims from different systems** (`u_b17dbadf1381` /
  `u_8ebe800c3bd7`), useful for checking that blinding and per-claim
  independence hold in practice.

## Deviation from the ADR-0013 sizing estimate — report to maintainer, not silently absorbed

ADR-0013 estimated "10 claims across ~5 questions." The literal first question
alone already carries 11 claims (both systems pooled). Measured density across
the first 6 questions of the actual shipped batch is 11, 8, 15, 11, 12, 12 —
mean ≈ 11.5 claims/question, roughly triple ADR-0013's planning estimate. Taking
literally "~5 questions" instead would produce a ~57-claim, multi-hour pilot,
contradicting the ADR's own "~1 h" target. This selection follows the tighter,
still-explicit "10 claims, ~1 h" anchor and treats "~5 questions" as the
approximation that was wrong, since claim density is exactly the kind of
planning assumption the pilot exists to correct (ANNOTATOR_GUIDE.md:176-177:
"that number is our estimate, not a measurement — which is exactly what the
pilot exists to check"). **Flag for the maintainer:** if 11 claims takes
meaningfully longer than 1 h per annotator, the 10–16 h main-pass budget
(ADR-0016) should be re-derived from the measured per-claim time, not the
original estimate.

## Procedure from here (unchanged from ADR-0006 / ADR-0013 / ADR-0016)

1. All three annotators complete exactly this one question (order_index 0) in
   their existing forms and stop — do not proceed to question 2 until cleared.
2. Maintainer computes Krippendorff's α (binary collapse) on the 11×3 label set
   and reviews qualitatively (boundary calls, blinding integrity, `active_s`
   timing).
3. Clean → freeze `CONTEXT.md`/`ANNOTATOR_GUIDE.md` at their current commit,
   record the hash, and all three annotators continue from question 2 onward in
   the same session — question 0's labels count toward the ≥250-claim gold set.
4. Ambiguous / α below the ADR-0011 0.6 heuristic → revise guidelines, discard
   question 0's labels (ADR-0013, ADR-0016), re-freeze, and all three annotators
   restart from question 0 under the revised guide.
