# ADR-0005 — The attribution unit is the decontextualized atomic claim

**Status:** Accepted · **Date:** 2026-07-30 · **Decided in:** grilling session Q9a–c

## Context

The roadmap simultaneously **fixed** granularity in its thesis ("answers as atomic claims") and
**swept** it as an ablation (C7: `sentence` / `atomic` / `decontextualized-atomic`). Both are fine —
but one setting must be the **headline configuration**, and it was named nowhere.

This is forced now rather than later because **the gold set is annotated at exactly one
granularity**. Annotators label claims a specific decomposer emitted. Changing the headline
granularity after W6 (Sep 7) orphans the gold set entirely.

ALCE — the closest comparison — scores attribution on **sentences**, and `teach/GLOSSARY.md`
faithfully records that, keeping "statement" and "claim" as distinct units.

## Decision

**The attribution unit is the decontextualized atomic claim.** Statement and claim are **merged**;
`sentence` and bare `atomic` survive only as C7 ablation rows.

Two supporting decisions, made at the same time:

- **Support is labeled on a 4-way scale** — `SUPPORTED` / `PARTIAL` / `NOT_SUPPORTED` /
  `CONTRADICTED` — with **G4's α ≥ 0.6 computed on the binary collapse** (SUPPORTED+PARTIAL vs
  NOT_SUPPORTED+CONTRADICTED), the quantity C4 actually consumes.
- **Citations use ALCE multi-citation semantics with a hard cap of 3.**

Full definitions live in [`CONTEXT.md`](../../CONTEXT.md).

## Consequences

- **Decomposition quality becomes an upstream confound on every headline number.** A malformed or
  over-split claim moves C2 and C3 for reasons unrelated to joint grounding.
  **Mitigation:** a `claim_validity` flag annotated alongside support (~10% extra annotation time),
  converting a hidden confound into a reportable decomposition-error rate.
- **The 3-citation cap must be identical across ours / post-hoc / vanilla.** Unequal caps make C2's
  gap an artifact of citation budget. This is the tightest instance of the baseline-fairness
  requirement (ADR-0002).
- **The annotation unit becomes the (claim, cited span) pair**, not the claim. 75 overlap claims
  ≈ 150–225 pair judgements plus 75 union judgements — which is why the annotator ask is ~3 h, not
  ~1–2 h (ADR-0006).
- **`CONTRADICTED` cannot be recovered later.** An annotator cannot be re-run; a label not collected
  in W6–W8 is gone. It is the payload of the biomedical failure-mode analysis and of ADR-0002's
  fallback framing.
- **A "Divergences from ALCE" paragraph is now required in the paper** — the merged statement/claim
  unit, and the verifier-vs-φ naming. Reviewers who know ALCE will look for exactly it. Drafted in
  `CONTEXT.md`.
- The thesis sentence in `research_roadmap.md` §1 must name the headline granularity explicitly.

## Alternatives rejected

- **Sentence as the headline unit.** Lowest annotation burden and closest to MiniCheck's training
  format, but too coarse: a sentence with one true half and one false half has no correct label,
  which directly weakens C3.
- **Bare atomic (not decontextualized).** Ruled out by evidence already in the plan — DnDScore shows
  atomic claims with dangling pronouns are unverifiable standalone. It is also incompatible with
  non-expert annotation (ADR-0006): *"it reduces mortality"* cannot be judged without outside
  knowledge; *"metformin reduces all-cause mortality in type 2 diabetes patients"* can.
- **Any-of citation semantics** (supported if *any* cited span entails). Easier to annotate, but it
  makes multi-citation free — three spans of which one entails would score identically to a precise
  single citation, erasing the exact quality difference C2 measures. It would also diverge silently
  from a metric the paper labels "ALCE-style."
