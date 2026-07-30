# ADR-0006 — Three annotators, deliberately non-expert

**Status:** Accepted · **Date:** 2026-07-30 · **Decided in:** grilling session Q6, Q6a

## Context

ADR-0002 makes attribution quality the headline and the method carries no trained component
(see `research_roadmap.md` §1) — so **the evaluation carries the entire paper**. The gold-attribution
set is therefore the project's tightest constraint (ADR-0001).

The roadmap assumed 3 annotators on an overlap subset but never verified that three people exist.
It also contained an incoherent fallback: *"single annotator + 30-claim overlap for α"* — with one
annotator there is no overlap and Krippendorff's α is undefined.

Sizing: 250–400 claims with a ~20–25% overlap subset. The primary annotator labels everything;
annotators 2 and 3 label **only the overlap**.

## Decision

**Three annotators, real inter-annotator α. Annotators 2 and 3 are deliberately non-experts**,
working under strictly prescriptive guidelines with a **no-outside-knowledge rule**.

| Role | Claims | Time |
|---|---|---|
| Primary (project author) | all 250–400 | ~4–10 h |
| Annotator 2 | overlap only (~75) | **~3 h** |
| Annotator 3 | overlap only (~75) | **~3 h** |

α is computed on the overlap with a **bootstrap CI** — on ~75 units it is not a point estimate.

## Rationale for non-experts

The task is *"does this cited span support this claim?"* — reading comprehension under a
no-outside-knowledge rule — **not** *"is this claim medically true?"* A claim can be **false but
supported** and **true but unsupported**; the gold set must label both by what the span says.

**Domain expertise is a liability here.** The dominant failure mode in attribution annotation is an
expert marking a claim supported because it is *correct* rather than because the span asserts it.
That is not noise — it is a systematic bias toward the method looking better than it is, and it
inflates exactly the number C4 depends on.

This is also the published precedent (AIS / Attributed-QA use trained non-expert annotators), so it
is a norm rather than a concession.

## Consequences

- **Recruiting is on the critical path and has the longest lead time in the project.** The W6 pilot
  is Sep 7–13; guidelines are drafted in W5 (Aug 31 – Sep 6); **the ask goes out by ~Aug 6.**
  Slipping this slips G4, not W6.
- **The ask is "two literate people, ~3 hours, early September."** Non-expert eligibility makes the
  pool effectively unlimited, which largely retires the recruiting risk.
- **The guidelines carry all the load.** They must let a non-expert resolve hedging ("may reduce"),
  partial support, numeric claims, and jointly-necessary citations without domain knowledge.
  The **`SUPPORTED` vs `PARTIAL` boundary is where the pilot will fail if it fails** — concentrate
  effort there.
- **The W6 pilot tests the guidelines, not the annotators.** Poor pilot α ⇒ revise guidelines.
- **`CONTEXT.md` must stay tracked in git.** It is an input to the annotation protocol — external
  annotators read it. `teach/GLOSSARY.md` is gitignored and cannot serve this role.
- **Replacement fallback** (the incoherent one is retired): if annotators fail to materialize by
  Sep 7 — LLM-assisted pre-labeling + **full** human adjudication of every claim + a blind
  self-agreement re-annotation of ~50 claims ≥2 weeks apart, yielding **intra-annotator** α.
  Weaker than inter-annotator, but honest and computable.
- **If any LLM pre-labeling is ever used, it must not be Opus 5** — that would contaminate the gold
  set against the very judge C4 evaluates it against. Different model family, stated explicitly.

## Alternatives rejected

- **Solo, no α.** Silently deletes gate G4 and leaves C4 without an interpretability anchor.
- **LLM-assisted + intra-annotator α as the plan** (rather than the fallback). Defensible, but
  strictly weaker than real inter-annotator agreement when three people are available.
- **Paid crowd (Prolific etc.).** ~$150–300 plus qualification design, with high quality risk on
  biomedical entailment from unvetted annotators.
- **Domain experts.** Small, slow-to-recruit pool, and actively harmful for the bias reason above.
