# ADR-0002 — Headline on attribution quality, cost as enabling modifier

**Status:** Accepted · **Date:** 2026-07-30 · **Decided in:** grilling session Q2

## Context

ADR-0001 limits the paper to one thesis. Three framings were available from the claim ledger, each
supported by the same run plan:

1. **Attribution quality** (C2 + C3) — joint per-claim grounding beats post-hoc citation.
2. **Cheap verifier** (C5 + C4) — a small entailment model matches an expensive LLM judge.
3. **Biomedical failure characterization** (C9 + C7) — what breaks, and why, in this domain.

## Decision

**Headline: attribution quality (C2 + C3), with cost as an enabling modifier.**

Title/abstract lead: *joint per-claim grounding beats post-hoc citation and cuts hallucination in
biomedical QA — at low overhead.*

## Consequences

- C5 (low overhead) is the **modifier**, not the claim. It must be measured well enough to survive
  scrutiny but does not carry the paper. See ADR-0004 and the overhead methodology in
  `research_roadmap.md` §4 Phase 3.
- The cost story is denominated in dollars, which is why Table 4 reports tokens and $ as primary.
- **Live risk:** C2 can return null at **gate G2 (Sep 6)**, leaving ~5 weeks and no thesis. The
  failure-characterization framing (#3) remains the natural fallback and shares the same run plan.
  It was deliberately **not** pre-declared as a formal contingency — but it is why the
  `CONTRADICTED` label is collected from the start (see `CONTEXT.md`), so the fallback stays
  available without a re-annotation.
- Baseline fairness becomes load-bearing: if C2 is the headline, a weak post-hoc baseline is fatal.
  See ADR-0004's consequences and `research_roadmap.md` Phase 2.

## Alternatives rejected

- **Cheap-verifier headline (C5 + C4).** Largely a domain-port of MiniCheck's already-proven result.
  Lower novelty in the lane `related_work.md` §7 identifies as open.
- **Biomedical-failure-characterization headline (C9 + C7).** Nearly unfailable, but abandons the
  system contribution entirely. Retained as the fallback, not the plan.
