# ADR-0001 — Target a workshop with a hard November deadline

**Status:** Accepted · **Date:** 2026-07-30 · **Decided in:** grilling session Q1

## Context

Project 2 needed a venue route before any table could be designed, because page limits determine
how many tables the paper can carry — and the roadmap's claim ledger assumed nine.

Two routes were genuinely available:

- **Workshop / conference with a fall deadline** — BioNLP-style workshop or a medical-AI venue.
- **Journal** (*JBI*, *JAMIA*, applied-AI) — rolling deadlines, stronger line on a CV, acceptance
  lands in 2027.

## Decision

**Workshop, hard November date.** A dated outcome inside this semester.

## Consequences

- **~8 pages ⇒ 4–5 tables, not 9.** The claim ledger must be **cut**, not merely prioritized. This
  is the origin of the reduced ledger in `research_roadmap.md` §1.
- **One thesis, not three.** See ADR-0002.
- **The human gold-attribution set becomes the tightest constraint in the project** — it has the
  longest lead time and cannot be compressed. See ADR-0006.
- Table and figure count is a hard budget, enforced by the cut order in `research_roadmap.md` §8.
- A journal extension in 2027 remains open, carrying the BioASQ generalization and a larger gold
  set. This is the natural home for work cut here.

## Alternatives rejected

**Journal.** Stronger publication, but acceptance in 2027 and no dated outcome this semester. The
deciding factor was wanting a result inside the current cycle; the journal route is deferred rather
than discarded.
