# ADR-0018 — A re-decomposed claim's span is its source sentence, and `atomic` is bare by instruction

**Status:** Accepted · **Date:** 2026-08-15 · **Decided in:** `decompose.py` implementation session
**Refines** ADR-0005 (the attribution unit) · **Constrained by** ADR-0009 §8 (Sep 3 freeze)

## Context

`decompose.py` re-cuts an already-generated answer into C7's three ablation rows — `sentence`, bare
`atomic`, `decontextualized_atomic` (ADR-0005). Two choices made while writing it are not
implementation details: they change what `claim_validity` annotation can mean for a C7 row, and
what a C7 post-mortem can conclude. Both were decided in a docstring; this ADR is the half-page
record that makes them findable and revisable on purpose rather than by accident.

## Decision

### 1. A claim's span is the span of the sentence that produced it, and it is exact

`Claim.source_start` / `source_end` are offsets into the *sentence list the decomposer was shown*,
not into the claim's own text. This holds for every row that calls a model: `atomic` and
`decontextualized_atomic` both **rewrite** their input — "Metformin reduces mortality" is nowhere in
"It reduces mortality" — so there is no honest substring of the answer for the span to point at.
`sentence_units()` is used both to build the numbered list the decomposer reads and to cut the
`sentence` row, so the ablation row and the decomposer's own input are the same cut by construction.

The offsets therefore answer "which text produced this claim", not "which text does this claim
quote". `_tighten()` still makes `answer[start:end]` exact — it is an exact pointer to the source,
never a lossy or approximate one, and it never claims to be a quotation.

**Consequence:** `claim_validity` annotation (ADR-0005) on a C7 row cannot use span-vs-text
character equality as a well-formedness signal the way it implicitly can on the C2 headline row's
`sentence`-adjacent original claims. A decomposition-error post-mortem on `atomic` or
`decontextualized_atomic` reads the span as provenance, and reads the claim text on its own merits.

### 2. `atomic` is bare by instruction, and that is the ablation's validity condition

The `atomic` row exists to isolate what decontextualization buys, holding atomicity fixed. That only
measures decontextualization if `atomic` claims are *actually* bare — an instruction-tuned model
asked only to split naturally tidies away pronouns and implicit subjects on its own, and a model
that does so silently turns `atomic` into a second `decontextualized_atomic` row wearing a different
name. `BARE_ATOMIC_RULE` withholds decontextualization explicitly rather than by omission:
"do not resolve pronouns... copy the words as they stand."

This is not a generation-quality preference; it is the condition under which the three-row
comparison means what C7 claims it means. If a live run finds `atomic` claims converging with
`decontextualized_atomic` claims despite the rule, the ablation has failed its own validity
condition, not merely produced a disappointing number — see `scripts/decompose_smoke.py`, which
measures the fraction of `atomic` claims that differ from their source sentence beyond a split, for
exactly this failure mode.

## Consequences

- A C7 annotator reading `claim_validity` on a re-decomposed claim judges the claim text alone; the
  span is not evidence for or against well-formedness, only for "what did this come from."
- The `atomic` row's validity is conditional and measurable, not assumed. `decompose_smoke.py`'s
  divergence check is load-bearing for the paper's C7 table, not an optional diagnostic.
- Neither decision is reversible without re-annotating: both constrain what the Sep 7 gold set can
  be read to mean, same as ADR-0005's granularity freeze.

## Alternatives rejected

- **Span as a best-effort substring match inside the rewritten claim** (e.g. locate the longest
  common subsequence). Produces an offset that is sometimes exact and sometimes fabricated, with no
  way for a reader to tell which — worse than an offset that is honestly "the sentence, not the
  claim" every time.
- **No span at all for model-decomposed rows.** Discards the one thing that makes a decomposition
  error auditable (ADR-0005's stated purpose for carrying spans in the first place).
- **Trust the model to stay bare without an explicit rule.** Rejected on the general lesson already
  recorded twice in this codebase (`prompts.py`'s CITE-numbering history): an instruction-tuned model
  does not withhold a capability it has just because the prompt does not ask for it — withholding
  has to be stated, or it silently does not happen.
