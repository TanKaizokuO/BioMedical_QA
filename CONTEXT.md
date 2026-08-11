# CONTEXT — Project domain language

**Project:** Evidence-Grounded, Claim-Attributable Biomedical QA
**Status:** definitions locked 2026-07-30 (grilling session Q9a–c). **These four units *are* the
frozen schema.** Changing one after the gold set is annotated (W6, Sep 7) orphans the gold set.

> **Scope.** This file is the *project's* language, and it is **authoritative wherever it conflicts
> with other documents**. `teach/GLOSSARY.md` is a separate artifact: the *course* glossary, which
> describes the **literature's** vocabulary (ALCE, AIS, MiniCheck). It is gitignored and is not
> updated by project decisions. Where the two disagree, the disagreement is deliberate and is
> recorded under [Divergences from ALCE](#divergences-from-alce).
>
> This file is also an **input to the annotation protocol** — external annotators read these
> definitions. Keep it readable by someone with no biomedical background.

---

## The four units

### Claim — *the attribution unit*

A **decontextualized atomic claim**: a single factual assertion, stated so that it is fully
self-contained and can be judged without reading anything else in the answer.

- **Atomic** — one assertion. *"Metformin reduces mortality and improves glycaemic control"* is two
  claims, not one.
- **Decontextualized** — no dangling references. Every pronoun, definite description, and implicit
  subject is resolved.

| Not a claim (bare atomic) | A claim (decontextualized atomic) |
|---|---|
| "It reduces all-cause mortality." | "Metformin reduces all-cause mortality in patients with type 2 diabetes." |
| "This was not observed in the elderly." | "Metformin's mortality benefit was not observed in patients over 75." |

**Why decontextualized:** a bare atomic claim with an unresolved pronoun is unverifiable in
isolation (DnDScore) — and an annotator instructed to use no outside knowledge cannot judge it
either. The verifier takes `premise = cited span, hypothesis = claim`; if the hypothesis is not
self-contained, "does the span entail it?" has no determinate answer.

`sentence` and bare `atomic` remain available as granularity settings, but only as **ablation
rows** (claim C7). The headline configuration is decontextualized atomic.

---

### Claim validity

A binary flag on every claim, annotated alongside support: **is this a well-formed, self-contained
claim at all?**

Not a judgement about the claim's truth or its support — only about whether decomposition produced
something judgeable. Malformed, over-split, fragmentary, or still-context-dependent output is
`invalid`.

**Why it exists:** decomposition quality is an upstream confound on every headline number. A
malformed claim moves C2 and C3 for reasons unrelated to joint grounding. This flag converts a
hidden confound into a **reportable decomposition-error rate**, and permits headline numbers to be
computed over well-formed claims only.

#### A claim that declines to answer is still `valid`

You will occasionally meet a claim like *"The question of whether prophylaxis helps all patients is
not addressed by the provided passages."* It carries **no citation**, because there is nothing for it
to cite.

**Mark it `valid`.** It is well-formed and self-contained — it just reports an absence instead of
asserting a fact. It is a *correct* thing for a system to say, and marking it invalid would
misattribute it to a decomposition failure.

Note the difference from an ordinary negative claim. *"Metformin does not reduce mortality"* is a
substantive assertion about the world and is labelled for support in the normal way; *"the passages
do not mention mortality"* is a statement about the passages themselves. Only the second is a
declining-to-answer claim.

*(Scoring keeps these separate automatically and leaves them out of the citation-recall count —
ADR-0010. Nothing about that changes what you do here.)*

---

### Citation

A **character span in a retrieved passage**: `{passage_id, char_start, char_end}`.

A claim carries a **list** of citations — **at most 3**.

**Multi-citation semantics (ALCE):**

- **Citation recall** — the claim is covered iff the **union** of its cited spans entails it:
  `recall(c) = 1 iff C ≠ ∅ ∧ φ(concat(C), c) = 1`
- **Citation precision** — a citation `x` is *irrelevant* iff it fails alone **and** the rest
  already suffice: `φ(x, c) = 0 ∧ φ(concat(C \ {x}), c) = 1`. Precision is the fraction of
  citations that are not irrelevant.
- **Citation F1** — harmonic mean of corpus-level precision and recall. This is the number that
  resists cite-everything gaming; recall alone does not.

**The 3-citation cap is a fairness control, not a formatting detail.** Uncapped, recall is trivially
gamed by citing every retrieved passage. **The cap must be identical in the prompt for every
system** — ours, post-hoc, and vanilla. An unequal cap makes C2's gap an artifact of citation budget
rather than of joint grounding.

*Jointly necessary citations are legitimate:* a claim whose dose comes from one span and whose
outcome comes from another is correctly cited with both. The remove-it-and-see rule above handles
this — neither citation is irrelevant, because removing either breaks entailment.

---

### Supported — *the label set*

Human annotation assigns each **(claim, cited span)** pair exactly one of four labels:

| Label | Meaning |
|---|---|
| `SUPPORTED` | The span asserts the claim. |
| `PARTIAL` | The span asserts part of the claim, or asserts it more weakly/narrowly than stated. |
| `NOT_SUPPORTED` | The span does not address the claim. |
| `CONTRADICTED` | The span asserts the **opposite** of the claim. |

**Judge attribution, not truth.** A claim may be **false but supported** (the passage says it) and
**true but unsupported** (correct medicine, absent from the cited span). Both must be labeled by
what the span says. Annotators use **no outside knowledge**.

**Why four labels rather than two:**
- `CONTRADICTED` is the payload of the biomedical failure-mode analysis (negation, numerics,
  scope/population are exactly the cases where a span asserts the opposite, not merely nothing).
  Collapsing it into `NOT_SUPPORTED` destroys that analysis **at annotation time**, and an
  annotator cannot be re-run.
- `PARTIAL` is what makes citation precision honest. Without it, every borderline case is silently
  forced into one bucket by the annotator and the split is never visible.

**Collapse rule.** The binary quantity the verifier is evaluated against is:

```
supported_binary = (SUPPORTED | PARTIAL)   vs   (NOT_SUPPORTED | CONTRADICTED)
```

Krippendorff's α for **gate G4 (α ≥ 0.6)** is computed on this **binary collapse** — it is the
quantity C4 actually consumes. The 4-way ordinal α is reported as a **secondary** number. Gate on
what you measure with; report the finer-grained agreement honestly alongside it.

**The hard boundary is `SUPPORTED` vs `PARTIAL`.** `NOT_SUPPORTED` vs `CONTRADICTED` is
mechanical ("does the span assert the opposite?"). Guideline-writing effort in W5 concentrates on
the first, and it is where the W6 pilot will fail if it fails.

---

## Annotation record

Per (claim, cited span) pair — three fields, no free text required:

```jsonc
{
  "claim_id": "...",
  "citation": { "passage_id": "...", "char_start": 0, "char_end": 0 },
  "claim_validity": true,                 // per claim
  "support_label": "SUPPORTED"            // SUPPORTED | PARTIAL | NOT_SUPPORTED | CONTRADICTED
}
```

Plus one **union judgement per claim** (does the concatenation of all its cited spans entail it?),
which is what citation recall is scored on.

**Never binarize at write time.** Store the 4-way label, raw verifier scores, and character offsets.
Thresholds and collapses live in scoring. Binarizing on write destroys the AUROC sweep and the
calibration bins irrecoverably.

---

## Divergences from ALCE

Declare these explicitly in the paper. Reviewers who know ALCE will look for exactly this section.

| | ALCE / `teach/GLOSSARY.md` | This project | Why |
|---|---|---|---|
| **Attribution unit** | *Statement* = each **sentence** of the answer; "claim" reserved for decomposed atomic facts — the two are **distinct** | Statement and claim are **merged**: the attribution unit **is** the decontextualized atomic claim | The method generates claims directly; there is no separate sentence layer to score. Sentence-level attribution is retained only as a C7 ablation row. |
| **Verifier vs. φ** | "NLI judge (φ)" is the entailment primitive; *"avoid: verifier, fact-checker"* | Both terms used, with fixed roles: **φ** = the entailment primitive; **the verifier** = the component built on it (MiniCheck-Flan-T5-Large + threshold + scoring) | The paper needs to name a system component, not only a function. |

Citation precision/recall semantics are **unchanged** from ALCE and are reused verbatim.

---

## Related decisions

Architectural decisions that produced or constrain these definitions live in [`docs/adr/`](docs/adr/).
The most relevant are ADR-0005 (attribution unit), ADR-0006 (annotation protocol) and ADR-0016
(all three annotators label the full gold set). Nothing on this page changes with ADR-0016 — it
settles who labels how much, not what a label means.
