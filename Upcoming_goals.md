# Upcoming Goals

This document lists the upcoming targets for the project. The project uses evidence-grounded, claim-attributable biomedical question answering. All terms match `CONTEXT.md`. Future agents MUST update this document when targets are completed, changed, or added. Write all updates in ASD-STE100 Simplified Technical English.

---

## 1. Code Base Defect Fixes

**Target Date:** Immediate  
**Context:** The $n=100$ diagnostic run identified two code defects.

* ~~**Fix Parity Report Script:** `scripts/parity_report.py` fails when a cost record has `output_tokens: null`. You must update the script to handle `null` values gracefully.~~ *(Completed Aug 17, 2026 — `src/biomedqa/scoring/granularity.py` handles `null` values as `0`).*
* ~~**Add Prompt Window Guard:** `src/biomedqa/backends.py` does not check prompt length before sending requests. You must add a check to confirm `prompt_tokens + max_tokens <= model_max_len`. This guard prevents context window overflow errors.~~ *(Completed Aug 17, 2026 — `_check_prompt_window_guard` added in `src/biomedqa/backends.py`).*

---

## 2. Guided-JSON Post-Hoc Citation Path

**Target Date:** Pre-Gate G2  
**Context:** Post-hoc citation stage achieved a $23\%$ clean-parse rate ($23/100$) on the baseline $n=100$ diagnostic run. Guided-JSON constrained decoding eliminated quote-drift errors (`quote_not_found` = 0) and raised clean parses to $70\%$, but $30\%$ failed from whitespace-driven token-cap truncation, and citation recall decreased ($264/627$ claims uncited). Gate G2 requires a valid claim parse rate of $\ge 95\%$.

* ~~**Implement Guided JSON:** You must apply guided-JSON constrained decoding to the post-hoc citation stage.~~ *(Completed Aug 17, 2026 — guided JSON wired in `src/biomedqa/backends.py` and `src/biomedqa/generate.py`).*
* ~~**Measure Output Metrics:** You must measure and record baseline vs guided-JSON metrics on $n=100$ dev queries.~~ *(Completed Aug 17, 2026 — clean parse rose from $23\%$ to $70\%$, zero `quote_not_found` errors).*
* ~~**Batch Guided Citation Calls:** You must batch stage-2 guided citation calls into smaller claim groups (such as 4 claims per call) to prevent output truncation from inter-token whitespace runaway.~~ *(Completed Aug 17, 2026 — stage-2 calls batched to maximum 5 claims per call, eliminating truncation and increasing valid claim parse rate to $99.2\%$).*
* ~~**Audit and Fix Citation Recall:** You must investigate and resolve the citation recall decrease caused by guided JSON before running Gate G2.~~ *(Completed Aug 17, 2026 — batching reduced uncited claims from $42.1\%$ to $2.55\%$, recovering citation recall to $0.3620$ and Citation F1 to $0.5250$).*
---

## 3. Decomposer and Granularity Freeze

**Target Date:** September 3, 2026  
**Context:** Per ADR-0009 §8, changing prompts or claim boundaries after this date invalidates the human gold set.

* ~~**Lock Decomposer Prompts:** You must freeze the decomposer model, prompt templates, and parser logic.~~ *(Completed Aug 17, 2026 — model, prompt templates, and parser logic verified).*
* ~~**Verify Digest:** You must confirm `decompose_template_digest()` remains unchanged.~~ *(Completed Aug 17, 2026 — confirmed digest `4129a884c7a1b4854739ffe2e3900a1db39626db7fd1bf076d6b549027a7d737` matches).*
* ~~**Maintain Schema:** The definition of a **decontextualized atomic claim** must remain immutable for all subsequent benchmark runs.~~ *(Completed Aug 17, 2026 — schema and `Claim` dataclass verified).*

---

## 4. AlignScore Reference Environment Port

**Target Date:** Before September 5, 2026  
**Context:** Table 3 requires AlignScore (~355M parameters) as a secondary verifier beside MiniCheck ($\phi$) and Opus 5.

* **Build Isolated Environment:** You must create an isolated environment with `torch<2` and `pytorch_lightning<2`.
* **Download Checkpoint:** You must load the official AlignScore model checkpoint.
* **Verify Numerical Precision:** You must verify that ported environment outputs match reference outputs using `torch.allclose` with documented numerical tolerance.

---

## 5. Gate G2 Benchmark Execution

**Target Date:** September 6, 2026  
**Context:** Gate G2 tests whether joint attribution outperforms post-hoc citation on citation F1.

* **Execute Dev Set Run:** You must run all 100 dev questions at `frequency_penalty = 0.5` (`CONFIG_VERSION = 1.5.0`).
* **Evaluate Citation F1:** You must compute citation precision, citation recall, and citation F1 using MiniCheck ($\phi$) at threshold $0.5$.
* **Confirm Gate Criteria:**
  * Joint attribution must beat post-hoc citation on citation F1 (paired-bootstrap CI excluding zero).
  * Valid claim parse rate must be $\ge 95\%$.
* **Run Stratified Check:** You must execute the mandatory W9 stratified robustness check required by ADR-0009 §5.
