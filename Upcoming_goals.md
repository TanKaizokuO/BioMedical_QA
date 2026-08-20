# Upcoming Goals

This document lists the upcoming targets for the project. The project uses evidence-grounded, claim-attributable biomedical question answering. All terms match `CONTEXT.md`. Future agents MUST update this document when targets are completed, changed, or added. Write all updates in ASD-STE100 Simplified Technical English.

---

## 1. Code Base Defect Fixes

**Target Date:** Completed  
**Context:** The $n=100$ diagnostic run identified two code defects.

* ~~**Fix Parity Report Script:** `scripts/parity_report.py` fails when a cost record has `output_tokens: null`. You must update the script to handle `null` values gracefully.~~ *(Completed Aug 17, 2026 — `src/biomedqa/scoring/granularity.py` handles `null` values as `0`).*
* ~~**Add Prompt Window Guard:** `src/biomedqa/backends.py` does not check prompt length before sending requests. You must add a check to confirm `prompt_tokens + max_tokens <= model_max_len`. This guard prevents context window overflow errors.~~ *(Completed Aug 17, 2026 — `_check_prompt_window_guard` added in `src/biomedqa/backends.py`).*

---

## 2. Guided-JSON Post-Hoc Citation Path

**Target Date:** Completed  
**Context:** Post-hoc citation stage achieved a $23\%$ clean-parse rate ($23/100$) on the baseline $n=100$ diagnostic run. Guided-JSON constrained decoding eliminated quote-drift errors (`quote_not_found` = 0) and raised clean parses to $70\%$, but $30\%$ failed from whitespace-driven token-cap truncation, and citation recall decreased ($264/627$ claims uncited). Gate G2 requires a valid claim parse rate of $\ge 95\%$. The final batched run achieved a valid claim parse rate of $99.2\%$, a citation F1 of $0.525$, and $0$ `quote_not_found` errors.

* ~~**Implement Guided JSON:** You must apply guided-JSON constrained decoding to the post-hoc citation stage.~~ *(Completed Aug 17, 2026 — guided JSON wired in `src/biomedqa/backends.py` and `src/biomedqa/generate.py`).*
* ~~**Measure Output Metrics:** You must measure and record baseline vs guided-JSON metrics on $n=100$ dev queries.~~ *(Completed Aug 17, 2026 — clean parse rose from $23\%$ to $70\%$, zero `quote_not_found` errors).*
* ~~**Batch Guided Citation Calls:** You must batch stage-2 guided citation calls into smaller claim groups (such as 4 claims per call) to prevent output truncation from inter-token whitespace runaway.~~ *(Completed Aug 17, 2026 — stage-2 calls batched to maximum 5 claims per call, eliminating truncation and increasing valid claim parse rate to $99.2\%$).*
* ~~**Audit and Fix Citation Recall:** You must investigate and resolve the citation recall decrease caused by guided JSON before running Gate G2.~~ *(Completed Aug 17, 2026 — batching reduced uncited claims from $42.1\%$ to $2.55\%$, recovering citation recall to $0.3620$ and Citation F1 to $0.5250$).*

---

## 3. Decomposer and Granularity Freeze

**Target Date:** Completed  
**Context:** Per ADR-0009 §8, changing prompts or claim boundaries after this date invalidates the human gold set.

* ~~**Lock Decomposer Prompts:** You must freeze the decomposer model, prompt templates, and parser logic.~~ *(Completed Aug 17, 2026 — model, prompt templates, and parser logic verified).*
* ~~**Verify Digest:** You must confirm `decompose_template_digest()` remains unchanged.~~ *(Completed Aug 17, 2026 — confirmed digest `4129a884c7a1b4854739ffe2e3900a1db39626db7fd1bf076d6b549027a7d737` matches).*
* ~~**Maintain Schema:** The definition of a **decontextualized atomic claim** must remain immutable for all subsequent benchmark runs.~~ *(Completed Aug 17, 2026 — schema and `Claim` dataclass verified).*

---

## 4. Joint Attribution Arm Parse Rate

**Target Date:** Immediate — code complete, measurement pending  
**Context:** Guided-JSON decoding covered only the post-hoc citation stage. The joint attribution arm used free-text decoding. On the $n=100$ batched run (`generate_fp05_n100_guided_batched`), the joint arm gives $34/100$ clean parses and $161$ `quote_not_found` errors, against $99/100$ clean parses and $0$ errors for post-hoc citation. Gate G2 requires a valid claim parse rate of $\ge 95\%$ for both arms. An unequal decoding constraint also confounds the citation F1 contrast.

* ~~**Apply Guided Decoding:** You must apply guided-JSON constrained decoding to the joint attribution arm.~~ *(Completed Aug 20, 2026 — `System.JOINT` guided branch in `src/biomedqa/generate.py`, `JOINT_JSON_TEMPLATE` in `src/biomedqa/prompts.py`, and `build_citation_response_format(..., is_joint=True)` which emits a `decision` field and a `text` field per claim).*
* ~~**Batch Citation Calls:** You must use the same batch size of $5$ claims per call that the post-hoc citation stage uses.~~ *(Withdrawn Aug 20, 2026 — batching does not apply to the joint arm. The post-hoc arm can batch because stage 1 makes the claims and stage 2 cites them, so the claim count is known before the citation call. The joint arm makes claims and citations in one call, and the stage-count check requires the joint arm to stay at one call per query. Truncation control for the joint arm is the bounded schema (`max_claims = 30`, `max_citations = 3`) plus output-cap headroom, not batching).*
* **Measure Output Metrics:** You must measure the joint arm clean parse rate, valid claim parse rate, and `quote_not_found` count on the $100$ dev questions. *(Blocked — needs a live A4000 run with `--guided-decoding`; the vLLM server was not reachable from the writing host).*
* **Confirm Uniform Constraints:** You must confirm that both arms use the same decoding constraint before you run gate G2. *(Blocked on the measurement above).*

---

## 5. Run Configuration Defects

**Target Date:** Before Gate G2 — one of three items complete  
**Context:** Two checks in `scripts/generate_smoke.py` and the served context window configuration did not agree with the batched citation path.

* ~~**Update Stage Count Check:** You must update the stage-count check because the batched post-hoc citation arm makes $153$ stage-2 calls and the check demands two calls per record.~~ *(Completed Aug 20, 2026 — the check now demands at least two calls per post-hoc query, and exactly one for the joint and vanilla arms).*
* **Adjust Context Window:** You must increase the served context window to $14336$ tokens, or decrease the output cap, because the largest stage-2 prompt is $4464$ tokens and only $144$ tokens of headroom remain in the $8192$ token window. *(Blocked — a server-side change on the A4000. `docs/harvest/runbooks/wsl-vllm-a4000.md` and `vllm-8b.service` still serve `--max-model-len 8192`).*
* **Verify Window Guard:** You must confirm that the pre-flight prompt window guard rejects or shrinks a request before the server returns a $400$ error. *(Blocked — the guard has unit tests in `tests/test_backends.py`, but no live server confirmation).*

---

## 6. Joint Attribution Citation F1 Measurement

**Target Date:** Before Gate G2 — diagnostic read complete, gate read pending  
**Context:** Citation F1 for the joint attribution arm at `frequency_penalty = 0.5` is now measured on the batched run: joint $0.5344$ against post-hoc $0.5250$, delta $+0.0094$ $[-0.0536, +0.0729]$. The confidence interval includes zero, so contrast C2 is not established. The earlier read from `parity_iter1b` at `frequency_penalty = 0.0` gave delta $+0.011$ $[-0.117, +0.137]$: the penalty raised both arms by approximately $0.11$ F1 and halved the interval width, but it did not move the delta.

* ~~**Score Joint Arm Records:** You must score the joint arm records of the batched run with MiniCheck ($\phi$) at threshold $0.5$.~~ *(Completed Aug 20, 2026 — 100 paired queries, 0 dropped, 2131 $\phi$ pairs).*
* ~~**Compute Confidence Interval:** You must compute the paired-bootstrap confidence interval for the joint minus post-hoc citation F1 delta.~~ *(Completed Aug 20, 2026 — 10000 resamples, cluster unit query, seed 0).*
* ~~**Record Harvest Results:** You must record the result in `docs/harvest/`.~~ *(Completed Aug 20, 2026 — `docs/harvest/joint_citation_f1_fp05.md` and `docs/harvest/generate_fp05_n100_guided_batched.citation_f1.minicheck.json`).*
* **Repeat With Both Arms Guided:** You must repeat the read after the joint arm runs under guided decoding. The present read compares a guided post-hoc arm against an unguided joint arm, so the decoding constraint confounds it, and the joint arm parses $34/100$.

---

## 7. AlignScore Reference Environment Port

**Target Date:** Completed  
**Context:** Table 3 requires AlignScore (~355M parameters) as a secondary verifier beside MiniCheck ($\phi$) and Opus 5.

* ~~**Build Isolated Environment:** You must create an isolated environment with `torch<2` and `pytorch_lightning<2`.~~ *(Completed Aug 20, 2026 — Python 3.10.20, `torch 1.13.1+cu117`, `pytorch_lightning 1.9.5`, `protobuf 3.20.0`, `transformers 4.29.2`. The environment lives at `/tmp/alignscore_venv` and is rebuilt from the one-line commands in `docs/harvest/alignscore_reference_port.md`).*
* ~~**Download Checkpoint:** You must load the official AlignScore model checkpoint.~~ *(Completed Aug 20, 2026 — `AlignScore-large.ckpt`, RoBERTa-large backbone, SHA256 `ff4336312b377edcbcdad5694a2d09d73dc4225422c0422d810aa7e78485e32d`).*
* ~~**Verify Numerical Precision:** You must verify that ported environment outputs match reference outputs using `torch.allclose` with documented numerical tolerance.~~ *(Completed Aug 20, 2026 — six test pairs, maximum absolute difference $0.0$ at `rtol = 1e-5`, `atol = 1e-7`. Recorded in `docs/harvest/alignscore_reference_port.md`).*

---

## 8. Gate G2 Benchmark Execution

**Target Date:** September 6, 2026  
**Context:** Gate G2 tests whether joint attribution outperforms post-hoc citation on citation F1.

* **Complete Prerequisite Goals:** You must complete goals 4, 5, and 6 before you start this run.
* **Execute Dev Set Run:** You must run all 100 dev questions at `frequency_penalty = 0.5` (`CONFIG_VERSION = 1.5.0`).
* **Evaluate Citation F1:** You must compute citation precision, citation recall, and citation F1 using MiniCheck ($\phi$) at threshold $0.5$.
* **Confirm Gate Criteria:**
  * Joint attribution must beat post-hoc citation on citation F1 (paired-bootstrap CI excluding zero).
  * Valid claim parse rate must be $\ge 95\%$.
* **Run Stratified Check:** You must execute the mandatory W9 stratified robustness check required by ADR-0009 §5.

---

## 9. Gate G3 Cheap Verifier Preparation

**Target Date:** September 20, 2026  
**Context:** Gate G3 requires a verifier AUROC of $\ge 0.75$ for unsupported claim detection, at a cost $10\times$ lower than the Opus 5 judge baseline. This gate is not started.

* **Select Claim Set:** You must select the labeled claim set that gives the AUROC reading.
* **Measure Verifier AUROC:** You must measure the MiniCheck ($\phi$) and AlignScore AUROC on that set.
* **Measure Verifier Cost:** You must measure the cost per claim of each verifier against the Opus 5 judge baseline.

---

## 10. Gate G4 Human Gold Set Preparation

**Target Date:** September 27, 2026  
**Context:** Gate G4 requires $\ge 250$ labeled claims and a Krippendorff $\alpha$ point estimate of $\ge 0.6$ on the binary collapse over the triple-labeled set (ADR-0016). The decomposer freeze is complete, so claim boundaries are now stable.

* **Build Annotation Batch:** You must build the annotation batch from the frozen decomposer output.
* **Start Gold Annotation:** You must start the three non-expert annotators on the full gold set.
* **Monitor Annotator Agreement:** You must monitor inter-annotator agreement during the annotation.

---

## 11. Mandatory W9 Stratified Robustness Check

**Target Date:** Before Gate G2 sign-off — run once, must repeat on the gate run  
**Context:** ADR-0009 §5 makes this check mandatory. A passing parity result does not cancel it. The check is per run, so the Gate G2 run of record needs its own execution.

* ~~**Run Stratified Check:** You must run the stratified robustness check on the granularity parity result.~~ *(Completed Aug 20, 2026 on `generate_fp05_n100_guided_batched` — all three schemes PASS. Compound structure: simple $+14.3\%$, compound $+11.8\%$. Claim length: five powered bands, all inside tolerance. Query claim volume: two powered bands, the $11+$ band empty).*
* ~~**Record Check Artifacts:** You must record the result beside the gate G2 artifacts.~~ *(Completed Aug 20, 2026 — `docs/harvest/w9_stratified_parity.md`, with the limitation that the claim-length scheme bins claims by the same quantity it compares).*
* **Repeat On The Gate Run:** You must run the check again on the Gate G2 run of record, because a stratified result does not transfer across runs.

---

## Priority Order

1. **Goal 4:** Measure the joint arm under guided decoding on the A4000. Code is complete.
2. **Goal 5:** Raise the served context window to $14336$ tokens and confirm the window guard against a live server.
3. **Goal 6:** Measure joint arm citation F1 for contrast C2.
4. **Goal 8:** Execute Gate G2 benchmark on dev set.
5. **Goal 11:** Repeat the stratified robustness check on the Gate G2 run of record.
6. **Goal 9:** Prepare cheap verifier AUROC benchmark for Gate G3.
7. **Goal 10:** Annotate human gold set for Gate G4.

Goals 1, 2, 3, and 7 are complete.
