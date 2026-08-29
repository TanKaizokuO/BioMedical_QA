# Upcoming Goals

This document lists the upcoming targets for the project. The project uses evidence-grounded, claim-attributable biomedical question answering. All terms match `CONTEXT.md`. Update this document when targets are completed, changed, or added. Write all updates in ASD-STE100 Simplified Technical English.

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

**Target Date:** Completed
**Context:** Guided-JSON decoding covered only the post-hoc citation stage. The joint attribution arm used free-text decoding. On the $n=100$ batched run (`generate_fp05_n100_guided_batched`), the joint arm gives $34/100$ clean parses and $161$ `quote_not_found` errors, against $99/100$ clean parses and $0$ errors for post-hoc citation. Gate G2 requires a valid claim parse rate of $\ge 95\%$ for both arms.

* ~~**Apply Guided Decoding:** You must apply guided-JSON constrained decoding to the joint attribution arm.~~ *(Completed Aug 20, 2026 — `System.JOINT` guided branch in `src/biomedqa/generate.py`, `JOINT_JSON_TEMPLATE` in `src/biomedqa/prompts.py`, and `build_citation_response_format(..., is_joint=True)` which emits a `decision` field and a `text` field per claim).*
* ~~**Batch Citation Calls:** You must use the same batch size of $5$ claims per call that the post-hoc citation stage uses.~~ *(Withdrawn Aug 20, 2026 — batching does not apply to the joint arm. The post-hoc arm can batch because stage 1 makes the claims and stage 2 cites them, so the claim count is known before the citation call. The joint arm makes claims and citations in one call, and the stage-count check requires the joint arm to stay at one call per query. Truncation control for the joint arm is the bounded schema (`max_claims = 30`, `max_citations = 3`) plus output-cap headroom, not batching).*
* ~~**Measure Output Metrics:** You must measure the joint arm clean parse rate, valid claim parse rate, and `quote_not_found` count on the $100$ dev questions.~~ *(Completed Aug 20, 2026 on the A4000, run `generate_fp05_n100_guided_both` — joint arm $89/100$ clean parses, $0$ `quote_not_found`, $11$ malformed-JSON call failures. Post-hoc $99/100$ clean, vanilla $99/100$ clean. $89\%$ is under the Gate G2 $\ge 95\%$ bar).*
* ~~**Confirm Uniform Constraints:** You must confirm that both arms use the same decoding constraint before you run gate G2.~~ *(Completed Aug 20, 2026 — both arms now use guided-JSON decoding on the same run. The remaining blocker is the joint arm's $89\%$ parse rate, not an unequal constraint).*
* ~~**Fix Malformed-JSON Failures:** You must investigate and reduce the $11$ malformed-JSON replies from the joint arm's guided decoder before a Gate G2 run of record, because $89\%$ is under the $\ge 95\%$ bar.~~ *(Completed Aug 20, 2026 — root cause was an xgrammar whitespace "death loop": greedy decoding walked into unbounded indentation-token runs inside JSON strings, burning the $3584$-token completion cap on tabs. Server-side `disable_any_whitespace` crashes vLLM 0.26.0's xgrammar backend on startup, so it was reverted. Fix is a bounded escape-valve retry in `generate_one` (`src/biomedqa/generate.py`) — a zero-claim, no-decision malformed-JSON reply at `temperature=0.0` retries up to twice at `temperature=0.3` then `0.7` — plus a claim-length target in `JOINT_JSON_TEMPLATE`. Run `generate_fp05_n100_guided_v5` reads $95/100$ clean parses, clearing the $\ge 95\%$ bar).*

---

## 5. Run Configuration Defects

**Target Date:** Completed
**Context:** Two checks in `scripts/generate_smoke.py` and the served context window configuration did not agree with the batched citation path.

* ~~**Update Stage Count Check:** You must update the stage-count check because the batched post-hoc citation arm makes $153$ stage-2 calls and the check demands two calls per record.~~ *(Completed Aug 20, 2026 — the check now demands at least two calls per post-hoc query, and exactly one for the joint and vanilla arms).*
* ~~**Adjust Context Window:** You must increase the served context window to $14336$ tokens, or decrease the output cap, because the largest stage-2 prompt is $4464$ tokens and only $144$ tokens of headroom remain in the $8192$ token window.~~ *(Completed Aug 20, 2026 — `serve_8b.sh` on the A4000 now serves `--max-model-len 14336`, confirmed live against `/v1/models`. `_MODEL_MAX_LEN` in `src/biomedqa/backends.py` raised to match, so the client-side pre-flight guard uses the same figure as the server).*
* ~~**Verify Window Guard:** You must confirm that the pre-flight prompt window guard rejects or shrinks a request before the server returns a $400$ error.~~ *(Completed Aug 20, 2026 — the guard is a client-side check that runs before any HTTP call is constructed (`_check_prompt_window_guard` in `src/biomedqa/backends.py`), so it structurally cannot race a server-side $400$. Unit tests confirm this for both a bare oversized prompt and a stage-2 prompt sized against the new $14336$-token window. No live $400$ was provoked or needed to confirm the ordering).*

---

## 6. Joint Attribution Citation F1 Measurement

**Target Date:** Completed — diagnostic read complete on both a confounded and an unconfounded run
**Context:** Citation F1 for the joint attribution arm at `frequency_penalty = 0.5` was first measured on the batched run, where post-hoc was guided and joint was not: joint $0.5344$ against post-hoc $0.5250$, delta $+0.0094$ $[-0.0536, +0.0729]$, interval includes zero. With both arms guided (`generate_fp05_n100_guided_both`), the read is joint $0.6137$ against post-hoc $0.5055$, delta $+0.1083$ $[+0.0432, +0.1722]$, interval **excludes zero**.

* ~~**Score Joint Arm Records:** You must score the joint arm records of the batched run with MiniCheck ($\phi$) at threshold $0.5$.~~ *(Completed Aug 20, 2026 — 100 paired queries, 0 dropped, 2131 $\phi$ pairs).*
* ~~**Compute Confidence Interval:** You must compute the paired-bootstrap confidence interval for the joint minus post-hoc citation F1 delta.~~ *(Completed Aug 20, 2026 — 10000 resamples, cluster unit query, seed 0).*
* ~~**Record Harvest Results:** You must record the result in `docs/harvest/`.~~ *(Completed Aug 20, 2026 — `docs/harvest/joint_citation_f1_fp05.md` and `docs/harvest/generate_fp05_n100_guided_batched.citation_f1.minicheck.json`).*
* ~~**Repeat With Both Arms Guided:** You must repeat the read after the joint arm runs under guided decoding.~~ *(Completed Aug 20, 2026 — 89 paired queries, 11 dropped (zero claims in joint arm), delta $+0.1083$ $[+0.0432, +0.1722]$, excludes zero. Recorded in `docs/harvest/joint_citation_f1_fp05_both_guided.md` and `docs/harvest/generate_fp05_n100_guided_both.citation_f1.minicheck.json`. Still a diagnostic reading, not a gate figure, because the joint arm parses $89/100$, under the $\ge 95\%$ bar).*

---

## 7. AlignScore Reference Environment Port

**Target Date:** Completed  
**Context:** Table 3 requires AlignScore (~355M parameters) as a secondary verifier beside MiniCheck ($\phi$) and Opus 5.

* ~~**Build Isolated Environment:** You must create an isolated environment with `torch<2` and `pytorch_lightning<2`.~~ *(Completed Aug 20, 2026 — Python 3.10.20, `torch 1.13.1+cu117`, `pytorch_lightning 1.9.5`, `protobuf 3.20.0`, `transformers 4.29.2`. The environment lives at `/tmp/alignscore_venv` and is rebuilt from the one-line commands in `docs/harvest/alignscore_reference_port.md`).*
* ~~**Download Checkpoint:** You must load the official AlignScore model checkpoint.~~ *(Completed Aug 20, 2026 — `AlignScore-large.ckpt`, RoBERTa-large backbone, SHA256 `ff4336312b377edcbcdad5694a2d09d73dc4225422c0422d810aa7e78485e32d`).*
* ~~**Verify Numerical Precision:** You must verify that ported environment outputs match reference outputs using `torch.allclose` with documented numerical tolerance.~~ *(Completed Aug 20, 2026 — six test pairs, maximum absolute difference $0.0$ at `rtol = 1e-5`, `atol = 1e-7`. Recorded in `docs/harvest/alignscore_reference_port.md`).*

---

## 8. Gate G2 Benchmark Execution

**Target Date:** ~~September 6, 2026~~ — **PASSED August 23, 2026**
**Context:** Gate G2 tests whether joint attribution is better than post-hoc citation on citation F1. The gate has **two** criteria, not three: `research_roadmap.md` requires a citation-F1 margin that is larger than the paired-bootstrap CI, and a valid claim parse rate of $\ge 95\%$. Earlier sessions added a third criterion, that the ADR-0009 §5 W9 stratified check must pass, and gave ADR-0009 §5 as the source. §5 does not say this. §5 makes the check mandatory to run and to disclose. §1 calls parity one quantity that you measure and disclose whatever it says, and §3 says the tolerance does not need to be reachable. Run `generate_fp05_n100_guided_v4` meets both real criteria: citation F1 joint $0.6651$ against post-hoc $0.5248$, delta $+0.1403$ $[+0.0751, +0.2066]$, and a valid claim parse rate of $97/100$.

* ~~**Complete Prerequisite Goals:** You must complete goals 4, 5, and 6 before you start this run.~~ *(Completed Aug 20, 2026).*
* ~~**Execute Dev Set Run:** You must run all 100 dev questions at `frequency_penalty = 0.5` (`CONFIG_VERSION = 1.5.0`).~~ *(Completed Aug 20, 2026 — `generate_fp05_n100_guided_v4`, both arms guided, escape-valve retry applied, and no claim-length target).*
* ~~**Evaluate Citation F1:** You must compute citation precision, citation recall, and citation F1 using MiniCheck ($\phi$) at threshold $0.5$.~~ *(Completed Aug 23, 2026 on `generate_fp05_n100_guided_v4` — joint $0.6651$, post-hoc $0.5248$, delta $+0.1403$ $[+0.0751, +0.2066]$, excludes zero, $99$ paired queries, $1$ dropped for zero claims in the joint arm).*
* ~~**Confirm Gate Criteria:**~~
  * ~~Joint attribution must beat post-hoc citation on citation F1 (paired-bootstrap CI excluding zero).~~ *(Met on `generate_fp05_n100_guided_v4` — $[+0.0751, +0.2066]$).*
  * ~~Valid claim parse rate must be $\ge 95\%$.~~ *(Met on `generate_fp05_n100_guided_v4` — record-level $97/100$ [97.0%], claim-level $399/406$ [98.3%, preregistered criterion ADR-0019], `quote_not_found` = 0. Both MET. See definitions in `docs/harvest/joint_citation_f1_fp05_guided_v4.md`).*
* ~~**Run Stratified Check:** You must execute the mandatory W9 stratified robustness check required by ADR-0009 §5.~~ *(Completed Aug 23, 2026 on `generate_fp05_n100_guided_v4` — verdict FAIL at $+30.8\%$, disclosed, and discharged. See goal 11).*
* ~~**Sign Off Gate G2:** You must sign off Gate G2 when the citation-F1 contrast and the $\ge 95\%$ parse-rate bar both pass on the same run.~~ *(Completed Aug 23, 2026 on `generate_fp05_n100_guided_v4`. Recorded in `docs/harvest/joint_citation_f1_fp05_guided_v4.md`).*

---

## 9. Gate G3 Cheap Verifier Preparation

**Target Date:** September 20, 2026  
**Context:** Gate G3 requires a verifier AUROC of $\ge 0.75$ for unsupported claim detection, at a cost $10\times$ lower than the Opus 5 judge baseline. **Status: machinery ready, evidence pending.** Machinery (`gate_g3`, `join_scores_and_labels`, annotation ingest, `scripts/g3_report.py`) is implemented and exercised end-to-end; evaluation run on real MiniCheck scores gives verdict `passes: false` blocked on: (a) human labels (annotation opens 2026-09-07), (b) judge cost evidence, (c) verifier pricing pending wall_s timing + cited GPU-hour rate. Judge sweep estimate ($1.02–$1.21 floor/ceiling, `docs/harvest/runbooks/judge_cost_estimate.json`) is an estimate, not measured cost. Canonical runbook pointer: `docs/harvest/runbooks/g3_runbook.md`.

* **Select Claim Set:** You must select the labeled claim set that gives the AUROC reading.
* **Measure Verifier AUROC:** You must measure the MiniCheck ($\phi$) and AlignScore AUROC on that set.
* **Measure Verifier Cost:** You must measure the cost per claim of each verifier against the Opus 5 judge baseline.

---

## 10. Gate G4 Human Gold Set Preparation

**Target Date:** September 27, 2026  
**Context:** Gate G4 requires $\ge 250$ labeled claims and a Krippendorff $\alpha$ point estimate of $\ge 0.6$ on the binary collapse over the triple-labeled set (ADR-0016). The decomposer freeze is declared (`docs/harvest/w6_decomposer_freeze.md`, 2026-08-23, 8 days ahead of the Sep 3 target), so claim boundaries are stable. The annotation batch is built. The two remaining prerequisites before opening the pilot are closed: pilot claim set selected (`docs/harvest/w6_pilot_claims.md` — the literal first question of the shared order, 11 claims) and the decomposer freeze recorded above. Still open: annotators actually running the pilot and the maintainer's $\alpha$/qualitative review before the guideline freeze that gates the main pass.

* ~~**Build Annotation Batch:** You must build the annotation batch from the frozen decomposer output.~~ *(Completed Aug 23, 2026 — `scripts/build_annotation_ui.py --records docs/harvest/generate_fp05_n100_guided_v4.records.jsonl`. The batch holds $100$ questions, $1009$ claims, and $1257$ span labels for each annotator. The order hash `42a52170009b` is the same in all three forms (ADR-0016 §2). The three forms show no system, model, or run identity (ADR-0016 §4). `annotation/keyfile.jsonl` holds the $1009$ de-blinding rows and stays with the maintainer, because `.gitignore` excludes `annotation/`. Do not rebuild the forms after an annotator starts: a new order makes ADR-0016 §2 invalid and clears the saved progress).*
* **Start Gold Annotation:** You must start the three non-expert annotators on the full gold set.
* **Monitor Annotator Agreement:** You must monitor inter-annotator agreement during the annotation.

---

## 11. Mandatory W9 Stratified Robustness Check

**Target Date:** Completed — run and discharged on the Gate G2 run of record (`generate_fp05_n100_guided_v4`)
**Context:** ADR-0009 §5 makes this check mandatory to run and to disclose. It does not make passing it a condition, and it is not a Gate G2 criterion. §1 calls parity one quantity that you measure and disclose whatever it says, and §3 says the tolerance does not need to be reachable. On `generate_fp05_n100_guided_v4` the check reads FAIL at $+30.8\%$ (joint median $13.0$ w/c against post-hoc $17.0$). The check exists to detect one confound: that post-hoc's coarser claims are harder to entail, so C2's gap appears without joint grounding doing any work. A gap is a confound only if it reaches citation F1. It does not. At matched claim length the contrast gets **larger**, not smaller.

* ~~**Run Stratified Check:** You must run the stratified robustness check on the granularity parity result.~~ *(Completed Aug 20, 2026 on `generate_fp05_n100_guided_batched` — all three schemes PASS. Compound structure: simple $+14.3\%$, compound $+11.8\%$. Claim length: five powered bands, all inside tolerance. Query claim volume: two powered bands, the $11+$ band empty. Superseded as the Gate G2 baseline once both arms became guided — see below).*
* ~~**Record Check Artifacts:** You must record the result beside the gate G2 artifacts.~~ *(Completed Aug 20, 2026 — `docs/harvest/w9_stratified_parity.md`, with the limitation that the claim-length scheme bins claims by the same quantity it compares).*
* ~~**Repeat On The Gate Run:** You must run the check again on the Gate G2 run of record, because a stratified result does not transfer across runs.~~ *(Completed Aug 23, 2026 on `generate_fp05_n100_guided_v4` — verdict **FAIL** at $+30.8\%$. Compound structure FAILS (simple $+23.1\%$), claim length PASSES ($5/5$ bands), query claim volume FAILS ($1$–$5$-claims stratum $+38.5\%$). Recorded in `docs/harvest/generate_fp05_n100_guided_v4.w9_stratified_parity.txt`).*
* ~~**Restore Claim-Length Parity:** You must add a claim-length floor or a prompt-level nudge to the joint arm's guided schema so its median claim length returns to parity with post-hoc, then repeat this check.~~ *(**Withdrawn Aug 23, 2026. This task was itself the defect.** It directs you to tune the joint arm's granularity. ADR-0009 §4 permits that lever only on `POST_HOC_ANSWER_TEMPLATE`, and §6's blind lifted Aug 14, so any such edit now steers granularity with citation F1 in view. Five edits were made against this task (`045a96c`, `95dd958`, `dab7a68`, `dc08914`, `b29e74c`), producing runs `v5` to `v9`. All five are reverted. See ADR-0009's Fourth amendment).*
* ~~**Discharge The Check By Measurement:** You must show whether the granularity gap reaches citation F1, instead of tuning the gap away.~~ *(Completed Aug 23, 2026 — `scripts/w9_length_standardized_contrast.py` re-weights the joint arm's citation recall to the post-hoc arm's own claim-length distribution. On `v4` the joint arm leads in four of five length bands and ties in the shortest, and the recall lead **grows** with claim length ($+0.139$, $+0.158$, $+0.202$, $+0.333$). The standardised delta is $+0.1495$ $[+0.0786, +0.2244]$ against $+0.1403$ unstandardised. The gap works **against** C2, so the coarser post-hoc claims were making the joint arm's lead look smaller than it is. Recorded in `docs/harvest/w9_stratified_parity_guided_v4.md`).*
* ~~**Disclose The Miss:** You must report the parity gap as a miss with its size, and not tune it away.~~ *(Completed Aug 23, 2026 — reported at $+30.8\%$. ADR-0009 §1 chose this outcome over a fifth enforced condition for exactly this reason).*

---

## Priority Order

1. ~~**Goal 4:** Fix the joint arm's malformed-JSON call failures so the valid claim parse rate reaches $\ge 95\%$.~~ *(Completed Aug 20, 2026 — $97/100$ on `generate_fp05_n100_guided_v4`).*
2. ~~**Goal 11:** Run the W9 stratified check on the Gate G2 run of record and discharge it.~~ *(Completed Aug 23, 2026 — FAIL at $+30.8\%$, disclosed, and discharged by length standardisation).*
3. ~~**Goal 8:** Sign off Gate G2 once the citation-F1 contrast and the parse-rate bar both pass on the same run.~~ *(Completed Aug 23, 2026 on `generate_fp05_n100_guided_v4`, two weeks before the Sep 6 date).*
4. **Goal 9:** Prepare cheap verifier AUROC benchmark for Gate G3. *(Machinery ready, evidence pending).*
5. **Goal 10:** Annotate human gold set for Gate G4.

Goals 1 to 8 and goal 11 are complete. Goal 9 (Gate G3, Sep 20: machinery ready, evidence pending) and goal 10 (Gate G4, Sep 27) are open. Gate G2 closed two weeks early, so that time is now available to them.

One rule carries forward. Do not edit any arm's prompt to move a granularity number. ADR-0009 §4
permits that lever only on the post-hoc template, and §6's blind lifted on Aug 14, so no legitimate
granularity lever remains on either arm. A guided-decoding parse fix is still allowed, but it must
not change claim-length guidance.
