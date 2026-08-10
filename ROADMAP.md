# BioMedical QA — Research Roadmap & Milestone Progress

**Project:** Evidence-Grounded, Claim-Attributable Biomedical QA  
**Target Submission:** November 2–6, 2026 (Workshop Venue)  
**Current Date:** August 10, 2026  
**Status Summary:** Week 1 Complete (Splits & 2M Corpus Frozen; G0 Passed). Week 2 complete (Aug 10). **W3: Table 1 is complete, G1 was executed 13 days early on Aug 10, and its disposition is settled — ADR-0015 gates it at hit@10 (0.94, Wilson lower 0.8752, passes).** At k=5 it fails: **0.86, Wilson [0.7786, 0.9147]**, short on both clauses, and that reading is reported in the paper beside the relaxed gate. Reranking is worth +0.13 hit@5 and +0.21 hit@1 over RRF and is a strict permutation of the fused pool (`not_in_pool` = 3 in both rows, hit@100 = 0.97); 8 of the 14 misses sit at ranks 6–10, which is why k=10 is the rung that clears. The chunker rung was spent first, not skipped: all 7 configs were upper-bounded inside Table 1's own row-4 pool for the cost of one GPU pass instead of ~14 h of index builds, and the best **eligible** arm reaches 0.89, so no build can pass and all 7 are refused. One arm did clear 0.90 — `section`, at 0.94 — falsifying the prediction registered before the run; it is disqualified because it cuts gold on real PubMedQA section boundaries while every distractor stays a whole abstract, which is ADR-0014 §2's rejected signal, and `chunk_text` proves the point exactly: with `sections=None` the strategy *is* `abstract`, 0/100 texts differ. The matched control settles it — `sentence_window_5_2` hands gold the identical 3.21 chunks/query and reads 0.87, because it cuts the other 352 candidates too. The leak check now runs automatically on every arm (`gold_cut_asymmetry`), so no future reader has to remember the ADR. **Thresholds 0.90 / 0.85 are unchanged; only k moved, once, on the record.** Next: joint claim-grounded generation prompts on a **10-passage** context, and the two reranked confusability-probe arms, which are still unrun.

---

## 1. Major Gates & Status Overview

| Gate | Target Date | Status | Pass Condition & Result |
|---|---|---|---|
| **G0** — Generator Bake-off & Compute Preflight | **Aug 4, 2026** | **PASSED (2026-08-04)** | ~~Llama-3.1-8B-Instruct-AWQ-INT4 selected; MedCPT throughput 343.6 abs/s at 0.80 GB VRAM (2M encode = 1.6 h). Issue #1 closed.~~ |
| **G1** — Retrieval Gate | **Aug 23, 2026** | **RELAXED TO k=10 AND PASSED (2026-08-10, ADR-0015)** | **hit@10 0.94, Wilson lower 0.8752** on dev at `(abstract, τ untouched)`, index `57ab89e445f8`. **At k=5 it fails: 0.86, Wilson [0.7786, 0.9147]**, and that reading is reported beside the relaxed gate in the paper. The chunker rung was spent first: all 7 arms bounded inside Table 1's own row-4 pool, best *eligible* arm 0.89 UB, so no build can reach 0.90 and all 7 are refused. The one arm that cleared it (`section`, 0.94 UB) cuts gold on real section boundaries while every distractor stays a whole abstract — ADR-0014 §2's rejected signal — and is disqualified. Thresholds 0.90 / 0.85 unchanged; only k moved, once, on the record. |
| **G2** — Joint Attribution Gate | **Sep 6, 2026** | **Unstarted** | On dev, joint attribution beats post-hoc citation on citation-F1 by margin > paired-bootstrap CI; ≥95% valid claim parse. |
| **G3** — Cheap Verifier Gate | **Sep 20, 2026** | **Unstarted** | Verifier AUROC ≥ 0.75 for unsupported claim detection at ≥10× lower cost than Opus 5 judge baseline. |
| **G4** — Human Gold Attribution Gate | **Sep 27, 2026** | **Unstarted** | ≥250 claims labeled; point estimate Krippendorff's α ≥ 0.6 on binary collapse across overlap set. |
| **G5** — Execution & Table Freeze | **Oct 11, 2026** | **Unstarted** | Every cell of Tables 1–5 populated from tracked run manifests with confidence intervals. |

---

## 2. Detailed Weekly Timeline & Milestone Progress

### Week 0 (Jul 30 – Aug 2, 2026) — Foundation & Setup
- ~~**Decision D1 (Compute Strategy):** Allocate exclusive RTX A4000 for local 8B AWQ generation and local verifiers; Claude Opus 5 as judge baseline (ADR-0004).~~ *(Completed Jul 31, 2026)*
- ~~**Decision D2 (Architecture Strategy):** Rebuild in `BioMedical_QA` harvesting PubMedQA loaders from legacy repository (ADR-0007).~~ *(Completed Jul 31, 2026)*
- ~~**Codebase Skeleton:** Create `src/biomedqa/` with `config.py`, `schema.py`, `data.py`, and least-processed output contracts.~~ *(Completed Aug 2, 2026)*
- ~~**Paper Skeleton:** Create `paper/skeleton.md` containing 9 section headers and 5 empty table shells with headers and captions.~~ *(Completed Aug 2, 2026)*
- ~~**Long-Lead Annotator Outreach:** Recruit 2 non-expert annotators for 3 h total allocation (~1 h pilot + ~2 h main pass).~~ *(Sent Aug 4, Accepted Aug 5, 2026 — Issue #7)*
- ~~**MedNLI / PhysioNet Application Decision:** Deliberately drop PhysioNet credentialing due to schedule mismatch; rely on AlignScore and cheap signal ensembles (ADR-0002, R7).~~ *(Completed Aug 4, 2026 — Issue #8)*
- ~~**Issue Tracker Setup:** File GitHub issues #1 through #8 for all gates and long-lead milestones.~~ *(Completed Jul 31 – Aug 4, 2026)*

### Week 1 (Aug 3 – Aug 9, 2026) — Gate G0, Splits & Corpus Build
- ~~**Gate G0 Execution (Aug 4, 2026):** Run 8B AWQ generator bake-off on RTX A4000. Select `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` based on atomic decontextualized claim conformance (9.2 claims/query, 3.42 s latency, 72.5 tok/s).~~ *(Passed Aug 4, 2026 — Issue #1)*
- ~~**MedCPT Benchmark (Aug 4, 2026):** Measure MedCPT encoder throughput (343.6 abstracts/s at batch 64, 0.80 GB VRAM peak). Confirm 2M encode time of 1.6 h, discharging R1 encode risk.~~ *(Passed Aug 4, 2026 — Issue #1)*
- ~~**Evaluation Splits Freeze (Aug 7, 2026):** Lock `data/splits.json` with 100 dev questions and 400 test questions drawn from `pqa_labeled` (Seed 20260807).~~ *(Completed Aug 7, 2026)*
- ~~**2M Retrieval Corpus Build (Aug 6, 2026):** Build 2,000,000 abstract corpus from `MedRAG/pubmed` (`data/corpus/corpus_manifest.json`, fingerprint `93321598f3f1`, 300 duplicate rows suppressed over 244 PMIDs).~~ *(Completed Aug 6, 2026)*
- ~~**Gold Context Deduplication:** Verify 1,000/1,000 PubMedQA gold PMIDs present and deduplicated against corpus draw.~~ *(Completed Aug 6, 2026)*
- ~~**Passage Format Decision (ADR-0014):** Index abstracts as title-free prose (`content` field) to prevent title-based artifact matching.~~ *(Completed Aug 6, 2026)*

### Week 2 (Aug 10 – Aug 16, 2026) — Retrieval Stack & 2M GPU Encode
- ~~Chunker sweep (`src/biomedqa/chunk.py`): abstract-level, sentence-window, and fixed-width boundaries.~~ *(Driver `scripts/chunker_sweep.py` delivered Aug 9, 2026 — 7 named configs; sweep **execution** is A4000-side.)*
- ~~Implement `bm25s` lexical index and MedCPT dense retriever in `src/biomedqa/retrieve.py`.~~ *(Completed Aug 9, 2026 — save/load round-trip and real-weights smoke test.)*
- ~~Execute 2M MedCPT embedding pass on A4000 GPU.~~ *(Completed Aug 10, 2026 — 2,162,838 passages in 1.99 h; 1,037/1,037 gold passages present; `dense.npy` 3.1 GB, `passage_texts.jsonl` 2.5 GB, `bm25/` built.)*
- ~~Implement Reciprocal Rank Fusion (RRF) candidate pipeline.~~ *(Completed Aug 9, 2026 — `_rrf_fuse`, arithmetic checked by hand against `1/(k+rank)`.)*
- ~~Execute distractor confusability probe (ADR-0012 §2) using MiniCheck on ~100 dev questions.~~ *(Completed Aug 10, 2026 — 427 non-gold passages over 100 dev questions, plus a **paired uniform-random control** at seed 12345, because the retrieved-side distribution alone could not be distinguished from MiniCheck's base rate. It cannot: at ≥0.3 random passages score **higher** than retrieved ones (67.7% vs 62.1%). Separation is in the tail — **14.5% vs 2.1% at ≥0.7, a 6.9× enrichment**, paired sign test p = 0.012. **τ_confusable = 0.7 set post-hoc**, where **35/100 questions carry a plausible mis-citation target against 8 by chance**. ADR-0012 §3 not triggered; re-confirm **both arms** post-rerank at W3. `docs/harvest/confusability_probe{,_control}.json`.)*
- ~~Measure empty title-segment convention impact on dev hit@5 (ADR-0014 §3).~~ *(Decided Aug 10, 2026 — **`empty` wins and is what the 2M index already holds, so no re-encode.** hit@5 0.59 vs 0.53 for `single`; paired gold rank better on 19 / worse on 39 / unchanged 33, sign test p = 0.012, mean +3.68 places worse. Measured by re-ranking Table 1's own 100-deep pools and re-encoding only the 9,832 pooled passages (~1 min) rather than two ~2 h index builds; the `empty` arm reproduced Table 1 row 2 exactly as a harness check. **The convention is not why dense hit@5 is 0.59.** Also closed the gap where `index_fingerprint()` could not tell the two indices apart — `RetrievalConfig.title_segment`, `CONFIG_VERSION` 1.3.0. `docs/harvest/title_convention_pool_eval.json`.)*
- ~~Implement `backends.py` vLLM / local Llama-3.1-8B AWQ generator adapter.~~ *(Completed Aug 9, 2026 — HTTP to vLLM; `vllm` deliberately absent from `pyproject.toml`.)*
- ~~Address non-blocking corpus findings from Issue #10.~~ *(All 6 findings closed Aug 9, 2026.)*
- ~~Measure baseline Table 1 rows 1–3.~~ *(Completed Aug 10, 2026 — dev, 2.16M index, evaluated over the full 100-deep pool so a near-miss is distinguishable from a retrieval failure. BM25 hit@5 0.71 / @10 0.77 / @100 0.90; Dense 0.59 / 0.70 / 0.91; RRF 0.73 / 0.81 / 0.97. `docs/harvest/table1_rows_1_3.{json,records.jsonl}`.)*

### Week 3 (Aug 17 – Aug 23, 2026) — Reranking & Retrieval Gate G1
- ~~Integrate cross-encoder reranker into cascade (top-100 pool → top-5).~~ *(Completed Aug 10, 2026 — `retrieve.py` `_rerank` was already written but untested against 2M; it now refuses a pool carrying no passage text rather than reranking identifier strings. `tests/test_rerank.py`.)*
- ~~**Gate G1 Execution:** Measure hit@5 ≥ 0.90 on dev with Wilson lower > 0.85.~~ *(Executed Aug 10, 2026, 13 days early. **FAILS at k=5: 0.86, Wilson [0.7786, 0.9147]** via `gate_g1()` on the records. **Gated at k=10 instead — 0.94, lower 0.8752, passes** (ADR-0015), after the chunker rung was spent and refused.)*
- ~~Complete Table 1 (Retrieval ablation & cascade lift).~~ *(Completed Aug 10, 2026 — rows 1–3 reproduced exactly (0.71 / 0.59 / 0.73) alongside row 4's 0.86. `docs/harvest/table1_rows_1_4.{json,records.jsonl}`, now carrying `index_fingerprint`.)*
- ~~Decide G1's disposition: run `scripts/chunker_pool_eval.py` for the upper bound on each chunker inside the recorded pool, then either owe a full ~2 h build to any arm clearing 0.90 or relax to hit@10 and say so in the paper.~~ *(Decided Aug 10, 2026 — **ADR-0015**: gate at hit@10. `docs/harvest/chunker_pool_eval.json` (harness passed: abstract arm 0.8600, gold promoted in 0 queries, demoted in 0, against the 1,466 sibling candidates the pool audit found re-chunking adds) and `docs/harvest/chunker_arm_eligibility.json` (`builds_owed: []`). `g1_miss_analysis.json`'s registered prediction is **falsified as written** — `section` hit 0.94 — but the falsifying arm is ineligible on a rule that predates the measurement, so its build is refused on that ground and not on its number. Matched control, exact: `sentence_window_5_2` gives gold the identical 3.21 chunks/query and reads 0.87, because it cuts the other 352 candidates too. ~14 h of A4000 not spent.)*
- ~~Paper: report hit@5 = 0.86 in Table 1 and in G1's row beside the relaxed k, per ADR-0015 §3. The relaxation is stated as a relaxation.~~ *(Done Aug 10, 2026 — `paper/skeleton.md` Table 1 now carries **both** k: hit@5 0.86 [0.78, 0.91] and hit@10 0.94 [0.88, 0.97] for the full cascade, and the gate block states the relaxation as a relaxation, names ADR-0015, prints the failing k=5 reading, and records that the thresholds 0.90 / 0.85 did not move and τ was not tuned. `recall_at_k`, `mrr`, and `ndcg` were the last three `NotImplementedError("W3")` stubs in `scoring/retrieval.py` and are now implemented, so Table 1's caption no longer names a function the repo cannot honour — relevance is the gold **chunk set**, recall@5's denominator keeps gold chunks the corpus never indexed, and nDCG's ideal is capped at k. Cells come from `scripts/table1_report.py` (CPU-only, re-scores the recorded 100-deep ranked lists) → `docs/harvest/table1_metrics.json`; it reproduces the recorded hit@5/@10 and both G1 readings exactly.)*
- ~~Generation stage takes a **10-passage** context wherever G1's k binds.~~ *(Encoded Aug 10, 2026 — `prompts.CONTEXT_DEPTH = 10`, cited to ADR-0015 and asserted by `tests/test_prompts.py`.)*
- ~~Draft initial joint claim-grounded generation prompts on dev set retrievals.~~ *(Drafted Aug 10, 2026 — `src/biomedqa/prompts.py`: joint, post-hoc (two stages), vanilla, plus the response grammar and its parser. Rendered against all 100 dev queries (`scripts/draft_prompts.py` → `docs/harvest/prompt_drafts.json`) and round-tripped end to end: prompt grammar → parsed claims → located spans → `QueryRecord.validate()` returns zero violations. **Citations are emitted as verbatim quotes, not char offsets** — a 4-bit 8B model cannot count characters but can copy, and `locate_quote()` recovers the span by exact search, so the offsets are right by construction. An uncopied quote is a counted parse error, never a fuzzy match, which is what keeps G2's ≥95%-valid-parse gate real. Post-hoc's first pass is not told citations are coming; a test enforces it, because a first pass that knows would already be grounding jointly and would close C2's gap for a non-reason.)*
- ~~Begin logging prompt-iteration counts for equal-effort baseline protocol.~~ *(Started Aug 10, 2026 — `prompts.PROMPT_ITERATIONS` is a per-system ledger of revision **cycles**, not file edits, so post-hoc's two stages are not charged double for one round of thinking. All three systems at cycle 1. `tests/test_prompts.py::test_joint_and_post_hoc_stay_on_equal_effort` fails the build when joint and post-hoc drift apart: spend the matching cycles or record the imbalance — do not delete the test. Vanilla is excluded from the parity requirement because it contributes nothing to citation-F1 (ADR-0010).)*
- [ ] **A4000, no GPU:** `uv run python scripts/dump_contexts.py --index-dir data/index/empty --depth 10`. Table 1's records carry text for ranks 1–5 only (written while G1 was a hit@5 gate), so the depth-10 context ADR-0015 requires cannot be rendered off the box. Prompt *wording* is already exercised at depth 5; prompt *sizing* is not.
- [ ] Re-confirm the distractor confusability probe post-reranking — **both arms**. A control-free re-run restates a number that is indistinguishable from MiniCheck's base rate below τ_confusable = 0.7; see ADR-0012 §2's 2026-08-10 result.
- [ ] Nudge annotators around Aug 19 ahead of Aug 20 response backstop.

### Week 4 (Aug 24 – Aug 30, 2026) — Joint Generation & Parity Loop
- [ ] Implement joint claim-grounded generation in `src/biomedqa/generate.py` and `decompose.py`.
- [ ] Validate end-to-end schema round-trip serialization (`src/biomedqa/schema.py`).
- [ ] Implement vanilla RAG and post-hoc citation baselines under equal-effort protocol.
- [ ] Run dry-run for question-clustered vs. unclustered bootstrap confidence intervals.
- [ ] Conduct blind granularity-parity loop (hard stop at 10 iterations or Aug 30).
- [ ] Freeze decomposer output granularity.

### Week 5 (Aug 31 – Sep 6, 2026) — Verification Setup & Gate G2
- [ ] Implement decontextualization pass and granularity settings (`sentence`, `bare atomic`, `decontextualized atomic`).
- [ ] Implement citation precision, recall, and F1 scoring (strict and lenient ALCE semantics) in `src/biomedqa/scoring/citation.py`.
- [ ] Compute initial citation-F1 baseline comparison (~Aug 31).
- [ ] Implement abstention scoring logic in `src/biomedqa/scoring/abstention.py`.
- [ ] Draft Pass 1 human annotation guidelines (Aug 31) and worked examples (Sep 3–6).
- [ ] **Decomposer & Granularity Freeze (Sep 3, 2026):** Lock claim decomposition model and prompts prior to annotation.
- [ ] **Gate G2 Execution (Sep 6, 2026):** Confirm joint attribution beats post-hoc citation on citation-F1 (paired-bootstrap CI excluding zero) with ≥95% valid claim parses.

### Week 6 (Sep 7 – Sep 13, 2026) — Verifier Integration & Pilot Annotation
- [ ] Wire MiniCheck-Flan-T5-Large verifier and Opus 5 LLM judge baseline in `src/biomedqa/verify.py`.
- [ ] Integrate AlignScore (~355M) as never-cut second row for Table 3.
- [ ] Conduct human annotation pilot pass (10 claims, 3 annotators, ~1 h each).
- [ ] Launch primary author annotation pass (~8–14 h total over W6–W7).

### Week 7 (Sep 14 – Sep 20, 2026) — Verifier Evaluation & Gate G3
- [ ] Perform verifier threshold sweep, AUROC, AUPRC, and ECE evaluation in `src/biomedqa/scoring/calibration.py`.
- [ ] Instrument clean overhead benchmarking on A4000 (tokens, $, wall-clock s per query).
- [ ] **Gate G3 Execution (Sep 20, 2026):** Achieve verifier AUROC ≥ 0.75 for unsupported claim detection at ≥10× lower cost than Opus judge.
- [ ] Draft Table 4 (cost & latency Pareto comparison).
- [ ] Complete human annotation main pass (2 h per annotator, ~19 questions / 38 claims overlap).

### Week 8 (Sep 21 – Sep 27, 2026) — Human Annotation Gate G4 & Code Freeze
- [ ] Complete all human label collection (Sun Sep 20).
- [ ] Compute inter-annotator agreement (Krippendorff's α on binary collapse).
- [ ] **Gate G4 Execution (Sep 27, 2026):** Validate human gold attribution set ≥250 claims labeled with Krippendorff's α ≥ 0.6.
- [ ] **Code Freeze & Tag (Sep 27, 2026):** Lock repository code state; all downstream work consists of evaluation runs and manuscript drafting.
- [ ] Finalize generator backend decision for full test evaluation.
- [ ] Begin Seed 1 test set evaluation runs.

### Week 9 (Sep 28 – Oct 4, 2026) — Evaluation Runs & Ablations
- [ ] Execute test set evaluation runs across Seeds 2 and 3 for all systems and baselines.
- [ ] Run ablation studies (granularity settings, sans verifier, sans decomposition).
- [ ] Execute generator backend swap check (~100 questions) to rule out model-specific prompt sensitivity.
- [ ] Execute stratified robustness check if triggered by parity gap.

### Week 10 (Oct 5 – Oct 11, 2026) — Gate G5 & Method Writing
- [ ] Perform statistical significance tests (McNemar for accuracy, paired bootstrap for citation/verifier metrics).
- [ ] Conduct biomedical failure-mode error analysis (negation, numerics, scope/population → Table 5).
- [ ] **Gate G5 Execution (Oct 11, 2026):** Verify all cells of Tables 1–5 are populated from execution manifests with CIs.
- [ ] Draft Paper: Method and Experimental Setup sections.

### Week 11 (Oct 12 – Oct 18, 2026) — Figures & Results Writing
- [ ] Generate Figure 1 (system architecture diagram).
- [ ] Generate Figure 2 (worked example: question → claims → citations → verifier verdicts).
- [ ] Generate Figure 3 (quality-vs-cost Pareto frontier plot).
- [ ] Draft Paper: Results and Analysis sections.

### Week 12 (Oct 19 – Oct 25, 2026) — Reproducibility & Section Drafting
- [ ] Perform repository cleanup and export run manifests.
- [ ] Write ML Reproducibility Checklist & Appendix.
- [ ] Draft Paper: Related Work, Limitations, Ethics & Clinical Risk Statement.

### Week 13 (Oct 26 – Nov 1, 2026) — Synthesis & Red-Teaming
- [ ] Draft Paper: Introduction and Abstract (written last).
- [ ] Internal red-team review against 8 key reviewer objections.
- [ ] Format paper to target venue template and prepare public repository release.
- [ ] Paper draft submission-ready.

### Week 14 (Nov 2 – Nov 8, 2026) — Final Buffer & Submission
- [ ] Schedule buffer for unexpected revisions.
- [ ] **Final Paper Submission (Target Window: Nov 2–6, 2026).**

---

## 3. Claim Ledger to Paper Tables

| Table | Claim | Target Metric | Target Gate |
|---|---|---|---|
| **Table 1** | **C1** (Retrieval adequacy) | hit@5, recall@5, MRR, nDCG@10 + Wilson CIs | G1 (Aug 23) |
| **Table 2** | **C2** (Joint vs. post-hoc attribution) & **C3** (Hallucination reduction) | Citation precision / recall / F1 (strict & lenient); Hallucination rate | G2 (Sep 6) |
| **Table 3** | **C4** (Cheap verifier vs. LLM judge) | AUROC, AUPRC, ECE, human agreement (MiniCheck vs. AlignScore vs. Opus 5) | G3 (Sep 20), G4 (Sep 27) |
| **Table 4** | **C5** (Overhead & Pareto efficiency) | Input/output tokens, $ cost per query, wall-clock latency (s) | G3 (Sep 20) |
| **Table 5** | **C9** (Biomedical failure modes) | Error distribution across Negation, Numerics, and Scope/Population | G5 (Oct 11) |

---

## 4. Key Deadlines & Standing Constraints

1. **Annotator Backstop:** Confirm annotator availability by **Thu Aug 20, 2026**.
2. **Decomposer Freeze:** **Thu Sep 3, 2026** (Changing decomposition prompt after this date invalidates the gold set).
3. **Annotation Window:** **Sep 7 – Sep 20, 2026** (Pilot W6, Main pass W7). All labeling ends Sun Sep 20.
4. **Code Freeze:** **Sun Sep 27, 2026** (Tag `v1.0.0-freeze`).
5. **Paper Submission:** **Nov 2 – Nov 6, 2026**.
