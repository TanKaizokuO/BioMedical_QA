# Project 2 — Research Roadmap to Submission

### Evidence-Grounded, Claim-Attributable Biomedical QA

**Written:** 2026-07-30 · **Revised:** 2026-07-30 after the grilling session · **Target submission:**
2026-10-26 → 2026-11-06 (workshop, see §6)
**Companions:** [`docs/project2_biomedical_attribution_rag_implementation_plan.md`](docs/project2_biomedical_attribution_rag_implementation_plan.md) (the *what*) ·
[`docs/related_work.md`](docs/related_work.md) (the *against what*) · [`docs/learning_roadmap.md`](docs/learning_roadmap.md) (the *concepts*, now fully taught — 8 lessons)
**Decisions:** [`CONTEXT.md`](CONTEXT.md) (domain language) · [`docs/adr/`](docs/adr/) (14 ADRs) ·
[`docs/grilling-handoff.md`](docs/grilling-handoff.md) (the reasoning behind them)

> This document is the *execution* layer: what gets built, in what order, with which numeric gates,
> and how each experiment maps to a table or figure in the paper. The study roadmap (§1.1–1.6) is
> **complete**; from here on, every hour should produce either a measurement or a paragraph.

---

## 0. Where the project actually stands (2026-07-30)

An honest audit, because the plan's stated starting point is optimistic:

| Asset | Reality | Consequence |
|---|---|---|
| "Existing RAG-over-PubMedQA pipeline, Slice 2 bug fixed" | Lives in **`~/Code/Research/RAG_Debate_Agent`**, not this repo. The fix (`e936d30`, index `pqa_labeled` not `pqa_artificial`) was applied to `rag_baseline.py` but **never re-executed**. `pubmedqa_baseline_v2` does not exist. | **Retired 2026-07-31 (ADR-0007).** No inherited retrieval measurement exists, and none is worth producing — a re-run would score hit@5 over the 1,000 gold contexts, which ADR-0003 rules out as a lexical gimme. `pubmedqa_baseline_v2` is **cancelled as a deliverable**. **G1 begins from zero by design, not by neglect.** Reference material extracted to [`docs/harvest/`](docs/harvest/README.md); the base repo is read-only history. |
| Retriever stack in that repo | ChromaDB + `all-MiniLM-L6-v2`, dense-only, top-5. No BM25, no RRF, no reranker. | Does not match the architecture the paper needs (hybrid + rerank). Porting it buys almost nothing. **But rebuilding from the notebooks is not free either** — see the promotion table below. |
| Generation stack in that repo | Local Ollama `qwen2.5:7b`, **~88s/query** (range 61–110s), CPU. | **The 88s/query risk is retired** — generation moves to the exclusive RTX A4000 (ADR-0004). **The row is not.** Two things are still unestablished, both due at **G0 (re-dated to Aug 4)**: (a) *which* 8B AWQ model — unchosen, and to be judged on **citation-format compliance**, not benchmark scores; (b) **no successful vLLM load on the A4000 has ever been recorded.** No document in this repo evidences that the box is reachable, driver-current, or able to hold ~9 GB of co-resident models. The laptop has **no CUDA fallback**, so (b) gates literally everything downstream and is the true first measurement — run it *before* the bake-off, not as part of it. **A4000 access starts Mon 2026-08-03** — the box is not available before then, which is why G0 moved. |
| This repo (`BioMedical_QA`) | Planning docs + 8 taught lessons + **8 runnable notebooks** already implementing BM25-from-scratch, MedCPT, RRF, cross-encoder rerank, citation P/R, decompose-then-verify, AUROC/calibration/CIs, Krippendorff's α, and a **working miniature eval harness** (`08_6_reproducible_eval_harness.ipynb`). | The notebooks *are* the codebase seed. `08_6` is the harness skeleton; promote it to `src/`. |
| Paper | Not started. No skeleton. Claim ledger now **cut to 5 tables** (§1); venue locked (§6). | Started in Week 0 — the paper is written **backwards from its tables**, not at the end. |

> **Next re-audit: 2026-08-23 (at G1).** §0 is a dated snapshot, not a standing description. The
> implementation plan's §4 went stale precisely because nothing ever forced it to be re-read; this
> line is that mechanism. Re-audit early if any gate slips.

### Notebook → module promotion, and what does **not** survive the move

"The notebooks *are* the codebase seed" is true and is the soft spot in row 4. **Every notebook runs
on a toy or simulated corpus** — that is correct for teaching and wrong for 2M abstracts. The
promotion work is real work; this table is what makes it estimable rather than a surprise in W2.

| Notebook | Promotes to | Scale/fidelity assumption that breaks |
|---|---|---|
| `01_1_retrieval_foundations` | `retrieve.py` | **N = 12 toy corpus.** `BM25Scratch` is pedagogical and **does not ship** — `bm25s` does (`rank_bm25` is borderline at 2M, §3). The MedCPT cells are marked *"run this when you have the model + network"*, i.e. **never executed**. |
| `02_1_chunking_granularity` | `chunk.py` | Uses **`all-MiniLM-L6-v2` as a stand-in** for MedCPT. Chunk boundaries validated against a toy corpus say nothing about `(chunker, τ)` behaviour at 2M, and nothing here emits the **char offsets** citations require. |
| `03_2_citation_precision_recall` | `scoring/citation.py` | Sound at any scale — pure functions over labels. But **φ is `cross-encoder/nli-deberta-v3-xsmall`, not MiniCheck**; the verifier swap is real work, and the ≤3-citation cap semantics must come from `CONTEXT.md`, not from the notebook. |
| `04_3_decompose_then_verify` | `decompose.py`, `verify.py` | Toy claims, MiniCheck **discussed but not run**. Decontextualization (the `CONTEXT.md` unit) is the hard part and is not implemented here. |
| `05_4_evaluation_auroc_calibration_ci` | `scoring/calibration.py` | **Simulated score vectors** (`A_hit[:160] = True`). Scale-free, so it promotes nearly as-is — the risk is that it has never seen a real, skewed score distribution. |
| `06_5_negation_numbers_scope` | failure-mode analysis, not a module | Toy strata, hand-built. Becomes an **analysis over real gold-set claims** in W6; there is no code to promote, only a taxonomy. |
| `07_4_human_eval_agreement` | `scoring/agreement.py` | Simulated 240 items — and, importantly, **3 labels**, while `CONTEXT.md` freezes a **4-way `support_label`** with a binary collapse for G4's α. **The notebook's label cardinality is wrong for this project.** Fix on promotion, not later: the collapse rule is what G4 gates on. |
| `08_6_reproducible_eval_harness` | `harness.py`, `config.py`, `schema.py` | Simulated n = 200, records held **in memory**; must become streamed `records.jsonl`. Its `sha256(json.dumps(cfg, sort_keys=True))` config hash is exactly the run-manifest primitive — and the answer to the base repo's count-based staleness bug (`docs/harvest/README.md`). |

**Not in any notebook, therefore net-new code in W1–W2:** the 2M corpus build and its
checkpoint/resume encode job, `backends.py`, `generate.py` (joint claim-grounded generation and both
baselines behind one API), and the cost/seed loop.

### Hardware (established, do not re-look-up)

| Machine | Specs | Role |
|---|---|---|
| **Laptop** (primary working dir) | Intel Ultra 9 285H, 16 cores, **no CUDA**, 30 GB RAM, 183 GB free, Java 21 | Development, writing |
| **Remote box** | **RTX A4000, 16 GB VRAM, Ampere**, 128 GB RAM, Xeon, 2 TB SSD — **exclusive** | All GPU work |

16 GB VRAM caps a local generator at ~7–8B fp16 (or ~14B at 4-bit). An 8B AWQ (~6 GB) leaves room
for MiniCheck-770M (~1.5 GB) and the cross-encoder (~1.3 GB) **concurrently** — which the overhead
measurement needs. A 70B model is out at any quantization.

**The Week-0 decisions (D1, D2) are now made — see §2.** Everything downstream is unblocked.

---

## 1. The paper this roadmap must produce

Lock this now; every experiment below exists to fill one of these slots.

**Working title:** *Cheap Per-Claim Grounding: Joint Claim-Attributed Generation with a Lightweight
Entailment Verifier for Biomedical QA*

**Thesis (one sentence):** In biomedical QA, generating answers as **decontextualized atomic claims**
each jointly attributed to a retrieved passage span — and screening them with a small entailment
verifier at generation time — yields substantially better attribution quality and lower hallucination
rate than post-hoc citation, at a small fraction of the cost of an LLM-judge, without regressing
answer accuracy.

**Headline** (ADR-0002): attribution quality (C2 + C3), with cost as the enabling modifier.

### The claim ledger — cut to 5 tables

**The workshop route (ADR-0001) allows ~8 pages ⇒ 4–5 tables, not 9.** The original nine-claim
ledger is preserved below the cut line for the 2027 journal extension. **A claim with no experiment
is cut. An experiment serving no claim is not run.**

| # | Claim | Experiment | Artifact |
|---|---|---|---|
| C1 | Retrieval is adequate, so attribution is meaningful | Retrieval gate: hybrid BM25+MedCPT+RRF+reranker vs. each ablated stage | **Table 1** (hit@5, recall@5, MRR, nDCG@10 + Wilson CIs) |
| **C2** | **Joint** claim-grounded generation beats **post-hoc** citation on attribution — *headline* | Ours vs. post-hoc baseline, **same retriever, same generator, same 3-citation cap, matched prompt-iteration budget** | **Table 2** (citation precision / recall / F1, ALCE-style; strict **and** lenient) |
| **C3** | Per-claim verification reduces hallucination — *headline* | Ours vs. vanilla RAG vs. ours−verifier | **Table 2** (hallucination rate = fraction of claims with no valid support) |
| C4 | The cheap verifier matches the expensive judge | Verifier vs. Opus 5 judge on the human-gold support set; **AlignScore as a second row** | **Table 3** (AUROC, AUPRC, ECE, agreement with human) |
| C5 | **…at low overhead** — the enabling modifier | **Tokens + \$ per query (primary)**, wall-clock secondary; ours vs. judge baseline | **Table 4** + **Fig. 3** (quality-vs-cost Pareto) |
| C9 | The failure modes are biomedical-specific and characterized | Stratified error analysis: negation, numerics, scope/population — driven by the `CONTRADICTED` label | **Table 5** + qualitative examples |

**Figures: Fig. 1** system diagram · **Fig. 2** worked example (question → claims → citations →
verifier verdicts, one supported and one caught-unsupported) · **Fig. 3** quality-vs-cost Pareto.

#### Cut, with where each goes

| # | Claim | Disposition |
|---|---|---|
| C6 | Attribution doesn't cost accuracy | **Demoted to prose + appendix.** Non-regression is a sentence with a McNemar *p*, not a table. Still run — it answers objection 4. |
| C7 | Decomposition granularity is a real design variable | **Folded into Table 2 as ablation rows** (sentence / bare-atomic / decontextualized-atomic), not its own table. ADR-0005 makes decontextualized-atomic the headline setting; the others are rows. |
| C8 | It isn't PubMedQA-specific | **Cut.** Already the designated cut in §8's cut order. State single-dataset scope in Limitations. → journal extension. |
| — | ROC + reliability diagram (was Fig. 2) | **Moved to the reproducibility appendix.** Table 3's AUROC/ECE carry the claim; the diagram is supporting evidence. |

**Why C9 survived the cut** while C6/C7/C8 did not: it is the designated fallback framing if C2
returns null at G2 (ADR-0002), it is the only claim that is *biomedical-specific* rather than
domain-portable, and its data is collected for free by the `CONTRADICTED` label (`CONTEXT.md`).
Cutting it would forfeit the fallback and waste an annotation field that cannot be recollected.

### Reviewer objections to pre-empt (answer inside the paper, not in rebuttal)

1. *"MiniCheck already showed cheap verifiers work."* → Ours wires it **into generation-time
   per-claim screening** in a domain it wasn't trained for; we report its biomedical degradation and
   what we do about it. Cite MiniCheck as backbone **and** baseline.
2. *"Decomposition is known not to always help."* → We ablate it (Table 2's granularity rows) and
   cite *Decomposition Dilemmas*.
3. *"Your attribution gold is small."* → Report α (Krippendorff) **with a bootstrap CI**, the human
   ceiling, gold-set sizing rationale, and CIs on every gold-derived number (Lesson 7 material).
4. *"You didn't beat SoTA accuracy."* → Explicitly out of scope and stated in the intro; MedRAG/MIRAGE
   reported as a reference point, not a target (C6 is a *non-regression* claim, reported in prose).
5. *"Overhead numbers are hardware-flattering."* → **Tokens and \$ are primary** (hardware-neutral and
   reader-reproducible); wall-clock secondary on named, exclusive hardware with batch policy stated;
   cost logged per run in the manifest (Lesson 8 material).
6. *"Your method is just prompt engineering on a weak model"* / *"your generator's format compliance
   is the confound."* → **The swap check** (ADR-0004): hold everything fixed, swap only the
   generator, re-run ours-vs-post-hoc on ~100 questions, report both gaps. What matters is whether
   the gap **persists**, not the absolute scores.
7. *"Your post-hoc baseline is a straw man."* → Same retriever, same generator, same citation cap,
   and a **matched, reported prompt-iteration budget** (§4 Phase 2).
8. *"This isn't ALCE's attribution unit."* → Correct, and stated: see the *Divergences from ALCE*
   section of [`CONTEXT.md`](CONTEXT.md).

---

## 2. Week 0 (Jul 30 – Aug 2): decisions **made**

Both blocking decisions are resolved. Recorded here; reasoning in [`docs/adr/`](docs/adr/).

### D1 — Compute for generation. **DECIDED (ADR-0004).**

The base repo's 88 s/query on local CPU Ollama was fatal at this scale. The **exclusive RTX A4000**
removes it.

- **Judge: Claude Opus 5 (`claude-opus-5`), ~\$23.** The expensive baseline C5 is measured against.
- **Generator: a local 8B AWQ model on the A4000, for all development iteration** — free, fast,
  unlimited, and **seedable** (the Claude API rejects `temperature`/`top_p`/`top_k` with a 400, so
  the ≥3-seed plan is only implementable locally).
- **The frozen-test-run backend is deferred to W8 code-freeze**, once the 8B model's citation-format
  compliance is measured. This is a config flag, not a second implementation — but it requires the
  **backend adapter in W2** (see §5).

Total ~\$28 typical, ceiling ~\$125–185. Cost was *not* the deciding axis — all options were
affordable; iteration friction and the seed plan were.

> **Candidates fixed 2026-08-04** (the winner is decided by the bake-off, not by this line):
> **A — `Qwen/Qwen2.5-7B-Instruct-AWQ`** · **B — `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`**.
>
> A is **7B, not 8B** — a deliberate choice, not drift. ADR-0004's binding constraint is the ~6 GB
> VRAM budget that leaves MiniCheck-770M and the cross-encoder co-resident, which a 7B AWQ meets.
> B is from a **different model family on purpose**: two Qwen checkpoints would not separate
> "this model follows the `[n]` format" from "this prompt elicits the format," and that confound is
> exactly what the W8 backend decision and the swap check later depend on being ruled out.
>
> Ranked on **citation-format compliance**, not benchmark scores. Latency breaks ties only.

#### G0 bake-off result — **DECIDED 2026-08-04: candidate B.**

**Generator: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`.**
Measured on 10 real `pqa_labeled` questions (seed 20260731), vLLM 0.26.0 under WSL2 on the A4000,
`--max-model-len 8192 --gpu-memory-utilization 0.85 VLLM_USE_V2_MODEL_RUNNER=0` (ADR-0008).

| | A — Qwen2.5-7B-Instruct-AWQ | **B — Llama-3.1-8B-Instruct-AWQ-INT4** |
|---|---|---|
| median latency | 1.64 s (0.78–2.84) | **3.42 s (1.57–4.67)** |
| throughput | 72.8 tok/s | 72.5 tok/s |
| claims / query | 3.8 | **9.2** |
| uncited claims | 1 of 38 | **1 of 92** |
| over-cap · out-of-range · malformed | 0 · 0 · 0 | 0 · 0 · 0 |

**The compliance column did not decide this, and should not be quoted as if it had.** Both models
produced *exactly one* uncited claim, both on the same query (`pubid 10781708`), and in both cases
the "violation" was a **correct abstention** — a statement that the passages do not answer the
question, which by construction has nothing to cite. The 0.991 vs 0.983 spread is `1/92` vs `1/38`,
a denominator artifact. See the scorer bug below.

**B was chosen on attribution-unit conformance (ADR-0005).** At 3.8 claims/query, A emits compound,
context-dependent claims — e.g. *"Among the patients with cancer, the incidence was 2.7%, compared
to 0.3% in the non-cancer group"* is two facts in one and does not stand alone. B splits exactly
that into two atomic, decontextualized claims. A also breaks format, appending prose outside the
claim list. **Under-atomization is unrecoverable** — two claims cannot be recovered from one after
the fact, and every ALCE precision/recall denominator is defined over that unit. B's opposite
failure, over-atomizing (splitting `RR = 10.0` from its `95% CI`), is prompt-tunable.

**Latency did not bind.** Projecting W5 at 500 questions × 3 systems × 3 seeds: ~4.3 h for B vs
~2.1 h for A, both trivial on an exclusive GPU. Against the retired pipeline's 88 s/query, B is
~26× faster. **D1's 88 s/query risk is now retired with a measurement, not an argument.**

*Honest limit:* n = 10. The granularity difference is qualitative and consistent across all ten
questions, so it is actionable. The compliance figures are **not** separable at this n and no claim
rests on them.

> **Scorer bug found by this run — fix before G2.** `score_compliance` counts an abstention as an
> uncited claim. Left unfixed, Table 2 will penalise a system for correctly declining to answer and
> reward one that confabulates a citation — inverting the property this paper exists to measure.

**Gate G0 (by Aug 4 — re-dated 2026-07-31):** the 8B generator is chosen, benchmarked on 10 real
queries on the A4000, and the measured per-call latency written into this file. Also benchmark
**MedCPT encode throughput on 1,000 abstracts** to convert ADR-0003's encode estimates into
measurements before committing to the 2M encode.

> **Why G0 moved, and why it costs nothing.** A4000 access starts **Mon Aug 3**, a day after G0's
> original Aug 2 date, so the gate was unmeetable as written. It moves to **Aug 4** — Monday for the
> preflight and install, Tuesday as buffer for a driver problem, which is the failure mode with the
> longest tail.
>
> **The slip does not touch the critical path**, because every other Week 0 and W1 deliverable is
> **CPU-only**: `src/biomedqa/` + `schema.py`, `paper/skeleton.md`, data load, and the split freeze
> need no GPU. Aug 1–2 therefore goes to items 4 and 5 of §8, which were already due Aug 2, and G0
> runs alongside W1's start rather than blocking it. Nothing is lost; the sequencing changes.
>
> **What it does consume is buffer.** W2 (Aug 10–16) contains the 2M encode, which must not begin
> until G0's throughput measurement exists — that gap is now 6 days rather than 8. **G1 stays Aug
> 23.** If the box is unusable on Aug 3, that is not a slip to absorb quietly: it invalidates
> ADR-0004's compute decision, and the response is R1/R4 territory, not rescheduling.

> Note: the verifier and reranker are small models and stay local — that's the point of C5. All three
> (8B AWQ ~6 GB + MiniCheck-770M ~1.5 GB + cross-encoder ~1.3 GB ≈ 9 GB of 16 GB) are co-resident,
> which the overhead measurement needs.

### D2 — Port vs. rebuild. **DECIDED: rebuild in this repo**, harvesting from the notebooks.

The base repo is a
*multi-agent debate* project with a different architecture (dense-only Chroma, no BM25/RRF/reranker,
agent-slice structure). The notebooks here already contain working versions of every component the
paper's architecture needs. Porting means adapting code that doesn't match the target design; rebuilding
means promoting code that does.

Harvest only these from `RAG_Debate_Agent`: the PubMedQA loading logic, the gold-passage tracking
fields added in `e936d30`, and the latency-benchmark methodology in `benchmark.py`.

> **Discharged 2026-07-31 (ADR-0007).** All three are extracted into
> [`docs/harvest/`](docs/harvest/README.md), each with the caveats on what survives the move to the
> 2M corpus. The base repo is now **read-only history**: nothing imports from it, no number it
> produced enters this repo or the paper, and it may be archived or deleted without effect. It is
> also not to be re-run — see ADR-0007 for why a starting hit@5 from it would be worse than none.
> One lesson came with them: **index freshness is a content hash, never a document count** — a
> `count == 1000` idempotency check is what let the original bug survive.

**Deliverable of Week 0** — the repo skeleton, derived from Lesson 8's harness design:

```
src/biomedqa/
  config.py          # every knob; base + diff, versioned, hashed into the run manifest
  data.py            # PubMedQA (pqa_labeled) load, split freeze, gold-context extraction
  chunk.py           # passage granularity (Lesson 2: hit@5 is only defined per (chunker, τ) pair)
  retrieve.py        # BM25 (bm25s) | MedCPT dense | RRF fusion | cross-encoder rerank
  generate.py        # joint claim-grounded generation; post-hoc + vanilla baselines behind one API
  backends.py        # vLLM (local 8B AWQ) | Anthropic — the W8 deferral lives here (ADR-0004)
  decompose.py       # decontextualized atomic claims; granularity is a config knob (Table 2 rows)
  verify.py          # MiniCheck-Flan-T5-Large (+ AlignScore) + Opus 5 judge baseline
  schema.py          # THE FROZEN OUTPUT SCHEMA — least-processed values only
  scoring/           # pure functions over the schema: retrieval, citation, faithfulness, calibration, accuracy
  harness.py         # seed loop, cost log, run manifest, config diff
runs/                # one directory per run: manifest.json + records.jsonl + costs.jsonl (gitignored)
paper/               # skeleton from day one
```

**The frozen schema is the single most important artifact in the repo.** Store `phi_score: 0.83`,
never `supported: true`; store char offsets and gold spans, never a precomputed hit@5. Binarizing at
write time destroys the AUROC sweep and calibration bins irrecoverably and turns re-chunking into a
re-*run*. (Lesson 8, §"least-processed-value rule".) **The rule extends to human labels** — store the
4-way `support_label`, never a collapsed boolean. An annotator cannot be re-run
([`CONTEXT.md`](CONTEXT.md)).

The schema's units are defined once, in [`CONTEXT.md`](CONTEXT.md), and `schema.py` implements them:
`claim` (decontextualized atomic) · `claim_validity` · `citation` (≤3, ALCE semantics) ·
`support_label` (4-way).

**Also in Week 0 — start the paper.** Create `paper/skeleton.md` with all nine section headings, the
cut claim ledger pasted in, and every table from §1 present as an **empty table with real column
headers and a caption**. The captions are written before the numbers exist. This is not ceremony: an
empty Table 4 with the columns "input tokens / output tokens / \$ per query / wall-clock (s)" forces
you to instrument for those columns in Week 5 instead of discovering the gap in October.

---

## 3. Corpus and evaluation set: freeze now, never touch again

### Retrieval corpus — **~2M PubMed abstracts** (ADR-0003)

All 1,000 PubMedQA gold contexts **plus ~2M distractors.** This is not a scale preference — it is
the condition under which the paper's central comparison is measurable at all. Indexing only the
1,000 gold contexts makes joint and post-hoc citation both score near-ceiling, and **G2 returns null
for setup reasons rather than scientific ones.** Citation precision only discriminates when
plausible-but-wrong passages are available to cite.

| Corpus | Encode (A4000, MedCPT fp16) | Embeddings fp16 | BM25 index | Peak disk |
|---|---|---|---|---|
| 1M | **49 min** | 1.4 GB | 1.2 GB | ~6 GB |
| **2M (chosen)** | **1.6 h** | 2.9 GB | 2.5 GB | ~12 GB |
| 23.9M (full MedRAG) | **19.3 h** | 34.2 GB | 20–30 GB | ~300 GB — deferred to 2027 |

**Measured at G0, 2026-08-04** — these are no longer estimates. MedCPT-Article-Encoder, fp16,
`max_length=512`, over 1,000 `pqa_labeled` abstracts on the A4000 under WSL2:

| batch | throughput | peak VRAM |
|---|---|---|
| 16 | 278.7 abstracts/s | 0.36 GB |
| 32 | 342.4 abstracts/s | 0.51 GB |
| **64 (best)** | **343.6 abstracts/s** | **0.80 GB** |

Throughput saturates by batch 32; 64 buys nothing but costs no VRAM either. **Peak VRAM is 0.80 GB**,
an order of magnitude below the budget in §2 — encode is not a memory-constrained job on this card.

**R1 is discharged. The 2M encode lands at 1.6 h, comfortably inside R1's ~4 h trigger, so the 1M
fallback is not needed and ADR-0003's 2M corpus proceeds as written.**

> **What this measurement does not cover, and why it still matters.** The probe encodes
> `pqa_labeled` abstracts, which may tokenise shorter than a real 2M PubMed dump — longer abstracts
> encode proportionally slower. The extrapolation also assumes no I/O stall, which a 2M streaming
> read will not fully honour. **Budget the W2 encode as a checkpoint/resume job regardless**; the
> headroom (1.6 h against a 4 h trigger) is wide enough to absorb a 2× miss on both counts, which is
> the actual reason R1 can be closed rather than merely deferred.

**`rank_bm25` is borderline at 2M: swapped for `bm25s`** (Java 21 is present if Pyserini is
preferred).

### Distractor selection — **uniform random, seeded** (ADR-0012; source and text form, ADR-0014)

ADR-0003 fixed the corpus *size* but never its **source or selection policy**, and the two gates pull
the same knob in opposite directions: G1 (hit@5 ≥ 0.90) wants easy distractors, G2 (citation
precision must discriminate) wants hard ones.

- **Uniform random 2M from `MedRAG/pubmed`, seeded**, with the ID list committed and hashed into
  `RunConfig.index_fingerprint()`. Uniform is unarguable to a reviewer; hand-picked hard negatives
  look like a corpus engineered around the gold set.
- **Read `data_files="chunk/*.jsonl"`, never the bare dataset id** (ADR-0014 §1). The Hub's
  auto-converted parquet export of `MedRAG/pubmed` is **partial** — 2,209,839 rows of 23,898,701,
  PMID-ascending, so the **oldest ~9% of PubMed**. It is within 10% of the 2M target, so the naive
  load succeeds and yields pre-1990 abstracts against 1990s–2010s gold: separable by era alone, G1
  excellent for the wrong reason, G2 with nothing plausible to mis-cite. `draw_corpus` refuses any
  scan whose row count is not 23,898,701, and refuses a full scan that collides with **no** gold PMID
  (the int32/int64 join failure, which otherwise reports "0 duplicates removed" and reads as success).
- **Deduplicate the gold contexts against the sample on PMID — W2 blocker, before the encode.**
  PubMedQA contexts *are* PubMed abstracts, so `MedRAG/pubmed` very likely already contains them; a
  naive union yields the same abstract under two `passage_id`s and **`gold_rank`/hit@5 silently
  miscount**. This is the shape of the bug ADR-0007 exists to remember.
- **Passages are abstract prose with no titles — MedRAG's `content`, not `contents`.** PubMedQA
  carries no title field, so gold is title-free whatever we do and the choice only ever concerns the
  distractors; titled distractors would make the empty title slot a gold/distractor format signal in
  the space hit@5 is measured in. Indexing the gold articles' real titles instead is the one clearly
  wrong option: **a PubMedQA question is its article's title verbatim** — over 60 sampled gold PMIDs
  the title covers the question's content tokens at median and mean 1.00, 60/60 at ≥ 0.8 (vs NCBI
  esummary, 2026-08-05) — so titled gold makes G1 a string match rather than a measurement. The cost
  is ~9% of corpus text and MedCPT off its trained *(title, abstract)* pair, symmetrically. The
  property bought: **if dev hit@5 lands under 0.90, the only levers left are the retriever's** —
  R2's ladder — because the corpus text cannot be quietly made easier to pass the gate.
- **W2–W3 confusability probe.** For ~100 dev questions: RRF-fused top-5 (no reranker until W3 —
  re-confirm after), drop the gold, and score the question's **gold claims** against the **non-gold**
  passages with the entailment model. Near-zero entailment means there is nothing plausible to
  mis-cite and citation precision cannot discriminate. **No threshold is pre-committed** — the first
  distribution is the first information anyone has about this quantity. Pulls MiniCheck forward from
  Phase 3 (~½ day). If the pool is too easy: sample more densely **within the gold questions' own
  MeSH terms — uniformly, seeded, declared** — never hand-picked.

> **Why this probe and not a topic judge.** A judge asked *"is anything in the top-5 on the same
> clinical topic?"* over a 2M biomedical corpus answers yes almost always, so a pre-committed
> threshold against it buys false comfort. A distractor damages citation precision only if it
> **plausibly entails a claim it should not** — that is what gets measured. It is also the **only
> remaining monitor** on ADR-0003's failure mode, since ADR-0009 §6 declined an early citation-F1
> read.

### Splits

| Split | Contents | Purpose | Frozen by |
|---|---|---|---|
| **dev** | 100 PubMedQA `pqa_labeled` questions | All development, prompt iteration, threshold tuning | Aug 7 |
| **test** | ~400–500 `pqa_labeled` questions, disjoint from dev | **Every number in the paper.** Run late, run once per system per seed. | Aug 7 |
| **gold-attribution** | **~4 claims × ~62 questions** ≈ 250 claims (ADR-0011), drawn from the **500 questions in neither dev nor test** — `data.gold_pool()`, derived from `splits.json`, never a second frozen file. **Not from `test`:** C4 asks whether the cheap verifier agrees with a human, and if it were measured on the questions the paper reports, the agreement and the result would stop being independent evidence. The 500 are idle, so this costs nothing. **Triple-labeled in full** — all three annotators label all ~250 claims, no overlap subset (ADR-0016) | C4 (verifier vs. human), attribution ground truth, α | Sep 27 (§4 Phase 4) |
| ~~transfer~~ | ~~BioASQ-Y/N~~ | **Cut with C8** (§1) | — |

Record split membership by question ID in a checked-in JSON with a hash in every run manifest. **If G0
forces a smaller test set, shrink it here and now** and report the reduced *n* honestly with CIs — a
300-question test set with confidence intervals is a real result; a 500-question set half-finished in
November is not.

---

## 4. Phased build plan

Each phase names a **gate** — a numeric condition with a stop rule. A missed gate triggers its
contingency, it does not silently roll forward.

### Phase 1 — Retrieval gate (Weeks 1–3, Aug 3–23) → **C1, Table 1**

The plan is right that attribution is meaningless without the right passage retrieved. But hit@5 is
**not** an absolute quantity — Lesson 2 established it is defined only relative to a `(chunker, τ)`
pair, where τ is the overlap threshold that counts a chunk as "containing the gold span." Fix and
document both before quoting any number.

1. Load `pqa_labeled`, freeze splits, extract gold contexts and gold spans (char offsets).
2. Chunker sweep — abstract-level, sentence-window, fixed-width. Report hit@5 per `(chunker, τ)`.
3. BM25 baseline → MedCPT dense → **RRF fusion** → **cross-encoder rerank**. Measure at each stage;
   the per-stage lift table *is* Table 1 and pre-empts "why the cascade?"
4. Report **Wilson** intervals, not Wald, on the gate proportion (Lesson 5).

> **Gate G1 (by Aug 23): hit@5 ≥ 0.90 on dev at a documented `(chunker, τ)`, with the Wilson lower
> bound reported.**
> *If missed:* do **not** loosen τ to manufacture a pass. In order: (a) widen first-stage candidate
> pool (top-50 → top-200 before rerank); (b) try the ModernBERT/ColBERT-style reranker; (c) revisit
> chunk granularity; (d) **relax the gate to hit@10 and say so in the paper**, reframing it as
> "attribution conditioned on retrieval success" with the retrieval-failure subset analyzed separately.
> Option (d) is an acceptable paper, and it is far better than a silently gamed 0.90.

### Phase 2 — Claim decomposition + joint attribution (Weeks 3–5, Aug 17 – Sep 6) → **C2, Table 2**

Overlaps Phase 1's tail deliberately; decomposition prompts can be developed on dev-set retrievals
while the reranker is still being tuned.

1. **Frozen output schema first** (`schema.py`), implementing [`CONTEXT.md`](CONTEXT.md)'s units:
   ```jsonc
   {question_id, answer_text,
    claims: [{claim_id, text, claim_validity,
              citations: [{passage_id, char_start, char_end}],   // ≤ 3
              verifier_score, support_label, ...}],
    retrieval: [...], costs: {...}, prompt_iteration_budget: {...}}
   ```
2. Joint generation: one call emits claims **with** citations. Prompt is versioned config (Lesson 8).
3. Decontextualization pass — bare atomic claims with dangling pronouns are unverifiable (DnDScore)
   **and unjudgeable by a non-expert annotator** (ADR-0005, ADR-0006).
4. Granularity knob → Table 2 ablation rows: `sentence` | `atomic` | **`atomic+decontextualized`
   (headline)**.
5. **Baselines built in the same module, behind the same API** (this matters — it's what makes the
   comparison fair and the harness clean): vanilla RAG (no attribution); post-hoc citation
   (generate, then attach citations).
6. Citation precision/recall scorers as pure functions over the schema (ALCE-style, from notebook
   `03_2`), reporting **strict** (SUPPORTED) and **lenient** (SUPPORTED+PARTIAL).
7. **Equal-effort baseline protocol (ADR-0002, ADR-0005).** Four conditions held identical and
   stated in the paper: same retriever/passages/*k* · same generator backend · same 3-citation cap
   and prompt token budget (within ~10%) · **matched prompt-iteration budget, counted and reported**
   ("joint: 14 revisions; post-hoc: 14 revisions"). **Log the iteration count from W3 onward** — it
   cannot be reconstructed later.
8. **Granularity parity — a measured diagnostic, reported *separately from* the four conditions
   (ADR-0009).** Joint emits claims natively; post-hoc emits them directly too, via
   `POST_HOC_ANSWER_TEMPLATE` (there is no prose→`decompose.py` step on the C2 path — see ADR-0009
   §4's 2026-08-13 amendment; `decompose.py` is the C7/Table-2 ablation tool). Coarser post-hoc
   claims are systematically harder to entail, so C2's gap could appear **without joint grounding
   doing any work** — a bias pointing toward the hypothesis.
   - It is **not** a fifth condition. The four above hold *by construction*; parity is an **outcome**
     of prompt tuning. Listing it alongside them invites a reader to assume it holds and turns a
     near-miss into a disclosed failure of our own protocol.
   - Measured on **median words/claim** (± **15%**, dev only, pre-committed now and never revised
     afterwards); **claims/query is reported, not gated**.
   - **Only the post-hoc arm is tuned — the lever is `POST_HOC_ANSWER_TEMPLATE`** (ADR-0009 §4
     amendment), never the shared `_claim_rules()` and never `decompose.py`. The joint prompt is out
     of bounds for the loop.
   - **Exactly 10 iterations, or Aug 30, whichever comes first** — hard, parity achieved or not.
   - **Fully blind: citation-F1 is not computed until the loop terminates.** First F1 ≈ **Aug 31**,
     six days before G2 — **R5's response is decided in advance, not improvised.**
   - Residual gap **favouring C2** → the **stratified robustness check (W9) becomes mandatory**;
     running against C2 → note and proceed. The asymmetry is **pre-registered in the paper's
     methods**, not only in the ADR.
   - Parity tuning is a **third disclosed ledger line, charged to neither system** — sound because
     the tuning is confined to the baseline, making reported baseline effort an *undercount*.
9. **Decomposer / granularity freeze: Sep 3** — a named dated artifact, three days before G2,
   protecting the gold set that launches Sep 7 (ADR-0005: changing granularity after W6 orphans it).
   Annotation guidelines are written in two passes: **unit-independent rules from Aug 31**
   (no-outside-knowledge, the SUPPORTED/PARTIAL boundary, hedging, numerics, jointly-necessary
   citations); **worked examples Sep 3–6**, built from frozen decomposer output.
10. **Abstention (ADR-0010, issue #9).** Abstention is **derived in `scoring/`, never stored** — no
    schema field, `SCHEMA_VERSION` stays 1.0.0. Abstention claims leave the **citation-recall
    denominator only** (precision is unaffected by construction). `abstention_rate` is a Table 2
    column, and **citation-F1 is reported on both denominators, always** — no threshold.

> **Gate G2 (by Sep 6): on dev, joint attribution beats post-hoc citation on citation-F1 by a margin
> exceeding the paired-bootstrap CI, and ≥95% of emitted claims parse into the schema with resolvable
> spans.**
> *If the margin is not there:* this is the paper's central contrast — investigate before proceeding.
> Usual causes: **the post-hoc baseline accidentally *weakened*** (unequal retriever, unequal
> citation cap, or — most likely — unequal prompt-iteration attention, since Weeks 3–5 are spent
> tuning the joint prompt), or joint prompts under-specified. **Same retriever is the minimum
> fairness condition, not a confound** — if the gap only survives when post-hoc gets a worse
> retriever, there is no C2. A genuinely null result means **repositioning around C3 + C9**
> (verification and biomedical failure characterization, ADR-0002) with C2 reported as a negative
> finding — decide this by Sep 6, not October.

### Phase 3 — Cheap verifier + overhead (Weeks 5–7, Aug 31 – Sep 20) → **C3, C4, C5**

**Instrument cost from the first line of code in this phase.** The low-overhead claim is the paper's
edge and it cannot be reconstructed after the fact.

1. **Wire `MiniCheck-Flan-T5-Large` (770M, MIT).** Premise = cited passage span, hypothesis = claim
   → score. It emits continuous `raw_prob`, not only its headline binary label — that is what the
   AUROC sweep and calibration bins consume. Fine-tuned from `google/flan-t5-large` on 21K ANLI +
   14K synthetic, i.e. **general-domain**, so biomedical degradation is the expected path (R7), not
   the tail case. *Bespoke-MiniCheck-7B is disqualified:* CC BY-NC, and at ~14 GB fp16 it neither
   fits alongside the generator nor leaves "cheap verifier" meaning anything.
2. **Do not binarize.** Store the raw score; threshold at scoring time.
3. **Opus 5 judge baseline** behind the identical API, with identical logging.
4. **Cost instrumentation, per call and per query — required manifest fields:** `input_tokens`,
   `output_tokens`, `usd_at_listed_rate`, `wall_clock_s`, `gpu_idle_confirmed`, hardware ID. For
   **both** verifier and judge. Retrofitting this in W7 means re-running Phase 3.
5. Threshold selection on **dev only**; report the ROC over the sweep, plus ECE (reliability diagram
   → appendix, §1).
6. **AlignScore (~355M) as a second row in Table 3** — one extra inference pass, no new infra
   (~½ day). Gives C4 a comparison instead of a single point, and Fig. 3 a second cheap Pareto point.
   ~~First thing to cut if W6 is tight.~~ **Never cut** — since `315f77b` dropped the MedNLI
   fine-tune, AlignScore is the **middle rung of R7's only remaining ladder**, and the cut would be
   forced in W6, a week before G3 (Sep 20) says whether it is needed. See §8 rule 8.
7. Biomedical degradation check: measure it and report it — a contribution, not an embarrassment.
8. **Overhead measurement (C5, Table 4).** Tokens and \$ **primary**; wall-clock **secondary** —
   median of ≥5 clean runs on the exclusive A4000, spread shown, GPU otherwise idle, batch policy
   fixed and stated. **Judge overhead is reported in tokens/\$ only**, since per-claim judge
   wall-clock includes network round-trip unrelated to model cost (ADR-0004).

> **Gate G3 (by Sep 20): verifier AUROC ≥ 0.75 on the gold set for unsupported-claim detection, at
> ≥10× lower per-claim cost than the judge baseline, both measured on stated hardware.**
> *If AUROC too low:* AlignScore, an ensemble of cheap signals, or a **biomedical-NLI fine-tune** —
> for which the window is one week (G3 Sep 20 → freeze Sep 27). **MedNLI (PhysioNet) access is
> licence-gated and human-reviewed: apply in W0, not W7.** Requires PhysioNet credentialing, CITI
> "Data or Specimens Only Research" training, and the Health Data Use Agreement 1.5.0.
> *If the cost ratio is thin:* the modifier weakens — the headline is already attribution quality
> (ADR-0002), so this costs the subtitle, not the paper. Decide by Sep 20.

### Phase 4 — Human gold-attribution set (Weeks 6–8, Sep 7–27) → **C4**, feeds C2

**Start the moment Phase 2 produces stable outputs — this has the longest lead time of anything in
the project and it is the classic October surprise.**

**Protocol (ADR-0006).** Three annotators; **annotators 2 and 3 are deliberately non-experts.** The
task is *"does this cited span support this claim?"* under a **no-outside-knowledge rule** — reading
comprehension, not clinical judgement. Domain expertise is a *liability*: experts mark claims
supported because they are *true*, which biases exactly the number C4 depends on.

| Role | Claims | Time |
|---|---|---|
| Primary (you) = Annotator 1 | **all ~250** | **~8–14 h** (ADR-0013 §4; the old 4 h lower bound was unreachable) |
| Annotator 2 | **the same ~250** | **~10–16 h** — no expertise discount (ADR-0016 §1) |
| Annotator 3 | **the same ~250** | **~10–16 h** |

**All three label the full set (ADR-0016).** There is no overlap subset, because the overlap *is*
the set: ≈750 annotator-level claim evaluations, α computed over the whole gold set rather than
estimated from a sample of it. ADR-0013's 3 h ceiling was sized to a two-volunteer constraint that
no longer holds — all three have agreed to the full pass. The cost is 30–49 annotator-hours in
Sep 7–27 and it is stated, not mitigated.

**The floor of this design is the ceiling of the old one.** All three work one shared seeded
question order, so the triple-labeled common prefix is always a complete unbiased subsample; if
capacity falls back to ~3 h each it shrinks to roughly the ~19 questions ADR-0013 bought
deliberately. Nothing is staked on the expanded commitment being met in full. The ~1 h pilot sitting
remains a separate uncosted session (W6 below).

- **Drawn from `data.gold_pool()`** — the 500 questions in neither dev nor test (§3). Never from
  `test`: C4 and the headline numbers must not share questions, or the agreement stops being
  independent evidence for the result. `tests/test_splits.py` asserts the disjointness.
- **Sampling: ~4 claims per question across ~62 questions** (ADR-0011), **not** every claim of ~27
  questions. At 9.2 claims/query a 250-claim budget otherwise covers only ~27 questions, and claims
  within a question are correlated — they share the question, passages, answer and topic. The
  250-claim budget is unchanged; question diversity rises ~2.3×. All three annotators label this
  same set, which is what makes a clustered bootstrap (§8 rule 10) mean anything.
- **There is no overlap subset** (ADR-0016). Every annotator labels all ~250 claims, so α is computed
  over the gold set rather than estimated from a sample of it: ~62 clusters instead of ~19, and an
  interval about $\sqrt{62/19} \approx 1.8\times$ narrower. Tripling raters adds precision, not
  units — 250 claims are still ~62 clusters, and the bootstrap still resamples questions.
- **The work is stoppable at any point with no loss** (ADR-0013 §3, generalised by ADR-0016 §2).
  **All three annotators work one shared seeded question order, question by question**, so the
  triple-labeled common prefix is always a complete unbiased subsample. This is a **W5 requirement
  on `data.py` and the annotation UI**, not a courtesy, and the UI must record per-question
  completion so a partial pass is separable from a half-finished question.
- **Blinding is load-bearing now that the author is one of three raters on identical units**
  (ADR-0016 §4). The UI presents claims stripped of system, model and run identity in a seeded
  shuffle that interleaves systems; the mapping to `(system, run_id, query_id)` lives outside the
  annotation file and is joined at scoring. No annotator sees another's judgements, or any aggregate
  over them, before finishing.
- **Disagreements are never adjudicated before α.** Raw per-annotator judgements are preserved
  (`HumanLabel.annotator_id`); choosing a single gold label per claim — majority, or the primary's —
  is a scoring-time decision made *after* α, with the no-majority rate reported either way.
- **The annotation unit is the (claim, cited span) pair.** Citations/claim was **1.01** on G0 (92
  claims, sections of one abstract); re-measured on the W4 end-to-end records in the real retrieval
  regime it is **1.13–1.50 parsed, 1.87–2.30 emitted** (`docs/harvest/generate_smoke_run4.md`,
  n = 3 questions, provisional). Re-run on the first ≥50-question batch before the W5 UI work. The
  union judgement is degenerate for single-citation claims.
- **Guidelines must say what to do with a sampled abstention claim** — at ~4 claims/question one can
  be drawn (ADR-0010).
- Labels are **4-way** (`SUPPORTED`/`PARTIAL`/`NOT_SUPPORTED`/`CONTRADICTED`) plus `claim_validity`.
  See [`CONTEXT.md`](CONTEXT.md). **Never collapse at write time** — an annotator cannot be re-run.
- Report **Krippendorff's α with a clustered bootstrap CI** (on a few dozen clusters it is not a
  point estimate), the **no-majority rate**, the **human ceiling**, the **decomposition-error rate**
  from `claim_validity`, and **per-annotator label distributions** — which full-set coverage makes
  informative for the first time. Report α for the first and second halves of the shared order
  separately: one order across all three annotators means fatigue drift is *correlated*, and
  correlated drift inflates α (ADR-0016 KW2).
- **Guidelines are written in W5 and carry all the load.** Concentrate effort on the
  **`SUPPORTED` vs `PARTIAL` boundary** — that is where the pilot fails if it fails. Include a
  worked example of jointly-necessary citations (dose in one span, outcome in another).
- Annotation UI: a static HTML form writing JSONL is sufficient — the `teach/assets` machinery is a
  fine starting point. Do not build a tool.

> **Gate G4 (by Sep 27): ≥250 claims labeled, and the *point* estimate α ≥ 0.6 computed on the
> binary collapse** (SUPPORTED+PARTIAL vs NOT_SUPPORTED+CONTRADICTED), the quantity C4 consumes,
> **over the triple-labeled common prefix** — the full gold set when all three finish, which is the
> target (ADR-0016 §3). The 4-way ordinal α is reported as a secondary number.
> **The clustered bootstrap CI is always reported, with its cluster count stated** (whatever was
> actually labeled — ~62 if the full pass lands, and never assumed), and is never used to soften the
> point.
> *If α < 0.6:* the guidelines are ambiguous, not the annotators. Revise guidelines, re-annotate.
> Report the final α whatever it is — a low α honestly reported with the ceiling stated is
> publishable; a hidden one is not. **This is the sole re-pilot trigger** (ADR-0011 dropped a second,
> CI-based one); it lands in **W9**.
> *Tripwire:* if the triple-labeled common prefix is under ~19 questions on **Sep 20**, the schedule
> is behind where the superseded overlap design would have been and G4 is at risk.

> **Why G4 gates on the point while G1 gates on point *and* Wilson lower bound.** Stated here so the
> asymmetry reads as a reason rather than a convenience. G1's gate is a **proportion over 100
> independent dev questions**, which supports a Wilson bound. G4's is **Krippendorff's α over a few
> dozen question-clusters** (§8 rule 10), where the bootstrap interval is wide enough that gating on it
> would make the gate a coin flip rather than a standard. We gate on the point and report the
> interval with its cluster count, so a reader can see exactly why it is wide.

### Phase 5 — Full runs, baselines, ablations (Weeks 8–10, Sep 21 – Oct 11) → **all tables**

1. **Freeze the code. Tag it.** Everything from here is runs and prose.
2. **Decide the frozen-run generator backend** (ADR-0004's deferred decision). Inputs: the 8B model's
   measured citation-format compliance rate. **Do not let this drift** — if it switches to an API
   backend, seeded variance is lost on switched systems, so headline systems stay local and the API
   is used for the swap check only.
3. Test-set runs, all systems: ours · vanilla RAG · post-hoc citation · ours−verifier ·
   ours−decomposition · judge-baseline. **≥3 seeds**, paired by question (Lesson 5: unpaired
   comparison picked the wrong winner 39% of the time in the notebook; paired, 0%).
4. **The swap check** (~\$2–10, under an hour): hold everything fixed, swap only the generator,
   re-run ours-vs-post-hoc on ~100 questions, report both gaps. Answers objections 6 in §1.
5. Significance: McNemar for accuracy, paired bootstrap for the rest. CIs on every headline number.
6. MedRAG/MIRAGE accuracy reference point for context (not a target — see C6).
7. Stratified error analysis for C9 (Table 5): negation, numerics, scope/population — driven by the
   `CONTRADICTED` labels (Lesson 6).

> **Gate G5 (by Oct 11): every cell of Tables 1–5 populated from a run manifest, with CIs. No number
> in the paper without a run ID behind it.**

### ~~Phase 6 — Generalization (C8)~~ — **CUT** (§1)

BioASQ-Y/N transfer is cut with C8. State single-dataset scope in Limitations; it moves to the 2027
journal extension. **Weeks 9–11 are freed** — absorb slippage from Phases 3–5 there rather than into
W14's buffer.

### Phase 7 — Writing (Weeks 10–13, Oct 5 – Nov 1)

Runs against the skeleton created in Week 0, in this order (tables outward, intro last):

- **W10:** Method + Experimental Setup (write while runs execute — you know the method already).
- **W11:** Results + Analysis, table by table, each paragraph anchored to a claim ID.
- **W12:** Related Work (mostly assembly from `related_work.md`, positioned per §7 of that file) ·
  Limitations · Ethics/clinical-risk statement (mandatory for a medical venue) · Reproducibility
  appendix (ML Reproducibility Checklist as the harness's test suite, Lesson 8).
- **W13:** Introduction + Abstract **last** — they're written from the results, never toward them.
  Then: internal red-team pass against the five objections in §1; format to venue template; public repo.

---

## 5. Week-by-week

| Week | Dates | Primary | Parallel | Due |
|---|---|---|---|---|
| **W0** | Jul 30 – Aug 2 | `src/biomedqa/` skeleton + `schema.py`; `paper/skeleton.md` — **all CPU-only; the A4000 is not available until Aug 3** | **Send the annotator ask**; **start MedNLI/PhysioNet application**; file the G0–G5 issues | (G0 moved → Aug 4) |
| **W1** | Aug 3–9 | `config.py`, data load, split freeze · **Mon–Tue: A4000 preflight, generator bake-off, MedCPT throughput** | Read/re-skim ALCE + MiniCheck with the schema in hand · distractor-pool construction | **G0 (Aug 4)** · Splits frozen (Aug 7) |
| **W2** | Aug 10–16 | **PMID dedup (blocks the encode)**; chunker sweep; **`bm25s`** + MedCPT + RRF; 2M encode; **empty title-segment convention — dev hit@5 measured both ways, then into the fingerprint** (ADR-0014 §3) | Harness: manifest, seed loop, cost log · **`backends.py` adapter (½ day)** · **confusability probe — pulls MiniCheck forward, ~½ day** | Table 1 rows 1–3 |
| **W3** | Aug 17–23 | Cross-encoder rerank; gate measurement + Wilson CIs | Decomposition prompt drafting on dev · **start logging prompt-iteration budget** · **re-confirm the probe post-rerank** | **G1** · Table 1 complete |
| **W4** | Aug 24–30 | Joint claim-grounded generation; schema round-trip · **granularity-parity loop, blind — hard stop at 10 iterations or Aug 30** | Vanilla RAG + post-hoc baselines, **equal-effort protocol** · **clustered-vs-unclustered CI dry-run** | First end-to-end record · **parity frozen** |
| **W5** | Aug 31 – Sep 6 | Decontextualization; granularity knob; citation P/R scorers (strict + lenient) · **first citation-F1 ever computed (≈ Aug 31)** | Verifier wiring; **cost instrumentation**; **guidelines pass 1 from Aug 31, worked examples Sep 3–6** · ~~MedNLI source~~ | **G2** · **decomposer freeze Sep 3** |
| **W6** | Sep 7–13 | MiniCheck + Opus 5 judge, identical APIs; AlignScore row | **Annotation pilot (10 claims, 3 annotators, ~1 h each)** — tests the *guidelines* · **all three main passes start here** (ADR-0013 §4, ADR-0016) | Gold set launched |
| **W7** | Sep 14–20 | Threshold sweep, AUROC, ECE; **overhead measurement (clean, GPU idle, ≥5 runs)** | Annotation in progress — **3 × 10–16 h is now the long pole** · **Sep 20 tripwire: common prefix ≥ ~19 questions** | **G3** · Table 4 draft |
| **W8** | Sep 21–27 | **Code freeze + tag. Decide the frozen-run backend.** Test runs begin, seed 1 | Annotation completes; α over the triple-labeled prefix | **G4** |
| **W9** | Sep 28 – Oct 4 | Seeds 2–3, all systems; ablations; **swap check** | ⚠ **Triple-booked:** stratified robustness check (ADR-0009, *likely* to trigger) · a G4 re-pilot if α < 0.6 · *and* its original role absorbing Phase 3–5 slippage | Raw results |
| **W10** | Oct 5–11 | Significance tests, CIs; stratified error analysis (Table 5) | **Write Method + Setup** | **G5** · Tables 1–5 final |
| **W11** | Oct 12–18 | Figures 1–3 | **Write Results + Analysis** | Results section |
| **W12** | Oct 19–25 | Repo cleanup, reproducibility appendix, run-manifest export | **Write Related Work, Limitations, Ethics** | Full draft |
| **W13** | Oct 26 – Nov 1 | Red-team pass; venue formatting; public repo release | **Write Intro + Abstract** | **Submission-ready** |
| **W14** | Nov 2–8 | **Buffer.** Submit. | — | **Submitted** |

W14 is real buffer, not a second W13. Something in Phases 3–5 will slip; this is where it lands.

> ⚠ **W9 is over-committed and it is the only slack in the schedule.** It now carries three claims:
> ADR-0009's stratified robustness check (whose triggering branch is the *likely* one, since joint
> emits finer claims natively), a possible G4 re-pilot, and Phase 3–5 slippage. **Unresolved.** The
> cut order (§8 rule 8) is the lever, and parity-loop iterations are the designated first cut.

---

## 6. Venue — **LOCKED: workshop, hard November deadline** (ADR-0001)

BioNLP-style workshop or a medical-AI venue with a fall deadline. A dated outcome this semester.

**Action by Aug 16** (not October — page limits and column format change how tables are built):
shortlist 2–3 and **verify official CFPs and page limits**. The 5-table budget in §1 assumes ~8
pages; a 4-page limit would force another cut, and it is better to know in August.

**The journal route is deferred, not discarded.** A 2027 extension carries the BioASQ generalization
(C8), the granularity study (C7) as a full table, a larger gold set, and — if it proves worthwhile —
the full 23.9M MedRAG corpus (ADR-0003).

**Decide the column format by Oct 5** — Tables 1 and 4 are wide, and a single-column template forces
restructuring if discovered late.

---

## 7. Risk register

| # | Risk | Trigger | Response |
|---|---|---|---|
| R1 | **Corpus / index build** — the 2M encode or `bm25s` index doesn't land | W2 encode exceeds ~4 h, or index build fails at 2M | **Encode half discharged at G0 (2026-08-04): measured 1.6 h for 2M, 343.6 abstracts/s.** The 1M fallback is not needed. *The `bm25s` half is still live* — no 2M index build has been attempted, so index failure remains the open leg of R1. **New in W2: PMID dedup must precede the encode** (ADR-0012) — a gold abstract present twice under two `passage_id`s makes `gold_rank`/hit@5 miscount silently, and the encode would have to be redone. **Never** fall back to the 1,000 gold contexts (ADR-0003) |
| R2 | Retrieval gate stalls | G1 missed Aug 23 | Escalation ladder in Phase 1; ultimately relax to hit@10 and reframe as conditional attribution — **never** by tuning τ |
| R3 | **Gold annotation slips** (highest-probability October surprise; **more likely under ADR-0016** — 30–49 annotator-hours in Sep 7–27) | Not launched by Sep 13, or common prefix < ~19 questions on Sep 20 | The shared seeded order (ADR-0016 §2) means a short pass degrades to the superseded overlap design, not to nothing — **that is the first-line response, and it is automatic**. Only if the set is never launched: cut to 150 claims and fall back to ADR-0006's protocol — LLM-assisted pre-label (**not Opus 5** — it would contaminate the gold against the judge) + full human adjudication + blind self-agreement re-annotation ≥2 weeks apart → **intra-annotator** α. Report reduced *n* and wider CIs |
| **R3b** | **Annotators never materialize** | Ask not accepted by ~Aug 20 | Same fallback as R3. **The ask goes out in W0 precisely to make this detectable in August, not September** |
| R4 | Verifier too heavy → the cost modifier dies | Cost ratio < 10× at G3 | The headline is already attribution quality (ADR-0002), so this costs the subtitle, not the paper. Decide **by Sep 20** |
| R5 | Joint ≈ post-hoc (null on C2) | G2 margin inside the CI | Check baseline fairness first (§4 Phase 2 — but note the equal-effort protocol makes an *artifactual* gap less likely, not more). If genuine, reposition around **C3 + C9** and publish C2 as a negative finding. **The response is decided before Aug 31, not on the day:** ADR-0009's blind parity loop means the first citation-F1 lands ≈ Aug 31 and G2 is Sep 6 — a **six-day window** with no earlier warning, since the burn-slice unblinding was declined. Clustering (§8 rule 10) also **widens** G2's margin, at an unchanged threshold |
| R6 | Crowded lane — a 2026 paper lands on this exact combination | Any time | Re-run the literature check **Sep 1** and **Oct 15**; differentiate on the biomedical + generation-time + cheap axis, and cite explicitly |
| R7 | Verifier degrades on biomedical text | Phase 3 measurement | **Expected, not exceptional** — MiniCheck is ANLI/synthetic-trained. Report the degradation, then mitigate. **The MedNLI fine-tune is off the table for this paper: the PhysioNet application was deliberately not started (decided 2026-08-04), and credentialing is slower than the remaining schedule.** The ladder is now **calibration/threshold → AlignScore (a W6 deliverable) → cheap-signal ensemble**, and it ends there. **Because it ends there, AlignScore left the cut order and is now never-cut (§8 rule 8)** — it was simultaneously the ladder's middle rung and the first thing dropped when behind, and the cut fires in W6, a week before G3. If all three leave AUROC < 0.75, G3 fails with no fourth option and C3 is reported as a negative result — which ADR-0002 already permits, since the headline is attribution quality and C3 is the modifier |
| R8 | Writing compressed into the last week | W10 Method not drafted | Method/Setup are written in W10 *by design*, while runs execute — protect that |
| R9 | Scope creep (graph RAG, iterative retrieval, agents…) | Any new component after Sep 6 | **Hard freeze after G2.** New ideas go in a `future_work.md`, not the pipeline |
| **R10** | **Decomposition quality confounds the headline** | Malformed/over-split claims move C2 and C3 for non-method reasons | The `claim_validity` flag (ADR-0005) makes it measurable: report the decomposition-error rate and, if needed, headline numbers over well-formed claims only |
| **R11** | **W8 backend decision drifts** | Code freeze passes without an explicit choice | It is a named W8 deliverable (§4 Phase 5). If the 8B model's format compliance is adequate, the default is "stay local" — and staying local preserves the seed story |

---

## 8. Standing rules

1. **No number without a run ID.** Every figure in the paper traces to `runs/<id>/manifest.json`.
2. **Never binarize at write time** — including human labels. Raw scores, 4-way labels, char offsets,
   gold spans. Thresholds and collapses live in scoring. An annotator cannot be re-run.
3. **Seeds by construction.** The harness loops seeds or variance never gets reported. **Headline
   systems stay on the local generator** — the API has no sampling knob to seed (ADR-0004).
4. **Pair every comparison** by question. Unpaired single-seed comparisons are actively misleading.
5. **Dev for tuning, test once.** Thresholds, prompts, and chunkers are chosen on dev, full stop.
6. **Cost is a first-class metric.** Logged per call, per run, always. **Tokens and \$ are primary.**
7. **Baselines get equal effort, and the effort is logged.** Prompt-iteration counts are a reported
   number, not a good intention (§4 Phase 2).
8. **Cut order when behind:** ~~BioASQ (C8)~~ *(already cut)* → ~~AlignScore second row~~
   *(promoted to never-cut, below)* → seeds 3→2 on ablations only → test set 500→300 → iterations of
   the granularity-parity loop (ADR-0009 §5's one-sided fallback already absorbs a miss).
   **Never cut:** the gold set · the CIs · the overhead measurement · **AlignScore** · **ADR-0012's
   confusability probe**.
   > **Why AlignScore moved.** `315f77b` dropped the MedNLI fine-tune, leaving R7's ladder at three
   > rungs — calibration/threshold → **AlignScore** → cheap-signal ensemble — with no fourth option.
   > AlignScore was simultaneously the ladder's middle rung and the first thing cut when behind, and
   > those conditions are positively correlated. Worse, the cut is triggered in **W6 (Sep 7–13)**
   > while G3 is **Sep 20**, so it would be cut a week before you learn whether you need it, with no
   > room to rebuild inside G3 → freeze (Sep 20–27). It costs ~½ day and one inference pass; as
   > insurance against G3 failing with no options, it is the cheapest thing on this list.
9. **`CONTEXT.md` is authoritative on the four units.** If code, guidelines, and prose disagree about
   what a claim is, `CONTEXT.md` wins and the others get fixed.
10. **Every bootstrap cluster-resamples questions, never claims** (ADR-0011). Claims within a
    question share passages, answer and topic; resampling them as independent units returns CIs that
    are too narrow. This applies to **every** interval in the paper — G2's margin, G4's α, and every
    CI in Tables 2–5 — and **every table caption naming a CI must name its resampling unit.**
    Pairing is not clustering: pairing compares two systems on the same question, clustering makes
    the question the resampling unit. **This widens intervals, including G2's, and the gate
    thresholds are unchanged.**

---

## 9. Immediate next actions

**The first two are external requests with lead times. They are not writing tasks and they block
things that cannot be compressed later.**

1. ~~**⏰ Send the annotator ask**~~ — **DONE 2026-08-04; both accepted.** Two literate
   **non-experts**, early September; not a biomedical qualification. **Superseded 2026-08-11
   (ADR-0016):** all three now label the full ~250-claim set, ~10–16 h each, on their own agreement.
   ADR-0013 §1's 3 h ceiling was sized to the previous commitment and no longer binds. Remaining:
   send the shared-order schedule and the stop-anytime guarantee — which is what makes a short pass
   degrade to the old overlap design rather than to nothing.
2. ~~**Start the MedNLI / PhysioNet application**~~ — **DROPPED 2026-08-04, deliberately.** The
   credentialing path (PhysioNet + CITI "Data or Specimens Only Research" + HDUA 1.5.0) is
   human-reviewed and slower than the schedule that remains, and the fine-tune it enables never fit
   its own window (G3 Sep 20 → freeze Sep 27, a week already holding code freeze, seed-1 test runs,
   and annotation completion). Dropped with the consequence understood and accepted: **G3 has no
   fourth fallback.** See R7. *Do not restart this in September — by then it buys nothing.*
3. **Pick the 8B AWQ generator** and benchmark 10 real queries on the A4000; write measured latency
   into §2. **Benchmark MedCPT encode on 1,000 abstracts** to convert §3's estimates. Scripts are
   written and committed (`scripts/g0_*`); this is now execution only. *(Aug 3–4 — the box is not
   available before Mon Aug 3)*
4. **Create `src/biomedqa/`** and promote `08_6_reproducible_eval_harness.ipynb` into `harness.py` +
   `schema.py`, with `schema.py` implementing `CONTEXT.md`'s four units. *(Aug 2)*
5. **Create `paper/skeleton.md`** — nine sections, the cut claim ledger, **five** empty tables with
   real captions. *(Aug 2)*
6. **Freeze the splits** and commit the ID lists. *(Aug 7)*
7. **File these as GitHub issues** per `CLAUDE.md` — one per gate (G0–G5), plus one each for actions
   1 and 2, labeled `ready-for-agent` where mechanical. *(Aug 3)*
