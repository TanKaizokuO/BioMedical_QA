# Project 2 — Research Roadmap to Submission

### Evidence-Grounded, Claim-Attributable Biomedical QA

**Written:** 2026-07-30 · **Target submission:** 2026-10-26 → 2026-11-06 (venue-dependent)
**Companions:** [`project2_biomedical_attribution_rag_implementation_plan.md`](project2_biomedical_attribution_rag_implementation_plan.md) (the *what*) ·
[`related_work.md`](related_work.md) (the *against what*) · [`learning_roadmap.md`](learning_roadmap.md) (the *concepts*, now fully taught — 8 lessons)

> This document is the *execution* layer: what gets built, in what order, with which numeric gates,
> and how each experiment maps to a table or figure in the paper. The study roadmap (§1.1–1.6) is
> **complete**; from here on, every hour should produce either a measurement or a paragraph.

---

## 0. Where the project actually stands (2026-07-30)

An honest audit, because the plan's stated starting point is optimistic:

| Asset | Reality | Consequence |
|---|---|---|
| "Existing RAG-over-PubMedQA pipeline, Slice 2 bug fixed" | Lives in **`~/Code/Research/RAG_Debate_Agent`**, not this repo. The fix (`e936d30`, index `pqa_labeled` not `pqa_artificial`) was applied to `rag_baseline.py` but **never re-executed**. `pubmedqa_baseline_v2` does not exist. | **hit@5 is currently unmeasured, not "nearly 90%."** Treat the gate as un-attempted. |
| Retriever stack in that repo | ChromaDB + `all-MiniLM-L6-v2`, dense-only, top-5. No BM25, no RRF, no reranker. | Does not match the architecture the paper needs (hybrid + rerank). Porting it buys almost nothing. |
| Generation stack in that repo | Local Ollama `qwen2.5:7b`, **~88s/query** (range 61–110s), CPU. | **This is the #1 schedule risk — larger than the retrieval gate.** See §2. |
| This repo (`BioMedical_QA`) | Planning docs + 8 taught lessons + **8 runnable notebooks** already implementing BM25-from-scratch, MedCPT, RRF, cross-encoder rerank, citation P/R, decompose-then-verify, AUROC/calibration/CIs, Krippendorff's α, and a **working miniature eval harness** (`08_6_reproducible_eval_harness.ipynb`). | The notebooks *are* the codebase seed. `08_6` is the harness skeleton; promote it to `src/`. |
| Paper | Not started. No skeleton, no claim ledger, no venue shortlist. | Started in Week 0 — the paper is written **backwards from its tables**, not at the end. |

**Two decisions to make in Week 0 (§2). Everything downstream depends on them.**

---

## 1. The paper this roadmap must produce

Lock this now; every experiment below exists to fill one of these slots.

**Working title:** *Cheap Per-Claim Grounding: Joint Claim-Attributed Generation with a Lightweight
Entailment Verifier for Biomedical QA*

**Thesis (one sentence):** In biomedical QA, generating answers as atomic claims each jointly
attributed to a retrieved passage — and screening them with a small entailment verifier at
generation time — yields substantially better attribution quality and lower hallucination rate than
post-hoc citation, at a small fraction of the cost of an LLM-judge, without regressing answer accuracy.

### The claim ledger

Each row is a claim the paper makes, the experiment that earns it, and where it lands. **A claim
with no experiment is cut. An experiment serving no claim is not run.**

| # | Claim | Experiment | Artifact |
|---|---|---|---|
| C1 | Retrieval is adequate, so attribution is meaningful | Retrieval gate: hybrid BM25+MedCPT+RRF+reranker vs. each ablated stage | **Table 1** (hit@5, recall@5, MRR, nDCG@10 + Wilson CIs) |
| C2 | **Joint** claim-grounded generation beats **post-hoc** citation on attribution | Ours vs. post-hoc-citation baseline, same retriever, same generator | **Table 2** (citation precision / recall / F1, ALCE-style) |
| C3 | Per-claim verification reduces hallucination | Ours vs. vanilla RAG vs. ours-minus-verifier | **Table 2** (hallucination rate = fraction of claims with no valid support) |
| C4 | The cheap verifier matches the expensive judge | Verifier vs. LLM-judge on the human-gold support set | **Table 3** (AUROC, AUPRC, ECE, agreement with human) + **Fig. 2** (ROC + reliability diagram) |
| C5 | **…at low overhead** — the headline | Latency / tokens / \$ per query, ours vs. judge-baseline, measured on identical hardware | **Table 4** + **Fig. 3** (quality-vs-cost Pareto) |
| C6 | Attribution doesn't cost accuracy | PubMedQA yes/no/maybe accuracy, ours vs. vanilla RAG, non-regression test | **Table 5** (secondary; McNemar / paired bootstrap) |
| C7 | Decomposition granularity is a real design variable, not a detail | Granularity ablation (sentence / atomic / decontextualized-atomic) | **Table 6** + short analysis |
| C8 | It isn't PubMedQA-specific | BioASQ-Y/N transfer run (**optional — cut first if behind**) | **Table 7** |
| C9 | The failure modes are biomedical-specific and characterized | Stratified error analysis: negation, numerics, scope/population (Lesson 6 material) | **Table 8** + qualitative examples |

**Figure 1** is the system diagram. **Fig. 4** (optional) is a worked example: question → claims →
citations → verifier verdicts, one supported and one caught-unsupported.

### Reviewer objections to pre-empt (answer inside the paper, not in rebuttal)

1. *"MiniCheck already showed cheap verifiers work."* → Ours wires it **into generation-time
   per-claim screening** in a domain it wasn't trained for; we report its biomedical degradation and
   what we do about it. Cite MiniCheck as backbone **and** baseline.
2. *"Decomposition is known not to always help."* → We ablate it (C7) and cite *Decomposition Dilemmas*.
3. *"Your attribution gold is small."* → Report α (Krippendorff), the human ceiling, gold-set sizing
   rationale, and CIs on every gold-derived number (Lesson 7 material).
4. *"You didn't beat SoTA accuracy."* → Explicitly out of scope and stated in the intro; MedRAG/MIRAGE
   reported as a reference point, not a target (C6 is a *non-regression* claim).
5. *"Overhead numbers are hardware-flattering."* → Same hardware, same batch policy, tokens **and**
   wall-clock **and** \$ reported, cost logged per run in the manifest (Lesson 8 material).

---

## 2. Week 0 (Jul 30 – Aug 2): the two blocking decisions

Nothing else starts cleanly until these are made. Both are yours; both have a recommendation.

### D1 — Compute for generation. **This is the schedule.**

The base repo's 88s/query on local CPU Ollama, at the scale this paper needs, is fatal. Rough arithmetic:

> ~500 PubMedQA questions × {ours, vanilla RAG, post-hoc citation, ours−verifier, ours−decomposition}
> = 5 systems ≈ **2,500 generation calls**, plus the judge-baseline over every claim (≈ 3–5 claims/answer
> → **~7,500+ judge calls**), plus reruns after bugs, plus ≥3 seeds where variance is claimed.
> At 88 s/call that is **hundreds of hours**. It does not fit before November.

**Decide one of:**
- **(A) Hosted API generator** (recommended — latest capable Claude model). Removes the bottleneck,
  makes the \$-per-query column honest and easy to log, and makes reruns cheap. Cost is real but bounded;
  budget it explicitly in Week 0 and log spend per run.
- **(B) GPU access** (university cluster / rented). Keeps everything local and free-at-point-of-use;
  costs setup time and queue latency.
- **(C) Small local model** (`qwen2.5:3b`). Cheapest, but a weak generator undermines C2/C3 — a bad
  generator makes attribution look easy or impossible for the wrong reasons.

**Gate G0:** by **Aug 2**, one generator is chosen, benchmarked on 10 real queries, and the measured
per-call latency is written into this file. If the measured throughput cannot complete the §5 run plan
in ≤ 72 h of wall-clock, shrink the eval set (§3) *now*, not in October.

> Note: the verifier and reranker are small models and stay local — that's the point of C5. Only the
> *generator* and the *judge baseline* need real compute, and both are measured on their own hardware
> with the hardware stated in the paper.

### D2 — Port vs. rebuild the pipeline.

**Recommendation: rebuild in this repo, harvesting from the notebooks.** The base repo is a
*multi-agent debate* project with a different architecture (dense-only Chroma, no BM25/RRF/reranker,
agent-slice structure). The notebooks here already contain working versions of every component the
paper's architecture needs. Porting means adapting code that doesn't match the target design; rebuilding
means promoting code that does.

Harvest only these from `RAG_Debate_Agent`: the PubMedQA loading logic, the gold-passage tracking
fields added in `e936d30`, and the latency-benchmark methodology in `benchmark.py`.

**Deliverable of Week 0** — the repo skeleton, derived from Lesson 8's harness design:

```
src/biomedqa/
  config.py          # every knob; base + diff, versioned, hashed into the run manifest
  data.py            # PubMedQA (pqa_labeled) load, split freeze, gold-context extraction
  chunk.py           # passage granularity (Lesson 2: hit@5 is only defined per (chunker, τ) pair)
  retrieve.py        # BM25 | MedCPT dense | RRF fusion | cross-encoder rerank
  generate.py        # joint claim-grounded generation; post-hoc + vanilla baselines behind one API
  decompose.py       # atomic claims; granularity is a config knob (C7)
  verify.py          # cheap entailment verifier (MiniCheck-family) + judge-LLM baseline
  schema.py          # THE FROZEN OUTPUT SCHEMA — least-processed values only
  scoring/           # pure functions over the schema: retrieval, citation, faithfulness, calibration, accuracy
  harness.py         # seed loop, cost log, run manifest, config diff
runs/                # one directory per run: manifest.json + records.jsonl + costs.jsonl (gitignored)
paper/               # skeleton from day one
```

**The frozen schema is the single most important artifact in the repo.** Store `phi_score: 0.83`,
never `supported: true`; store char offsets and gold spans, never a precomputed hit@5. Binarizing at
write time destroys the AUROC sweep and calibration bins irrecoverably and turns re-chunking into a
re-*run*. (Lesson 8, §"least-processed-value rule".)

**Also in Week 0 — start the paper.** Create `paper/skeleton.md` with all nine section headings, the
claim ledger above pasted in, and every table from §1 present as an **empty table with real column
headers and a caption**. The captions are written before the numbers exist. This is not ceremony: an
empty Table 4 with the columns "latency (s) / tokens / \$ per query / AUROC" forces you to instrument
for those columns in Week 5 instead of discovering the gap in October.

---

## 3. Evaluation set: freeze it now, never touch it again

| Split | Contents | Purpose | Frozen by |
|---|---|---|---|
| **dev** | 100 PubMedQA `pqa_labeled` questions | All development, prompt iteration, threshold tuning | Aug 7 |
| **test** | ~400–500 `pqa_labeled` questions, disjoint from dev | **Every number in the paper.** Run late, run once per system per seed. | Aug 7 |
| **gold-attribution** | 60–100 dev+test answers, ~250–400 claims, human-annotated support labels | C4 (verifier vs. human), attribution precision/recall ground truth | Sep 13 (§6) |
| **transfer** | BioASQ-Y/N subset | C8, optional | Sep 27 |

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

### Phase 2 — Claim decomposition + joint attribution (Weeks 3–5, Aug 17 – Sep 6) → **C2, C7**

Overlaps Phase 1's tail deliberately; decomposition prompts can be developed on dev-set retrievals
while the reranker is still being tuned.

1. **Frozen output schema first** (`schema.py`): `{question_id, answer_text, claims: [{claim_id, text,
   citations: [{passage_id, char_start, char_end}], verifier_score, ...}], retrieval: [...], costs: {...}}`.
2. Joint generation: one call emits claims **with** citations. Prompt is versioned config (Lesson 8).
3. Decontextualization pass — atomic claims with dangling pronouns are unverifiable (DnDScore).
4. Granularity knob for C7: `sentence` | `atomic` | `atomic+decontextualized`.
5. **Baselines built in the same module, behind the same API** (this matters — it's what makes the
   comparison fair and the harness clean): vanilla RAG (no attribution); post-hoc citation
   (generate, then attach citations).
6. Citation precision/recall scorers as pure functions over the schema (ALCE-style, from notebook `03_2`).

> **Gate G2 (by Sep 6): on dev, joint attribution beats post-hoc citation on citation-F1 by a margin
> exceeding the paired-bootstrap CI, and ≥95% of emitted claims parse into the schema with resolvable
> spans.**
> *If the margin is not there:* this is the paper's central contrast — investigate before proceeding.
> Usual causes: post-hoc baseline accidentally strengthened by using the same retriever, or joint
> prompts under-specified. A genuinely null result here means **repositioning the paper around C3+C5**
> (verification and cost) with C2 reported as a negative finding — decide this by Sep 6, not October.

### Phase 3 — Cheap verifier + overhead (Weeks 5–7, Aug 31 – Sep 20) → **C3, C4, C5**

**Instrument cost from the first line of code in this phase.** The low-overhead claim is the paper's
edge and it cannot be reconstructed after the fact.

1. Wire MiniCheck-family verifier: premise = cited passage span, hypothesis = claim → score.
2. **Do not binarize.** Store the raw score; threshold at scoring time so the AUROC sweep and
   calibration bins remain possible.
3. Judge-LLM baseline behind the identical API, with identical logging.
4. Cost instrumentation, per call and per query: wall-clock, input/output tokens, \$, and hardware ID.
5. Threshold selection on **dev only**; report the ROC over the sweep, plus ECE and a reliability
   diagram (notebook `05_4`).
6. Biomedical degradation check: verifiers trained general-domain will drop on biomedical text.
   Measure it and report it — it's a contribution, not an embarrassment.

> **Gate G3 (by Sep 20): verifier AUROC ≥ 0.75 on the gold set for unsupported-claim detection, at
> ≥10× lower per-claim cost than the judge baseline, both measured on stated hardware.**
> *If AUROC too low:* try AlignScore, a biomedical-NLI fine-tune, or an ensemble of cheap signals.
> *If the cost ratio is thin:* the headline weakens — pivot the framing toward attribution quality
> (C2) as primary and cost as secondary. Decide by Sep 20.

### Phase 4 — Human gold-attribution set (Weeks 6–8, Sep 7–27) → **C4**, feeds C2

**Start the moment Phase 2 produces stable outputs — this has the longest lead time of anything in
the project and it is the classic October surprise.**

- Protocol from SALAD (Lesson 7): per-claim support judgements, 3 annotators on an overlap subset.
- Report **Krippendorff's α** on the overlap, the **no-majority discard rate**, and the **human
  ceiling** — the ceiling caps what the verifier can be credited with achieving.
- Size the set from the CI width you need on Table 3, not from convenience.
- Annotation UI: a static HTML form writing JSONL is sufficient — the `teach/assets` machinery is a
  fine starting point. Do not build a tool.

> **Gate G4 (by Sep 27): ≥250 claims labeled, α ≥ 0.6 on the overlap subset.**
> *If α < 0.6:* the guidelines are ambiguous, not the annotators. Revise guidelines, re-annotate the
> overlap. Report the final α whatever it is — a low α honestly reported with the ceiling stated is
> publishable; a hidden one is not.

### Phase 5 — Full runs, baselines, ablations (Weeks 8–10, Sep 21 – Oct 11) → **all tables**

1. **Freeze the code.** Tag it. Everything from here is runs and prose.
2. Test-set runs, all systems: ours · vanilla RAG · post-hoc citation · ours−verifier ·
   ours−decomposition · judge-baseline. **≥3 seeds**, paired by question (Lesson 5: unpaired
   comparison picked the wrong winner 39% of the time in the notebook; paired, 0%).
3. Significance: McNemar for accuracy, paired bootstrap for the rest. CIs on every headline number.
4. MedRAG/MIRAGE accuracy reference point for context (not a target — see C6).
5. Stratified error analysis for C9: negation, numerics, scope/population (Lesson 6).

> **Gate G5 (by Oct 11): every cell of Tables 1–6 populated from a run manifest, with CIs. No number
> in the paper without a run ID behind it.**

### Phase 6 — Generalization (Weeks 9–11, Sep 28 – Oct 18, **optional**) → **C8**

BioASQ-Y/N transfer. **This is the designated cut.** If anything upstream slipped, drop it without
regret and state single-dataset scope in Limitations. It strengthens the paper; it does not carry it.

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
| **W0** | Jul 30 – Aug 2 | **D1 compute decision + benchmark; D2 rebuild** | `paper/skeleton.md` with empty captioned tables | **G0** |
| **W1** | Aug 3–9 | `src/` skeleton, `config.py`, `schema.py`, data load, split freeze | Read/re-skim ALCE + MiniCheck with the schema in hand | Splits frozen (Aug 7) |
| **W2** | Aug 10–16 | Chunker sweep; BM25 + MedCPT + RRF | Harness: manifest, seed loop, cost log | Table 1 rows 1–3 |
| **W3** | Aug 17–23 | Cross-encoder rerank; gate measurement + Wilson CIs | Decomposition prompt drafting on dev | **G1** · Table 1 complete |
| **W4** | Aug 24–30 | Joint claim-grounded generation; schema round-trip | Vanilla RAG + post-hoc baselines | First end-to-end record |
| **W5** | Aug 31 – Sep 6 | Decontextualization; granularity knob; citation P/R scorers | Verifier wiring begins; **cost instrumentation** | **G2** |
| **W6** | Sep 7–13 | Verifier + judge baseline, identical APIs | **Annotation guidelines + pilot (10 claims, 3 annotators)** | Gold set launched |
| **W7** | Sep 14–20 | Threshold sweep, AUROC, ECE, reliability diagram; overhead measurement | Annotation in progress | **G3** · Fig. 2, Table 4 draft |
| **W8** | Sep 21–27 | **Code freeze + tag.** Test runs begin, seed 1 | Annotation completes | **G4** |
| **W9** | Sep 28 – Oct 4 | Seeds 2–3, all systems; ablations | (Optional) BioASQ setup | Raw results |
| **W10** | Oct 5–11 | Significance tests, CIs; stratified error analysis | **Write Method + Setup** | **G5** · Tables 1–6 final |
| **W11** | Oct 12–18 | Figures 1–4; (optional) BioASQ run | **Write Results + Analysis** | Results section |
| **W12** | Oct 19–25 | Repo cleanup, reproducibility appendix, run-manifest export | **Write Related Work, Limitations, Ethics** | Full draft |
| **W13** | Oct 26 – Nov 1 | Red-team pass; venue formatting; public repo release | **Write Intro + Abstract** | **Submission-ready** |
| **W14** | Nov 2–8 | **Buffer.** Submit. | — | **Submitted** |

W14 is real buffer, not a second W13. Something in Phases 3–5 will slip; this is where it lands.

---

## 6. Venue

**Action by Aug 16** (not October — page limits and column format change how tables are built):
shortlist 2–3 and verify official CFPs.

| Route | Candidates | Trade-off |
|---|---|---|
| **Workshop / conference** (dated outcome this cycle) | BioNLP-style workshop; medical-AI venue with a fall deadline | Decision inside the semester; shorter page limit forces ruthless table selection. **Recommended if a dated outcome matters.** |
| **Journal** | *JBI*, *JAMIA*, applied-AI journal | Stronger line, rolling deadlines (schedule pressure drops), but acceptance lands 2027 |

Hybrid worth considering: workshop submission in November for the dated outcome and feedback, journal
extension in 2027 with the BioASQ generalization and a larger gold set added.

**Decide the format by Oct 5** — Table 1 and Table 4 are wide, and a single-column template will force
restructuring if discovered late.

---

## 7. Risk register

| # | Risk | Trigger | Response |
|---|---|---|---|
| R1 | **Generation throughput** — the plan doesn't fit the compute | G0 benchmark shows > 72 h for the §5 run plan | Shrink test set to 300 with CIs; drop to 2 seeds on ablations only (never on headline systems); hosted API |
| R2 | Retrieval gate stalls | G1 missed Aug 23 | Escalation ladder in Phase 1; ultimately relax to hit@10 and reframe as conditional attribution — **never** by tuning τ |
| R3 | **Gold annotation slips** (highest-probability October surprise) | Not launched by Sep 13 | Cut gold set to 150 claims, single annotator + 30-claim overlap for α; report reduced *n* and wider CIs |
| R4 | Verifier too heavy → headline dies | Cost ratio < 10× at G3 | Reframe: attribution quality primary, cost secondary. Decide **by Sep 20** |
| R5 | Joint ≈ post-hoc (null on C2) | G2 margin inside the CI | Investigate baseline strength first; if genuine, publish as a negative finding around C3/C5 |
| R6 | Crowded lane — a 2026 paper lands on this exact combination | Any time | Re-run the literature check **Sep 1** and **Oct 15**; differentiate on the biomedical + generation-time + cheap axis, and cite explicitly |
| R7 | Verifier degrades on biomedical text | Phase 3 measurement | This is a *result*, not a failure. Report the degradation, then mitigate (biomedical NLI fine-tune) |
| R8 | Writing compressed into the last week | W10 Method not drafted | Method/Setup are written in W10 *by design*, while runs execute — protect that |
| R9 | Scope creep (graph RAG, iterative retrieval, agents…) | Any new component after Sep 6 | **Hard freeze after G2.** New ideas go in a `future_work.md`, not the pipeline |

---

## 8. Standing rules

1. **No number without a run ID.** Every figure in the paper traces to `runs/<id>/manifest.json`.
2. **Never binarize at write time.** Raw scores, char offsets, gold spans. Thresholds live in scoring.
3. **Seeds by construction.** The harness loops seeds or variance never gets reported.
4. **Pair every comparison** by question. Unpaired single-seed comparisons are actively misleading.
5. **Dev for tuning, test once.** Thresholds, prompts, and chunkers are chosen on dev, full stop.
6. **Cost is a first-class metric.** Logged per call, per run, always — it's the headline claim.
7. **Cut order when behind:** BioASQ (C8) → decomposition-granularity ablation (C7) → seeds 3→2 on
   ablations → test set 500→300. **Never cut:** the gold set, the CIs, or the overhead measurement.

---

## 9. Immediate next actions

1. **Make D1** — pick the generator, benchmark 10 real queries, write the measured latency into §2. *(Today–Aug 2)*
2. **Make D2** — confirm rebuild-in-this-repo; create `src/biomedqa/` and promote `08_6_reproducible_eval_harness.ipynb` into `harness.py` + `schema.py`. *(Aug 2)*
3. **Create `paper/skeleton.md`** — nine sections, claim ledger, nine empty tables with real captions. *(Aug 2)*
4. **Freeze the splits** and commit the ID lists. *(Aug 7)*
5. **File these as GitHub issues** per `CLAUDE.md` — one per gate (G0–G5), labeled `ready-for-agent` where they're mechanical. *(Aug 3)*
