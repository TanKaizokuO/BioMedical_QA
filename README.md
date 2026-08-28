# Biomedical QA — Evidence-Grounded, Claim-Attributable

A biomedical question-answering system that answers as a set of atomic, decontextualized claims,
each attributed to the retrieved passage(s) that support it, with a lightweight faithfulness
verifier. Evaluation leads on attribution quality and faithfulness; accuracy is secondary.

The headline claim: joint per-claim grounding (generating claims and their citations together)
produces higher citation recall than post-hoc attribution (generate, then attach citations) at
comparable precision, measured against ~2M PubMed abstracts with a real distractor pool.

---

## Pipeline

```
Question
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Retrieval cascade  (retrieve.py)                    │
│  1. BM25 (bm25s) + MedCPT dense → pool of 100       │
│  2. RRF fusion                                       │
│  3. MedCPT cross-encoder rerank → top-k passages     │
└────────────────────────┬─────────────────────────────┘
                         │ top-k passages
                         ▼
┌──────────────────────────────────────────────────────┐
│  Generation  (generate.py + backends.py)             │
│  Joint:     one call, claims + citations together    │
│  Post-hoc:  two calls — answer, then cite            │
│  Vanilla:   one call, no attribution                 │
└────────────────────────┬─────────────────────────────┘
                         │ QueryRecord (schema.py)
                         ▼
┌──────────────────────────────────────────────────────┐
│  Claim decomposition  (decompose.py)                 │
│  Decontextualized atomic claims — C7 ablation only   │
│  (headline systems emit CLAIM lines directly)        │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  Faithfulness verification  (verify.py)              │
│  MiniCheck-Flan-T5-Large: (passage, claim) → [0,1]  │
│  Opus 5 judge baseline for cost comparison           │
└────────────────────────┬─────────────────────────────┘
                         │ VerifierScore per citation
                         ▼
┌──────────────────────────────────────────────────────┐
│  Scoring  (scoring/)                                 │
│  retrieval · citation · calibration · cost · strata  │
│  agreement · accuracy · granularity · abstention     │
└──────────────────────────────────────────────────────┘
```

All run artifacts are written as `.manifest.json` / `.records.jsonl` / `.costs.jsonl` triples
under `docs/harvest/`. Every scoring function is a pure function over those files; re-chunking,
threshold changes, or a different k are re-scores, not re-runs.

---

## Repository layout

```
.
├── pyproject.toml          # uv environment; Python >=3.12,<3.14
├── .env.example            # copy to .env and fill in API keys
├── src/biomedqa/           # the package (see Modules below)
├── tests/                  # 31 test_*.py modules
├── scripts/                # one-off run scripts (g0_, g1_, g3_, parity_, table1_, ...)
├── docs/
│   ├── adr/                # 21 architecture decision records
│   └── harvest/            # committed run manifests, records, and result JSON
├── data/
│   ├── splits.json         # dev (100 q) / test (400 q) split, seed 20260807
│   ├── gold_pmids.json     # 1,000 PubMedQA gold PMIDs
│   └── corpus/             # corpus draw manifest (2M-abstract JSONL sample)
├── paper/skeleton.md       # paper outline and table captions
├── annotator_guidance/     # annotation interface spec and annotator guide
└── research_roadmap.md     # week-by-week plan and gate ledger
```

---

## Modules

### `src/biomedqa/`

| Module | Responsibility |
|---|---|
| `schema.py` | Frozen output types: `QueryRecord`, `Claim`, `Citation`, `VerifierScore`, `HumanLabel`, `CostRecord`. Enforces the least-processed-value rule — continuous scores and 4-way labels are never collapsed on write. |
| `config.py` | Every tunable knob as frozen dataclasses, hashed into run manifests. `RunConfig.canonical_hash()` is the trace key for every result. |
| `data.py` | Loads PubMedQA `pqa_labeled`, the dev/test split, and instances. Stringifies `pubid` consistently to avoid silent str/int join failures against MedRAG rows. |
| `corpus.py` | Draws the ~2M-abstract distractor corpus from `MedRAG/pubmed` JSONL shards. Guards against the partial parquet export (2.2M rows) and str/int PMID mismatches — both failure modes return plausible-looking numbers without raising. |
| `chunk.py` | Passage granularity strategies: `abstract`, `section`, `sentence_window`, `fixed_width`. Every chunk carries char offsets into the source text; `text[c.char_start:c.char_end] == c.text` is asserted for all strategies. |
| `retrieve.py` | BM25 (bm25s) + MedCPT asymmetric dense + RRF fusion + MedCPT cross-encoder rerank. Each stage is ablatable via `RetrievalConfig` because Table 1 is those ablations. |
| `backends.py` | HTTP transport to the vLLM OpenAI-compatible server (local 8B AWQ) and the Anthropic API (Opus 5 judge). vLLM is not imported — see design decisions below. |
| `generate.py` | Stage orchestration: joint (one call), post-hoc (two calls — answer then cite, first call withholds citation rules), vanilla. Cost accounting per stage. Runs `QueryRecord.validate()` every query and returns violations rather than silently repairing them. |
| `decompose.py` | Re-cuts a generated answer into decontextualized atomic units for C7 ablation rows. Not in the headline path — headline systems emit `CLAIM` lines directly. |
| `verify.py` | MiniCheck-Flan-T5-Large and the Claude Opus 5 judge, behind one `Verifier.score_pairs` API. Scores are continuous `[0,1]`; thresholds live in `scoring/`. |
| `harness.py` | Run identity: writes `.manifest.json` before the run starts and `.records.jsonl` / `.costs.jsonl` during it. `verify_run()` reports provenance gaps; never repairs them. |
| `prompts.py` | All system prompts, response format specs, and the parse logic. Includes parity-loop state (`PARITY_LOOP_CLOSED`) and granularity constants. |
| `annotate.py` | Constructs annotation task files — one seeded question order shared by all three annotators, blinded (no system identity in the annotator file). Emits self-contained HTML forms. |

### `src/biomedqa/scoring/`

| Module | Feeds | What it computes |
|---|---|---|
| `retrieval.py` | Table 1 | hit@k, recall@k, MRR, nDCG, Wilson intervals |
| `citation.py` | Table 2 + G2 | ALCE citation precision / recall / F1 (φ = MiniCheck at threshold 0.5) |
| `calibration.py` | Table 3 + G3 | AUROC, ECE, threshold sweep, question-clustered bootstrap CI |
| `cost.py` | Table 4 | input/output tokens, USD per query, wall-clock |
| `strata.py` | Table 5 | Negation / numerics / scope error analysis (not yet implemented — due W10) |
| `agreement.py` | G4 | Krippendorff's α on binary collapse and 4-way; question-clustered |
| `accuracy.py` | Secondary (C6) | PubMedQA yes/no/maybe accuracy and confusion matrix |
| `granularity.py` | Parity diagnostic | Joint vs post-hoc median words/claim, ±15% tolerance — blind (reads `Claim.text` only) |
| `abstention.py` | Recall denominator | Detects abstention claims; excluded from recall denominator per ADR-0010 |
| `verifier_scores.py` | Tables 2–3 | Attaches `VerifierScore` records to `QueryRecord`s from a scores file |

---

## Evaluation design

### Three systems compared (C2)

All three use the same retriever, generator, and citation cap (≤ 3 citations per claim).

- **Joint** — generates claim text and citations in one call; the system being evaluated.
- **Post-hoc** — generates a plain answer, then attaches citations in a second call.
- **Vanilla** — retrieves and generates; no attribution. Citation recall is 0 by construction.

### Gate structure

| Gate | Condition | Status |
|---|---|---|
| G0 | Generator bake-off: choose the local model | Passed (Llama 3.1 8B AWQ selected) |
| G1 | Retrieval hit@10 ≥ 0.90, Wilson lower > 0.85 | Passed (hit@10 = 0.94 on 100-q dev) |
| G2 | Joint citation F1 > post-hoc, 95% CI excludes zero | Ongoing — human annotation pending |
| G3 | MiniCheck AUROC ≥ 0.75 against gold labels | Pending human labels |
| G4 | Inter-annotator α ≥ 0.6 (binary collapse, Krippendorff) | Pending annotation |

Gates are not adjusted after measurement. The escalation ladder for a failing G1 ends at relaxing
to hit@10 and reporting it as conditional attribution — it does not include moving the threshold
(ADR-0015).

G1 was originally set at k = 5. The full cascade reached hit@5 = 0.86 (Wilson lower 0.7786),
failing the gate. The gate was relaxed to k = 10 (ADR-0015), where hit@10 = 0.94 passes.

### What is measured

**Attribution quality (C2)** — ALCE citation F1, joint vs post-hoc, question-clustered bootstrap
95% CI on the paired gap.

**Faithfulness (C3, C4)** — MiniCheck-Flan-T5-Large scores each (cited passage, claim) pair.
Threshold sweep, AUROC and ECE against human majority labels.

**Cost (C5)** — tokens and USD per query. MiniCheck is local compute ($ = 0 by construction);
Opus 5 judge cost is reported in tokens/$ only, since wall-clock includes round-trip latency.

**Accuracy (C6, secondary)** — PubMedQA yes/no/maybe. Reported second; the paper's claim is
attribution quality, not SoTA accuracy.

**Granularity ablation (C7)** — `decompose.py` re-cuts headline answers into `sentence` and bare
`atomic` units; citation F1 at each granularity quantifies what decontextualization costs.

**Parity diagnostic** — median words/claim, joint vs post-hoc, tolerance ±15%. Blind: reads
`Claim.text` only, no citation or annotation data. Gate must pass before G2 unblinding.

**Error analysis (Table 5)** — stratified by negation, numerics, and scope — the three strata
where `CONTRADICTED` (rather than `NOT_SUPPORTED`) is expected to concentrate.

### Attribution unit

The attribution unit is the **decontextualized atomic claim** (ADR-0005): one assertion per claim,
with every pronoun and implicit subject resolved. The verifier takes `(cited span, claim)` as a
standalone hypothesis; a dangling pronoun makes that hypothesis indeterminate.

Citations use ALCE multi-citation semantics with a hard cap of 3. The cap is identical across all
three systems; an unequal cap would make C2's gap an artifact of citation budget.

### Human annotation

Three non-expert annotators independently label the full gold set (~250 claims across ~62 questions,
~4 claims per question). Labels are on a 4-way scale: `SUPPORTED` / `PARTIAL` / `NOT_SUPPORTED` /
`CONTRADICTED`. G4 gates on the binary collapse `(SUPPORTED | PARTIAL) vs (NOT_SUPPORTED |
CONTRADICTED)`. The 4-way label is always stored; `CONTRADICTED` is the payload of the error
analysis and an annotator cannot be re-run (ADR-0006, ADR-0016).

---

## Committed results

The following numbers come from `docs/harvest/` and represent the current state of the dev split
(100 questions). Human annotation is not yet complete; G2, G3, and G4 remain open.

**Retrieval — Table 1 (100-question dev split, `pubmed-2m-v1` corpus)**

| Row | Configuration | hit@5 | Wilson lower | G1 (k=10) |
|---|---|---|---|---|
| 1 | BM25 only | 0.71 | 0.615 | — |
| 2 | Dense (MedCPT) only | 0.59 | 0.492 | — |
| 3 | BM25 + Dense + RRF | 0.73 | 0.636 | — |
| 4 | BM25 + Dense + RRF + Rerank | **0.86** | 0.779 | **passes (hit@10 = 0.94)** |

Source: `docs/harvest/table1_rows_1_4.json`, `docs/harvest/chunker_pool_eval.json`

**Citation F1 — MiniCheck φ, threshold 0.5 (dev split, `generate_fp05_n100_guided_v4`)**

| System | Precision | Recall | F1 |
|---|---|---|---|
| Joint | 0.956 | 0.510 | 0.665 |
| Post-hoc | 0.954 | 0.362 | 0.525 |
| Delta (joint − post-hoc) | — | — | **+0.140** [0.075, 0.207] 95% CI |

Source: `docs/harvest/generate_fp05_n100_guided_v4.citation_f1.minicheck.json` (99 paired queries).
The gap excludes zero at 95% confidence (question-clustered bootstrap, 10,000 iterations). This is a
dev-split result measured before human annotation; G2's gate runs on the test split with human labels.

**Parity (granularity fairness diagnostic, parity_iter1b)**

| Basis | Joint median words/claim | Post-hoc median | Gap | Gate (±15%) |
|---|---|---|---|---|
| All 100 records | 15 | 17 | +13.3% | PASS |
| Untruncated, same 78 queries | 15 | 16 | +6.7% | PASS |

Source: `docs/harvest/parity_iter1b.md`. Parity loop closed 2026-08-14; granularity unit is frozen.

---

## Key design decisions

### vLLM is a network boundary, not a dependency ([`pyproject.toml`](pyproject.toml), [`docs/adr/0004-local-generator-frontier-judge.md`](docs/adr/0004-local-generator-frontier-judge.md))

vLLM pins torch to an exact version, conflicting with `torch>=2.13.0` and backtracking the
resolver to vllm 0.2.5 with pydantic 1.10.x. Instead, vLLM runs on the A4000 in its own
environment as an OpenAI-compatible server; `backends.py` talks to it over HTTP. The generator
backend is a network boundary, not an import.

### The attribution unit is the decontextualized atomic claim ([`docs/adr/0005-attribution-unit-decontextualized-atomic-claim.md`](docs/adr/0005-attribution-unit-decontextualized-atomic-claim.md))

Sentence-level attribution is too coarse: a sentence with one true half and one false half has no
correct label. Bare atomic claims with unresolved pronouns are unverifiable standalone — the
verifier receives `(cited span, claim)` and cannot evaluate a non-self-contained hypothesis.
Decontextualized atomic is also the one unit non-expert annotators can judge without outside
domain knowledge.

### Abstention is derived at scoring time, not stored ([`docs/adr/0010-abstention-is-derived-at-scoring-time-not-stored.md`](docs/adr/0010-abstention-is-derived-at-scoring-time-not-stored.md))

A `Claim.is_abstention` boolean would binarize a value derived by a rule and freeze it into the
record — exactly what the least-processed-value rule forbids. More concretely, G0 showed abstention
arriving in two shapes: Llama emitted it as a claim in the list (capturable by `Claim.text`), Qwen
emitted it as prose outside the list (where no `Claim` field could hold it, but
`QueryRecord.raw_generation` does). Storing the abstention detection rule version in `RunConfig`
means revising the rule is a re-score, not a re-run.

### The corpus is 2M PubMed abstracts, read from JSONL shards ([`docs/adr/0003-retrieval-corpus-2m-pubmed-abstracts.md`](docs/adr/0003-retrieval-corpus-2m-pubmed-abstracts.md), [`docs/adr/0014-corpus-text-is-jsonl-shards-indexed-as-title-free-prose.md`](docs/adr/0014-corpus-text-is-jsonl-shards-indexed-as-title-free-prose.md))

A 1,000-abstract corpus (the source questions' own contexts) makes retrieval trivial and citation
precision/recall both ceiling: there is nothing plausible to mis-cite. Citation precision only
discriminates when plausible-but-wrong passages exist. The HuggingFace parquet export of
`MedRAG/pubmed` is a partial export (2.2M of 23.9M rows — the oldest ~9% of PubMed). `corpus.py`
reads `data_files="chunk/*.jsonl"` and raises unless the full scan sees exactly 23,898,701 rows.

### The corpus is indexed as title-free abstract prose ([`docs/adr/0014-corpus-text-is-jsonl-shards-indexed-as-title-free-prose.md`](docs/adr/0014-corpus-text-is-jsonl-shards-indexed-as-title-free-prose.md))

PubMedQA has no title field. If gold passages are indexed with titles but distractors are not, gold
becomes separable by format alone. More specifically: a PubMedQA question is its article's title
verbatim — a titled gold passage would turn G1 into a string match.

### Gold-set sampling clusters on questions, not claims ([`docs/adr/0011-gold-set-sampling-clustered-bootstrap-g4-gates-on-the-point.md`](docs/adr/0011-gold-set-sampling-clustered-bootstrap-g4-gates-on-the-point.md))

Claims from the same question share the question, retrieved passages, answer, and topic — they are
not independent. Bootstrapping on claims produces confidence intervals that are too narrow. Every
bootstrap in the paper resamples questions; `calibration.bootstrap_ci` implements this.

---

## Getting started

```bash
cp .env.example .env
# fill in ANTHROPIC_API_KEY, HF_TOKEN, and VLLM_BASE_URL if using a local server

uv sync              # install the environment
uv run pytest        # run the test suite (31 modules; no GPU required)
```

The retrieval tests and generation tests require models from HuggingFace (`NCBI/MedCPT-*`,
`lytang/MiniCheck-Flan-T5-Large`). Set `HF_TOKEN` and `HF_HOME` in `.env` first.

The generator backend (`backends.py`) expects a vLLM server at `VLLM_BASE_URL`
(default `http://localhost:8000`). An Anthropic key is required for `JudgeVerifier` and for
post-hoc generation using the Opus 5 judge.

Building the corpus index is a multi-hour operation on the A4000 (RTX A4000, 16 GB VRAM):
`encode_corpus.py` benchmarks MedCPT throughput first; `scripts/g0_medcpt_throughput.py` is
the reference for that measurement.

---

## Project status

| Component | Status |
|---|---|
| Schema, config, data loading | Complete |
| Retrieval cascade (Table 1) | Complete; G1 passed at k=10 |
| Generation — joint, post-hoc, vanilla | Complete |
| Claim decomposition (C7 ablation) | Complete |
| Faithfulness verifier — MiniCheck | Complete |
| Faithfulness verifier — Opus 5 judge | Complete |
| Evaluation harness (run manifests) | Complete |
| Scoring — retrieval, citation, calibration, cost, agreement, accuracy, granularity, abstention | Complete |
| Scoring — strata (Table 5) | Not yet implemented (W10) |
| Human annotation | In progress |
| G2 gate (citation F1 on test split with human labels) | Pending annotation |
| G3 gate (AUROC with human labels) | Pending annotation |
| G4 gate (inter-annotator agreement) | Pending annotation |

Dev-split citation F1 (MiniCheck φ): joint 0.665, post-hoc 0.525, gap +0.140 [0.075, 0.207].
This is a pre-annotation result on 100 questions; the paper's claim rests on the test split with
human labels.
