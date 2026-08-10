# Evidence-Grounded, Claim-Attributable Biomedical QA

**Venue:** workshop, hard November deadline (ADR-0001) · ~8 pages · **5-table budget**
**Status:** skeleton created 2026-07-31 (W0). Table 1 is populated from a real run (W3); every
other table is still empty, and no number may be invented here.

> **How this document is used.** The paper is written **backwards from its tables**. Every table
> below is present now, with real column headers and a real caption, before any number exists. The
> caption of each names the `scoring/` function that will populate it — so if that function stops
> existing, the caption is a claim the repo cannot honour and it fails loudly. An empty Table 4
> whose columns say *input tokens / output tokens / $ per query / wall-clock* is what forces
> `backends.py` to instrument for those columns in W2, instead of the gap being discovered in
> October.
>
> Prose sections are written W10–W13, in this order: Method + Setup, then Results + Analysis, then
> Related Work / Limitations / Ethics, and **Intro + Abstract last**.

---

## Claim ledger (cut to five — ADR-0001)

Every claim maps to exactly one table. A claim with no table is not in this paper.

| | Claim | Evidence | Table |
|---|---|---|---|
| **C1** | Retrieval is adequate, so attribution is meaningful | Hybrid BM25+MedCPT+RRF+rerank vs. each ablated stage | Table 1 |
| **C2** | **Joint claim-grounded generation attributes better than post-hoc citation** *(the headline — ADR-0002)* | Citation P/R/F1, matched retriever / generator / cap / prompt-iteration budget | Table 2 |
| **C3** | A small local verifier detects unsupported claims usefully well | AUROC, ECE against the human gold set | Table 3 |
| **C4** | The verifier agrees with humans about as well as humans agree with each other | Verifier vs. gold; Krippendorff's α on the overlap subset | Table 3 |
| **C5** | It does so at a small fraction of a frontier judge's cost | Tokens, USD, wall-clock, overhead ratio | Table 4 |

**Cut, and stated as cut:** C6 (accuracy) is reported as a secondary guard, not a claim. C7
(granularity) survives as ablation rows inside Table 2. **C8 (BioASQ generalization) is cut
entirely** and deferred to the 2027 journal extension — do not reintroduce it in September.

---

## 1. Introduction

*Written last (W13).* One paragraph per: the attribution problem in biomedical QA; why post-hoc
citation is the wrong default; what joint claim-grounded generation is; the cheap-verifier result;
contributions as a bulleted list that maps 1:1 onto C1–C5.

## 2. Related work

*W12.* Position against, explicitly and by name: ALCE-style attribution evaluation, MiniCheck and
lightweight fact-verification, LatentAudit-style faithfulness monitoring, graph-guided medical RAG,
and current SoTA PubMedQA systems. **Differentiate on joint claim-grounded generation + a cheap
verifier — not on beating accuracy.** The lane is crowded; the framing is the defence.

## 3. Method

*W10.* The pipeline, then the three definitional commitments, each of which a reviewer will look
for:

- **The attribution unit is a decontextualized atomic claim**, not ALCE's sentence-level statement.
  See *Divergences from ALCE* — declare it, do not let it be discovered.
- **Citations are character spans**, at most three per claim, with ALCE's multi-citation semantics
  reused verbatim. The cap is a fairness control and is identical across all three systems.
- **φ vs. the verifier**: φ is the entailment primitive; the verifier is the component built on it.

## 4. Experimental setup

*W10.* Corpus (~2M PubMed abstracts + all 1,000 gold contexts — and **why not 1,000 alone**:
citation precision cannot discriminate without plausible-but-wrong passages to cite). Frozen
dev/test splits by pubid with a hash. Seeds (≥3, local backend). The matched prompt-iteration
budget, reported as a number. Hardware: one RTX A4000.

## 5. Results

*W11.* Tables 1–4, in order, each with confidence intervals. Prose says what the table shows and
nothing the table does not.

### Table 1 — Retrieval cascade → **C1**

*Caption:* Gold-passage retrieval on the 100-question dev split, `chunker = abstract`, over the ~2M
abstract corpus (`index_fingerprint 57ab89e445f8`, `config_version 1.3.0`). Wilson intervals, not
Wald. Relevance is the **gold chunk set** — one abstract cuts into 2–7 chunks — so hit@k asks
whether any gold chunk surfaced and recall@5 asks what share of them did; MRR is over the first
gold chunk. **Populated by `scoring/retrieval.py::gate_g1`, `hit_at_k`, `recall_at_k`, `mrr`,
`ndcg`, via `scripts/table1_report.py` → `docs/harvest/table1_metrics.json`.**

| System | hit@5 | 95% CI (Wilson) | hit@10 | 95% CI (Wilson) | recall@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| BM25 only | 0.71 | [0.61, 0.79] | 0.77 | [0.68, 0.84] | 0.22 | 0.62 | 0.29 |
| MedCPT dense only | 0.59 | [0.49, 0.68] | 0.70 | [0.60, 0.78] | 0.19 | 0.45 | 0.23 |
| BM25 + dense (RRF) | 0.73 | [0.64, 0.81] | 0.81 | [0.72, 0.87] | 0.24 | 0.61 | 0.30 |
| **+ cross-encoder rerank (full)** | **0.86** | [0.78, 0.91] | **0.94** | [0.88, 0.97] | 0.28 | 0.79 | 0.38 |

> **G1, as gated: hit@10 ≥ 0.90 with the Wilson lower bound above 0.85 — 0.9400, lower 0.8752,
> passes.** This is a **relaxation of k, made once and on the record** (ADR-0015). The gate was
> written at k=5 and the full cascade reads **hit@5 = 0.86, Wilson lower 0.7786, which fails both
> clauses**; that reading is printed above and is not withdrawn. The thresholds, 0.90 and 0.85, are
> unchanged. τ was not tuned, and the chunker sweep that preceded the relaxation refused all seven
> of its candidate index builds on evidence (`docs/harvest/chunker_arm_eligibility.json`) — the one
> arm that cleared 0.90 cuts gold differently from distractors and measures its own privilege.
> Attribution is therefore reported as **conditional on a 10-passage context**, which is the
> context every system in Table 2 receives.

### Table 2 — Attribution quality → **C2** *(the headline)*

*Caption:* Citation precision, recall, and F1 on the dev split. All systems share a retriever, a
generator, a ≤3-citation cap, and a matched prompt-iteration budget (reported). **Populated by
`scoring/citation.py::citation_f1`.**

| System | Citation P | Citation R | **Citation F1** | 95% CI | Claims / answer | Decomposition-error rate |
|---|---|---|---|---|---|---|
| Vanilla RAG (no attribution) | — | — | — | — | | |
| Post-hoc citation | | | | | | |
| **Joint claim-grounded (ours)** | | | | | | |
| *ablation:* sentence granularity (C7) | | | | | | |
| *ablation:* bare atomic granularity (C7) | | | | | | |

> **G2:** joint beats post-hoc on citation-F1 by a margin whose CI excludes zero. This is the
> paper's central contrast — if the margin is absent, investigate before proceeding.

### Table 3 — Verifier quality → **C3, C4**

*Caption:* Unsupported-claim detection against the human gold set (*n* claims, *k* triple-labeled).
Thresholds swept on dev, fixed once. **Populated by `scoring/calibration.py::auroc`, `ece`,
`threshold_sweep`; α by `scoring/agreement.py::krippendorff_alpha_binary`.**

| Verifier | AUROC | 95% CI | ECE | P / R / F1 at the chosen threshold | α vs. human gold |
|---|---|---|---|---|---|
| MiniCheck-Flan-T5-Large | | | | | |
| + MedNLI fine-tune *(if credentialing lands)* | | | | | |
| AlignScore | | | | | |
| Opus 5 judge *(reference ceiling)* | | | | | |
| *human–human agreement* | — | — | — | — | |

> **G3:** AUROC ≥ 0.75. **G4:** ≥250 claims labeled, α ≥ 0.6 on the binary collapse of the overlap
> subset. Degradation on biomedical text is **expected** (MiniCheck is ANLI-trained) — report it,
> then mitigate.

### Table 4 — Cost and overhead → **C5**

*Caption:* Per-query cost of verification, local verifier vs. routing to a frontier judge. Clean GPU,
idle, ≥5 runs, range reported alongside the median. **Populated by `scoring/cost.py::per_query_cost`
and `overhead_ratio`.**

| Verifier | Input tokens | Output tokens | $ / query | Wall-clock (s) | Overhead vs. generation |
|---|---|---|---|---|---|
| MiniCheck (local, A4000) | | | | | |
| Opus 5 judge (API) | | | | | |
| **ratio** | | | | | |

> If the ratio lands under 10×, this costs the subtitle, not the paper — the headline is already
> attribution quality (ADR-0002).

## 6. Analysis

*W11.* **Table 5** plus the secondary accuracy guard.

### Table 5 — Where attribution fails → biomedical failure modes

*Caption:* Error rates by stratum, over gold-set claims. `CONTRADICTED` is reported separately from
`NOT_SUPPORTED` — collapsing them would have destroyed this table at annotation time. **Populated by
`scoring/strata.py::error_rates_by_stratum`.**

| Stratum | *n* claims | Citation F1 | Verifier AUROC | % CONTRADICTED | % PARTIAL |
|---|---|---|---|---|---|
| Negation | | | | | |
| Numerics / dosage | | | | | |
| Scope / population | | | | | |
| Remainder | | | | | |

**Secondary (C6, not a claim):** PubMedQA yes/no/maybe accuracy with Wilson intervals, reported to
show attribution does not cost correctness. **`scoring/accuracy.py::accuracy`.** Inline prose, no
table — the budget is five.

## 7. Limitations

*W12.* Written honestly and early, because each of these is a reviewer's first question:
single-corpus (PubMedQA only, C8 cut); non-expert annotators (ADR-0006) with α reported; a single
8B generator; reduced *n* wherever a gate forced a cut; MiniCheck's domain shift.

## 8. Ethics and broader impact

*W12.* Non-expert annotation of biomedical text and what it does and does not license. No clinical
deployment claim. Annotator compensation and consent. Licence terms for MedNLI/PhysioNet data if
used.

## 9. Reproducibility

*W12.* Public repo, tagged at code freeze (W8). Run manifests exported: every table cell traces to a
config hash, an index fingerprint, and a split hash. **G5: no number appears here that a manifest
cannot produce.**

---

## Appendix — Divergences from ALCE

Reproduce the table from `CONTEXT.md` verbatim. Reviewers who know ALCE will look for exactly this
section, and finding it stated is very different from finding it omitted.
