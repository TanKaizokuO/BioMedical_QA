# HANDOFF — 2026-08-04 (end of day)

Snapshot for resuming in a fresh session. Regenerate with `/handoff`; do not append.

`main`, working tree clean. **4 commits unpushed** — `origin/main` is at `dbd9ed4`.

---

## 1. Where the project is

**Gate G0 PASSED (2026-08-04).** Both deliverables measured. Numbers and reasoning live in the
commits and the roadmap; not repeated here.

| | |
|---|---|
| A4000 is a **Windows/WDDM box**; vLLM runs under WSL2 | `docs/adr/0008-*.md` · `docs/harvest/runbooks/wsl-vllm-a4000.md` |
| Generator: **`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`**, 3.42 s median | `d11e847` · `research_roadmap.md` §2 D1 |
| MedCPT 343.6 abstracts/s → 2M in 1.6 h; R1's encode leg discharged | `6561f60` · §3 |
| MedNLI/PhysioNet **deliberately dropped**; G3's ladder ends at three rungs | `315f77b` · R7 · issue #8 (closed, not planned) |
| Annotator ask **sent**; awaiting two confirmations | issue #7 (open) |
| Scorer bug found by the bake-off | issue #9 |

**W2 (Aug 10–16) is next**: chunker sweep, `bm25s` + MedCPT + RRF, the 2M encode. **G1 stays
Aug 23.**

---

## 2. Sixteen decisions awaiting confirmation — the critical section

From a `/grilling` session on 2026-08-04. **These exist in no ADR, roadmap section, schema, or
issue.** This document is their only record. The user had not confirmed them when the session
ended — **do not implement before confirmation.**

### A. Granularity parity — a NEW fifth condition on ADR-0002's equal-effort protocol

*Motivation:* G0 measured **9.2 claims/query (Llama) vs 3.8 (Qwen)** — same prompt, same passages.
Joint emits claims natively; post-hoc goes through `decompose.py`. Two different mechanisms produce
the unit Table 2's citation-F1 is computed over. Coarser post-hoc claims are harder to entail, so
post-hoc is systematically penalised and **C2's gap appears without joint grounding doing any
work** — a bias pointing *toward* the hypothesis.

1. Add **granularity parity** as a fifth condition to ADR-0002's equal-effort protocol.
2. Enforce on **median words/claim** — the quantity driving per-claim entailment difficulty.
   **claims/query is reported, not gated** (it tracks answer length, not granularity). Compound
   claims are caught at annotation by `claim_validity`, which ADR-0005 already built.
3. Tolerance **±15%**, dev only, frozen **before the first W4 run (Aug 24)**.
4. Parity tuning is a **third disclosed line** in the prompt-iteration ledger, charged to neither
   system — a fairness-control cost, not method development.
5. **Bounded at ~10 iterations.** One-sided fallback: residual gap favouring C2 (post-hoc coarser)
   → the stratified robustness check becomes mandatory; running against C2 → note and proceed.
   Never tune until it passes.
6. The parity loop is **blind** — citation-F1 is not computed until parity is frozen.
7. **Granularity / decomposer prompt freezes Sep 3**, a named dated artifact, three days before G2.
   Protects the gold set launching Sep 7 (ADR-0005: changing granularity after W6 orphans it).
8. Guidelines in **two passes** — unit-independent rules (no-outside-knowledge, SUPPORTED/PARTIAL
   boundary, hedging, numerics, jointly-necessary citations) from Aug 31; **worked examples only**
   Sep 3–6, from frozen decomposer output.

### B. Abstention — issue #9 is larger than the scorer patch it describes

*Evidence:* both models abstained on the same question (`pubid 10781708`), and both did so
**partially** — ten good cited claims plus one claim flagging the gap.

9. Primary citation-F1 on the **common answered set** (all three systems answered), consistent with
   G2's existing paired bootstrap. `abstention_rate` becomes a Table 2 column; the full-set number
   is a robustness check. Full abstention (zero substantive claims) will be rare, so this is cheap
   insurance rather than a frequent exclusion.
10. Add **`Claim.is_abstention`**; bump **`SCHEMA_VERSION` 1.0.0 → 1.1.0 now** — free today, costs
    the gold set after Sep 7. Detection rule versioned in `RunConfig`. Claim-level, not
    record-level, because partial abstention is the real behaviour. Do **not** overload
    `claim_validity`; that would corrupt ADR-0005's decomposition-error rate.

### C. Gold set and statistics

11. Gold set sampled as **~4 claims/question × ~62 questions**, not all claims of ~27. ADR-0006's
    250-claim budget and ~3 h ask are unchanged; question diversity rises ~2.3×. The overlap
    subset is sampled the same way → ~75 claims across ~19 questions instead of ~8.
12. **Every bootstrap cluster-resamples QUESTIONS, not claims** — a standing rule in §8, matching
    what G2 already does. Claims within a question share passages, answer and topic; resampling
    claims treats correlated units as independent and returns CIs that are too narrow.
13. **G4 gates on point α ≥ 0.6.** The clustered CI is always reported and never used to soften the
    number. Pre-committed: CI lower bound < 0.4 → revise guidelines and re-pilot rather than
    proceed.

### D. Corpus — ADR-0003 fixed the SIZE (2M) but never the source or selection policy

*Unnamed tension:* G1 (hit@5 ≥ 0.90) wants easy distractors; G2 (citation precision must
discriminate) wants hard ones. Same knob, opposite directions.

14. **Uniform random 2M from `MedRAG/pubmed`**, seeded and reproducible. Uniform is unarguable to a
    reviewer; hand-picked hard negatives look like a corpus engineered around the gold set.
15. **W2 hardness diagnostic**, threshold pre-committed now: for each of 100 dev questions take the
    **RRF-fused top-5** (no reranker until W3 — re-confirm after), drop the gold, and ask a
    **non-Opus** judge (Haiku/Sonnet, ~100 calls) whether ≥1 remaining passage is on the same
    clinical topic. Require **≥ 70%**. Below → escalate to injected hard negatives *in W2*, four
    weeks before G2. Opus 5 is avoided to keep the C5 judge uncontaminated.

### E. G3 insurance — **already implemented**

16. Ladder reordered to calibration/threshold → AlignScore (already a W6 deliverable) →
    cheap-signal ensemble. Done in `315f77b`; PhysioNet dropped, so it ends at three rungs.

### Flagged, not resolved
Decisions 1–8 add real work to W4–W5, a stretch already carrying joint generation, both baselines,
decontextualization, the granularity knob, the scorers, cost instrumentation and G2. **W9 is the
only slack in the schedule.**

### Proposed write-up, once confirmed
Three ADRs (granularity parity 1–8; gold sampling + clustered bootstrap + G4 gate 11–13; distractor
pool + diagnostic 14–15) · `schema.py` at 1.1.0 with tests · `research_roadmap.md` §4 Phase 2, §5
(add the Sep 3 freeze), §8, §9 · `CONTEXT.md` (annotators read it) · update issue #9, which
currently describes only the scorer patch. The user may want this scoped differently — ask.

---

## 3. Housekeeping — small, blocking nothing

- **Push** the four commits.
- `runs/g0/*.json` are untracked (`.gitignore` excludes `runs/`) but are the evidence for a gate
  decision that reaches the paper → copy to `docs/harvest/g0/`. `g0_medcpt_throughput.json` is
  still only on the box and needs scp back.
- **Close issue #1** (G0 complete).
- `scripts/g0_smoke.sh` **has never run successfully** and is now wrong — it assumes a POSIX login
  shell; the box answers with `cmd.exe`. Recommend deleting; its job is done and evidenced by the
  runbook. User has not decided.

---

## 4. What to read

1. `CONTEXT.md` — project language and the frozen annotation protocol
2. `research_roadmap.md` §0, §2, §5
3. `docs/adr/0003`, `0004`, `0005`, `0007`, `0008`
4. `docs/harvest/runbooks/wsl-vllm-a4000.md` — before touching the box
5. `src/biomedqa/schema.py` — the frozen contract; read before writing anything that emits data
6. `paper/skeleton.md` — the five tables and the C1–C5 ledger every result must land in

Not needed: `docs/project2_biomedical_attribution_rag_implementation_plan.md` (§4, §8, §9 superseded
and banner-marked) and `notebooks/` (toy/simulated; `07_4` simulates 3 labels where `CONTEXT.md`
freezes 4 — a correctness bug, not a scale assumption).

---

## 5. Standing constraints

- **Least-processed value.** Store `phi_score: 0.83`, never `supported: true`. Store `gold_rank` or
  the ranked list, never a precomputed hit@5. Store the 4-way `support_label`, never its collapse.
- **Wilson, not Wald**, on gate proportions. G1 passes iff point ≥ 0.90 **and** Wilson lower > 0.85.
- **vLLM never enters `pyproject.toml`**, not even an optional group — it pins torch exactly and
  backtracks the workspace to pydantic 1.10.x. It is a network boundary and now a separate OS.
- **`RAG_Debate_Agent` is retired.** Never re-run it; cite `docs/harvest/`.
- **Index identity is a content hash**, never a document count (the ADR-0007 lesson).
- **≤3 citations per claim**, identical across all three systems.
- **`validate()` reports violations and never repairs them.**

---

## 6. Working mode — violating these wastes a turn

- **The A4000 is copy-paste only.** No SSH from the agent environment; `scp` to `vllm-box` fails
  with `Permission denied (publickey…)`. Hand the user commands and wait for pasted output. Prefer
  designs that keep work on the box, or the user-opened tunnel
  (`ssh -L 8000:localhost:8000 vllm-box`) for HTTP-only work.
- **Never inspect `~/.ssh/`.** Declined once.
- `docs/` is gitignored via `docs/*` with `!docs/adr/` and `!docs/harvest/`. Docs elsewhere are
  silently untracked — verify with `git check-ignore`.
- **VRAM drifts** on the A4000 (WDDM, display attached). Always launch vLLM with
  `--gpu-memory-utilization 0.85` and `VLLM_USE_V2_MODEL_RUNNER=0`.
- **Do not install an NVIDIA driver inside WSL** — the Windows driver is passed through.
- Repo is **private**; W12 calls for a public release.
- Dates have drifted in conversation before. Today is **2026-08-04**; November is hard.

---

## 7. Suggested skills

- **`/handoff`** — regenerate this file; it goes stale fast.
- **`/grilling`** — for the W4–W5 load problem (§2, "Flagged, not resolved") or to push C2's design
  further. Today's pattern: look facts up from the repo and the run JSONs, put only *decisions* to
  the user, one at a time, each with a recommendation.
- **`/domain-modeling`** — the natural fit for the three ADRs and the `CONTEXT.md` edits.
- **`/tdd`** — for `schema.py` 1.1.0. `tests/test_schema_roundtrip.py` already covers lossless
  round-trip, unknown-field rejection and continuous verifier scores; `is_abstention` needs the same
  plus a test that an abstaining answer does not score as non-compliant (named in issue #9).
- **`/code-review`** — before the W2 encode commits, once `retrieve.py` and `chunk.py` are real.

Not needed yet: `dataviz` (figures are W11), `claude-api` (the Opus 5 judge is wired in W6).

---

## 8. Immediate next actions

1. Get the user's **confirm / push-back on the sixteen decisions** (§2). Everything else is small.
2. On confirmation, write them up.
3. Housekeeping (§3).

**User-side, open:** two annotators must confirm — issue #7, checkpoint **~Aug 20**, hard trigger
**Sep 7**. No strong fallback; ADR-0006's replacement is explicitly weaker.
