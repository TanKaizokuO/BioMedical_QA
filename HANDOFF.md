# HANDOFF — 2026-08-14 (end of twelfth session)

Snapshot for resuming in a fresh session. Regenerate wholesale; **do not append** — a stale line here
is worse than a missing one, because the next session will trust it.

`main` · **working tree clean, `HEAD` == `origin/main`.** Everything this session is committed and
pushed.

Tests: `uv run --with pytest python -m pytest tests/ -q` → **306 passed**. `pyproject.toml`'s
`pythonpath` is `["src", "scripts"]` — `tests/test_corpus.py` imports from `scripts/build_corpus.py`.

---

## 1. Where the project is

**W3 (Aug 17–23), Phase 1 — Retrieval gate (Aug 3–23) → C1, Table 1.**

| Gate | Date | State |
|---|---|---|
| **G0** — 8B AWQ generator chosen, A4000 measured | Aug 4 | **PASSED 2026-08-04.** Issue #1 closed. |
| **G1** — hit@10 ≥ 0.90, Wilson lower > 0.85 (ADR-0015) | **Aug 23** | **PASSED 2026-08-10.** Row 4 hit@5 = 0.86, **hit@10 = 0.9400** (Wilson lower 0.8752). |
| **G2** — citation-F1 / attribution quality | Sep 6 | Unblinded first read on `parity_iter1b`: Joint 0.264 vs Post-hoc 0.345 (delta -0.081). |
| G3 · G4 · G5 | Sep 20 · Sep 27 · Oct 11 | Unstarted, with due weeks. |

**W1, W2, and Phase 1 Retrieval (G0 & G1) are complete.** The 2M index exists on the box, Table 1
rows 1–4 are measured, ADR-0012 §2's probe has run with its control pre- and post-rerank, ADR-0014 §3 is decided, and ADR-0015 is accepted.

### Table 1 rows 1–4 — dev, 2.16M index, `docs/harvest/table1_rows_1_4.json`

| row | hit@1 | **hit@5** | **hit@10** | hit@100 | not in pool |
|---|---|---|---|---|---|
| BM25 | 0.55 | 0.71 | 0.77 | 0.90 | 10 |
| Dense (MedCPT) | 0.32 | 0.59 | 0.70 | 0.91 | 9 |
| RRF | — | 0.73 | 0.81 | **0.97** | 3 |
| **RRF + Rerank** | — | **0.86** | **0.94** | 0.97 | 3 |

**The cross-encoder reranker lifts hit@5 from 0.73 to 0.86 and hit@10 to 0.94.** Under ADR-0015, G1 re-gated at k=10 and passed with Wilson lower 0.8752 > 0.85.

**Dense underperforming BM25 (0.59 vs 0.71) is genuine and unexplained.** Two hypotheses are dead:
normalisation (`diag_dense_metric.py`, `df14e82` — CLS norms CV 0.0229, L2 vs raw dot identical to
1 ULP) and the title-segment convention (§3 below). It is retriever quality. Do not re-litigate the
two dead ones.

### The key W2/W3 measurements recorded

- **ADR-0012 §2's probe ran pre- and post-rerank with paired controls.** Reranked output recorded in `docs/harvest/confusability_probe_reranked{,_control}.json`.
- **ADR-0014 §3 is decided: `empty`**, and the index already on disk is the winner, so no re-encode.
  hit@5 0.59 vs 0.53; paired gold rank better on 19 / worse on 39 / unchanged 33, sign test p = 0.012.
- **ADR-0015 is accepted:** G1 re-gated at k=10 (hit@10 = 0.9400, Wilson lo = 0.8752). Chunker sweep arm `section` disqualified due to gold-only cut asymmetry; all eligible chunker arms upper-bounded ≤0.89 at k=5.

Everything past retrieval is `NotImplementedError` with a due week in its module docstring. That is
by design, not drift. The exceptions are `scoring/abstention.py`, `retrieve.py` and `backends.py`.

### ADR-0009 Granularity-Parity Loop — CLOSED, and the blind is lifted (2026-08-14)

**The loop terminated at 1 of 10 iterations on `parity_iter1b`, and the first citation-F1 has been
computed.** Read `docs/harvest/parity_iter1b.md` (the gate), ADR-0009's *Termination* section (the
decision), and `docs/harvest/first_citation_f1.md` (the unblinding read) — in that order.

- **The loop is closed in code, not just in prose:** `prompts.PARITY_LOOP_CLOSED` carries the run, the
  verdict, the interval and the SHA-256 of `POST_HOC_ANSWER_TEMPLATE` as it stood; `parity_loop_is_open()`
  is False; `scoring.citation.citation_f1` **raises** while the loop is open; `tests/test_prompts.py`
  fails if the frozen template is edited. `parity_budget_remains()` is still True and **that is not
  permission** — the freeze is what governs.
- **Iterations used: 1 of 10**, five days inside the Aug 30 drop-dead. `parity_iter1b` charged none: it
  re-measures the *same* prompt at a shared cap of 3584 (server `--max-model-len 14336`), which is run
  config, not a prompt edit (`parity_iter0.md`'s precedent).
- **`parity_iter1b` verdict, three bases:** all records **+13.3%** (joint 15 / post-hoc 17) ·
  untruncated per arm **+14.3%** (was +21.4% FAIL at 2560) · untruncated on the same 78 queries both
  arms **+6.7%**. All PASS. The baseline of record fails all three (+25.0% / +42.9% / +37.9%).
- **The basis disagreement is closed** — the higher cap took post-hoc cite truncation 26 → 16 of 100,
  which was the collider iteration 1 argued it was.
- **The gate's resolution is one word (~6.7%), and ±15% is two words wide.** `parity_iter1` and
  `parity_iter1b` ran the **same post-hoc prompt** and read **+0.0%** and **+13.3%** on the same
  basis. So verdicts are now reported with a query-level bootstrap (`gap_bootstrap_ci`): all-records
  95% interval **[+0.0%, +14.3%]** for `iter1b` against **[+18.8%, +40.0%]** for `parity_iter0b` —
  non-overlapping, so the movement is real; the residual is one grid step and is not resolvable.
- **Not answering less:** post-hoc holds 10 claims/query and parses **1242** claims against joint's
  **719**.
- **The joint control drifted 23/100 across the cap change** (same prompt, `temperature=0.0`) — the
  second instance of the vLLM non-determinism recorded at iteration 1 (vanilla, 19/100). Checked
  instead on the 77 byte-identical records, where joint's claim lists are *identical* (357 claims,
  median 15 in both runs): joint's apparent 16 → 15 median shift is composition, not content.
  **Byte-identity is a control check only across runs with matching server config.**
- **W9 stratified robustness check remains mandatory** — the residual favours C2 on every basis, and
  a pre-registered asymmetric check is not retracted because a later iteration passed.
- **Defect found and deliberately deferred to W5/W6 (out of bounds under §4):** joint query
  **21074975** yields a single 731-word "claim" from an `and …, and …` repetition loop whose length
  scales with the cap (164 words at 2560). 4.7% of joint claims exceed 30 words vs post-hoc's 3.1%;
  `_claim_rules()` splits on "and" and did not split this. **It now costs a measured number** — see
  the first-F1 section below.
- **Code:** `src/biomedqa/scoring/granularity.py` (gate, per-stage token verification,
  `gap_bootstrap_ci`/`GapInterval`) · `scripts/parity_report.py` (three bases + intervals) ·
  `scripts/first_citation_f1.py` (the unblinding read) · `tests/` at **306 passed**.

### The first citation-F1 — R5 early warning, and it runs against C2 (2026-08-14)

`docs/harvest/first_citation_f1.md` · artifact `parity_iter1b.citation_f1.json` · command:
`uv run python scripts/first_citation_f1.py docs/harvest/parity_iter1b --n-boot 2000 --max-tokens 3584`

| system | P | R | **F1** | 95% CI (question-clustered) |
|---|---|---|---|---|
| joint | 0.902 | 0.154 | **0.264** | [0.205, 0.331] |
| post_hoc | 0.866 | 0.215 | **0.345** | [0.286, 0.403] |

**joint − post_hoc = −0.081, 95% [−0.157, +0.005]**, paired on 100 questions. **C2's direction is not
established and the point estimate favours the baseline.** §6's pre-armed R5 trigger is live — and
because the loop closed early, the window to G2 is **23 days, not six**.

- **φ is interim and not MiniCheck:** `cross-encoder/nli-deberta-v3-xsmall`, `argmax == entailment`,
  4,904 pairs. `verify.py` still raises (W6). R7's predicted degradation is visible — recall 0.15–0.22
  against precision 0.87–0.90 — so **the levels are not interpretable and none of this is a G2
  number.** The contrast is what to read.
- **The contrast is not the granularity residual.** Post-hoc leads in **every** claim-length band
  (widest at 16–20 words: 0.189 vs 0.090), and joint entails **0 of 34** claims over 30 words. On the
  78 untruncated questions it is 0.303 vs 0.376 — the gap survives dropping joint's runaway records.
  Citations/claim are equal (1.48 vs 1.45) and post-hoc leaves *more* claims uncited (30.9% vs 26.7%),
  both of which run against post-hoc.
- **Leading alternative explanation, untested:** post-hoc quotes longer spans (median 23 words vs 19),
  and a longer premise entails more easily under a sentence-pair NLI model. This is exactly what
  MiniCheck is meant to fix, so it is the first thing to re-read at W6.
- **The joint runaway-claim defect is now load-bearing**, not cosmetic (0/34 entailed above 30 words).
  Fix the splitter and the non-terminating generation **before** the G2 read.

Unresolved, and **not needing re-derivation**: **W9 is triple-booked** (`research_roadmap.md` §5 ⚠)
· `SPLIT_SEED = 20260807` while the draw happened Aug 5 — **left alone deliberately**; the constant
documents the deadline it was written against, and changing it would change the draw for nothing.

---

## 2. What exists on the box, and the corpus as built

**Nothing is blocking.** The 2M index exists and every W2/W3 number has been taken from it.

### The index (built 2026-08-10 on the A4000)

| | |
|---|---|
| location | `data/index/empty` — **the `empty` convention, which ADR-0014 §3 has now confirmed is the right one** |
| passages | **2,162,838** encoded in **1.99 h** (the 1.6 h G0 estimate was close) |
| gold | **1,037 gold passages present** — sourced by `_iter_passages`, not from the corpus file (trap 12) |
| artifacts | `dense.npy` 3.1 GB · `passage_texts.jsonl` 2.5 GB · `bm25/` · all **box-only, gitignored** |

**There is no `data/index/single`, and there does not need to be.** ADR-0014 §3 was answered inside
the pool the existing index already produced; the second ~2 h build is not owed. Do not run it.

### The corpus (built 2026-08-06 on the A4000)

| | |
|---|---|
| `fingerprint` | **`93321598f3f1`** — this is the corpus. The earlier `41cf7a6c9160` is the duplicate-bearing draw and **must not be used**. |
| `gold collisions` | **1,000 of 1,000.** Every PubMedQA gold PMID is in MedRAG. The overlap ADR-0012 §1 guessed at is **total** — without draw-time exclusion every gold abstract would have been indexed twice. |
| `gold in the draw` | **0 of 1,000** — this is the *design*, not a defect. See trap 12. |
| `duplicate rows` | **300 suppressed** over 244 PMIDs (trap 3). |
| scanned | 23,898,701 — exact |
| artifacts | `data/corpus/corpus_manifest.json` (tracked) · `corpus.jsonl` 5.5G, `prescan.jsonl` 5.6G (gitignored, **box only**, 12 GB total) |

`load_splits()` returns `{dataset, seed, dev, test, hash}`; `dev` is 100 pubid **strings** and the
split hash is **`71c46cc5b0ca`**.

**`prescan.jsonl` can now be deleted from the box if space is needed** — it existed to allow a redraw
without a second 54 GB read, and the encode it was protecting is done. If it is kept, `--from-prescan`
still refuses unless the manifest records `n_prescan_rows` matching the file on disk (`198172c`); a
count taken *after* a truncation certifies the truncation, which is why it refuses rather than
back-filling.

**If the corpus is ever rebuilt from scratch**, budget **~3 h wall**, not the 1 h first estimated.
**Network to HF is flaky**: two read timeouts hit retry 2/5 on the 2026-08-05 run, the budget is 5
retries per file, and the build is **not resumable** (trap 2). Four guards can stop it — row count ≠
23,898,701 · zero gold collisions · short heap · a draw short on distinct PMIDs. **Each one means a
corpus that must not be encoded.** Ask for the traceback rather than working around it.

---

## 3. Reasoning that the diffs and ADRs do not carry

`git log` and `docs/adr/` hold the rest; this is only what neither recovers.

### From earlier sessions

- **A diagnostic without a null is the defect ADR-0012 §2 was written to avoid, wearing a new hat.**
  The probe's retrieved-side distribution — mean 0.4245, 62/100 questions with a non-gold passage at
  ≥0.5 — reads as a confusable pool and is **nearly indistinguishable from MiniCheck's base rate**.
  Measured against a paired uniform-random draw: **at ≥0.3 random passages beat retrieved ones,
  67.7% vs 62.1%.** Separation only appears in the tail (2.13× at 0.5, **6.89× at 0.7**, 8× at 0.8).
  §2 rejected the LLM topic judge because "a threshold pre-committed against a measurement with no
  established discriminating power buys false comfort" — a probe with no control has the same defect,
  and the number would have gone into the paper's setup section unchallenged.
- **The pairing is what makes the contrast readable, and it was not incidental.** Each score is a max
  over the question's gold sentences (mean 8.8), which inflates any base rate. Drawing the *same
  number* of control passages per question makes that inflation identical on both arms so it cancels.
  An unpaired control would have measured the aggregation, not the retrieval.
- **`τ_confusable = 0.7`, set post-hoc and licensed by §2 because the probe gates nothing.** Not 0.5,
  MiniCheck's nominal operating point, where a 15.9% random base rate still contaminates a third of
  what gets counted. At 0.7: **35/100 dev questions carry a plausible mis-citation target against 8
  by chance.** That is the reportable number; the mean is not.
- **A question that costs 4 h of GPU can sometimes be answered inside data already on disk.**
  ADR-0014 §3 nominally needed two 2 h index builds. It was answered in ~1 min by re-ranking the
  100-deep dense pools Table 1 had already recorded and re-encoding only the 9,832 pooled passages.
  **The trick generalises**: the least-processed artifacts (`*.records.jsonl`) are re-analysable, so
  ask what the recorded pool can already answer before booking the box.
- **A cheap proxy needs a check that it reproduces the expensive thing.** The pool re-rank's `empty`
  arm re-derives Table 1 row 2 from the same vectors, so it *must* return hit@5 = 0.59;
  `--expect-hit5` fails the run otherwise. That check is the only reason the `single` arm is
  believable, and it is worth building into any future proxy measurement.
- **State the scope of a conditioned measurement, then check the asymmetry runs your way.** Fixing
  the candidate set to `empty`'s pool holds recall constant (both arms hit@100 = 0.91) and cannot see
  a passage `single` would have surfaced from 2M. That gap is harmless *here* only because switching
  would have required `single` to win and it lost. Had it won, the full build would have been owed.
- **`index_fingerprint()` hashed the weights and not the call.** `dense_encoder` names the checkpoint;
  `tok("", abstract)` and `tok(abstract)` run it differently over the same text and produce different
  vectors (mean cosine 0.9797, max component diff 0.0649). So two indices representing **two separate
  2 h encodes hashed identically**, while `encode_corpus.py`'s resume guard refused to mix them from
  its own local state and its comment claimed every one of its knobs was in the fingerprint. Fixed:
  `RetrievalConfig.title_segment`, `CONFIG_VERSION` 1.3.0, with a test. **When a config names a
  model, ask whether it also names how the model is called.**
- **Read the paired test, not the marginal intervals.** ADR-0014 §3's hit@5 is 0.59 [0.492, 0.681]
  against 0.53 [0.433, 0.625] — heavily overlapping, which is what paired data looks like summarised
  marginally. Same queries, same candidates: the paired rank test has the power, and it separates at
  p = 0.012. A reviewer will look at the overlapping CIs first, so the write-up says this explicitly.

### Carried forward from earlier sessions

- **The dedup is exclusion at draw time, not reconciliation afterwards.** There is never a moment
  when two copies of a gold abstract exist, so nothing has to decide which survives and no
  half-deduped index can reach the encoder. **Its cost is trap 12** — the index has to source gold
  from somewhere, and that somewhere is `encode_corpus.py`, not the corpus file.
- **The surviving gold copy is PubMedQA's, not MedRAG's**, and that is forced, not preferred:
  citations are char spans into `Instance.abstract_text`, so indexing MedRAG's string would
  invalidate every gold offset and every annotation record written against one.
- **Axis 3 reversed its own shape under measurement.** It began as "titles on both sides or neither."
  The measurement that killed the middle option: **a PubMedQA question is its article's title,
  verbatim** — 60 sampled gold PMIDs, title covers the question's content tokens at **median and mean
  1.00, 60/60 at ≥ 0.8**. ADR-0003 called retrieval here "a lexical gimme"; it is stronger than that.
- **Citations/claim measures 1.01 on G0** (89 of 92 claims cite exactly one), but G0's passages were
  sections of one abstract, which is not the retrieval regime. **Re-measured on the W4 end-to-end
  records (ADR-0013 KW2, discharged 2026-08-11):** 1.13–1.50 parsed, 1.87–2.30 emitted, n = 3
  questions — provisional, re-run on the first ≥50-question batch
  (`docs/harvest/generate_smoke_run4.md`).
- **Per-question annotation cost is sublinear in claims; per-claim cost is linear** (2 sampled claims
  cite 1.52 distinct passages, 4 cite 2.03; 4,000 resamples/question over the G0 answers). This is
  *why* ADR-0013 sized the overlap at 2 claims/question — **superseded by ADR-0016**, which drops the
  overlap entirely: all three annotators label all ~250 claims at 4 claims/question, ~10–16 h each.
  The model still governs the schedule risk.
- **How the corpus failure was diagnosed, because the method transfers.** The write step raised
  `wrote 2,000,000 rows for 1,999,703 drawn PMIDs`, whose message named **two** possible causes. The
  guard's *direction* discriminated them without another 3 h run. Everything after came from the
  on-disk `prescan.jsonl`, not the network — four throwaway scripts, each narrowing: how many repeat
  (244/2,041,867) → do their rows differ (129 yes) → **is the difference chunking or revision** → is
  the draw contained in the superset. **Step 3 was nearly skipped as unnecessary and was the one that
  mattered.** Look before assuming the cheap fix is the right one.
- **Sub-agent completion reports are claims, not evidence.** Five W2 slices were delegated in
  parallel and every one came back "verified". Checking the artifacts found, in code reported as
  working: four names used at runtime and never imported; an encoder writing `embeddings.npy` against
  a loader reading `dense.npy`; `passage_texts.jsonl` never written; three sweep configs collapsing
  onto one. **The delegation was still right — the verification is the part that cannot be delegated.**
- **The repo went PUBLIC on 2026-08-05.** **No names, emails or PII appear in any tracked file or
  issue — checked, not assumed.**
- **Both annotators accepted on 2026-08-05.** Issue #7's recruiting deliverable is met, so
  **ADR-0011 §1's prohibition on revising the ask upward is live.**

**ADR house rule.** The default is still that accepted ADRs are not edited — supersede instead. **One
narrow exception exists, written down in `docs/agents/domain.md`:** when a *premise* inside an
accepted ADR is wrong but the *decision* it supported is unchanged, a dated in-place amendment is
allowed (original text stays with a pointer; the amendment says what did not change; the header
records the edit). Used once — **ADR-0014 §2's Amendment, 2026-08-06**. Separately, ADR-0012 §2 and
ADR-0014 §3 now carry dated **Result / Decided** subsections: those are not amendments but the
measurements the sections *asked for*, which is why they are additive and change no decision text.
ADR-0011 §1's now-stale open note is the provenance of ADR-0013 and is left standing on purpose.
**Do not "tidy" it.**

---

## 4. The thing owed to a person — sent 2026-08-06, **replied 2026-08-14; closed**

**Message to the two annotators, sent by the user on the original channel, not GitHub.** It exists
nowhere in the repo. Its content, so it can be reconstructed — and so a **reply** can be read against
what was actually promised:

- **two sittings, 3 hours total** — ~1 h pilot (guidelines + 10 practice items), ~2 h main pass
- the amount of material is **sized to fit those hours and will not grow**
- **stop whenever — everything finished stays useful**; there is no wasted partial work
- a closing question: what would make it sit easier (timing, sitting length, how it is split)

**The reply, 2026-08-14** (user-side channel, ahead of §5's Thu 2026-08-20 backstop):

1. **No worry was raised — both accepted.** §4's closing question had three candidate causes (hours,
   September timing, open-endedness) and only hours was priced. None of the three was the cause. The
   three follow-up constraints that used to bind any nudge are **discharged**, not carried.
2. **No hour limit — "whatever the project requires."**
3. **Both are free from Sep 5 onward.**

Consequences:

- **(3) fits the window as designed — no schedule change.** `annotate.ANNOTATION_SEED = 20260907` and
  ADR-0013 §1 put the pilot in W6 (from Sep 7) and the main pass in W7–W8, all of it after Sep 5. The
  Sep 3 decomposer freeze still precedes the pilot, so claims are frozen units before anyone reads
  them. **Nothing is owed to either annotator before Sep 7**, and no date moves.
- **(2) does not license more material.** ADR-0013 §1 makes 3 h a **ceiling by design** — *"the ask can
  never be revised upward, because the design absorbs the overrun instead of the annotator."* Offered
  hours do retire ADR-0011 Known weakness 1 as an *annotator* risk (it stays a schedule risk), and they
  do put ADR-0011 Known weakness 2 — the ~19-cluster overlap that makes G4's α interval wide — back on
  the table. But widening the overlap changes a **pre-registered sampling quantity**, so it needs an ADR
  and it must be decided **before any label is collected**, never after seeing α.

§4's original allowance stands and is now exercised: if either offers more time, taking it is allowed —
the ceiling binds the project, not them.

---

## 5. Standing constraints — easy to violate by accident

- **Least-processed value.** Store `phi_score: 0.83`, never `supported: true`. Store `gold_rank` or
  the ranked list, never a precomputed hit@5. Store the 4-way `support_label`, never its collapse.
  *This is the rule that decided ADR-0010, and the rule that made ADR-0014 §3 answerable for free.*
- **Wilson, not Wald**, on gate proportions. G1 passes iff point ≥ 0.90 **and** Wilson lower > 0.85.
  **G4 is deliberately different** — it gates on the point alone; ADR-0011 §4 defends why.
- **Every bootstrap clusters on the question, never the claim** (ADR-0011; §8 rule 10). Every table
  caption naming a CI must name its resampling unit.
- **vLLM never enters `pyproject.toml`**, not even an optional group — it pins torch exactly and
  backtracks the workspace to pydantic 1.10.x. **It is a network boundary and a separate OS.**
  `backends.py` reaches it over HTTP via `httpx` at `{VLLM_BASE_URL|http://localhost:8000}`.
- **`RAG_Debate_Agent` is retired** (ADR-0007). Never re-run it; cite `docs/harvest/`.
- **Index identity is a content hash**, never a document count (the ADR-0007 lesson). It now covers
  `corpus_id`, `corpus_fingerprint`, the chunker, `dense_encoder` **and `title_segment`** — the last
  added this session, because the encoder name does not say how the encoder was called. **A redraw
  must change the `corpus_fingerprint` default**; the test pinning it to the committed manifest is
  what fails if it does not.
- **Passages carry no titles, gold or distractor** (ADR-0014 §2). `MEDRAG_TEXT_FIELD = "content"` is
  load-bearing, not a default.
- **MedCPT is asymmetric.** `NCBI/MedCPT-Query-Encoder` for queries, `NCBI/MedCPT-Article-Encoder`
  for passages, and **the title slot never receives the question** (trap 4).
- **≤3 citations per claim**, identical across all three systems.
- **`validate()` reports violations and never repairs them.**
- **`AlignScore` is never-cut** — the middle rung of R7's only remaining ladder.
- **Never tune τ to pass a gate.** R2's ladder ends at relaxing to hit@10 and *saying so in the
  paper*, never at moving a threshold quietly. **`τ_confusable = 0.7` is not an exception**: it was
  set after seeing a distribution, which ADR-0012 §2 licenses precisely because the probe gates
  nothing. If anything ever starts gating on it, it must be re-derived before the fact.
- **`SCHEMA_VERSION` stays `1.0.0`; `CONFIG_VERSION` is now `1.3.0`** (was 1.2.0 — `title_segment`).
- **Schema field names that have bitten.** `QueryRecord` uses **`query_id`**, not `question_id`.
  `CostRecord` is `run_id, query_id, component, backend, input_tokens, output_tokens, usd, wall_s`.
  `RetrievedPassage` is `passage_id, rank` (1-indexed), `score, retriever, text`. Check the dataclass
  before writing a record; three plausible-sounding alternatives are all wrong.
- **The annotator ask is a ceiling, never revised upward** (ADR-0011 §1, live since both accepted).
- **The stop-anytime guarantee has a live dependency in W5.** It is true *only* because ADR-0013 §3
  puts annotators 2 and 3 on the **same randomized question order**, so any common prefix is a
  complete unbiased subsample. That requirement lands on `data.py` and the annotation UI in W5. **If
  it is dropped or simplified there, a sentence the user has already sent becomes false.**
- **The repo is public.** **Nothing outward-facing goes out without the user's word, and push
  authorization does not carry across sessions.**

### Dates set outside the repo — record them, they are nowhere else

- **Annotator reply: hard backstop Thu 2026-08-20.** Derived backwards from the closing question, not
  the schedule: a bad answer is recovered by finding a replacement against R3's **Sep 7** hard
  trigger, whose fallback (intra-annotator α, 150 claims) is explicitly weaker. **Silence past Aug 20
  costs the same as a bad answer** — worth a nudge around Aug 19.
- **All labeling ends Sun 2026-09-20.** Pilot **Sep 7–13 (W6)**, main pass **Sep 14–20 (W7)**;
  **W8 (Sep 21–27) is α, adjudication and G4 — not annotation time.**
- **The dependency that makes it hold: the pilot must actually happen in W6.** If it slips, the main
  pass slides into W8 and ADR-0011's α < 0.6 branch loses its only re-run. **The gap between the two
  sittings is load-bearing, not padding.**

### How this user works

- **One decision at a time, lettered, each with a recommendation.** Replies are terse, often a single
  letter or two words. **A one-word answer may address only part of a multi-part question** — re-ask
  the remainder rather than assuming. This has fired in six sessions.
- **`do it` / `go` means "the next thing you just named."** **Name the next action explicitly at the
  end of every turn**, or that instruction is ambiguous.
- **Ambiguous confirmations about the box are usually about a different machine.** "done" and
  "git push is done" both arrived this session meaning something other than what was asked — a
  re-paste of an earlier digest, and a push executed *on the A4000* rather than on this laptop.
  **Verify with `git fetch` and `git rev-list --left-right --count` rather than believing the
  report**, and say which machine a command belongs to in the command block itself.
- **Terse instructions are decisions, not openings for discussion.** When one cuts against a repo
  convention: **state the conflict in two sentences, do it anyway, in the shape that preserves the
  convention's purpose.**
- **Brevity while driving the box.** Lead with the command, keep reasoning to what changes the next
  action. This never meant the reasoning should go unrecorded: the commit messages are long and none
  has been queried.
- **Look facts up; ask only decisions.** Every sharp finding across seven sessions came from reading
  a file, fetching a shard, querying NCBI or running the code — none from reasoning.
- **Argue against your own earlier recommendation when the evidence changes.** Reversals have been
  accepted immediately every time. This session reversed the *previous handoff's* recommendation to
  defer ADR-0014 §3, and the reversal was right: deferring would have left the index's identity
  undecided under W3's reranker numbers.
- **The A4000 is copy-paste only.** No SSH from the agent environment. Hand over commands, wait for
  pasted output. **Never inspect `~/.ssh/`** — declined once. **Run throwaway diagnostics through a
  syntax check before handing them over**; a one-line slip costs a full round trip.
- **Long multi-line command blocks lose their flags.** Two runs this session were wasted because a
  `git pull` and a `--random-control` were dropped from a pasted block. **Prefer short blocks, put
  the load-bearing flag on its own line, and where a dropped flag would silently produce a wrong
  artifact, make the script refuse** (as `confusability_probe.py` now does — trap 15).

### The box's environment

- **Git auth uses a fine-grained PAT** (Contents: read/write). `credential.helper store` may have
  been run, which writes it in plaintext to `~/.git-credentials`. **Never read or echo that file.**
- **`git config --global user.email` on the box differs** from the address on the user's other
  commits. Cosmetic, but worth one check rather than a surprise in November.
- **The box pushes directly to `origin/main`**, so `git pull --rebase` here before any push; it
  happened twice this session and once produced a commit that had to be reverted (trap 15).
- **This laptop has no CUDA.** Both MedCPT encoders are in the local HF cache, so smoke tests re-run
  in seconds. `transformers` warns `torch_dtype is deprecated! Use dtype instead` — cosmetic.

---

## 6. Traps — these have gone wrong once and would again

1. **The partial parquet.** `load_dataset("MedRAG/pubmed")` resolves to an auto-converted **partial**
   export — 2,209,839 rows of 23,898,701, PMID-ascending, the **oldest ~9% of PubMed**. It is within
   10% of the 2M target, so the naive load *succeeds* and yields pre-1990 abstracts against
   1990s–2010s gold: separable by era alone, G1 excellent for the wrong reason, G2 with nothing
   plausible to mis-cite. **Read `data_files="chunk/*.jsonl"`, never the bare dataset id.** ADR-0014 §1.
2. **The str/int join.** PubMedQA's `pubid` is int32 and `data.py` stringifies it; MedRAG's `PMID` is
   int64. `{"21645374"} & {21645374}` is empty, so a broken dedup reports **"0 duplicates removed" —
   which reads as good news.** `draw_corpus` raises on non-`int` keys and raises again on a full scan
   that collides with *no* gold PMID. **This is why the build is not resumable.** The split itself is
   unresolved design debt, noted in `corpus.py`'s docstring; Issue #10 finding 5 left it there
   deliberately, because it is a modelling decision and not a bug.
3. **The repeated PMID.** PubMed re-publishes revised records and MedRAG keeps each revision as its
   own row — **244 of 2,041,867 drawn PMIDs.** Duplicates share a `selection_key`, so both copies
   enter the bottom-k together and the draw quietly becomes 2M rows over 1,999,703 articles: **one
   abstract under two `passage_id`s.** The revisions are whole abstracts, not chunks, so **ADR-0014
   §2's "a row is an article" holds** — its unstated "and appears once" does not. One row survives per
   article: **longest `content`, ties on smallest `id`.** The manifest key is
   **`n_duplicate_rows_in_draw`**, renamed rather than recounted: the counter only sees PMIDs already
   in the running bottom-k, so 300 is a **lower bound**. **The 26 MB committed manifest was left
   alone** — only newly written manifests carry the new key.
4. **Indexing the question.** `scripts/g0_medcpt_throughput.py:46` puts `row["question"]` in MedCPT's
   title slot as a throughput stand-in. **Copying that into the real encode would index the query
   against itself.**
5. **Fetching gold titles to "fix" the title asymmetry.** The intuitive repair, and the one option
   that is definitely wrong — the titles *are* the questions.
6. **A `pytest.skip` on a missing fixture is invisible in a green run.** The G0 records were under
   gitignored `runs/`, so `tests/test_abstention.py` was silently skipping three load-bearing tests
   on every other machine. **They now live in `docs/harvest/g0/`.** **ADR-0010's Validation section
   and ADR-0013's evidence table still cite `runs/g0/`** — left stale on purpose (a path is neither a
   wrong premise nor a changed decision). Read them as `docs/harvest/g0/`.
7. **`docs/` is gitignored** via `docs/*` with `!docs/adr/`, `!docs/harvest/` and `!docs/agents/` as
   the **only** exceptions, and `*.jsonl` wins inside them unless separately negated
   (`!docs/harvest/**/*.jsonl`, added after `.gitignore` silently ate
   `table1_rows_1_3.records.jsonl`). Docs written anywhere else are **silently untracked** — verify
   with `git check-ignore`. **Any new exception needs the same two-line pair**: the negation, and a
   note saying what it exists for.
8. **VRAM drifts on the A4000** (WDDM, display attached). Always launch vLLM with
   `--gpu-memory-utilization 0.85` and `VLLM_USE_V2_MODEL_RUNNER=0`. **Do not install an NVIDIA
   driver inside WSL** — the Windows driver is passed through.
9. **There is no box preflight script any more.** `scripts/g0_smoke.sh` was deleted 2026-08-06.
   **Its thresholds are the right ones** if one is rewritten: ≥ 10 GB free VRAM of 16 and ≥ 60 GB
   free disk. Recover with `git show c213b4d:scripts/g0_smoke.sh`.
10. **Scratchpad handoffs get deleted.** Three already have been. **The tracked root file is the
    convention**; fold any temp-directory output in here and delete it.
11. **A guard can be unreachable by construction and still read as a guard.** `--from-prescan`
    checked that every drawn key fell below the prescan cutoff — but the prescan only ever wrote rows
    already under that cutoff, so the comparison could not fail. `draw_corpus` had the same shape:
    handed the surviving rows as its own `expected_rows`, it compared the file to itself. Both fixed
    (`198172c`). **When reading a guard, ask what varies that it could see.**
12. **The corpus contains no gold, and an index built from it alone scores hit@5 = 0.00.** 2,000,000
    drawn PMIDs, 1,000 gold PMIDs, **0 in common** — ADR-0012 §1 working. The consequence is easy to
    miss: hit@5 is defined over `Instance.gold_passage_ids` (`f"{pubid}:{i}"`) while `corpus.jsonl`
    chunks into `f"{row_id}:{i}"`, **so an index over the corpus alone contains no id in the gold
    space and every configuration scores exactly 0.00 while reading as a broken retriever.**
    `_iter_passages` yields gold **first**, then distractors, so a `--limit` run still contains all of
    it. **The default is gold ON**, and `with_gold` is in the resume guard.
13. **A missing file that does not raise.** `RetrievalIndex.load` looks `dense.npy` up **by name** and
    leaves `dense_embeddings = None` when absent. The encoder wrote `embeddings.npy`. Nothing raised —
    Table 1's dense and RRF rows would have been **silently BM25-only**. Sibling bug:
    `passage_texts.jsonl` was written only under `--build-bm25`, and with empty texts `_rerank` scores
    the query against the literal string `"pubmed23n0001_0:0"`. Both fixed; a length-agreement guard
    now `sys.exit`s if ids/texts/embeddings disagree. **Where a loader resolves by name and tolerates
    absence, the writer's name is not a preference.**
14. **A summary in the slot where the value should be.** `table1_baseline.py` wrote a key literally
    named `gold_rank` holding `{rank_mean, rank_median, …}`, with the 100 per-question ranks
    discarded. It passes every eye test. But `rank_mean` cannot be re-thresholded at hit@10 (**R2's
    ladder ends exactly there**) or bootstrapped clustered on the question. The related censoring bug:
    `retrieve.py` truncated the pool to `top_k=5` *before* Table 1 recorded ranks, conflating "rank 6"
    with "not retrieved" — Table 1 now evaluates the full 100-deep pool and nulls text past rank 5.
    **That fix is what made ADR-0014 §3 answerable without a second index build.**
15. **A dropped flag can produce a confidently mislabelled artifact.** A run missing
    `--random-control` re-scored the ordinary retrieved-side probe and wrote it to
    `confusability_probe_control.json`; the summary was byte-identical to the real probe and it
    reached `main` as `d9d6a13` before being reverted in `c2d8e56`. **The only tell was an absent
    `seed` key.** A file named `control` holding the uncontrolled run is the kind of artifact that
    gets quoted later as a null result. `confusability_probe.py` now **refuses** to write to an
    `--out` containing "control" without `--random-control`. **Where a dropped flag silently changes
    what an artifact means rather than failing, the script must refuse.**
16. **A hardness diagnostic with no null measures the model, not the corpus.** See §3. The retrieved
    scores were reported as evidence the pool is confusable; against a paired random draw, everything
    below 0.5 is base rate and **at 0.3 random passages score higher**. Any future
    "is this hard/plausible/on-topic?" measurement gets a control drawn from the same population in
    the same run, or it does not get reported.

---

## 7. What to read — the shortest ordered list

1. **`docs/adr/0015`, `0014`, `0013`, `0012`** — G1 re-gating, the corpus's text form and source, the annotation budget, the distractor pool.
2. **`src/biomedqa/corpus.py`'s module docstring** — the longest-form write-up of traps 1–3.
3. **`scripts/encode_corpus.py`'s `_iter_passages` docstring** — the long form of trap 12.
4. `CONTEXT.md` — the four frozen units and the annotation protocol; authoritative on the units.
5. `research_roadmap.md` §3 (distractor selection), §5 (the week grid), §7 (R1–R7), §8 rules 8 and 10.
6. `docs/adr/0009`–`0011` — parity, abstention, the gold set.
7. `docs/adr/0003`–`0008` — the decisions the newer ones refine.
8. `docs/agents/domain.md` — the ADR conventions, including the amendment exception.
9. `src/biomedqa/schema.py` — the frozen contract, still **1.0.0** and deliberately so.
10. `scripts/title_convention_pool_eval.py`'s docstring — the pattern for answering an expensive
    question from recorded pools, including what it cannot see.
11. `docs/harvest/runbooks/wsl-vllm-a4000.md` — before touching the box.
12. `paper/skeleton.md` — the five tables and the C1–C5 ledger every result must land in.

Not needed: `docs/project2_biomedical_attribution_rag_implementation_plan.md` (superseded,
banner-marked) · `notebooks/` (toy/simulated; `07_4` simulates 3 labels where `CONTEXT.md` freezes 4
— a correctness bug, not a scale assumption).

---

## 8. Open work, in the order recommended

### 1. W5/W6 generator non-termination — **parse guard landed; decoder value needs the A4000**

Joint `21074975` emitted 13 CLAIM lines where each re-emitted its predecessor plus one more clause,
ending at 731 words. Every such claim scored 0.0 recall, so a decoder failure was being charged to
joint's grounding. `CONFIG_VERSION` is now **1.4.0**; `PARITY_LOOP_CLOSED`'s template SHA-256 is
untouched and no prompt text changed.

**Done, prompt-free, 317 tests passing (was 306):**
- `prompts.MAX_CLAIM_WORDS = 50` + `parse_response(..., max_claim_words=)`. An over-length claim is
  **flagged and kept**, never truncated or dropped — same rule as over-cap citations.
- `ScoringConfig.max_claim_words` (imported from `prompts`, one copy). It is a **scoring** rule, not
  a generation knob: parse errors are re-derived from `raw_generation`, so revising the number
  re-scores existing records and never forces a re-run (ADR-0010).
- Threshold is **50, not 30**. Claim-length p95 is 29 / 29 / 34 words for joint / post-hoc /
  vanilla; a guard at 30 flags 4.73% / 3.06% / 9.43%, which fails G2's ≥95% valid-parse bar on
  vanilla by itself and taxes the three arms at three different rates — moving C2's gap by
  instrument. Replayed over `parity_iter1b.records.jsonl`, 50 flags **20 / 3 / 4 claims across 3 / 2
  / 3 queries**: joint's excess is its own degeneracy, which is the finding.
- `GenerationConfig.frequency_penalty` + `stop`, forwarded to the vLLM payload; `stop` also maps to
  Anthropic's `stop_sequences`, and a non-zero `frequency_penalty` on that backend **raises** rather
  than being silently dropped.
- `scripts/generate_smoke.py --frequency-penalty`, with `over_length_claims`, `quote_not_found` and
  `longest_claim_words` in the summary.

**`frequency_penalty`, never `repetition_penalty`** — this corrects the previous session's plan.
Verified in vLLM source (`vllm/model_executor/layers/utils.py::apply_penalties`, and the Triton
`_penalties_kernel`): `repetition_penalty` is applied over `prompt_mask | output_mask`, so it
down-weights every token appearing **in the prompt** — precisely the tokens a citation must copy
verbatim for `locate_quote` to find its span. It would trade a decoding defect for a citation
defect, and Table 2 would read the loss as failed grounding. `frequency_penalty` and
`presence_penalty` are computed from `output_tokens_tensor` alone. Neither is bypassed at
`temperature: 0.0` — penalties run in `Sampler.forward` before `greedy_sample`, gated on
`no_penalties` (value ≠ default), not on temperature. `tests/test_backends.py` asserts
`repetition_penalty` is never sent, so it cannot be reintroduced as the obvious knob.

**Blocked, and the only blocked part: the value.** `frequency_penalty` defaults to `0.0`, the
OpenAI no-op, so 1.4.0 changes no token yet. A positive value also reaches the verbatim quotes —
they are generated tokens too — so a quote whose common words were already emitted can be pushed
off its exact wording. The value cannot be chosen from the desk; it needs one sweep on the box,
reading `over_length_claims` (must fall) against `quote_not_found` (must not rise):

```
# WSL2 / A4000, vLLM already serving. Copy-paste; repeat per value.
for FP in 0.0 0.1 0.2 0.4; do
  uv run python scripts/generate_smoke.py --model <served-id> --n 20 \
    --frequency-penalty $FP --out-prefix docs/harvest/freqpen_$FP
done
```

Then set the default in `GenerationConfig` and re-run the three arms. **Consequence to decide:** the
first citation-F1 pair (joint 0.264 / post-hoc 0.345) was measured under 1.3.0 with the runaway
claims included. Either it is recomputed under 1.4.0 or the paper reports the 1.3.0 pair and the
repair is a documented deviation — the guard alone changes G2's parse rate for records that already
exist, because parse errors are re-derived.

### 2. W6 MiniCheck integration (`src/biomedqa/verify.py`)

Replace interim `cross-encoder/nli-deberta-v3-xsmall` φ with MiniCheck-Flan-T5-Large on the box.
Re-evaluate `scripts/first_citation_f1.py` with MiniCheck to test premise-length sensitivity hypothesis (post-hoc quotes 23w median vs joint 19w).

### 3. W9 Stratified robustness check

Mandatory pre-registered asymmetric check stands (residual parity gap favours C2 on all bases).

### 4. Overlap width — decide before the Sep 7 pilot, not after

The annotators' unlimited-hours offer (§4) reopens ADR-0011 Known weakness 2: the overlap subset is
~19 question-clusters, which is what makes G4's α interval wide, and no amount of correctness narrows
it. More hours can buy more clusters. This changes a **pre-registered sampling quantity**, so it needs
an ADR and it must land **before any label exists** — deciding it after seeing α is not available.
Declining is a legitimate outcome; leaving it undecided past Sep 7 is not.

---

**Suggested skills.** `/tdd` for anything that has to fail before it can be trusted — traps 12, 13
and 15 are all cases where the code ran green and produced a wrong number or a wrong artifact.
`tests/test_corpus.py` is the house model: each test's docstring names the silent failure it
prevents, and its fixture is a **real MedRAG row embedded verbatim**. `/grilling` is the user's
preferred instrument and produced ADR-0009–0013; live target is the **W9 triple-booking**.
`/domain-modeling` for any new ADR, and for the PMID `str`/`int` split (trap 2) if it is judged worth
a type. Not needed yet: `dataviz` (figures are W11) · `claude-api` (the Opus 5 judge is wired in W6).
