# HANDOFF — 2026-08-04 (end of second session)

Snapshot for resuming in a fresh session. Regenerate with `/handoff`; do not append.

`main`. **8 commits unpushed** — `origin/main` is at `dbd9ed4`. **Working tree is clean** (§3).
Pushing was offered and declined this session; the decision stands until asked again.

> **If you read the previous version of this file:** its §2 held sixteen unconfirmed decisions and
> said "do not implement before confirmation." **All sixteen are now resolved**, amended, and written
> up as `docs/adr/0009`–`0012`. That section is gone. Several decisions changed shape in ways the old
> text contradicts — the ADRs are authoritative.

---

## 1. Where the project is

**Gate G0 PASSED (2026-08-04).** Numbers and reasoning live in the commits and the roadmap.

| | |
|---|---|
| A4000 is a **Windows/WDDM box**; vLLM runs under WSL2 | `docs/adr/0008-*.md` · `docs/harvest/runbooks/wsl-vllm-a4000.md` |
| Generator: **`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`**, 3.42 s median | `d11e847` · `research_roadmap.md` §2 D1 |
| MedCPT 343.6 abstracts/s → 2M in 1.6 h; R1's encode leg discharged | `6561f60` · §3 |
| MedNLI/PhysioNet **deliberately dropped**; G3's ladder ends at three rungs | `315f77b` · R7 · issue #8 |
| **The sixteen open decisions are closed** — four ADRs, §2 below | `1c7aa58` |
| Annotator ask **sent**; awaiting two confirmations | issue #7 (open) |
| Scorer bug: design half decided, body not yet rewritten | issue #9 · ADR-0010 |

**W2 (Aug 10–16) is next**: PMID dedup *(blocks the encode)*, chunker sweep, `bm25s` + MedCPT + RRF,
the 2M encode, the confusability probe. **G1 stays Aug 23.**

---

## 2. The four new ADRs — read these, don't reconstruct them

Produced by a `/grilling` session on the sixteen decisions. None was wrong; four carried live
contradictions with the repo, and several needed their shape, bound, or date changed.

| ADR | Was | What changed |
|---|---|---|
| `0009-granularity-parity-is-a-measured-diagnostic-not-a-condition.md` | decisions 1–8 | Parity is a **measured diagnostic**, not a fifth equal-effort condition — the other four hold by construction, this one is an outcome. **±15% kept** and pre-committed. **Hard 10 iterations or Aug 30.** **Post-hoc decomposer only.** Fully blind. |
| `0010-abstention-is-derived-at-scoring-time-not-stored.md` | decisions 9–10 | **No schema field, no version bump** — `SCHEMA_VERSION` stays 1.0.0. Recall denominator **only**. Both F1 numbers always reported, no threshold. Decision 9 **dropped**. |
| `0011-gold-set-sampling-clustered-bootstrap-g4-gates-on-the-point.md` | decisions 11–13 | Clustering is a **change**, not existing practice — pairing ≠ clustering, and it **widens G2's margin** at an unchanged threshold. G4's **0.4 CI trigger dropped** as self-contradictory. |
| `0012-distractor-pool-is-a-uniform-sample-with-a-confusability-probe.md` | decisions 14–15 | Topic judge **replaced**, not demoted. **PMID dedup** added as a W2 blocker. |

Each has **Known weaknesses** and **Alternatives rejected**. Read both before proposing an
improvement — it was probably considered and declined for a stated reason.

**Two problems flagged, not solved:**

- **W9 is triple-booked** — ADR-0009's stratified robustness check (its triggering branch is the
  *likely* one), a possible G4 re-pilot, and absorbing Phase 3–5 slippage. It is the only slack in
  the schedule. ⚠ in `research_roadmap.md` §5.
- **Six-day window before G2.** The blind parity loop puts the first citation-F1 at ≈ **Aug 31**;
  G2 is **Sep 6**. R5's response is decided in advance, not on the day. `research_roadmap.md` §7 R5.

---

## 3. What landed this session, and what is still unpushed

The tree is **clean**. Three commits on 2026-08-04, on top of the five that were already unpushed:

| | |
|---|---|
| `1c7aa58` | the four ADRs + `CONTEXT.md` (a declining-to-answer claim is `valid`) + `research_roadmap.md` §3, §4 Phase 2/3/4, §5, §7, §8 |
| `5af697c` | `scoring/abstention.py` + `tests/test_abstention.py` + `config.py` (`ScoringConfig`; `CONFIG_VERSION` → 1.1.0, `SCHEMA_VERSION` untouched) |
| this one | `HANDOFF.md` |

**`origin/main` is still at `dbd9ed4` — all eight commits exist only on this machine.** A push was
recommended and the user chose commit-only. Offer it again rather than assuming it was forgotten;
the argument was offsite backup of the G0 measurements and the four ADRs on a hard-deadline project.

**Tests:** `uv run --with pytest python -m pytest tests/ -q` → **44 passed**. There is no bare
`python` on this box and no installed `pytest`; that invocation is the one that works.

**`scoring/abstention.py` is implemented, not stubbed** — unusual, since every other `scoring/`
module is `NotImplementedError` with a due week. It had to be: ADR-0010 keeps `SCHEMA_VERSION` at
1.0.0 *conditional on* a validation that has now run. `tests/test_abstention.py` asserts the
predicate against both real `runs/g0/*.json` records — Llama's list-item abstention and Qwen's
trailing-prose form. **Those two tests are load-bearing.** If either starts failing, the raw material
is insufficient and ADR-0010 must be revisited **before Sep 7**, not after.

---

## 4. Open work, in deadline order

1. **PMID dedup** between the 2M sample and the 1,000 gold contexts. **W2 opens Aug 10 and this gates
   the encode** — a gold abstract present under two `passage_id`s makes `gold_rank`/hit@5 miscount
   silently, and the encode would have to be redone. Specified in ADR-0012 §1; **not written.**
2. **Annotator-hours re-derivation.** ADR-0006's ~3 h ask was derived for ~75 claims, not for the ~19
   overlap *questions* ADR-0011 implies. **Must land before the two annotators confirm** (issue #7) —
   revising upward after someone accepts is the worst outcome. Arithmetic, not a decision; offered to
   the user, not yet answered.
3. **Issue #9's body is stale** — it still describes only the scorer patch. ADR-0010 is the design
   half. Deliberately withheld: it is an outward-facing edit and was not in the go-ahead. **Ask.**
4. **Deferred by the user's own triage to W4–W5:** old decisions 2 (words/claim vs claims/query as
   the gated quantity — see ADR-0009's Known weaknesses), 7 (the Sep 3 freeze), 8 (the guideline
   two-pass calendar).
5. **Housekeeping** — push (declined once, §3) · `runs/g0/*.json` → `docs/harvest/g0/` **— note this
   is not a file move: `tests/test_abstention.py` resolves `G0_DIR` as `runs/g0`, and those two
   tests are load-bearing, so the move carries a code change** · `g0_medcpt_throughput.json` is
   still only on the box and needs scp back · close issue #1 · `scripts/g0_smoke.sh` has never run
   successfully and is now wrong (it assumes a POSIX login shell; the box answers with `cmd.exe`) —
   recommend deleting, user has not decided.

---

## 5. Standing constraints

- **Least-processed value.** Store `phi_score: 0.83`, never `supported: true`. Store `gold_rank` or
  the ranked list, never a precomputed hit@5. Store the 4-way `support_label`, never its collapse.
  *This is the rule that decided ADR-0010.*
- **Wilson, not Wald**, on gate proportions. G1 passes iff point ≥ 0.90 **and** Wilson lower > 0.85.
  **G4 is deliberately different** — it gates on the point alone, and ADR-0011 §4 defends why.
- **Every bootstrap clusters on the question, never the claim** (ADR-0011, §8 rule 10). Every table
  caption naming a CI must name its resampling unit.
- **vLLM never enters `pyproject.toml`**, not even an optional group — it pins torch exactly and
  backtracks the workspace to pydantic 1.10.x. It is a network boundary and now a separate OS.
- **`RAG_Debate_Agent` is retired.** Never re-run it; cite `docs/harvest/`.
- **Index identity is a content hash**, never a document count (the ADR-0007 lesson).
- **≤3 citations per claim**, identical across all three systems.
- **`validate()` reports violations and never repairs them.**
- **`AlignScore` is never-cut** as of this session — it is the middle rung of R7's only remaining
  ladder, and the cut fired in W6, a week before G3 (§8 rule 8).

---

## 6. Working mode — violating these wastes a turn

- **The A4000 is copy-paste only.** No SSH from the agent environment; `scp` to `vllm-box` fails with
  `Permission denied (publickey…)`. Hand the user commands and wait for pasted output. Prefer designs
  that keep work on the box, or the user-opened tunnel (`ssh -L 8000:localhost:8000 vllm-box`).
- **Never inspect `~/.ssh/`.** Declined once.
- `docs/` is gitignored via `docs/*` with `!docs/adr/` and `!docs/harvest/`. Docs elsewhere are
  silently untracked — verify with `git check-ignore`.
- **VRAM drifts** on the A4000 (WDDM, display attached). Always launch vLLM with
  `--gpu-memory-utilization 0.85` and `VLLM_USE_V2_MODEL_RUNNER=0`.
- **Do not install an NVIDIA driver inside WSL** — the Windows driver is passed through.
- Repo is **private**; W12 calls for a public release.
- Dates have drifted in conversation before. Today is **2026-08-04**; November is hard.

### How this user works

- **One decision at a time, each with a recommendation.** Bundled questions get partial answers.
- **Look facts up; ask only decisions.** This session's sharpest findings all came from reading files
  rather than reasoning: the false "matching what G2 already does," the AlignScore cut-order
  contradiction, and the two structurally different G0 abstention shapes.
- Replies are terse, often a single letter. **A one-word answer may address only part of a multi-part
  question** — re-ask the remainder rather than assuming. One reply was `s`, not an option letter;
  asking rather than guessing was correct.
- The user reverses a recommendation when the argument is good, **including the assistant's own
  earlier advice** — ±15% was restored after having been told to drop it. Argue against your own
  prior position when the evidence changes.

---

## 7. Suggested skills

- **`/grilling`** — the instrument that produced §2, and the user's preferred mode. Natural next
  targets: the **W9 triple-booking**, or the W4–W5 load, which the four new ADRs made *heavier*.
- **`/tdd`** — for the PMID dedup and the confusability probe. `tests/test_abstention.py` is the
  model: assert against real artifacts in `runs/g0/` rather than fixtures where possible.
- **`/code-review`** — before the W2 encode commits, once `retrieve.py` and `chunk.py` are real.
- **`/domain-modeling`** — if more ADRs are needed; it matches the house style of `docs/adr/`.
- **`/handoff`** — regenerate this file; it goes stale fast.

Not needed yet: `dataviz` (figures are W11) · `claude-api` (the Opus 5 judge is wired in W6).

---

## 8. What to read

1. **`docs/adr/0009`–`0012`** — this session's output, and the answer to almost any "why is it like
   this?" about parity, abstention, the gold set, or the corpus.
2. `CONTEXT.md` — the four frozen units and the annotation protocol; authoritative on the units.
3. `research_roadmap.md` §3 (distractor selection), §4 Phase 2 items 8–10, §5, §8 rules 8 and 10.
4. `docs/adr/0003`, `0004`, `0005`, `0006`, `0007`, `0008` — the decisions the new four refine.
5. `src/biomedqa/schema.py` — the frozen contract, still at **1.0.0** and deliberately so.
6. `src/biomedqa/scoring/abstention.py` + `tests/test_abstention.py` — the only implemented scoring
   module, and why.
7. `docs/harvest/runbooks/wsl-vllm-a4000.md` — before touching the box.
8. `paper/skeleton.md` — the five tables and the C1–C5 ledger every result must land in.

Not needed: `docs/project2_biomedical_attribution_rag_implementation_plan.md` (superseded,
banner-marked) and `notebooks/` (toy/simulated; `07_4` simulates 3 labels where `CONTEXT.md` freezes
4 — a correctness bug, not a scale assumption).

---

## 9. Immediate next actions

1. **PMID dedup** — W2 opens Aug 10 and it blocks the encode.
2. **Re-derive the annotator hours**; get the number to the user before issue #7's confirmations land.
3. Ask about **issue #9's** body.
4. Re-offer the **push** (§3).

**User-side, open:** two annotators must confirm — issue #7, checkpoint **~Aug 20**, hard trigger
**Sep 7**. ADR-0006's replacement fallback is explicitly weaker.
