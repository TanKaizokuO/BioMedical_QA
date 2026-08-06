# HANDOFF — 2026-08-06 (end of sixth session)

Snapshot for resuming in a fresh session. Regenerate with `/handoff`; **do not append** — a stale
line here is worse than a missing one, because the next session will trust it.

`main` · working tree clean · **`origin/main` == `HEAD` == `0c45a09`. Nothing is unpushed.**
Tests: `uv run --with pytest python -m pytest tests/ -q` → **82 passed**. There is no bare `python`
on this box and no installed `pytest`; that invocation is the one that works.

> **This file supersedes three scratchpad handoffs** (sessions 3, 4 and 5, all in
> `/tmp/claude-1000/.../scratchpad/`). Everything load-bearing in them is folded in here — including
> the annotator message draft (§4) and the three dates in §5, which exist nowhere else in the repo.
> The session-3 file has already been lost to a cleared scratchpad. **Do not go looking for them.**

---

## 1. Where the project is

**W1 (Aug 3–9), Phase 1 — Retrieval gate (Aug 3–23) → C1, Table 1.**

| Gate | Date | State |
|---|---|---|
| **G0** — 8B AWQ generator chosen, A4000 measured | Aug 4 | **PASSED 2026-08-04.** Issue #1 still open; close it. |
| **G1** — hit@5 ≥ 0.90, Wilson lower > 0.85 | **Aug 23** | **Unstarted, on plan.** Retrieval is W2–W3 work; nothing about G1 is late. |
| G2 · G3 · G4 · G5 | Sep 6 · Sep 20 · Sep 27 · Oct 11 | Unstarted, with due weeks. |

**W1's list is complete** — splits frozen two days early, PMID dedup pulled forward a week from W2,
Axis 3 decided, ADR-0014 written. **W2 opens Aug 10** and holds: PMID dedup verification, the
chunker sweep, `bm25s` + MedCPT + RRF, the **2M encode**, ADR-0012 §2's confusability probe, and
ADR-0014 §3's empty title-segment measurement. That is not a light week.

Everything past retrieval is `NotImplementedError` with a due week in its module docstring. That is
by design, not drift. The exception is `scoring/abstention.py` (§5).

Unresolved, and **not needing re-derivation**: **W9 is triple-booked** (`research_roadmap.md` §5 ⚠)
· the blind parity loop leaves six days between the first citation-F1 (≈Aug 31) and G2 (Sep 6)
· `SPLIT_SEED = 20260807` while the draw happened Aug 5 — **left alone deliberately**; the constant
documents the deadline it was written against, and changing it would change the draw for nothing.

---

## 2. What is blocking right now

**Nothing is blocking. The corpus is built, and the annotator message was sent on 2026-08-06**
(user-reported), a day inside its deadline — §4. What is now outstanding there is a *reply*, not a
send: the closing question's answer is what prices the September worry, and it is not owed by a date.
W2 work does not wait on it.

### The corpus, as built (2026-08-06, on the A4000)

| | |
|---|---|
| `fingerprint` | **`93321598f3f1`** — this is the corpus. The earlier `41cf7a6c9160` is the duplicate-bearing draw and **must not be used**. |
| `gold collisions` | **1,000 of 1,000.** Every PubMedQA gold PMID is in MedRAG. The overlap ADR-0012 §1 guessed at is **total** — without draw-time exclusion every gold abstract would have been indexed twice. |
| `duplicate rows` | **300 suppressed** over 244 PMIDs (§6 trap 3). |
| scanned | 23,898,701 — exact |
| artifacts | `data/corpus/corpus_manifest.json` (tracked) · `corpus.jsonl`, `prescan.jsonl` (gitignored, box only) |

The scan ran once and completed; the draw was then redone from its on-disk superset via
`--from-prescan` after the repeated-PMID fix. `prescan.jsonl` is worth keeping on the box while W2
runs — it is the only way to redraw without a second 54 GB read.

**If the corpus is ever rebuilt from scratch**, budget **~3 h wall**, not the 1 h first estimated —
a JSON parse per row sits on top of the network, and two read timeouts hit retry 2/5 on the
2026-08-05 run. It is **not resumable** (§6 trap 2). Four guards can stop it — row count ≠
23,898,701 · zero gold collisions · short heap · a draw short on distinct PMIDs. **Each one means a
corpus that must not be encoded.** Ask for the traceback rather than working around it.

Agent-side there is no blocker. The next agent-side actions are in §8.

---

## 3. What changed since the last handoff (`269af6a` → `0c45a09`)

Eight commits. **The commit messages and the ADRs carry the reasoning; it is not repeated in full
here.** What follows is only what a diff does not recover.

| Commit | What |
|---|---|
| `66bc824` | `corpus.py` + `scripts/build_corpus.py` + `build_gold_pmids.py` + `data/gold_pmids.json` — the 2M draw, with gold excluded at draw time |
| `2a81dbf` | **ADR-0013** — annotation sized to a fixed time budget; roadmap §3, §4 Phase 4, §5 W6–W8, §9 |
| `23aa4cb` | `data/splits.json` — dev and test **frozen** |
| `3a1c65b` | `data.gold_pool()` + `tests/test_splits.py` — gold-attribution questions drawn from outside `test` |
| `9f7e2c9` | **Axis 3** — distractors indexed as abstract prose, no titles anywhere |
| `fecfeb8` | **ADR-0014** + roadmap (partial-parquet bullet, W2 row) |
| `0d3a0ed` | one row per article — MedRAG's repeated PMIDs (§6 trap 3) |
| `0c45a09` | `--from-prescan` — redraw off a completed scan, never a partial one |

**Reasoning that is not in the diffs:**

- **The dedup is exclusion at draw time, not reconciliation afterwards.** There is never a moment
  when two copies of a gold abstract exist, so nothing has to decide which survives and no
  half-deduped index can reach the encoder.
- **The surviving gold copy is PubMedQA's, not MedRAG's**, and that is forced, not preferred:
  citations are char spans into `Instance.abstract_text`, so indexing MedRAG's string would
  invalidate every gold offset and every annotation record written against one.
- **Axis 3 reversed its own shape under measurement.** It began as "titles on both sides or neither."
  The measurement that killed the middle option: **a PubMedQA question is its article's title,
  verbatim** — 60 sampled gold PMIDs, title covers the question's content tokens at **median and mean
  1.00, 60/60 at ≥ 0.8** (`Instance.question` vs NCBI esummary, 2026-08-05). ADR-0003 called
  retrieval here "a lexical gimme"; it is stronger than that. **Neither is the only reachable parity.**
- **Citations/claim measures 1.01, not the 2–3 assumed** (89 of 92 G0 claims cite exactly one; the G0
  prompt did not suppress multi-citation). But **G0's passages were sections of one abstract**, which
  is not the retrieval regime. **ADR-0013 KW2: re-measure on the W4 end-to-end records.** This is the
  single largest uncertainty left in the annotation plan.
- **Per-question annotation cost is sublinear in claims; per-claim cost is linear** (2 sampled claims
  cite 1.52 distinct passages, 4 cite 2.03; 4,000 resamples/question over the G0 answers). This is
  *why* 2 claims/question beats ADR-0011's 4×19 at equal cost.
- **The repo went PUBLIC on 2026-08-05** (`gh repo view` → `isPrivate: false`). Any earlier note
  calling issues "internal records the annotators cannot read" is **dead**. Issue #7's false
  parenthetical was corrected on 2026-08-05; nothing else in #7 was touched. **No names, emails or
  PII appear in any tracked file or issue — checked, not assumed.**
- **Both annotators accepted on 2026-08-05** (user-reported). Issue #7's recruiting deliverable is
  met, so **ADR-0011 §1's prohibition on revising the ask upward is live.**

**ADR-0014 is new since the last handoff.** House convention, verified rather than assumed: **no ADR
in this repo is ever edited after acceptance** — every cross-reference lives in the *newer* ADR's
header. ADR-0011 §1's now-stale open note is the provenance of ADR-0013 and is left standing on
purpose. **Do not "tidy" it.**

---

## 4. The thing owed to a person — sent 2026-08-06, reply outstanding

**Message to the two annotators, sent by the user on the original channel, not GitHub.** It exists
nowhere in the repo. Its content, so it can be reconstructed — and so a **reply** can be read against
what was actually promised:

- **two sittings, 3 hours total** — ~1 h pilot (guidelines + 10 practice items), ~2 h main pass
- the amount of material is **sized to fit those hours and will not grow**
- **stop whenever — everything finished stays useful**; there is no wasted partial work
- a closing question: what would make it sit easier (timing, sitting length, how it is split)

Three constraints, which outlived the send — they now bind **any follow-up** to the annotators:

1. It must **never present this as hours going up.** They are not — the W6 pilot was simply never
   inside ADR-0006's ~3 h. It is an uncosted session, not a mis-estimate.
2. **The stop-anytime line is a real guarantee, not reassurance** — see §5.
3. The closing question exists because the annotators raised a worry whose *cause* is unknown. Hours,
   September timing and open-endedness are three different problems and **only the first has been
   priced.**

If either offers more time, **taking it is allowed** — the ceiling binds the project, not them.

**The schedule is publicly readable in issue #7 right now.** That was the reason to send promptly,
and it was never a reason to edit #7. Now that the message is out, the race is over: the annotators
heard it from the user first.

---

## 5. Standing constraints — easy to violate by accident

- **Least-processed value.** Store `phi_score: 0.83`, never `supported: true`. Store `gold_rank` or
  the ranked list, never a precomputed hit@5. Store the 4-way `support_label`, never its collapse.
  *This is the rule that decided ADR-0010.*
- **Wilson, not Wald**, on gate proportions. G1 passes iff point ≥ 0.90 **and** Wilson lower > 0.85.
  **G4 is deliberately different** — it gates on the point alone; ADR-0011 §4 defends why.
- **Every bootstrap clusters on the question, never the claim** (ADR-0011; §8 rule 10). Every table
  caption naming a CI must name its resampling unit.
- **vLLM never enters `pyproject.toml`**, not even an optional group — it pins torch exactly and
  backtracks the workspace to pydantic 1.10.x. **It is a network boundary and a separate OS.**
- **`RAG_Debate_Agent` is retired** (ADR-0007). Never re-run it; cite `docs/harvest/`.
- **Index identity is a content hash**, never a document count (the ADR-0007 lesson). The empty
  title-segment convention (ADR-0014 §3) is part of that identity and goes in the fingerprint.
- **Passages carry no titles, gold or distractor** (ADR-0014 §2). `MEDRAG_TEXT_FIELD = "content"` is
  load-bearing, not a default.
- **≤3 citations per claim**, identical across all three systems.
- **`validate()` reports violations and never repairs them.**
- **`AlignScore` is never-cut** — the middle rung of R7's only remaining ladder, and the cut fired in
  W6, a week before G3 (§8 rule 8).
- **Never tune τ to pass a gate.** R2's ladder ends at relaxing to hit@10 and *saying so in the
  paper*, never at moving a threshold quietly.
- **The annotator ask is a ceiling, never revised upward** (ADR-0011 §1, live since both accepted).
- **The stop-anytime guarantee has a live dependency in W5.** It is true *only* because ADR-0013 §3
  puts annotators 2 and 3 on the **same randomized question order**, so any common prefix is a
  complete unbiased subsample. That requirement lands on `data.py` and the annotation UI in W5. **If
  it is dropped or simplified there, a sentence the user will already have sent becomes false.**
- **The repo is public.** Outward-facing now means the whole repo, not just issues. **Nothing
  outward-facing goes out without the user's word.**

### Dates set outside the repo — record them, they are nowhere else

- **Annotator message: due Fri 2026-08-07, sent Thu 2026-08-06.** Met. The **Thu 2026-08-20 hard
  backstop still stands, and now applies to the answer, not the send** — the deadline was derived
  backwards from the closing question, not from the schedule. It asks two people an open question,
  and a bad answer is recovered by finding a replacement against R3's **Sep 7** hard trigger, whose
  fallback (intra-annotator α, 150 claims) is explicitly weaker. **Silence past Aug 20 costs the same
  as a bad answer**, so it is worth a nudge before then rather than after.
- **All labeling ends Sun 2026-09-20.** Pilot **Sep 7–13 (W6)**, main pass **Sep 14–20 (W7)**;
  **W8 (Sep 21–27) is α, adjudication and G4 — not annotation time.**
- **The dependency that makes it hold: the pilot must actually happen in W6.** If it slips, the main
  pass slides into W8 and ADR-0011's α < 0.6 branch loses its only re-run. **The gap between the two
  sittings is load-bearing, not padding** — the pilot exists to test the guidelines, so the guideline
  revision happens in that gap.

### How this user works

- **One decision at a time, lettered, each with a recommendation.** Replies are terse, often a single
  letter. **A one-word answer may address only part of a multi-part question** — re-ask the remainder
  rather than assuming. This has fired in three separate sessions and asking was correct every time.
- **`do it` / `go` means "the next thing you just named."** **Name the next action explicitly at the
  end of every turn**, or that instruction is ambiguous.
- **Look facts up; ask only decisions.** Every sharp finding across the last four sessions came from
  reading a file, fetching a shard or querying NCBI — none from reasoning.
- **Argue against your own earlier recommendation when the evidence changes.** Reversals have been
  accepted immediately every time. Do not defend a prior position.
- **The A4000 is copy-paste only.** No SSH from the agent environment; `scp` to `vllm-box` fails with
  `Permission denied (publickey…)`. Hand over commands, wait for pasted output. **Never inspect
  `~/.ssh/`** — declined once.

---

## 6. Traps — these have gone wrong once and would again

1. **The partial parquet.** `load_dataset("MedRAG/pubmed")` resolves to an auto-converted **partial**
   export — 2,209,839 rows of 23,898,701, PMID-ascending, the **oldest ~9% of PubMed**. It is within
   10% of the 2M target, so the naive load *succeeds* and yields pre-1990 abstracts against
   1990s–2010s gold: separable by era alone, G1 excellent for the wrong reason, G2 with nothing
   plausible to mis-cite. **Read `data_files="chunk/*.jsonl"`, never the bare dataset id.**
   ADR-0014 §1.
2. **The str/int join.** PubMedQA's `pubid` is int32 and `data.py` stringifies it; MedRAG's `PMID` is
   int64. `{"21645374"} & {21645374}` is empty, so a broken dedup reports **"0 duplicates removed" —
   which reads as good news.** `draw_corpus` raises on non-`int` keys and raises again on a full scan
   that collides with *no* gold PMID. **This is why the build is not resumable:** a partial scan could
   otherwise satisfy the row-count guard. `--from-prescan` is the one exception and it refuses unless
   an existing manifest already records a completed 23,898,701-row scan — it restates guards, never
   manufactures them.
3. **The repeated PMID.** PubMed re-publishes revised records and MedRAG keeps each revision as its
   own row — **244 of 2,041,867 drawn PMIDs, twice inside a single shard in at least one case.**
   Duplicates share a `selection_key`, so both copies enter the bottom-k together and the draw
   quietly becomes 2M rows over 1,999,703 articles: **one abstract under two `passage_id`s**, which
   is ADR-0012 §1's failure arriving from inside MedRAG rather than from gold. The revisions are
   whole abstracts, not chunks (`22367489` is `b-subunit` vs `beta-subunit`; `22453897` appends an
   abbreviation list), so **ADR-0014 §2's "a row is an article" holds** — its unstated "and appears
   once" does not, and its 15,377/15,377 was one shard. One row survives per article: **longest
   `content`, ties on smallest `id`.**
4. **Indexing the question.** `scripts/g0_medcpt_throughput.py:46` puts `row["question"]` in MedCPT's
   title slot as a throughput stand-in. **Copying that into the real encode would index the query
   against itself.** The title slot never receives the question.
5. **Fetching gold titles to "fix" the title asymmetry.** The intuitive repair, and the one option
   that is definitely wrong — the titles *are* the questions (§3).
6. **`runs/g0/` → `docs/harvest/g0/` is not a file move.** `tests/test_abstention.py` resolves
   `G0_DIR` as `runs/g0`, and **those two tests are load-bearing**: they assert the abstention
   predicate against both real G0 record shapes (Llama's list-item form, Qwen's trailing prose). If
   either starts failing, ADR-0010's decision to hold `SCHEMA_VERSION` at 1.0.0 must be revisited
   **before Sep 7**, not after.
7. **`docs/` is gitignored** via `docs/*` with `!docs/adr/` and `!docs/harvest/`. Docs written
   anywhere else are **silently untracked** — verify with `git check-ignore`.
8. **VRAM drifts on the A4000** (WDDM, display attached). Always launch vLLM with
   `--gpu-memory-utilization 0.85` and `VLLM_USE_V2_MODEL_RUNNER=0`. **Do not install an NVIDIA
   driver inside WSL** — the Windows driver is passed through.
9. **`scripts/g0_smoke.sh` has never run successfully and is now wrong** — it assumes a POSIX login
   shell; the box answers with `cmd.exe`. Recommend deleting; the user has not decided.
10. **Scratchpad handoffs get deleted.** One already was. The tracked root file is the convention;
   `/handoff` writes to the OS temp directory by default.

---

## 7. What to read — the shortest ordered list

1. **`docs/adr/0014`, `0013`, `0012`** — the corpus's text form and source, the annotation budget,
   the distractor pool. The answer to almost any "why is it like this?" about W1–W2.
2. **`src/biomedqa/corpus.py`'s module docstring** — the longest-form write-up of traps 1–5, and the
   only surviving record of the session-3 findings.
3. `CONTEXT.md` — the four frozen units and the annotation protocol; authoritative on the units.
4. `research_roadmap.md` §3 (distractor selection), §5 (the week grid), §7 (R1–R7), §8 rules 8 and 10.
5. `docs/adr/0009`–`0011` — parity, abstention, the gold set.
6. `docs/adr/0003`–`0008` — the decisions the newer ones refine.
7. `src/biomedqa/schema.py` — the frozen contract, still **1.0.0** and deliberately so.
8. `src/biomedqa/retrieve.py`'s docstring — every settled W2 constraint, and the one open question.
9. `docs/harvest/runbooks/wsl-vllm-a4000.md` — before touching the box.
10. `paper/skeleton.md` — the five tables and the C1–C5 ledger every result must land in.

Not needed: `docs/project2_biomedical_attribution_rag_implementation_plan.md` (superseded,
banner-marked) · `notebooks/` (toy/simulated; `07_4` simulates 3 labels where `CONTEXT.md` freezes 4
— a correctness bug, not a scale assumption).

---

## 8. Open work, in the order recommended

1. ~~Send the annotator message~~ — **sent 2026-08-06.** What remains is the **reply** to §4's
   closing question, which prices the September worry. User-side, undated, and **not a blocker**.
2. **Issue #9's body is stale** (ADR-0010 is its design half). **Ask before editing** — it was
   withheld from a previous go-ahead, and the repo being public **strengthens** that caution.
3. **W2 build work** — the corpus is built and waiting: chunker sweep · `bm25s` + MedCPT + RRF · the 2M encode ·
   ADR-0012 §2's confusability probe (pulls MiniCheck forward, ~½ day) · ADR-0014 §3's title-segment
   measurement · `backends.py` adapter (~½ day).
4. **`corpus.py` and `data.py` have never been reviewed** — the first substantial non-scoring modules.
   Worth doing before the W2 encode commits land on top.
5. **Deferred by the user's own triage to W4–W5:** words/claim vs claims/query as the gated quantity
   (ADR-0009 Known weaknesses) · the Sep 3 freeze · the guideline two-pass calendar.
6. **Housekeeping** — `runs/g0/` → `docs/harvest/g0/` (**carries a code change**, trap 6) ·
   `g0_medcpt_throughput.json` is still only on the box · `scripts/g0_smoke.sh` (trap 8) · **close
   issue #1** (G0 passed Aug 4).

**Suggested skills:** `/grilling` is the user's preferred instrument and produced ADR-0009–0013 —
live target is the **W9 triple-booking**. `/tdd` for the chunker sweep and the probe;
`tests/test_corpus.py` is the strongest house example, because its fixture is a **real MedRAG row
embedded verbatim** — the whole question was what those three text fields actually contain, and an
invented fixture could have asserted a convenient fiction. `/domain-modeling` for any new ADR.
`/code-review` for item 5. Not needed yet: `dataviz` (figures are W11) · `claude-api` (the Opus 5
judge is wired in W6).
