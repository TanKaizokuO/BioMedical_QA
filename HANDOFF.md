# HANDOFF — 2026-08-09 (end of ninth session)

Snapshot for resuming in a fresh session. Regenerate wholesale; **do not append** — a stale line here
is worse than a missing one, because the next session will trust it.

`main` · **working tree dirty and `HEAD` is one commit ahead of `origin/main`.** Nothing this session
has been committed or pushed, and **push authorization does not carry across sessions** — ask before
either. Modified: `src/biomedqa/retrieve.py`, `src/biomedqa/backends.py`, `src/biomedqa/corpus.py`,
`scripts/build_corpus.py`, `tests/test_corpus.py`. Untracked: `ROADMAP.md`, `scripts/encode_corpus.py`,
`scripts/chunker_sweep.py`, `scripts/confusability_probe.py`, `scripts/title_convention_eval.py`,
`scripts/table1_baseline.py`. The unpushed commit is `a7d6915` (the four chunkers).

Tests: `uv run --with pytest python -m pytest tests/ -q` → **114 passed**. (The previous handoff said
86; that predates the chunker tests. 114 is the current true count.) There is no bare `python` on this
box and no installed `pytest`; that invocation is the one that works. `pyproject.toml`'s `pythonpath`
is `["src", "scripts"]` — `tests/test_corpus.py` imports from `scripts/build_corpus.py`.

---

## 1. Where the project is

**W2 (Aug 10–16), Phase 1 — Retrieval gate (Aug 3–23) → C1, Table 1.**

| Gate | Date | State |
|---|---|---|
| **G0** — 8B AWQ generator chosen, A4000 measured | Aug 4 | **PASSED 2026-08-04.** Issue #1 closed. |
| **G1** — hit@5 ≥ 0.90, Wilson lower > 0.85 | **Aug 23** | **Unstarted, on plan.** The retrieval stack now exists; G1 is W3. |
| G2 · G3 · G4 · G5 | Sep 6 · Sep 20 · Sep 27 · Oct 11 | Unstarted, with due weeks. |

**W1 is complete.** **W2's code half is complete and verified; W2's execution half is A4000-side.**
Five of the nine W2 items are done (chunker-sweep driver, `retrieve.py`, RRF, `backends.py`,
Issue #10). The other four — the 2M encode, the confusability probe, the title-convention
measurement, Table 1 rows 1–3 — are **runs, not code**: every script is written, `--help`-clean and
smoke-tested against real MedCPT weights, and each one waits only on the box. §8 holds the
copy-paste block, in order.

Everything past retrieval is `NotImplementedError` with a due week in its module docstring. That is
by design, not drift. The exceptions are `scoring/abstention.py`, `retrieve.py` and `backends.py`,
which now exist and are tested.

Unresolved, and **not needing re-derivation**: **W9 is triple-booked** (`research_roadmap.md` §5 ⚠)
· the blind parity loop leaves six days between the first citation-F1 (≈Aug 31) and G2 (Sep 6)
· `SPLIT_SEED = 20260807` while the draw happened Aug 5 — **left alone deliberately**; the constant
documents the deadline it was written against, and changing it would change the draw for nothing.

---

## 2. What is blocking, and the corpus as built

**Nothing is blocking agent-side.** What is *waiting* is the box: nothing downstream of the 2M encode
can produce a number until it runs.

### The corpus (built 2026-08-06 on the A4000)

| | |
|---|---|
| `fingerprint` | **`93321598f3f1`** — this is the corpus. The earlier `41cf7a6c9160` is the duplicate-bearing draw and **must not be used**. |
| `gold collisions` | **1,000 of 1,000.** Every PubMedQA gold PMID is in MedRAG. The overlap ADR-0012 §1 guessed at is **total** — without draw-time exclusion every gold abstract would have been indexed twice. |
| `gold in the draw` | **0 of 1,000** — verified this session, and this is the *design*, not a defect. See trap 12; it is the reason `encode_corpus.py` indexes gold itself. |
| `duplicate rows` | **300 suppressed** over 244 PMIDs (trap 3). |
| scanned | 23,898,701 — exact |
| artifacts | `data/corpus/corpus_manifest.json` (tracked) · `corpus.jsonl` 5.5G, `prescan.jsonl` 5.6G (gitignored, **box only**, 12 GB total) |

`load_splits()` returns `{dataset, seed, dev, test, hash}`; `dev` is 100 pubid **strings** and the
split hash is **`71c46cc5b0ca`**.

The scan ran once and completed; the draw was then redone from its on-disk superset via
`--from-prescan` after the repeated-PMID fix. **Keep `prescan.jsonl` on the box until the encode is
done** — it is the only way to redraw without a second 54 GB read.

> **`--from-prescan` will refuse on the box until one number is added to the manifest.** Since
> `198172c` the redraw matches `prescan.jsonl`'s row count against a `n_prescan_rows` field, and the
> manifest on the box predates the field. **Only if that prescan has not been re-scanned since the
> Aug 5 run**, record it:
>
> ```
> wc -l data/corpus/prescan.jsonl
> ```
>
> then add `"n_prescan_rows": <that number>` to `data/corpus/corpus_manifest.json` on the box. A
> count taken *after* a truncation certifies the truncation, which is why the code refuses rather
> than back-filling it automatically. The tracked copy of the manifest carries no `prescan.jsonl`
> and does not need the field.

**If the corpus is ever rebuilt from scratch**, budget **~3 h wall**, not the 1 h first estimated — a
JSON parse per row sits on top of the network. **Network to HF is flaky**: two read timeouts hit
retry 2/5 on the 2026-08-05 run, the budget is 5 retries per file, and the build is **not resumable**
(trap 2), so a rebuild is a real risk rather than a formality. Four guards can stop it — row count ≠
23,898,701 · zero gold collisions · short heap · a draw short on distinct PMIDs. **Each one means a
corpus that must not be encoded.** Ask for the traceback rather than working around it.

---

## 3. Reasoning that the diffs and ADRs do not carry

`git log` and `docs/adr/` hold the rest; this is only what neither recovers.

- **The dedup is exclusion at draw time, not reconciliation afterwards.** There is never a moment
  when two copies of a gold abstract exist, so nothing has to decide which survives and no
  half-deduped index can reach the encoder. **Its cost is trap 12** — the index has to source gold
  from somewhere, and that somewhere is `encode_corpus.py`, not the corpus file.
- **The surviving gold copy is PubMedQA's, not MedRAG's**, and that is forced, not preferred:
  citations are char spans into `Instance.abstract_text`, so indexing MedRAG's string would
  invalidate every gold offset and every annotation record written against one. The same fact is why
  `_iter_passages` chunks gold via `chunk_instance` rather than re-fetching it.
- **Axis 3 reversed its own shape under measurement.** It began as "titles on both sides or neither."
  The measurement that killed the middle option: **a PubMedQA question is its article's title,
  verbatim** — 60 sampled gold PMIDs, title covers the question's content tokens at **median and mean
  1.00, 60/60 at ≥ 0.8** (`Instance.question` vs NCBI esummary, 2026-08-05). ADR-0003 called
  retrieval here "a lexical gimme"; it is stronger than that. **Neither is the only reachable parity.**
- **The two MedCPT title conventions are not equivalent** — measured this session on real weights:
  same passages, `tok(["", text])` vs `tok(text)`, **max abs embedding difference 0.0349**. ADR-0014
  §3's open question is therefore a real measurement and not a no-op, which was not known before.
- **Citations/claim measures 1.01, not the 2–3 assumed** (89 of 92 G0 claims cite exactly one; the G0
  prompt did not suppress multi-citation). But **G0's passages were sections of one abstract**, which
  is not the retrieval regime. **ADR-0013 KW2: re-measure on the W4 end-to-end records.** This is the
  single largest uncertainty left in the annotation plan.
- **Per-question annotation cost is sublinear in claims; per-claim cost is linear** (2 sampled claims
  cite 1.52 distinct passages, 4 cite 2.03; 4,000 resamples/question over the G0 answers). This is
  *why* 2 claims/question beats ADR-0011's 4×19 at equal cost.
- **How the corpus failure was diagnosed, because the method transfers.** The write step raised
  `wrote 2,000,000 rows for 1,999,703 drawn PMIDs`, whose message named **two** possible causes. The
  guard's *direction* discriminated them without another 3 h run: a prescan that failed to contain
  the draw writes *fewer* rows than PMIDs, more means the source holds repeats. Everything after came
  from the on-disk `prescan.jsonl`, not the network — four throwaway scripts, each narrowing:
  how many repeat (244/2,041,867) → do their rows differ (129 yes) → **is the difference chunking or
  revision** → is the draw contained in the superset (exact, not statistical). **Step 3 was nearly
  skipped as unnecessary and was the one that mattered**: had the rows been chunks of one abstract,
  ADR-0014 §2, `chunk.py`'s input contract and `passage_text` would all have needed revisiting.
  **Look before assuming the cheap fix is the right one.**
- **Sub-agent completion reports are claims, not evidence.** Five W2 slices were delegated in
  parallel and every one came back "verified". Checking the artifacts myself found, in code reported
  as working: four names used at runtime and never imported in `retrieve.py`; an encoder writing
  `embeddings.npy` against a loader reading `dense.npy`; `passage_texts.jsonl` never written at all;
  and three sweep configs that would have collapsed onto one. **The delegation was still right — the
  verification is the part that cannot be delegated with it.**
- **The repo went PUBLIC on 2026-08-05.** Any earlier note calling issues "internal records the
  annotators cannot read" is **dead**. **No names, emails or PII appear in any tracked file or
  issue — checked, not assumed.**
- **Both annotators accepted on 2026-08-05** (user-reported). Issue #7's recruiting deliverable is
  met, so **ADR-0011 §1's prohibition on revising the ask upward is live.**

**ADR house rule.** The default is still that accepted ADRs are not edited — supersede instead. **One
narrow exception exists and is written down in `docs/agents/domain.md`:** when a *premise* inside an
accepted ADR is wrong but the *decision* it supported is unchanged, a dated in-place amendment is
allowed, under three conditions (original text stays with a pointer; the amendment says what did not
change; the header records the edit). Used exactly once — **ADR-0014 §2's Amendment, 2026-08-06**.
ADR-0011 §1's now-stale open note is the provenance of ADR-0013 and is left standing on purpose.
**Do not "tidy" it.**

---

## 4. The thing owed to a person — sent 2026-08-06, reply outstanding

**Message to the two annotators, sent by the user on the original channel, not GitHub.** It exists
nowhere in the repo. Its content, so it can be reconstructed — and so a **reply** can be read against
what was actually promised:

- **two sittings, 3 hours total** — ~1 h pilot (guidelines + 10 practice items), ~2 h main pass
- the amount of material is **sized to fit those hours and will not grow**
- **stop whenever — everything finished stays useful**; there is no wasted partial work
- a closing question: what would make it sit easier (timing, sitting length, how it is split)

Three constraints, which outlived the send — they now bind **any follow-up**:

1. It must **never present this as hours going up.** They are not — the W6 pilot was simply never
   inside ADR-0006's ~3 h. It is an uncosted session, not a mis-estimate.
2. **The stop-anytime line is a real guarantee, not reassurance** — see §5.
3. The closing question exists because the annotators raised a worry whose *cause* is unknown. Hours,
   September timing and open-endedness are three different problems and **only the first has been
   priced.**

If either offers more time, **taking it is allowed** — the ceiling binds the project, not them.

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
  `backends.py` reaches it over HTTP via `httpx` at `{VLLM_BASE_URL|http://localhost:8000}` and
  re-raises `httpx.ConnectError` with a runbook pointer.
- **`RAG_Debate_Agent` is retired** (ADR-0007). Never re-run it; cite `docs/harvest/`.
- **Index identity is a content hash**, never a document count (the ADR-0007 lesson). The empty
  title-segment convention (ADR-0014 §3) is part of that identity and goes in the fingerprint.
  `index_fingerprint()` honours this as of `dc69341` — the corpus enters as
  `RetrievalConfig.corpus_fingerprint`, the draw's hash over the 2M PMIDs, because `corpus_id` is
  the literal `"pubmed-2m-v1"` at every seed. **A redraw must change that default**; the test
  pinning it to `data/corpus/corpus_manifest.json` is what fails if it does not.
- **Passages carry no titles, gold or distractor** (ADR-0014 §2). `MEDRAG_TEXT_FIELD = "content"` is
  load-bearing, not a default.
- **MedCPT is asymmetric.** `NCBI/MedCPT-Query-Encoder` for queries, `NCBI/MedCPT-Article-Encoder`
  for passages, and **the title slot never receives the question** (trap 4).
- **≤3 citations per claim**, identical across all three systems.
- **`validate()` reports violations and never repairs them.**
- **`AlignScore` is never-cut** — the middle rung of R7's only remaining ladder, and the cut fired in
  W6, a week before G3 (§8 rule 8).
- **Never tune τ to pass a gate.** R2's ladder ends at relaxing to hit@10 and *saying so in the
  paper*, never at moving a threshold quietly.
- **`SCHEMA_VERSION` stays `1.0.0`; `CONFIG_VERSION` stays `1.2.0`.**
- **Schema field names that have bitten.** `QueryRecord` uses **`query_id`**, not `question_id`.
  `CostRecord` is `run_id, query_id, component, backend, input_tokens, output_tokens, usd, wall_s`.
  `RetrievedPassage` is `passage_id, rank` (1-indexed), `score, retriever, text`. Check the dataclass
  before writing a record; three plausible-sounding alternatives are all wrong.
- **The annotator ask is a ceiling, never revised upward** (ADR-0011 §1, live since both accepted).
- **The stop-anytime guarantee has a live dependency in W5.** It is true *only* because ADR-0013 §3
  puts annotators 2 and 3 on the **same randomized question order**, so any common prefix is a
  complete unbiased subsample. That requirement lands on `data.py` and the annotation UI in W5. **If
  it is dropped or simplified there, a sentence the user has already sent becomes false.**
- **The repo is public.** Outward-facing means the whole repo, not just issues. **Nothing
  outward-facing goes out without the user's word, and push authorization does not carry across
  sessions.**

### Dates set outside the repo — record them, they are nowhere else

- **Annotator reply: hard backstop Thu 2026-08-20.** The message was due Fri Aug 7 and sent Thu
  Aug 6; the backstop now applies to the *answer*. It was derived backwards from the closing
  question, not the schedule: a bad answer is recovered by finding a replacement against R3's
  **Sep 7** hard trigger, whose fallback (intra-annotator α, 150 claims) is explicitly weaker.
  **Silence past Aug 20 costs the same as a bad answer** — worth a nudge around Aug 19.
- **All labeling ends Sun 2026-09-20.** Pilot **Sep 7–13 (W6)**, main pass **Sep 14–20 (W7)**;
  **W8 (Sep 21–27) is α, adjudication and G4 — not annotation time.**
- **The dependency that makes it hold: the pilot must actually happen in W6.** If it slips, the main
  pass slides into W8 and ADR-0011's α < 0.6 branch loses its only re-run. **The gap between the two
  sittings is load-bearing, not padding** — the pilot exists to test the guidelines, so the guideline
  revision happens in that gap.

### How this user works

- **One decision at a time, lettered, each with a recommendation.** Replies are terse, often a single
  letter or two words. **A one-word answer may address only part of a multi-part question** — re-ask
  the remainder rather than assuming. This has fired in five sessions and asking was correct each time.
- **`do it` / `go` means "the next thing you just named."** **Name the next action explicitly at the
  end of every turn**, or that instruction is ambiguous.
- **Terse instructions are decisions, not openings for discussion.** When one cuts against a repo
  convention, the accepted pattern is: **state the conflict in two sentences, do it anyway, and do it
  in the shape that preserves the convention's purpose.** Do not stop and re-ask.
- **Brevity while driving the box.** "Just tell me what to do in very brief" — lead with the command,
  keep reasoning to what changes the next action. This never meant the reasoning should go
  unrecorded: the commit messages are long and none has been queried.
- **Look facts up; ask only decisions.** Every sharp finding across six sessions came from reading a
  file, fetching a shard, querying NCBI or running the code — none from reasoning.
- **Argue against your own earlier recommendation when the evidence changes.** Reversals have been
  accepted immediately every time. Do not defend a prior position.
- **The A4000 is copy-paste only.** No SSH from the agent environment; `scp` to `vllm-box` fails with
  `Permission denied (publickey…)`. Hand over commands, wait for pasted output. **Never inspect
  `~/.ssh/`** — declined once. **Run throwaway diagnostics through a syntax check before handing them
  over**; a one-line slip costs a full round trip.

### The box's environment

- **Git auth uses a fine-grained PAT** (Contents: read/write). `credential.helper store` may have
  been run, which writes it in plaintext to `~/.git-credentials`. **Never read or echo that file.**
- **`git config --global user.email` on the box differs** from the address on the user's other
  commits. Unconfirmed whether it was changed; commits pushed from the box may not attribute
  correctly. Cosmetic, but worth one check rather than a surprise in November.
- **This laptop has no CUDA** (`WARNING: CUDA not available; encoding on CPU`). Both MedCPT encoders
  are now in the local HF cache, so smoke tests re-run in seconds. `transformers` warns
  `torch_dtype is deprecated! Use dtype instead` from `_load_model` — cosmetic, worth switching.

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
   an existing manifest already records a completed 23,898,701-row scan **and a `n_prescan_rows` that
   still matches the superset on disk** (`198172c` — §2). The split itself is unresolved design debt,
   noted in `corpus.py`'s module docstring; Issue #10 finding 5 left it there deliberately, because
   it is a modelling decision and not a bug.
3. **The repeated PMID.** PubMed re-publishes revised records and MedRAG keeps each revision as its
   own row — **244 of 2,041,867 drawn PMIDs, twice inside a single shard in at least one case.**
   Duplicates share a `selection_key`, so both copies enter the bottom-k together and the draw
   quietly becomes 2M rows over 1,999,703 articles: **one abstract under two `passage_id`s**, which
   is ADR-0012 §1's failure arriving from inside MedRAG rather than from gold. The revisions are
   whole abstracts, not chunks (`22367489` is `b-subunit` vs `beta-subunit`; `22453897` appends an
   abbreviation list), so **ADR-0014 §2's "a row is an article" holds** — its unstated "and appears
   once" does not, and its 15,377/15,377 was one shard. One row survives per article: **longest
   `content`, ties on smallest `id`.** Recorded in **ADR-0014 §2's Amendment (2026-08-06)**. The
   manifest key is now **`n_duplicate_rows_in_draw`**, renamed rather than recounted: the counter
   only sees PMIDs already in the running bottom-k, so 300 is a **lower bound**, and the rename makes
   the manifest honest without touching the draw. **The 26 MB committed manifest was left alone** —
   only newly written manifests carry the new key.
4. **Indexing the question.** `scripts/g0_medcpt_throughput.py:46` puts `row["question"]` in MedCPT's
   title slot as a throughput stand-in. **Copying that into the real encode would index the query
   against itself.** The title slot never receives the question.
5. **Fetching gold titles to "fix" the title asymmetry.** The intuitive repair, and the one option
   that is definitely wrong — the titles *are* the questions (§3).
6. **A `pytest.skip` on a missing fixture is invisible in a green run.** The G0 records were under
   gitignored `runs/`, so the only copies were on this laptop and `tests/test_abstention.py` was
   silently skipping its three load-bearing tests on every other machine. **They now live in
   `docs/harvest/g0/`** and `g0_generator_bakeoff.py` writes there. Those tests assert the abstention
   predicate against both real record shapes (Llama's list-item form, Qwen's trailing prose); if
   either fails, ADR-0010's decision to hold `SCHEMA_VERSION` at 1.0.0 must be revisited **before
   Sep 7**. **ADR-0010's Validation section and ADR-0013's evidence table still cite `runs/g0/`** —
   left stale on purpose (a path is neither a wrong premise nor a changed decision, so the
   `docs/agents/domain.md` exception does not cover it). Read them as `docs/harvest/g0/`.
7. **`docs/` is gitignored** via `docs/*` with `!docs/adr/`, `!docs/harvest/` and `!docs/agents/` as
   the **only** exceptions. Docs written anywhere else are **silently untracked** — verify with
   `git check-ignore`. `docs/harvest/` therefore holds two unrelated things and its README says which
   rules govern which. **Any new exception needs the same two-line pair**: the negation, and a note
   saying what it exists for. `chunker_sweep.py` writes `docs/harvest/chunker_sweep.json`, which is
   inside an exception and therefore tracked — deliberately.
8. **VRAM drifts on the A4000** (WDDM, display attached). Always launch vLLM with
   `--gpu-memory-utilization 0.85` and `VLLM_USE_V2_MODEL_RUNNER=0`. **Do not install an NVIDIA
   driver inside WSL** — the Windows driver is passed through.
9. **There is no box preflight script any more.** `scripts/g0_smoke.sh` was deleted 2026-08-06: it
   never ran successfully, assumed a POSIX login shell against a box that answers with `cmd.exe`, and
   its job was gating G0. **Its thresholds are the right ones** if one is ever rewritten: ≥ 10 GB
   free VRAM of 16 (8B AWQ ~6 + MiniCheck-770M ~1.5 + cross-encoder ~1.3, co-resident at G3) and
   ≥ 60 GB free disk. Recover with `git show c213b4d:scripts/g0_smoke.sh`.
10. **Scratchpad handoffs get deleted.** Three already have been. **The tracked root file is the
    convention**; `/handoff` writes to the OS temp directory by default, so fold its output in here
    and delete it.
11. **A guard can be unreachable by construction and still read as a guard.** `--from-prescan`
    checked that every drawn key fell below the prescan cutoff — but the prescan only ever wrote
    rows already under that cutoff, so the comparison could not fail, and it printed *"draw sits N%
    inside the cutoff"* on the way to certifying nothing. **The tell was that the guard's input came
    from the same place its subject did.** `draw_corpus` had the same shape in the redraw path: it
    was handed the surviving rows as its own `expected_rows`, so the row-count guard compared the
    file to itself. Both are fixed (`198172c`). **When reading a guard, ask what varies that it
    could see** — here nothing did, and the thing that actually varied (how much of the 12 GB
    superset was still on disk) had no check at all.
12. **The corpus contains no gold, and an index built from it alone scores hit@5 = 0.00.** Verified:
    2,000,000 drawn PMIDs, 1,000 gold PMIDs, **0 in common.** That is ADR-0012 §1 working — gold is
    excluded *at draw time* so no abstract is indexed twice (trap 3's failure from the other
    direction). The consequence is easy to miss and was: hit@5 is defined over
    `Instance.gold_passage_ids`, which is `f"{pubid}:{i}"`, while `corpus.jsonl` chunks into
    `f"{row_id}:{i}"` — **so an index over the corpus alone contains no id in the gold space at all,
    and every configuration scores exactly 0.00 while reading as a broken retriever.** The fix is
    inside the encoder: `_iter_passages` yields gold from `chunk_instance(load_instances())` **first**,
    then distractors, so a `--limit` run still contains all of it. `--no-gold` exists to reproduce the
    empty case on purpose; **the default is gold ON**, and `with_gold` is in the resume guard because
    resuming across it would splice two id spaces into one `dense.npy`.
13. **A missing file that does not raise.** `RetrievalIndex.load` looks `dense.npy` up **by name** and
    leaves `dense_embeddings = None` when it is absent. The encoder wrote `embeddings.npy`. Nothing
    raised — the index loaded cleanly, retrieved plausibly, and Table 1's dense and RRF rows would
    have been **silently BM25-only**. The sibling bug: `passage_texts.jsonl` was written only under
    `--build-bm25`, and with empty texts `_rerank` falls back to `p.text or p.passage_id` and scores
    the query against the literal string `"pubmed23n0001_0:0"`. Both fixed; a length-agreement guard
    now `sys.exit`s if ids/texts/embeddings disagree. **Where a loader resolves by name and tolerates
    absence, the writer's name is not a preference.**
14. **A summary in the slot where the value should be.** `table1_baseline.py` wrote a key literally
    named `gold_rank` — holding `{rank_mean, rank_median, rank_distribution, …}`, with the 100
    per-question ranks discarded. It passes every eye test: the rule says "store `gold_rank`", and a
    key called `gold_rank` was there. But `rank_mean` cannot be re-thresholded at hit@10 (**R2's
    ladder ends exactly there**) and cannot be bootstrapped clustered on the question (ADR-0011), and
    recovering it costs an index load plus 3x100 retrievals. `title_convention_eval.py` had the worse
    version: both conventions run the **same** 100 questions, so the test that answers ADR-0014 §3 is
    *paired*, and two independent means cannot support a paired test at all — after two full index
    builds. Both now emit `gold_rank_per_query` keyed by `query_id`, and Table 1 additionally writes
    the ranked lists to `<out>.records.jsonl` (300 records). **Caught by dry-running the scripts
    against a real 1,200-passage index before handing them to the box, not by reading them.**

---

## 7. What to read — the shortest ordered list

1. **`docs/adr/0014`, `0013`, `0012`** — the corpus's text form and source, the annotation budget,
   the distractor pool. The answer to almost any "why is it like this?" about W1–W2.
2. **`src/biomedqa/corpus.py`'s module docstring** — the longest-form write-up of traps 1–3, and the
   only surviving record of the session-3 findings.
3. **`scripts/encode_corpus.py`'s `_iter_passages` docstring** — the long form of trap 12.
4. `CONTEXT.md` — the four frozen units and the annotation protocol; authoritative on the units.
5. `research_roadmap.md` §3 (distractor selection), §5 (the week grid), §7 (R1–R7), §8 rules 8 and 10.
6. `docs/adr/0009`–`0011` — parity, abstention, the gold set.
7. `docs/adr/0003`–`0008` — the decisions the newer ones refine.
8. `docs/agents/domain.md` — the ADR conventions, including the amendment exception.
9. `src/biomedqa/schema.py` — the frozen contract, still **1.0.0** and deliberately so.
10. `src/biomedqa/retrieve.py`'s docstring — every settled W2 constraint, and the one open question.
11. `docs/harvest/runbooks/wsl-vllm-a4000.md` — before touching the box.
12. `paper/skeleton.md` — the five tables and the C1–C5 ledger every result must land in.

Not needed: `docs/project2_biomedical_attribution_rag_implementation_plan.md` (superseded,
banner-marked) · `notebooks/` (toy/simulated; `07_4` simulates 3 labels where `CONTEXT.md` freezes 4
— a correctness bug, not a scale assumption).

---

## 8. Open work, in the order recommended

### 1. Commit and (with the user's word) push the W2 code

Nothing from this session is committed. Six new scripts, four modified modules, one modified test.
**Push authorization does not carry across sessions** — ask.

### 2. The A4000 block — this is the whole remaining W2

Copy-paste, in order. Each step's output is the input to the next; nothing here can be run from this
laptop.

```
# (a) the real index — gold + 2M distractors, ~1.6 h, resumable
python scripts/encode_corpus.py --title-convention empty --build-bm25 \
  --out data/index/empty --resume

# (b) the second index, for ADR-0014 §3 only — same corpus, other convention
python scripts/encode_corpus.py --title-convention single --build-bm25 \
  --out data/index/single --resume

# (c) ADR-0014 §3: dev hit@5 both ways. Decided by measurement, not taste.
python scripts/title_convention_eval.py --index-dir-empty data/index/empty \
  --index-dir-single data/index/single

# (d) Table 1 rows 1-3 — BM25, dense, RRF
python scripts/table1_baseline.py --index-dir data/index/empty

# (e) ADR-0012 §2's distractor confusability probe, ~100 dev questions
python scripts/confusability_probe.py --index-dir data/index/empty
```

`--resume` is safe to re-issue after any interruption: resume is verified **byte-equivalent** on real
weights, and the guard refuses a resume whose `title_convention`, `strategy`, `max_chars`,
`window_sentences`, `stride_sentences` or `with_gold` differs from `progress.json`.

The chunker sweep (`scripts/chunker_sweep.py`, 7 named configs) is a **separate, longer** run — it
encodes once per config. Run it with `--sample` first; **a number measured against a truncated corpus
is not a Table 1 row**, and the script prints that warning itself.

### 3. Once (c) has a number

Decide whether the winning title convention is recorded in `RetrievalConfig` and noted in ADR-0014
§3. It is already inside `index_fingerprint()`, so this is documentation of a settled question, not a
behaviour change — but ADR-0014 §3 is currently *open* and should not stay open past a measurement.

### 4. The annotator reply

User-side, undated, **backstop Thu 2026-08-20** (§5). Not a blocker on anything agent-side. Nudge
around Aug 19.

### 5. Deferred by the user's own triage to W4–W5

Words/claim vs claims/query as the gated quantity (ADR-0009 Known weaknesses) · the Sep 3 freeze ·
the guideline two-pass calendar.

### 6. Optional, user-side

`g0_medcpt_throughput.json` exists only on the box. Every number in it is transcribed into
`research_roadmap.md` §3, so losing it costs nothing — but if convenient, `scp` it into
`docs/harvest/g0/` beside the two bake-off records.

---

**Suggested skills.** `/tdd` for anything that has to fail before it can be trusted — traps 12 and 13
are both cases where the code ran green and produced a wrong number, and neither test could have been
written honestly after the fix. `tests/test_corpus.py` is the house model: each test's docstring names
the silent failure it prevents, and its fixture is a **real MedRAG row embedded verbatim**.
`/grilling` is the user's preferred instrument and produced ADR-0009–0013; live target is the **W9
triple-booking**. `/domain-modeling` for any new ADR, and for the PMID `str`/`int` split (trap 2) if
it is judged worth a type. Not needed yet: `dataviz` (figures are W11) · `claude-api` (the Opus 5
judge is wired in W6).
