# ADR-0014 — The corpus is read from the JSONL shards, and indexed as title-free abstract prose

**Status:** Accepted · **Date:** 2026-08-05 · **Decided in:** W1 corpus build (Axis 3)
**Amended 2026-08-06** — §2's uniqueness premise was wrong across shards; see *Amendment* under §2.
The decision it supported is unchanged. This is the **first** in-place edit of an accepted ADR in
this repo; it was made on the user's explicit instruction, in preference to an ADR-0015.
**Refines** ADR-0012 §1, which fixed the distractor pool's *source and selection policy* but left
both the **bytes read** and the **text indexed** unstated

## Context

ADR-0012 §1 says "a uniform, seeded 2M sample of `MedRAG/pubmed`". Building it exposed two questions
that sentence does not answer, and both have a wrong answer that leaves every number looking sane:

- **Which copy of `MedRAG/pubmed`?** The dataset ships JSONL shards; the Hub also serves an
  auto-converted parquet export. They are not the same corpus.
- **Which of a row's three text fields?** A MedRAG row carries `title`, `content` and `contents`.

Both are recorded here rather than in ADR-0012 because ADR-0012 is accepted and this project does not
edit accepted ADRs. They are one ADR rather than two because they are the same decision seen twice —
*what text enters the index* — and each is a way to end up with a corpus that must not be encoded.

## Decision

### 1. Read `chunk/*.jsonl`, and refuse any scan whose row count is not 23,898,701

The corpus is read as `data_files="chunk/*.jsonl"`, **never as the bare dataset id**.

The Hub's auto-converted parquet export of `MedRAG/pubmed` is **partial** — 2,209,839 rows of
23,898,701 — and the shards are PMID-ascending, so it is the **oldest ~9% of PubMed**. The trap is
the size: 2.2M is within 10% of our 2M target, so `load_dataset("MedRAG/pubmed")` followed by "take
2M" succeeds, returns the right row count, and yields a corpus of pre-1990 abstracts sitting against
1990s–2010s gold.

That corpus is separable by era and vocabulary alone. **G1's hit@5 would look excellent for the wrong
reason, and G2 would have nothing plausible to mis-cite** — ADR-0003's fatal scenario, reached by the
one path on which no number looks wrong.

Two guards make it loud, both in `corpus.py`:

- `draw_corpus` raises unless the full scan sees exactly `MEDRAG_TOTAL_ROWS = 23_898_701`. Any other
  count means the glob missed shards or resolved to the parquet export.
- **The str/int join is guarded separately.** PubMedQA's `pubid` is int32 and `data.py` stringifies
  it; MedRAG's `PMID` is int64. `{"21645374"} & {21645374}` is empty, so a broken dedup reports
  *"0 duplicates removed"* — which reads as good news. `draw_corpus` raises on non-`int` keys, and
  raises again if a full scan collides with **no** gold PMID at all.

A full scan that trips either guard has produced a corpus that must not be encoded. **Ask for the
traceback; do not work around it.**

### 2. Index `content` — abstract prose, with no title on any passage

`content` is title-free and `contents == title + " " + content`; verified on a real shard
(`chunk/pubmed23n0001.jsonl`, 5,000/5,000 rows, no empty titles, no empty content, 2026-08-05).
A row is a whole article, not a snippet — 15,377 rows over 15,377 distinct PMIDs in that shard.
*(That last clause is qualified by the Amendment below: whole-article holds; one-row-per-article
does not.)*

**Gold is title-free under every option.** PubMedQA has no title field, and the surviving gold copy
is PubMedQA's because citations are char spans into `Instance.abstract_text` (ADR-0005; `corpus.py`).
So the choice was only ever about the distractors, and only one option reaches parity.

Indexing `contents` would make an **empty title the one property every gold passage shares and no
distractor has** — a systematic format signal sitting in exactly the space hit@5 and ADR-0012 §2's
confusability probe are measured in.

**The tempting repair is the one option that is definitely wrong.** Fetching the gold articles' real
titles and indexing those fails because **a PubMedQA question is its article's title, verbatim**:
over 60 sampled gold PMIDs, the title covers the question's content tokens at **median and mean 1.00,
60/60 at ≥ 0.8** (`Instance.question` vs NCBI esummary, 2026-08-05). ADR-0003 called retrieval here
"a lexical gimme"; it is stronger than that — titled gold makes G1 a string match rather than a
measurement.

#### Amendment, 2026-08-06 — a row is a whole article, but an article is not one row

**The sentence "15,377 rows over 15,377 distinct PMIDs" was measured on one shard, and uniqueness
does not survive across them.** PubMed re-publishes revised records and MedRAG keeps every revision
as its own row, so one article can arrive two or three times — **244 repeated PMIDs in 2,041,867
drawn**, on the 2026-08-05 prescan, **129 of them with differing `content`**, sometimes twice inside
a single shard.

**What this does not change: the decision.** The revisions differ by *added text*, not by chunking —
`22367489` is `b-subunit` against `beta-subunit`; `22453897` is the same abstract with an
abbreviation list appended. Each row is still a whole article, so `content` remains the right field,
`chunk.py`'s input contract holds, and `passage_text` is untouched. **Had they been chunks of one
abstract, all three would have needed revisiting** — which is why the classifier that decided
revision-vs-chunk was worth writing rather than assuming.

**What it does change: the draw.** Duplicates share a `selection_key`, so they entered the bottom-k
*together* and the draw became 2M rows over **1,999,703 articles** — one abstract under two
`passage_id`s. That is exactly the miscount ADR-0012 §1 exists to prevent, arriving from **inside
MedRAG** rather than from gold, which is the direction §1's dedup does not look.

**The guards, both in `corpus.py` / `scripts/build_corpus.py`:** `draw_corpus` admits a PMID once and
refuses a draw whose distinct count is short; `build_corpus.py` picks the single row each drawn
article contributes — **longest `content`, ties broken by the lexicographically smallest `id`**.
Longest because a revision that adds text is the more complete record, and both observed cases add
rather than cut. The tie-break exists only to make the rule **total**: `corpus_id` promises
seed → ID list, and a rule that leaves 244 rows to dict ordering does not keep that promise. The
write-step guard that caught this originally named *two* possible causes —
a prescan not containing the draw, or repeats in the source. **Its direction discriminates them**:
fewer rows than PMIDs means the first, more means the second. It now names one.

### 3. The empty title segment is one convention, measured at W2 — not decided here

Dropping titles puts MedCPT's article encoder off its trained *(title, abstract)* pair. That cost is
accepted in §2 because it is paid **symmetrically, by every passage**. It leaves one question open:

**`tok("", abstract)` or single-segment `tok(abstract)`** — one convention, applied to every passage,
**recorded in the index fingerprint**, because it is part of the index's identity. **Pick it by
measuring dev hit@5 both ways at W2.** Not deferred to taste: it is a measurement W2 already has the
apparatus for, and pre-deciding it would be guessing.

**Whatever is chosen, the title slot never receives the question.**
`scripts/g0_medcpt_throughput.py:46` puts `row["question"]` there as a throughput stand-in; copying
that into the real encode would index the query against itself.

#### Decided, 2026-08-10 — `empty`, and the index already built is the winner

**`tok("", abstract)`.** Measured on the dev split against the 2.16M index:
`docs/harvest/title_convention_pool_eval.json`.

| | hit@1 | **hit@5** | hit@10 | hit@20 | hit@50 | hit@100 |
|---|---|---|---|---|---|---|
| `empty`  | 0.32 | **0.59** | 0.70 | 0.74 | 0.81 | 0.91 |
| `single` | 0.35 | **0.53** | 0.64 | 0.69 | 0.79 | 0.91 |

Paired on gold rank over the 91 queries whose gold is in the pool: `single` ranks it **better on 19,
worse on 39**, unchanged on 33 — sign test **p = 0.012**, mean rank delta **+3.68 places worse**. The
two conventions are close but not equivalent: mean cosine 0.9797, max abs component diff 0.0649.

**Read the paired test, not the two intervals.** Marginally, hit@5 is 0.59 [0.492, 0.681] against
0.53 [0.433, 0.625] — CIs that overlap heavily, which is what paired data looks like when summarised
marginally. The queries and the candidates are identical across arms, so the paired rank test is the
one with the power here, and it separates.

**Measured the cheap way.** `scripts/title_convention_pool_eval.py` re-ranks the 100-deep dense pools
Table 1 already recorded, re-encoding only the 9,832 pooled passages (~1 min) instead of the two ~2 h
index builds `title_convention_eval.py` needs. Its `empty` arm re-derives Table 1 row 2 from the same
vectors and must reproduce it exactly; `--expect-hit5 0.59` fails the run otherwise, and it passed.

**Scope, stated.** The candidate set is fixed to `empty`'s pool, so recall is held constant by
construction — both arms sit at hit@100 = 0.91 — and this cannot see a passage `single` would have
pulled in from 2M that `empty` missed. That gap does not matter for the decision, because the
asymmetry runs the right way: switching would require `single` to win on ordering, and it lost. Had
it won, the full two-index build would have been owed.

**Two consequences beyond the convention.** The 2M index on disk is already `empty`, so §3 closes
with **no re-encode at all**. And **the segment convention is not why dense hit@5 is 0.59** — that
hypothesis is dead, the weak dense row is genuine retriever quality, and the W3 cross-encoder is the
remaining lever, exactly as the Consequences below anticipate.

**The fingerprint gap this exposed.** §3 says the convention is "recorded in the index fingerprint",
and it was not: `RunConfig.index_fingerprint()` hashed `dense_encoder`, which names the *weights* and
not the *call*, so an `empty` and a `single` index — same corpus, same chunker, two separate 2 h
encodes, different vectors — hashed **identically**. `encode_corpus.py`'s resume guard refused to mix
them from its own local state while its comment claimed every one of its knobs was inside the
fingerprint. `RetrievalConfig.title_segment` now is (`CONFIG_VERSION` 1.3.0), with a test.

## Consequences

- **~9% of the corpus text is discarded** — median 88 title chars against 905 of content. Bought
  deliberately: if dev hit@5 comes in under 0.90, **the only levers left are the retriever's**,
  because the corpus text cannot be quietly made easier to pass a gate. R2's ladder is unchanged.
- **The build is not resumable, by design.** A dropped run restarts the 54 GB read (~3 h wall, not
  the 1 h first estimated — a JSON parse per row sits on top of the network). Resumability would mean
  a partial scan could satisfy the row-count guard, which is the guard §1 exists for.
- **The gold-collision count is a first measurement, not just a guard.** It is the first number anyone
  has for the gold/MedRAG overlap, and belongs in the run manifest beside the fingerprint.
- **`MEDRAG_TEXT_FIELD = "content"` is a load-bearing constant**, not a default. `chunk.py` and
  `retrieve.py` are written against title-free passages.
- **W2 gains one measurement** — the §3 title-segment comparison — on a week that already holds the
  chunker sweep, `bm25s`, RRF, the 2M encode and ADR-0012 §2's probe.

## Alternatives rejected

- **`load_dataset("MedRAG/pubmed")` and trust the row count.** The partial export; §1.
- **Index `contents` (title + abstract) for distractors.** Empty title becomes a gold-only signal.
- **Fetch the gold articles' real titles from NCBI and index titled text on both sides.** Titles are
  the questions; G1 degenerates to a string match. This was the intuitive repair, and measuring it
  reversed the decision's shape — from "titles on both sides or neither" to "**neither is the only
  reachable parity**".
- **Index `title` alone, or a title-weighted field.** Same defect as `contents`, more sharply.
- **Reconcile duplicate gold copies after the draw rather than excluding at draw time.** Leaves a
  moment when two `passage_id`s carry one abstract, and something must then decide which survives;
  see ADR-0012 §1 and `corpus.py`.
- **Make the build resumable to cut the 3 h.** Trades the row-count guard for wall time.
