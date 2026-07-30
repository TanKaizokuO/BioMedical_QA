# Harvested — PubMedQA loading and gold-context extraction

For `src/biomedqa/data.py`. Source: `rag_baseline.py` @ `e936d30`.

## What the base repo established

The dataset is `qiaojin/PubMedQA`, config `pqa_labeled`, split `train` — 1,000 rows, the entire
labeled set. It is small enough to load whole; no streaming.

```python
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
```

Row shape, as actually observed (this is the part worth having written down):

| Field | Content |
|---|---|
| `pubid` | PubMed ID, integer. **The stable instance identifier.** Cast to `str` before use as a key. |
| `question` | The question, derived from the source article's title. |
| `context` | A dict, **not** a string. `context["contexts"]` is a **list of passage strings** — the abstract's labeled sections. |
| `context["labels"]` | Section labels (`BACKGROUND`, `METHODS`, `RESULTS`, …), parallel to `contexts`. |
| `long_answer` | The article's own conclusion sentence. |
| `final_decision` | `yes` / `no` / `maybe` — the accuracy label. |

The gold context extraction, with the defensive fallback the base repo needed:

```python
if "contexts" in row["context"]:
    passage = " ".join(row["context"]["contexts"])
else:
    passage = str(row["context"])   # fallback
```

**The gold passage for question `pubid` is the abstract in that same row.** This is the fact the
whole evaluation rests on, and it is why the fix commit exists: the original code indexed
`pqa_artificial` (a disjoint ~211k synthetic set) while evaluating `pqa_labeled` questions, so no
question's gold abstract was in the index at all.

## What changes here

**`" ".join(contexts)` does not survive.** The base repo flattened each abstract into one string
because its unit of retrieval was the whole abstract. This project's unit is set by `chunk.py`, and
`context["labels"]` is discarded by the join — those section labels are the natural boundary for the
sentence-window and section-level chunkers, and `hit@5` is only defined per `(chunker, τ)` pair.
**Keep `contexts` as a list and keep `labels` alongside it.** Do the joining, if any, in `chunk.py`.

**Character offsets must be preserved.** Citations are `{passage_id, char_start, char_end}`
(`CONTEXT.md`), so `data.py` has to record, for each emitted chunk, its offset within the source
abstract. The base repo never needed this and does not have it.

**The 1,000 abstracts are gold contexts, not the corpus.** They are inserted into the ~2M-abstract
corpus (ADR-0003); they do not constitute it. `INDEX_SIZE = 1000` has no analogue here.

**Splits are frozen, not sliced by iteration order.** The base repo took its eval set as the first
50 rows encountered (`for i, row in enumerate(eval_ds): if i >= EVAL_SIZE: break`). This project
freezes dev (100) and test (~400–500) membership by `pubid` in a checked-in JSON, hashed into every
run manifest (`research_roadmap.md` §3). Iteration order is not a split.
