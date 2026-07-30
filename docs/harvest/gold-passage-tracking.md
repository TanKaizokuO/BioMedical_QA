# Harvested — gold-passage tracking fields (`e936d30`)

For `src/biomedqa/schema.py` and `scoring/retrieval.py`. Source: `rag_baseline.py` @ `e936d30`.

## What the base repo wrote per query

```python
retrieved_ids = [p["id"] for p in result["retrieved_passages"]]
gold_retrieved = q_id in retrieved_ids
gold_rank = (retrieved_ids.index(q_id) + 1) if gold_retrieved else None

record = {
    "query_id": q_id,
    "retrieved_passages": [...],   # {id, passage, score}
    "gold_pubid": q_id,
    "gold_retrieved": gold_retrieved,   # bool
    "gold_rank": gold_rank,             # 1-indexed, or None
    "overlap_caveat": True,
}
```

Two things here are right and worth keeping:

- **Rank is recorded, not just the hit.** `gold_rank` is the least-processed value — hit@1, hit@5,
  MRR, and nDCG are all derivable from it, and none of them are derivable from each other.
- **Gold identity is carried on the record**, so scoring never has to re-join against the dataset.

## What changes here

**`gold_retrieved` is not stored.** It is `gold_rank is not None and gold_rank <= k` — a
binarization at a fixed `k`, computed at write time, which is exactly what the least-processed-value
rule forbids (`research_roadmap.md` §2). Storing it fixes `k = 5` into the run and makes the
hit@10 fallback in G1's escalation ladder a re-*run* rather than a re-*score*. Store `gold_rank`;
let `scoring/retrieval.py` threshold it.

**`gold_pubid == query_id` stops being an identity.** In the base repo the retrieved unit was the
whole abstract and its Chroma id was the `pubid`, so "did we retrieve the gold?" was a string
equality. Under chunking, one abstract becomes many passages with ids like `{pubid}:{chunk_idx}`,
and gold membership is a **set**:

```
gold_passage_ids(pubid) = { chunk ids derived from that pubid's abstract }
gold_rank = min rank over that set, or None
```

This makes `gold_rank` well-defined per `(chunker, τ)` and nothing else — which is the point.

**Distractors change the meaning of a miss.** With 2M abstracts indexed, a miss is a real retrieval
failure rather than an indexing bug, and the ranks of the *non*-gold passages start carrying
information (they are the plausible-but-wrong passages citation precision is measured against). Keep
every retrieved passage's id and score, not only the gold's.

**`overlap_caveat` is dropped.** It flagged that the gold document was in the index by design — a
caveat that only made sense when the corpus *was* the gold set. Under ADR-0003 the gold contexts are
1,000 documents among ~2M, and the caveat no longer describes anything.

## The comment worth preserving verbatim

From the fix commit, on why the gold document being in the index is not leakage:

> `overlap_caveat` now correctly reflects that the gold doc IS in the index by design (that's the
> fix) — this is expected/desired, not leakage, since we're testing whether retrieval finds it.

Still true, and still the answer to the reviewer question "isn't your gold passage planted?" Yes,
deliberately: retrieval cannot be evaluated against a corpus that lacks the answer. What changed is
that it is now planted in a haystack.
