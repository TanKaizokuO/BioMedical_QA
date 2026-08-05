"""Passage granularity — Table 1 rows, one per `(chunker, τ)`.

**Not yet implemented.** Due W2 (Aug 10–16).

Design constraints that are already settled, so that implementing this is not a design exercise:

- **hit@5 is only defined per `(chunker, τ)` pair** (Lesson 2). A chunker is therefore part of the
  index identity, which is why `ChunkConfig` feeds `RunConfig.index_fingerprint()`.
- **Every emitted chunk carries its char offsets** into `Instance.abstract_text`. Citations are
  `{passage_id, char_start, char_end}` (`CONTEXT.md`); a chunker that loses offsets makes the
  attribution unit unrepresentable.
- **Distractors are abstract prose with no title**, read through `corpus.passage_text` and never by
  indexing a MedRAG field directly. Gold has no title — PubMedQA carries none — so a titled
  distractor is a gold/distractor format difference in the space hit@5 is measured in. The gold
  articles' *real* titles are not the repair: a PubMedQA question is its article's title verbatim
  (measured, `corpus.py`), so titled gold turns G1 into a string match.
- **Section labels survive.** `data.py` keeps PubMedQA's BACKGROUND/METHODS/RESULTS boundaries
  precisely so the section and sentence-window strategies can use them.
- Promoted from `notebooks/02_1_chunking_granularity.ipynb`, which validates against a toy corpus
  using `all-MiniLM-L6-v2` as a stand-in for MedCPT — neither assumption survives 2M abstracts
  (`research_roadmap.md` §0, promotion table).
"""

from __future__ import annotations

from .config import ChunkConfig
from .data import Instance


def chunk_instance(instance: Instance, config: ChunkConfig) -> list:
    """Split one abstract into retrievable passages, preserving char offsets."""
    raise NotImplementedError("W2 — see module docstring for the settled constraints")
