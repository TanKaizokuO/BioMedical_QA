"""The pool-restricted chunker evaluation's three load-bearing reductions.

The GPU work in `chunker_pool_eval.py` is a cross-encoder call. Everything that can be *wrong*
about the answer is in the plumbing around it: which abstracts the pool contains, which chunks
count as gold, and which path each abstract is chunked through. All three are silent failures —
each one produces a complete, plausible hit@5 — so each is pinned here.
"""

from __future__ import annotations

import json

import pytest

import chunker_pool_eval as cpe
from biomedqa.config import ChunkConfig
from biomedqa.data import GoldPassage, Instance


def _record(row: int, query_id: str, passage_ids: list[str]) -> dict:
    return {
        "table1_row": row,
        "query_id": query_id,
        "retrieved": [
            {"passage_id": pid, "rank": i, "score": 1.0 / i, "retriever": "rerank", "text": None}
            for i, pid in enumerate(passage_ids, start=1)
        ],
    }


@pytest.fixture()
def records_file(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [
        _record(3, "q1", ["aaa:0", "bbb:0"]),
        _record(4, "q1", ["ccc:0", "ddd:0", "ccc:1"]),
        _record(4, "q2", ["eee:0"]),
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_reads_only_the_requested_row(records_file):
    """Rows 3 and 4 pool the same abstracts, but reading both would double every pool."""
    pools = cpe._pool_by_query(records_file, row=4)

    assert set(pools) == {"q1", "q2"}
    assert "aaa" not in pools["q1"]


def test_collapses_passage_ids_to_abstracts_in_rank_order(records_file):
    """Re-chunking is per abstract: two chunks of one abstract must not be re-chunked twice."""
    pools = cpe._pool_by_query(records_file, row=4)

    assert pools["q1"] == ["ccc", "ddd"]


def _instance(pubid: str = "999") -> Instance:
    first, second = "Aspirin lowers risk. ", "Effect was small."
    return Instance(
        pubid=pubid,
        question="Does aspirin lower risk?",
        passages=[
            GoldPassage(f"{pubid}:0", pubid, 0, "BACKGROUND", first.strip(), 0, len(first) - 1),
            GoldPassage(f"{pubid}:1", pubid, 1, "RESULTS", second, len(first), len(first) + len(second)),
        ],
    )


def test_gold_rank_is_the_first_chunk_of_the_gold_abstract():
    """Any chunk of the gold abstract is gold — membership is a set (`data.py`)."""
    inst = _instance()
    chunks = cpe.chunk_text("other text here.", "distractor", ChunkConfig())
    chunks += cpe.chunk_instance(inst, ChunkConfig(strategy="section"))

    assert cpe._gold_rank(chunks, "999") == 2
    assert cpe._gold_rank(chunks, "absent") is None


def test_gold_is_chunked_with_its_sections_and_distractors_without():
    """`section` splits the gold abstract on real labels; a MedRAG row has none and degrades.

    Chunking gold through the distractor path would silently turn the `section` arm into the
    `abstract` arm for the one abstract whose granularity the measurement is about.
    """
    inst = _instance()
    config = ChunkConfig(strategy="section")
    texts = {"distractor": "One sentence. Two sentence."}

    chunks = cpe._rechunk(["distractor", inst.pubid], texts, inst, config)

    gold = [c for c in chunks if c.source_id == inst.pubid]
    distractor = [c for c in chunks if c.source_id == "distractor"]
    assert len(gold) == 2
    assert [c.label for c in gold] == ["BACKGROUND", "RESULTS"]
    assert len(distractor) == 1
    assert distractor[0].label is None


def test_an_abstract_with_no_resolvable_text_is_skipped_not_faked():
    """A missing index text must drop the candidate, never chunk an empty string into the pool."""
    inst = _instance()

    chunks = cpe._rechunk(["missing", inst.pubid], {}, inst, ChunkConfig())

    assert {c.source_id for c in chunks} == {inst.pubid}
