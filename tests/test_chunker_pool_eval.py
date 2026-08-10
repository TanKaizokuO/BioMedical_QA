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


def test_reconstructs_an_abstract_the_index_split_into_several_chunks():
    """Taking only `X:0` truncates every long abstract — and long abstracts are the distractors.

    This was a live defect: it weakened the competition, inflated gold by +0.02 hit@5, and read as
    a chunker win until the harness check refused it.
    """
    rebuilt = cpe._reconstruct_abstracts(
        ["A:0", "A:1", "A:2", "B:0"], ["first ", "second ", "third", "other"], {"A"}
    )

    assert rebuilt == {"A": "first second third"}


def test_reconstruction_inverts_the_production_chunker():
    """`_enforce_max_chars` cuts consecutive spans and inserts nothing, so the join is exact."""
    text = "".join(f"Sentence number {i} is here. " for i in range(200))
    stored = cpe.chunk_text(text, "LONG", ChunkConfig())
    assert len(stored) > 1, "fixture must exceed max_chars or it tests nothing"

    rebuilt = cpe._reconstruct_abstracts(
        [c.passage_id for c in stored], [c.text for c in stored], {"LONG"}
    )

    assert rebuilt["LONG"] == text


def test_chunks_are_ordered_by_index_not_by_position_in_the_index_file():
    """A shard written out of order must not silently scramble the abstract."""
    rebuilt = cpe._reconstruct_abstracts(
        ["A:2", "A:10", "A:1"], ["three", "ten", "one"], {"A"}
    )

    assert rebuilt == {"A": "onethreeten"}


def test_audit_flags_a_pool_whose_abstracts_have_unpooled_siblings(tmp_path, records_file):
    """Re-chunking such an abstract adds a competitor row 4 never scored, so exactness is lost."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "passage_ids.json").write_text(
        json.dumps(["ccc:0", "ccc:1", "ddd:0", "eee:0", "unrelated:0"]), encoding="utf-8"
    )

    report = cpe.audit_pool(index_dir, records_file, row=4)

    assert report["pooled_abstracts"] == 3
    assert report["abstracts_with_unpooled_siblings"] == 0  # "ccc:1" is itself pooled
    assert report["harness_check_can_be_exact"] is True


def test_audit_is_exact_only_when_every_sibling_was_pooled(tmp_path, records_file):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "passage_ids.json").write_text(
        json.dumps(["ccc:0", "ccc:1", "ddd:0", "ddd:1", "eee:0"]), encoding="utf-8"
    )

    report = cpe.audit_pool(index_dir, records_file, row=4)

    assert report["abstracts_with_unpooled_siblings"] == 1
    assert report["extra_candidates_introduced"] == 1
    assert report["harness_check_can_be_exact"] is False



def _arm(pairs, hit5):
    """An 'abstract' arm carrying `(gold_rank, reference_gold_rank)` per query."""
    return {
        "chunker": "abstract",
        "hit_at_k_upper_bound": {"hit_at_5": {"point": hit5}},
        "gold_rank_per_query": [
            {"query_id": str(i), "gold_rank": got, "reference_gold_rank": ref}
            for i, (got, ref) in enumerate(pairs)
        ],
    }


def test_harness_demands_equality_when_the_audit_says_nothing_was_added():
    arm = _arm([(1, 1), (3, 3), (None, None)], hit5=0.86)

    assert cpe._harness_check(arm, 0.86, exact=True)["passed"] is True
    # Same aggregate, but a gold rank moved: with no new candidates in play, nothing may move.
    moved = _arm([(1, 1), (4, 3), (None, None)], hit5=0.86)
    assert cpe._harness_check(moved, 0.86, exact=True)["passed"] is False


def test_added_candidates_may_demote_gold_but_never_promote_it():
    """The whole point of the restated rule: demotion is expected, promotion is impossible."""
    demoted = _arm([(1, 1), (9, 3), (None, None)], hit5=0.85)
    check = cpe._harness_check(demoted, 0.86, exact=False)
    assert check["passed"] is True
    assert check["queries_gold_demoted"]["n"] == 1

    promoted = _arm([(1, 1), (2, 3), (None, None)], hit5=0.86)
    check = cpe._harness_check(promoted, 0.86, exact=False)
    assert check["passed"] is False
    assert [q["query_id"] for q in check["queries_gold_improved"]["queries"]] == ["1"]


def test_gold_found_where_row_4_never_found_it_is_a_promotion():
    """`None` is not a neutral value: finding gold that row 4 missed means the pool differs."""
    arm = _arm([(1, 1), (7, None)], hit5=0.86)
    check = cpe._harness_check(arm, 0.86, exact=False)

    assert check["passed"] is False
    assert check["queries_gold_improved"]["n"] == 1


def test_an_arm_above_row_4_fails_even_with_no_per_query_promotion():
    """A hit@5 above row 4's without a single promoted rank is arithmetically impossible, so it
    means the two are being computed over different query sets — refuse rather than reconcile."""
    arm = _arm([(1, 1), (3, 3)], hit5=0.90)

    assert cpe._harness_check(arm, 0.86, exact=False)["passed"] is False


def _inst(text="Aims here. Methods here. Results here.", spans=None):
    from biomedqa.data import GoldPassage, Instance

    spans = spans or [(0, 10, "BACKGROUND"), (11, 25, "METHODS"), (26, len(text), "RESULTS")]
    return Instance(
        pubid="p1",
        question="q?",
        passages=[
            GoldPassage(
                passage_id=f"p1:{i}",
                pubid="p1",
                index=i,
                label=label,
                text=text[a:b],
                char_start=a,
                char_end=b,
            )
            for i, (a, b, label) in enumerate(spans)
        ],
        long_answer="",
        final_decision="yes",
    )


def test_section_cuts_gold_unlike_every_distractor():
    """The disqualifying asymmetry, caught by the check rather than by a reader's memory.

    MedRAG rows carry no section labels, so `chunk_text(sections=None)` degrades 'section' to
    'abstract' for every distractor while gold splits on its real boundaries. That is ADR-0014 §2's
    rejected signal, and it is what made the `section` arm read 0.94.
    """
    report = cpe.gold_cut_asymmetry(cpe.SWEEP["section"], [_inst()])

    assert report["cuts_gold_unlike_distractors"] is True
    assert report["abstracts_cut_differently_than_a_distractor"] == 1
    assert report["gold_chunks_via_chunk_instance"] > report["gold_chunks_via_distractor_path"]


@pytest.mark.parametrize(
    "name", [n for n in cpe.SWEEP if n != "section"]
)
def test_every_other_strategy_cuts_gold_exactly_as_it_cuts_a_distractor(name):
    """One splitter cuts both, which is `chunk.py`'s stated invariant. Only 'section' breaks it."""
    report = cpe.gold_cut_asymmetry(cpe.SWEEP[name], [_inst()])

    assert report["cuts_gold_unlike_distractors"] is False
    assert report["gold_chunks_via_chunk_instance"] == report["gold_chunks_via_distractor_path"]