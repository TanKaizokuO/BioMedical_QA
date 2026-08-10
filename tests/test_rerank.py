"""Cross-encoder rerank — Table 1 row 4's only new stage.

The failure this file exists for is silent. `_rerank` used to fall back to `p.text or
p.passage_id`, so an index loaded without `passage_texts.jsonl` would score every question
against strings like `"12345:0"`, return a perfectly well-formed ranking, and report a hit@5
that measures nothing. Row 4 is the row Gate G1 reads, so that number has to be un-fakeable.

The cross-encoder itself is stubbed: what is under test is the contract around it — refuse
text-free pools, rank by descending score, keep the passage text for downstream scoring.
"""

from __future__ import annotations

import pytest

from biomedqa import retrieve
from biomedqa.schema import RetrievedPassage


class _StubCrossEncoder:
    """Scores by a caller-supplied table keyed on passage text."""

    def __init__(self, table: dict[str, float]):
        self.table = table
        self.seen: list[tuple[str, str]] = []

    def predict(self, pairs):
        self.seen = list(pairs)
        return [self.table[text] for _, text in pairs]


@pytest.fixture()
def stub_encoder(monkeypatch):
    def install(table: dict[str, float]) -> _StubCrossEncoder:
        stub = _StubCrossEncoder(table)
        monkeypatch.setattr(retrieve, "_get_cross_encoder", lambda name: stub)
        return stub

    return install


def _passage(pid: str, rank: int, text: str | None) -> RetrievedPassage:
    return RetrievedPassage(passage_id=pid, rank=rank, score=1.0 / rank, retriever="rrf", text=text)


def test_reorders_the_pool_by_cross_encoder_score(stub_encoder):
    """The reranker's job: a passage the fused pool ranked last can come first."""
    pool = [_passage("a:0", 1, "alpha"), _passage("b:0", 2, "beta"), _passage("c:0", 3, "gamma")]
    stub_encoder({"alpha": 0.1, "beta": 0.9, "gamma": 0.5})

    out = retrieve._rerank("q", pool, "stub-model")

    assert [p.passage_id for p in out] == ["b:0", "c:0", "a:0"]
    assert [p.rank for p in out] == [1, 2, 3]
    assert [p.score for p in out] == [0.9, 0.5, 0.1]
    assert {p.retriever for p in out} == {"rerank"}


def test_keeps_passage_text_for_downstream_scoring(stub_encoder):
    """Generation and the confusability probe both read `text` off the reranked list."""
    pool = [_passage("a:0", 1, "alpha"), _passage("b:0", 2, "beta")]
    stub_encoder({"alpha": 0.2, "beta": 0.8})

    out = retrieve._rerank("q", pool, "stub-model")

    assert [p.text for p in out] == ["beta", "alpha"]


def test_pairs_the_query_against_the_text_not_the_id(stub_encoder):
    pool = [_passage("a:0", 1, "alpha")]
    stub = stub_encoder({"alpha": 0.3})

    retrieve._rerank("does alpha help?", pool, "stub-model")

    assert stub.seen == [("does alpha help?", "alpha")]


@pytest.mark.parametrize("missing_text", [None, ""])
def test_refuses_a_pool_whose_passages_carry_no_text(stub_encoder, missing_text):
    """An index loaded without passage_texts must fail loudly, not rerank identifier strings."""
    pool = [_passage("a:0", 1, "alpha"), _passage("b:0", 2, missing_text)]
    stub_encoder({"alpha": 0.3})

    with pytest.raises(ValueError, match="b:0"):
        retrieve._rerank("q", pool, "stub-model")


def test_empty_pool_is_returned_untouched(stub_encoder):
    """A query whose retrieval found nothing must not reach the cross-encoder at all."""
    stub = stub_encoder({})

    assert retrieve._rerank("q", [], "stub-model") == []
    assert stub.seen == []
