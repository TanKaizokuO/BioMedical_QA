"""Config identity and the G1 gate arithmetic.

The index-fingerprint tests encode the lesson from the retired base pipeline: a count is not an
identity (`docs/harvest/README.md`).
"""

from __future__ import annotations

import math

import pytest

from biomedqa.config import RunConfig, canonical_hash, config_diff
from biomedqa.schema import QueryRecord, RetrievedPassage, System
from biomedqa.scoring.retrieval import (
    gate_g1,
    gold_rank,
    hit_at_k,
    mrr,
    ndcg,
    recall_at_k,
    wilson_interval,
)


class TestConfigIdentity:
    def test_hash_is_stable_across_equal_configs(self):
        assert RunConfig().hash() == RunConfig().hash()

    def test_field_order_does_not_change_a_hash(self):
        assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})

    def test_chunker_change_changes_the_index_fingerprint(self):
        base = RunConfig()
        other = base.ablate("sentence-window", **{"chunk.strategy": "sentence_window"})
        assert base.index_fingerprint() != other.index_fingerprint(), (
            "hit@5 is only defined per (chunker, tau) — a different chunker is a different index"
        )

    def test_encoder_change_changes_the_index_fingerprint(self):
        base = RunConfig()
        other = base.ablate("other-encoder", **{"retrieval.dense_encoder": "some/other-model"})
        assert base.index_fingerprint() != other.index_fingerprint()

    def test_title_segment_change_changes_the_index_fingerprint(self):
        """ADR-0014 §3: the convention is part of the index's identity, not a call detail.

        `dense_encoder` names the checkpoint; it does not say how the checkpoint was called.
        `tok("", abstract)` and `tok(abstract)` are the same weights over the same title-free text
        and still produce different vectors, so without this axis the `empty` and `single` indices —
        two separate 2 h encodes — hash the same. `encode_corpus.py`'s resume guard already refused
        to concatenate them; the fingerprint could not tell them apart.
        """
        base = RunConfig()
        other = base.ablate("single-segment", **{"retrieval.title_segment": "single"})
        assert base.index_fingerprint() != other.index_fingerprint()

    def test_a_different_corpus_draw_changes_the_index_fingerprint(self):
        """ADR-0012 §1 requires the drawn ID list, not just its name, to reach the fingerprint.
        `corpus_id` is the literal string `"pubmed-2m-v1"` for every draw at every seed, so on its
        own it makes the duplicate-bearing `41cf7a6c9160` draw and the corpus that replaced it —
        `93321598f3f1`, same name, same size, 300 rows different — byte-identical here. That is the
        ADR-0007 staleness bug exactly: an index preserved because its summary matched.
        """
        base = RunConfig()
        other = base.ablate("earlier-draw", **{"retrieval.corpus_fingerprint": "41cf7a6c9160"})
        assert base.index_fingerprint() != other.index_fingerprint()

    def test_the_default_corpus_fingerprint_is_the_committed_draw(self):
        """The default is a literal, so a redraw that does not update it would leave every run
        claiming the old corpus. This is the line that fails when that happens."""
        import json
        from pathlib import Path

        manifest = json.loads(
            (Path(__file__).resolve().parents[1] / "data/corpus/corpus_manifest.json").read_text()
        )
        assert RunConfig().retrieval.corpus_fingerprint == manifest["fingerprint"]
        assert RunConfig().retrieval.corpus_id == manifest["corpus_id"]

    def test_generation_change_does_not_change_the_index_fingerprint(self):
        base = RunConfig()
        other = base.ablate("hotter", **{"generation.temperature": 0.7})
        assert base.index_fingerprint() == other.index_fingerprint(), (
            "the index does not depend on the generator; rebuilding it would waste two hours"
        )

    def test_ablation_diff_names_only_what_changed(self):
        base = RunConfig()
        no_rerank = base.ablate("no-rerank", **{"retrieval.rerank": False})
        diff = config_diff(base, no_rerank)
        assert diff["retrieval.rerank"] == (True, False)
        assert set(diff) == {"retrieval.rerank", "name"}


class TestWilson:
    def test_matches_a_known_value(self):
        # 90/100 at 95%: plain Wilson (no continuity correction) is (0.8256, 0.9448).
        point, lower, upper = wilson_interval(90, 100)
        assert point == pytest.approx(0.90)
        assert lower == pytest.approx(0.8256, abs=1e-3)
        assert upper == pytest.approx(0.9448, abs=1e-3)

    def test_is_not_wald(self):
        """Wald would give 0.9 ± 1.96*sqrt(.9*.1/100) = (0.841, 0.959) — narrower and mis-centred."""
        _, lower, upper = wilson_interval(90, 100)
        assert lower != pytest.approx(0.8412, abs=1e-3)
        assert upper < 0.9588

    def test_stays_inside_zero_one_at_the_boundary(self):
        """Where Wald visibly breaks: p = 1 gives an upper bound above 1."""
        _, lower, upper = wilson_interval(100, 100)
        assert 0.0 <= lower <= 1.0 and upper == 1.0

    def test_empty_sample_is_nan_not_a_crash(self):
        point, lower, upper = wilson_interval(0, 0)
        assert all(math.isnan(v) for v in (point, lower, upper))


def _record(ranked: list[str], gold: list[str], query_id: str = "1") -> QueryRecord:
    """A retrieval-only record: ranked passage ids, 1-indexed and contiguous, plus the gold set."""
    return QueryRecord(
        run_id="run-test",
        query_id=query_id,
        question="?",
        system=System.JOINT,
        seed=0,
        retrieved=[
            RetrievedPassage(passage_id=pid, rank=i, score=1.0 / i, retriever="rerank")
            for i, pid in enumerate(ranked, start=1)
        ],
        gold_passage_ids=gold,
    )


class TestGoldRankAndHits:
    def test_gold_rank_is_the_best_ranked_member_of_the_gold_set(self):
        """Gold is a set of chunks, so the rank is the minimum over it, not the first id tried."""
        r = _record(["a", "g2", "b", "g1"], gold=["g1", "g2"])
        assert gold_rank(r) == 2

    def test_gold_rank_is_none_when_no_gold_chunk_was_retrieved(self):
        assert gold_rank(_record(["a", "b"], gold=["g1"])) is None

    def test_hit_at_k_counts_the_boundary_rank_as_a_hit(self):
        records = [_record(["a", "b", "c", "d", "g"], gold=["g"], query_id=str(i)) for i in range(3)]
        assert hit_at_k(records, 5) == (3, 3)
        assert hit_at_k(records, 4) == (0, 3)

    def test_gate_needs_both_clauses(self):
        """94/100 passes; 90/100 clears the point estimate but its Wilson lower bound does not."""
        passing = [_record(["g"], gold=["g"], query_id=str(i)) for i in range(94)]
        passing += [_record(["x"], gold=["g"], query_id=f"m{i}") for i in range(6)]
        assert gate_g1(passing, 10)["passes"] is True

        borderline = [_record(["g"], gold=["g"], query_id=str(i)) for i in range(90)]
        borderline += [_record(["x"], gold=["g"], query_id=f"m{i}") for i in range(10)]
        gate = gate_g1(borderline, 10)
        assert gate["hit_at_k"] == pytest.approx(0.90)
        assert gate["passes"] is False, "0.90 with a Wilson lower of 0.826 must not pass"


class TestRecallAtK:
    def test_denominator_is_the_whole_gold_set(self):
        assert recall_at_k([_record(["g1", "a", "b"], gold=["g1", "g2"])], 5) == pytest.approx(0.5)

    def test_unreachable_gold_chunks_stay_in_the_denominator(self):
        """A gold chunk the corpus never indexed cannot be retrieved; recall must show that."""
        r = _record(["g1", "a"], gold=["g1", "never-indexed"])
        assert recall_at_k([r], 5) == pytest.approx(0.5)

    def test_is_macro_averaged_so_a_finely_cut_abstract_cannot_dominate(self):
        many = _record(["g1", "g2", "g3", "g4"], gold=["g1", "g2", "g3", "g4"], query_id="1")
        few = _record(["x", "y"], gold=["g"], query_id="2")
        # Micro-averaging would give 4/5 = 0.8; macro gives (1.0 + 0.0) / 2.
        assert recall_at_k([many, few], 5) == pytest.approx(0.5)

    def test_respects_k(self):
        r = _record(["g1", "a", "b", "c", "d", "g2"], gold=["g1", "g2"])
        assert recall_at_k([r], 5) == pytest.approx(0.5)
        assert recall_at_k([r], 10) == pytest.approx(1.0)

    def test_a_query_with_no_gold_raises_rather_than_being_dropped(self):
        with pytest.raises(ValueError, match="undefined"):
            recall_at_k([_record(["a"], gold=[])], 5)

    def test_empty_input_is_nan(self):
        assert math.isnan(recall_at_k([], 5))


class TestMRR:
    def test_uses_the_first_gold_chunk(self):
        r = _record(["a", "g2", "g1"], gold=["g1", "g2"])
        assert mrr([r]) == pytest.approx(0.5)

    def test_a_miss_contributes_zero_and_still_counts_in_the_denominator(self):
        hit = _record(["g"], gold=["g"], query_id="1")
        miss = _record(["x"], gold=["g"], query_id="2")
        assert mrr([hit, miss]) == pytest.approx(0.5)


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        r = _record(["g1", "g2", "a", "b"], gold=["g1", "g2"])
        assert ndcg([r], 10) == pytest.approx(1.0)

    def test_discounts_by_rank(self):
        r = _record(["a", "g1"], gold=["g1"])
        assert ndcg([r], 10) == pytest.approx(1 / math.log2(3))

    def test_ideal_is_capped_at_k_so_an_unreachable_gain_is_not_charged(self):
        """Three gold chunks and k=2: the best any ranking can do is the top two."""
        r = _record(["g1", "g2", "g3"], gold=["g1", "g2", "g3"])
        assert ndcg([r], 2) == pytest.approx(1.0)

    def test_gold_below_k_contributes_nothing(self):
        r = _record(["a", "b", "g1"], gold=["g1"])
        assert ndcg([r], 2) == 0.0
