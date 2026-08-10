"""Config identity and the G1 gate arithmetic.

The index-fingerprint tests encode the lesson from the retired base pipeline: a count is not an
identity (`docs/harvest/README.md`).
"""

from __future__ import annotations

import math

import pytest

from biomedqa.config import RunConfig, canonical_hash, config_diff
from biomedqa.scoring.retrieval import wilson_interval


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
