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
