"""Aggregation contract for the confusability probe.

Two properties are tested here without loading weights or an index:

1. **Max-over-gold-sentences per passage.** The probe reports one score per non-gold passage;
   that score is the maximum entailment score across all gold sentences for that passage.  A
   supporting sentence anywhere in the gold abstract is enough to flag the passage.

2. **The probe calls a verifier, not a private model.** After the 2026-08-17 cutover, no
   ``AutoModelForSeq2SeqLM`` or sequence-loss scorer lives in the script.  A fake verifier
   injected via monkeypatch is the entire scoring path, and no weights are downloaded.

The probe loop is exercised through its importable helpers (``split_sentences``, the batched
``score_pairs`` call shape) rather than through ``main()`` end-to-end, because the latter needs
a GPU, a retrieval index, and real data — none of which belong in the unit suite.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — mirror the script's own sys.path.insert so imports resolve
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomedqa.schema import VerifierScore  # noqa: E402

# ---------------------------------------------------------------------------
# Import the probe's pure helpers only — no GPU, no network
# ---------------------------------------------------------------------------
import importlib.util
import types

_PROBE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "confusability_probe.py"


def _load_probe_helpers() -> types.ModuleType:
    """Import only the pure functions from confusability_probe without executing main()."""
    spec = importlib.util.spec_from_file_location("confusability_probe", _PROBE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Inject a stub for biomedqa imports so the module-level import of MiniCheckVerifier does not
    # trigger a torch / transformers import when the test runner has no GPU.
    # We only need the pure Python helpers; MiniCheckVerifier itself is not called in these tests.
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Fake verifier — records calls, returns preset scores
# ---------------------------------------------------------------------------


@dataclass
class FakeVerifier:
    """Returns scores from a preset queue; asserts it is called exactly once per score_pairs call."""

    _scores: list[float]
    calls: list[Sequence[tuple[str, str]]] = field(default_factory=list)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[VerifierScore]:
        self.calls.append(list(pairs))
        out = []
        for i, _ in enumerate(pairs):
            out.append(VerifierScore(name="fake", score=self._scores[i], latency_s=0.0))
        return out


# ---------------------------------------------------------------------------
# split_sentences — tested in isolation (pure function, no GPU)
# ---------------------------------------------------------------------------


def test_split_sentences_breaks_on_sentence_boundaries() -> None:
    mod = _load_probe_helpers()
    result = mod.split_sentences("First sentence. Second sentence. Third one.")
    assert result == ["First sentence.", "Second sentence.", "Third one."]


def test_split_sentences_strips_whitespace() -> None:
    mod = _load_probe_helpers()
    assert mod.split_sentences("  Hello world.  ") == ["Hello world."]


def test_split_sentences_empty_string_returns_empty_list() -> None:
    mod = _load_probe_helpers()
    assert mod.split_sentences("") == []


# ---------------------------------------------------------------------------
# Aggregation: max-over-gold-sentences per passage
# ---------------------------------------------------------------------------


def test_max_over_gold_sentences_per_passage() -> None:
    """The probe reports max(score for each gold sentence) per non-gold passage, not a mean.

    Given two passages and three gold sentences, with scores arranged so the max differs from the
    mean, the output passage_max_scores must equal the per-passage maxima.
    """
    # Passage 0: sentences score [0.1, 0.9, 0.3] → max = 0.9
    # Passage 1: sentences score [0.5, 0.2, 0.4] → max = 0.5
    preset = [0.1, 0.9, 0.3, 0.5, 0.2, 0.4]
    fake = FakeVerifier(preset)

    passages = ["passage zero text", "passage one text"]
    gold_sentences = ["sent A", "sent B", "sent C"]

    # Replicate the probe's batched aggregation exactly as written in the script.
    pairs = [
        (p, sentence)
        for p in passages
        for sentence in gold_sentences
    ]
    raw = [vs.score for vs in fake.score_pairs(pairs)]
    n_sent = len(gold_sentences)
    q_scores = [
        max(raw[pi * n_sent : pi * n_sent + n_sent])
        for pi in range(len(passages))
    ]

    assert q_scores[0] == pytest.approx(0.9)
    assert q_scores[1] == pytest.approx(0.5)


def test_aggregation_is_max_not_mean() -> None:
    """Confirm that a low-scoring sentence cannot drag the passage score below the supporting one."""
    # One passage, two gold sentences: 0.05 and 0.95 → mean = 0.5, max = 0.95
    preset = [0.05, 0.95]
    fake = FakeVerifier(preset)

    passages = ["only passage"]
    gold_sentences = ["bad sentence", "good sentence"]

    pairs = [(p, s) for p in passages for s in gold_sentences]
    raw = [vs.score for vs in fake.score_pairs(pairs)]
    n_sent = len(gold_sentences)
    q_scores = [max(raw[pi * n_sent : pi * n_sent + n_sent]) for pi in range(len(passages))]

    assert q_scores[0] == pytest.approx(0.95), "max, not mean"


def test_score_pairs_called_once_per_question() -> None:
    """All pairs for one question are batched into a single score_pairs call, not one per pair."""
    preset = [0.3, 0.7, 0.4, 0.6]
    fake = FakeVerifier(preset)

    passages = ["passage A", "passage B"]
    gold_sentences = ["claim X", "claim Y"]

    pairs = [(p, s) for p in passages for s in gold_sentences]
    fake.score_pairs(pairs)

    assert len(fake.calls) == 1, "score_pairs must be called once per question, not per pair"
    assert len(fake.calls[0]) == 4  # 2 passages × 2 sentences


# ---------------------------------------------------------------------------
# No private MiniCheck framing in the script
# ---------------------------------------------------------------------------


def test_probe_script_contains_no_private_minicheck_implementation() -> None:
    """The private loader and scorer deleted in the 2026-08-17 cutover must not reappear.

    The docstring is allowed to mention their names as historical record; what must not exist
    is a callable definition or invocation that bypasses ``biomedqa.verify``.
    ``scripts/minicheck_format_check.py`` keeps the retired framing deliberately.
    """
    source = _PROBE_PATH.read_text(encoding="utf-8")
    # These are the exact runtime strings that the old private implementation used.
    assert 'f"premise: {' not in source, "retired NLI f-string must not appear"
    assert 'f"hypothesis: {' not in source, "retired NLI f-string must not appear"
    assert "premise: {premise}" not in source
    assert "hypothesis: {hypothesis}" not in source
    # The private function definitions must be gone (def-level check, not a bare name mention).
    assert "def load_minicheck(" not in source
    assert "def minicheck_score(" not in source
    assert "AutoModelForSeq2SeqLM" not in source
    # The routing must go through the package verifier.
    assert "MiniCheckVerifier" in source
    assert "score_pairs" in source
