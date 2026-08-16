"""The entailment scorers behind Table 2's φ and Table 3's rows.

Three properties are load-bearing here, and every test is one of them or a way of breaking one:

1. **The MiniCheck call is the reference call.** The model is a `T5ForConditionalGeneration` with
   no classification head, so *any* prompt returns a number and a wrong prompt returns a wrong
   number silently. The rendered input, the decoder-start forward pass, and the two vocabulary ids
   the softmax is taken over are pinned here — `scripts/confusability_probe.py` shipped a
   plausible-looking `"premise: … hypothesis: …"` framing for six days, and no assertion caught it.
2. **Nothing is thresholded on write.** `score_pairs` returns floats; `phi_from_scores` is the only
   place a cutoff is applied and it takes the cutoff as an argument (G3 sweeps it).
3. **A missing score is not a refutation.** φ raises for a pair it never scored, and the judge
   raises for a reply it cannot parse. Both would otherwise read as "the passage does not support
   the claim", which is a number Table 2 and Table 3 cannot distinguish from grounding failure.

No test loads 770M parameters: the model and tokenizer are injected. The weights are exercised on
the A4000 by `scripts/minicheck_format_check.py`, whose committed artifact is the evidence that
these fixtures describe the real thing.
"""

from __future__ import annotations

import pytest
import torch

from biomedqa.config import VerifierConfig
from biomedqa.schema import Claim, CostRecord
from biomedqa.verify import (
    MINICHECK_SUPPORTED_TOKEN_ID,
    MINICHECK_UNSUPPORTED_TOKEN_ID,
    JudgeVerifier,
    MiniCheckVerifier,
    build_verifier,
    chunk_document,
    minicheck_input,
    parse_judge_score,
    phi_from_scores,
    score_map,
)

# ---------------------------------------------------------------------------------------------
# Fakes. The tokenizer records what it was asked to encode; the model returns logits chosen so the
# expected support probability of each row is exactly known.
# ---------------------------------------------------------------------------------------------

_VOCAB = 512


class FakeEncoding(dict):
    def to(self, _device):  # noqa: D102 — mimics transformers' BatchEncoding
        return self


class FakeTokenizer:
    eos_token = "</s>"

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.max_lengths: list[int] = []

    def encode(self, text, add_special_tokens=True):  # noqa: ARG002
        return {"0": [3, 632], "1": [209]}[text]

    def __call__(self, texts, *, max_length, truncation, padding, return_tensors):  # noqa: ARG002
        self.seen.extend(texts)
        self.max_lengths.append(max_length)
        n = len(texts)
        return FakeEncoding(
            input_ids=torch.zeros((n, 4), dtype=torch.long),
            attention_mask=torch.ones((n, 4), dtype=torch.long),
        )


class FakeModel:
    """Returns a preset support logit per call row, everything else at zero."""

    device = "cpu"

    def __init__(self, support_logits: list[float]) -> None:
        self.support_logits = list(support_logits)
        self.decoder_input_ids: list[torch.Tensor] = []

    def __call__(self, *, input_ids, attention_mask, decoder_input_ids):  # noqa: ARG002
        self.decoder_input_ids.append(decoder_input_ids)
        rows = input_ids.shape[0]
        logits = torch.zeros((rows, 1, _VOCAB))
        for row in range(rows):
            logits[row, 0, MINICHECK_SUPPORTED_TOKEN_ID] = self.support_logits.pop(0)
            logits[row, 0, MINICHECK_UNSUPPORTED_TOKEN_ID] = 0.0
        return type("Out", (), {"logits": logits})()


def _verifier(support_logits, **kwargs) -> MiniCheckVerifier:
    verifier = MiniCheckVerifier(**kwargs)
    verifier._tokenizer = FakeTokenizer()
    verifier._model = FakeModel(support_logits)
    return verifier


# ---------------------------------------------------------------------------------------------
# The input format
# ---------------------------------------------------------------------------------------------


def test_input_is_the_reference_format_not_an_nli_framing() -> None:
    """`predict: {document}</s>{claim}` — from `minicheck/inference.py`, not invented here."""
    rendered = minicheck_input("The sky is blue.", "The sky is azure.", "</s>")
    assert rendered == "predict: The sky is blue.</s>The sky is azure."
    assert "premise:" not in rendered and "hypothesis:" not in rendered


def test_scoring_renders_every_chunk_through_that_format() -> None:
    verifier = _verifier([0.0])
    verifier.score_pairs([("A cited span.", "A claim.")])
    assert verifier._tokenizer.seen == ["predict: A cited span.</s>A claim."]


def test_load_refuses_a_tokenizer_whose_decision_tokens_moved() -> None:
    """The softmax is taken at two hard-coded ids; if they stop meaning 0/1 the score is noise."""
    from biomedqa.verify import _check_decision_tokens

    class Moved(FakeTokenizer):
        def encode(self, text, add_special_tokens=True):  # noqa: ARG002
            return {"0": [7, 632], "1": [209]}[text]

    _check_decision_tokens(FakeTokenizer())  # the real ids pass
    with pytest.raises(RuntimeError, match="unsupported token moved"):
        _check_decision_tokens(Moved())


def test_the_real_tokenizer_still_encodes_the_two_pinned_ids() -> None:
    """The one test that touches a real tokenizer — small, cached, and the pin's whole point."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/flan-t5-large", local_files_only=True
        )
    except Exception:  # noqa: BLE001 — weights absent on this machine, not a failure of the code
        pytest.skip("flan-t5 tokenizer not in the local HF cache")
    assert tokenizer.encode("0", add_special_tokens=False)[0] == MINICHECK_UNSUPPORTED_TOKEN_ID
    assert tokenizer.encode("1", add_special_tokens=False) == [MINICHECK_SUPPORTED_TOKEN_ID]


# ---------------------------------------------------------------------------------------------
# The forward pass and its aggregation
# ---------------------------------------------------------------------------------------------


def test_score_is_the_softmax_over_the_two_decision_tokens() -> None:
    """logit 0 against logit 0 is 0.5; the arithmetic is the reference's, not a rescaling."""
    scores = _verifier([0.0, 2.0]).score_pairs([("p one.", "h"), ("p two.", "h")])
    assert scores[0].score == pytest.approx(0.5)
    assert scores[1].score == pytest.approx(torch.softmax(torch.tensor([0.0, 2.0]), 0)[1].item())


def test_the_decoder_is_started_from_a_single_pad_token() -> None:
    verifier = _verifier([0.0])
    verifier.score_pairs([("p.", "h")])
    (started,) = verifier._model.decoder_input_ids
    assert started.shape == (1, 1)
    assert int(started[0, 0]) == 0


def test_a_pair_scores_as_the_maximum_over_its_chunks() -> None:
    """Reference aggregation: the chunk that supports the claim decides, and no other dilutes it."""
    document = " ".join(f"Sentence {i} of the premise text." for i in range(4))
    verifier = _verifier([-3.0, 4.0], chunk_words=12)
    (score,) = verifier.score_pairs([(document, "h")])
    assert len(verifier._tokenizer.seen) == 2
    assert score.score == pytest.approx(torch.softmax(torch.tensor([0.0, 4.0]), 0)[1].item())


def test_scores_come_back_in_the_order_the_pairs_went_in_across_batches() -> None:
    pairs = [(f"Premise {i}.", "h") for i in range(5)]
    scores = _verifier([0.0, 1.0, 2.0, 3.0, 4.0], batch_size=2).score_pairs(pairs)
    assert [round(s.score, 6) for s in scores] == sorted(round(s.score, 6) for s in scores)


def test_no_pairs_means_no_model_call() -> None:
    assert MiniCheckVerifier().score_pairs([]) == []


def test_truncation_length_is_the_reference_context_not_512() -> None:
    """512 would silently cut a document-level premise, which is what MiniCheck exists to read."""
    verifier = _verifier([0.0])
    verifier.score_pairs([("p.", "h")])
    assert verifier._tokenizer.max_lengths == [2048]


# ---------------------------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------------------------


def test_a_cited_span_is_one_chunk() -> None:
    assert chunk_document("Metformin reduced all-cause mortality in the treated arm.") == [
        "Metformin reduced all-cause mortality in the treated arm."
    ]


def test_chunks_break_on_sentence_boundaries_not_mid_sentence() -> None:
    document = "One two three four. Five six seven eight. Nine ten eleven twelve."
    chunks = chunk_document(document, chunk_words=8)
    assert chunks == ["One two three four. Five six seven eight.", "Nine ten eleven twelve."]


def test_a_sentence_longer_than_the_budget_is_its_own_chunk() -> None:
    """Better one over-long chunk the tokenizer truncates than a claim cut in half."""
    long_sentence = " ".join(["word"] * 30) + "."
    assert chunk_document(long_sentence, chunk_words=5) == [long_sentence]


def test_an_empty_premise_still_produces_one_scoreable_chunk() -> None:
    """An uncited claim reaches φ with `concat([]) == ""`; it must score, and score low."""
    assert chunk_document("   ") == [""]


# ---------------------------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("87", 0.87), (" 0\n", 0.0), ("100", 1.0), ("62%", 0.62), ("5.", 0.05)],
)
def test_judge_percentages_become_probabilities(reply: str, expected: float) -> None:
    assert parse_judge_score(reply) == pytest.approx(expected)


@pytest.mark.parametrize(
    "reply",
    [
        "I cannot determine this.",
        "101",
        "-4",
        "",
        # The whole reply must be the number. Reading the *first* integer out of a sentence turns
        # a confused judge into a confident one: "3 out of 5" would score 0.03, a refutation.
        "I give this a 3 out of 5",
        "62 (the passage is partial)",
        "0.85",
    ],
)
def test_an_unusable_judge_reply_raises_rather_than_scoring_zero(reply: str) -> None:
    with pytest.raises(ValueError):
        parse_judge_score(reply)


def test_judge_calls_are_costed_as_judge_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """C5 compares the judge's dollars against MiniCheck's; a row stamped "generate" is invisible."""
    seen: list[str] = []

    def fake_complete(prompt, config, *, seed=0, run_id="", query_id=None, response_format=None):  # noqa: ARG001
        seen.append(prompt)
        return "62", CostRecord(
            run_id=run_id,
            query_id=query_id,
            component="generate",
            backend=f"anthropic:{config.model}",
            input_tokens=11,
            output_tokens=1,
            usd=0.002,
            wall_s=0.5,
        )

    monkeypatch.setattr("biomedqa.backends.complete", fake_complete)

    judge = JudgeVerifier(model="claude-opus-5", run_id="r1", query_id="q1")
    (score,) = judge.score_pairs([("The passage.", "The claim.")])

    assert score.score == pytest.approx(0.62)
    assert score.name == "judge:claude-opus-5"
    assert [c.component for c in judge.costs] == ["judge"]
    assert judge.costs[0].usd == 0.002
    assert "The passage." in seen[0] and "The claim." in seen[0]


def test_the_judge_is_asked_for_support_not_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The distinction `CONTEXT.md` rule 1 puts to the human annotators, put to the judge too."""
    prompts: list[str] = []

    def fake_complete(prompt, config, *, seed=0, run_id="", query_id=None, response_format=None):  # noqa: ARG001
        prompts.append(prompt)
        return "5", CostRecord(run_id="", query_id=None, component="generate", backend="x")

    monkeypatch.setattr("biomedqa.backends.complete", fake_complete)
    JudgeVerifier().score_pairs([("p", "h")])
    assert "support, not truth" in prompts[0]


# ---------------------------------------------------------------------------------------------
# Driving them
# ---------------------------------------------------------------------------------------------


def test_build_verifier_picks_the_configured_model_or_its_judge() -> None:
    config = VerifierConfig()
    assert build_verifier(config).name == config.model
    assert build_verifier(config, judge=True).name == f"judge:{config.judge_model}"


def test_a_repeated_pair_is_scored_once() -> None:
    """Two claims of one answer citing one span is the common case, and the model is deterministic."""
    calls: list[int] = []

    class Counting:
        name = "counting"

        def score_pairs(self, pairs):
            calls.append(len(pairs))
            return [type("S", (), {"score": 0.9})() for _ in pairs]

    pairs = [("a", "x"), ("a", "x"), ("b", "y")]
    scores = score_map(pairs, Counting())
    assert calls == [2]
    assert set(scores) == {("a", "x"), ("b", "y")}


def test_phi_thresholds_at_the_cutoff_it_is_given() -> None:
    scores = {("p", "h"): 0.5, ("p", "g"): 0.49}
    assert phi_from_scores(scores, 0.5)("p", "h") is True
    assert phi_from_scores(scores, 0.5)("p", "g") is False
    assert phi_from_scores(scores, 0.4)("p", "g") is True


def test_phi_refuses_a_pair_it_never_scored() -> None:
    """Returning False here would report a missing score as a grounding failure."""
    with pytest.raises(KeyError, match="never scored"):
        phi_from_scores({}, 0.5)("p", "h")


def test_verify_scores_one_pair_through_the_cached_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    from biomedqa import verify as verify_module

    seen: list[tuple[str, str]] = []

    class Stub:
        name = "stub"

        def score_pairs(self, pairs):
            seen.extend(pairs)
            return [verify_module.VerifierScore(name="stub", score=0.77) for _ in pairs]

    monkeypatch.setattr(verify_module, "_CACHED", {})
    monkeypatch.setattr(verify_module, "build_verifier", lambda config, **kw: Stub())

    claim = Claim(claim_id="c1", text="Metformin reduces mortality.")
    score = verify_module.verify(claim, "the cited span", VerifierConfig())

    assert score.score == pytest.approx(0.77)
    assert seen == [("the cited span", "Metformin reduces mortality.")]
