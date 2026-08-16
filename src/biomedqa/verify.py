"""MiniCheck-Flan-T5-Large and the Opus 5 judge baseline — **Table 3**, and φ for Table 2.

Both are *entailment scorers over (premise, hypothesis) pairs*, and they are reached through one
API — `Verifier.score_pairs` — so C5's cost comparison comes from the same call path rather than
from two implementations that differ in ways the table cannot see.

**Raw scores only.** `VerifierScore.score` is continuous and is never thresholded on write. The
threshold sweep, AUROC, ECE, and the calibration bins live in `scoring/`; a stored boolean fixes one
operating point and discards the sweep irrecoverably. `phi_from_scores` is the *one* place a
threshold is applied, at scoring time, and it takes the cutoff as an argument.

## The input format is the model's, not a plausible NLI framing

MiniCheck-Flan-T5-Large declares `pipeline_tag: text-classification` on the hub and is in fact a
`T5ForConditionalGeneration` with no classification head. The reference implementation
(`Liyan06/MiniCheck`, `minicheck/inference.py`) scores a pair by

1. rendering `"predict: " + document + <eos> + claim`,
2. running one decoder step from the decoder start token,
3. taking `softmax` over the *first-position* logits at exactly two vocabulary ids —
   `3` (`"▁"`, the first sub-token of `"0"` / unsupported) and `209` (`"▁1"` / supported),
4. and reporting column `1` as the support probability.

None of that is guessable, and getting it wrong does not raise: any prompt yields *a* number.
`scripts/confusability_probe.py` originally used `"premise: {p} hypothesis: {h}"` with a
sequence-loss comparison of the strings `"1"` and `"0"` — a reasonable-looking framing that the
model was never trained on. Every score it produced was off-distribution, which is why the probe is
re-run against this module rather than left to disagree with it. **There is one MiniCheck in this
package and it is this one.**

Long premises are handled the way the reference does: the document is cut into ~500-word chunks on
sentence boundaries (`chunk.sentence_spans`, so "sentence" means one thing repo-wide), every chunk
is scored, and the pair's score is the **maximum** over chunks. A cited span is far shorter than one
chunk, so on Table 2's pairs this is the identity — it exists so that a document-level premise, which
is what MiniCheck is built for and what `verify.py` is asked for at W7, is not silently truncated.

## The judge

`JudgeVerifier` asks Opus 5 for an integer 0–100 and divides by 100. That is a *self-reported*
probability, not a calibrated one, and it is the honest description of what a judge without logprobs
can give: Anthropic exposes no token probabilities, so the alternative is a binary label, which would
make the judge the one row of Table 3 that cannot be swept for AUROC. An unparsable reply raises
rather than defaulting — a judge that silently scores 0 would read as a confident refutation.

Every judge call emits a `CostRecord` with `component="judge"` (MiniCheck emits none: it is local
compute, timed in `VerifierScore.latency_s`, and Table 4's dollar column for it is zero by
construction).

## Not here

**AlignScore** (Table 3's second row) is *not* in this module. Its package pins `torch<2`,
`pytorch_lightning<2` and `protobuf<=3.20` against this project's `torch>=2.13`, and it loads a
Lightning `.ckpt` rather than a hub model, so it is an isolated-environment job or a manual
state-dict port — a separate W6 item, not a backend that can be added to `build_verifier` today.
Degradation of every one of these on biomedical text is expected, not exceptional (R7).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .chunk import sentence_spans
from .config import GenerationConfig, VerifierConfig
from .schema import Claim, CostRecord, VerifierScore

#: The two vocabulary ids MiniCheck's decision lives at, from `minicheck/inference.py`. `3` is
#: `"▁"` — the SentencePiece space that is the *first* sub-token of `"0"` (`[3, 632]`); `209` is
#: `"▁1"`, which encodes `"1"` on its own. They are asserted against the tokenizer at load time
#: rather than trusted, because a tokenizer swap upstream would otherwise move the decision
#: silently to two unrelated tokens.
MINICHECK_UNSUPPORTED_TOKEN_ID = 3
MINICHECK_SUPPORTED_TOKEN_ID = 209

#: Chunking and truncation, matching the reference for `flan-t5-large`.
MINICHECK_CHUNK_WORDS = 500
MINICHECK_MAX_MODEL_LEN = 2048

#: MiniCheck's *own* binarisation (`pred_label = 1 if p > 0.5`). It is the operating point to quote
#: when a single φ is needed before G3's sweep has chosen one — a default of the model, not a tuned
#: value of ours.
MINICHECK_DEFAULT_THRESHOLD = 0.5

#: `phi(premise, hypothesis) -> bool`, the primitive `scoring.citation` is written against.
Phi = Callable[[str, str], bool]

#: A (premise, hypothesis) pair. Premise first, everywhere, in this module and in `scoring/`.
Pair = tuple[str, str]


class Verifier(Protocol):
    """Anything that turns pairs into continuous scores. MiniCheck, the judge, and W6's AlignScore
    port all satisfy it, which is what lets Table 3's rows share a driver."""

    name: str

    def score_pairs(self, pairs: Sequence[Pair]) -> list[VerifierScore]:
        """One `VerifierScore` per pair, in order. Never thresholded."""
        ...


# ---------------------------------------------------------------------------------------------
# MiniCheck
# ---------------------------------------------------------------------------------------------


def minicheck_input(document: str, claim: str, eos_token: str) -> str:
    """The exact string the reference feeds the tokenizer: `predict: {document}{eos}{claim}`."""
    return f"predict: {document}{eos_token}{claim}"


def chunk_document(document: str, *, chunk_words: int = MINICHECK_CHUNK_WORDS) -> list[str]:
    """Sentence-aligned chunks of at most ~`chunk_words` words, and never empty.

    A sentence longer than the budget is its own chunk rather than being cut mid-sentence — the
    tokenizer's `max_length` is the backstop for that case, as it is in the reference.
    """
    text = document.strip()
    if not text:
        return [""]

    sentences = [text[a:b].strip() for a, b in sentence_spans(text, 0, len(text))]
    sentences = [s for s in sentences if s]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    budget = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and budget + words > chunk_words:
            chunks.append(" ".join(current))
            current, budget = [], 0
        current.append(sentence)
        budget += words
    if current:
        chunks.append(" ".join(current))
    return chunks


@dataclass
class MiniCheckVerifier:
    """`lytang/MiniCheck-Flan-T5-Large` over `transformers`, faithful to `minicheck/inference.py`.

    The official package is not a dependency: it would pull `nltk`, `datasets`, `pandas` and an
    `openai` client for forty lines of forward pass, and its `[llm]` extra pulls `vllm`, which
    `pyproject.toml` refuses for the whole workspace. The forward pass is reproduced here and the
    two token ids it turns on are checked against the tokenizer at load.
    """

    model_id: str = "lytang/MiniCheck-Flan-T5-Large"
    batch_size: int = 16
    device: str | None = None
    max_model_len: int = MINICHECK_MAX_MODEL_LEN
    chunk_words: int = MINICHECK_CHUNK_WORDS
    fp16: bool = True

    name: str = field(init=False)
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = self.model_id

    # -- loading ------------------------------------------------------------------------------

    def load(self) -> None:
        """Load weights and check that the decision tokens still mean what the constants say."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        _check_decision_tokens(tokenizer)

        # `dtype=`, not `torch_dtype=`: transformers 5 deprecates the old spelling with a warning
        # rather than an error, and a warning in a three-hour run's log is a warning nobody reads.
        dtype = torch.float16 if (self.fp16 and device.startswith("cuda")) else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, dtype=dtype)
        model.to(device).eval()

        self._tokenizer, self._model, self.device = tokenizer, model, device

    # -- scoring ------------------------------------------------------------------------------

    def score_pairs(self, pairs: Sequence[Pair]) -> list[VerifierScore]:
        if not pairs:
            return []
        self.load()
        import torch

        # Flatten to (pair index, rendered chunk input) so one batch can span pairs. A pair whose
        # premise is one cited span contributes exactly one row, which is the common case.
        flat: list[tuple[int, str]] = []
        for index, (premise, hypothesis) in enumerate(pairs):
            for chunk in chunk_document(premise, chunk_words=self.chunk_words):
                flat.append((index, minicheck_input(chunk, hypothesis, self._tokenizer.eos_token)))

        best = [0.0] * len(pairs)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(flat), self.batch_size):
                batch = flat[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [text for _, text in batch],
                    max_length=self.max_model_len,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(self._model.device)
                decoder_input_ids = torch.zeros(
                    (len(batch), 1), dtype=torch.long, device=self._model.device
                )
                logits = self._model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                ).logits.squeeze(1)
                probabilities = torch.softmax(
                    logits[
                        :, [MINICHECK_UNSUPPORTED_TOKEN_ID, MINICHECK_SUPPORTED_TOKEN_ID]
                    ].float(),
                    dim=-1,
                )[:, 1].tolist()
                for (index, _), probability in zip(batch, probabilities):
                    # Max over chunks, as the reference aggregates: the chunk that supports the
                    # claim decides, so an unrelated chunk cannot dilute a real match.
                    best[index] = max(best[index], probability)
        elapsed = time.perf_counter() - t0

        per_pair = elapsed / len(pairs)
        return [VerifierScore(name=self.name, score=s, latency_s=per_pair) for s in best]


def _check_decision_tokens(tokenizer: Any) -> None:
    """Fail loudly if the two ids the softmax is taken over stop encoding `"0"` and `"1"`."""
    unsupported = tokenizer.encode("0", add_special_tokens=False)
    supported = tokenizer.encode("1", add_special_tokens=False)
    if not unsupported or unsupported[0] != MINICHECK_UNSUPPORTED_TOKEN_ID:
        raise RuntimeError(
            f"MiniCheck's unsupported token moved: '0' encodes to {unsupported}, expected to start "
            f"at {MINICHECK_UNSUPPORTED_TOKEN_ID}. Scoring the old ids would return a number with "
            "no meaning."
        )
    if supported != [MINICHECK_SUPPORTED_TOKEN_ID]:
        raise RuntimeError(
            f"MiniCheck's supported token moved: '1' encodes to {supported}, expected "
            f"[{MINICHECK_SUPPORTED_TOKEN_ID}]."
        )


# ---------------------------------------------------------------------------------------------
# The Opus 5 judge baseline
# ---------------------------------------------------------------------------------------------

JUDGE_TEMPLATE = """You are grading whether a passage supports a claim.

PASSAGE:
{premise}

CLAIM:
{hypothesis}

Answer with a single integer from 0 to 100 and nothing else: the probability, in percent, that the
passage alone supports the claim. Judge support, not truth — a claim you believe is correct but
that the passage does not state scores low. Do not explain."""

#: The whole reply, not a number found inside it. `search` would read "I give this a 3 out of 5"
#: as 0.03 — a confident refutation extracted from a confused answer, and Table 3's AUROC would
#: carry it. A trailing `%` or full stop is the one liberty taken.
_JUDGE_REPLY = re.compile(r"\s*(\d{1,3})\s*%?\.?\s*")


def parse_judge_score(reply: str) -> float:
    """`"87"` → `0.87`. Raises on anything that is not one integer in `[0, 100]`.

    A judge that answers "I cannot determine this" must not become a confident 0.0 — that is a
    refutation, and it would move Table 3's AUROC in the direction of the judge looking decisive.
    """
    match = _JUDGE_REPLY.fullmatch(reply)
    if match is None:
        raise ValueError(f"judge reply is not a single percentage: {reply!r}")
    value = int(match.group(1))
    if not 0 <= value <= 100:
        raise ValueError(f"judge returned {value}, which is not a percentage: {reply!r}")
    return value / 100.0


@dataclass
class JudgeVerifier:
    """Opus 5 as an entailment scorer, behind the same `score_pairs` as MiniCheck.

    It is the *expensive baseline* C5 is measured against, so its cost rows matter as much as its
    scores: each call appends a `CostRecord` with `component="judge"` to `costs`, which the driver
    writes to the run's `costs.jsonl`.
    """

    model: str = "claude-opus-5"
    max_tokens: int = 8
    run_id: str = ""
    query_id: str | None = None

    name: str = field(init=False)
    costs: list[CostRecord] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.name = f"judge:{self.model}"

    def score_pairs(self, pairs: Sequence[Pair]) -> list[VerifierScore]:
        from .backends import complete

        config = GenerationConfig(
            backend="anthropic",
            model=self.model,
            max_tokens=self.max_tokens,
            frequency_penalty=0.0,
            stop=(),
        )

        scores: list[VerifierScore] = []
        for premise, hypothesis in pairs:
            prompt = JUDGE_TEMPLATE.format(premise=premise, hypothesis=hypothesis)
            text, cost = complete(
                prompt, config, run_id=self.run_id, query_id=self.query_id
            )
            # Same convention as `decompose.py` and `generate.py`: `complete` stamps "generate",
            # and the caller that knows what the call was for corrects it.
            cost.component = "judge"
            self.costs.append(cost)
            scores.append(
                VerifierScore(
                    name=self.name, score=parse_judge_score(text), latency_s=cost.wall_s
                )
            )
        return scores


# ---------------------------------------------------------------------------------------------
# Driving them
# ---------------------------------------------------------------------------------------------


def build_verifier(config: VerifierConfig, *, judge: bool = False, **kwargs: Any) -> Verifier:
    """The verifier `config` names, or its judge baseline when `judge=True`."""
    if judge:
        return JudgeVerifier(model=config.judge_model, **kwargs)
    return MiniCheckVerifier(model_id=config.model, **kwargs)


_CACHED: dict[tuple[str, bool], Verifier] = {}


def verify(claim: Claim, premise: str, config: VerifierConfig) -> VerifierScore:
    """Score one (premise = cited span, hypothesis = claim) pair. Continuous, never binarized.

    Convenience over `score_pairs`, and it holds the loaded model in a process-level cache so that
    a loop over claims does not reload 770M parameters per pair. Anything scoring more than a
    handful of pairs should call `score_pairs` directly and get the batching.
    """
    key = (config.model, False)
    verifier = _CACHED.get(key)
    if verifier is None:
        verifier = _CACHED.setdefault(key, build_verifier(config))
    return verifier.score_pairs([(premise, claim.text)])[0]


def score_map(pairs: Iterable[Pair], verifier: Verifier) -> dict[Pair, float]:
    """`{(premise, hypothesis): score}`, deduplicated before scoring.

    Repeated pairs are common — two claims of one answer citing the same span — and the model is
    deterministic, so scoring a duplicate twice buys nothing but wall-clock.
    """
    unique = list(dict.fromkeys(pairs))
    scored = verifier.score_pairs(unique)
    return {pair: result.score for pair, result in zip(unique, scored)}


def phi_from_scores(scores: Mapping[Pair, float], threshold: float) -> Phi:
    """The `scoring.citation` primitive at one operating point.

    **The only threshold in the module.** An unseen pair raises rather than returning `False`: a φ
    that answers "not entailed" for a pair it was never given would report a missing score as a
    grounding failure, and citation-recall cannot tell the two apart afterwards.
    """

    def phi(premise: str, hypothesis: str) -> bool:
        try:
            return scores[(premise, hypothesis)] >= threshold
        except KeyError:
            raise KeyError(
                "φ was asked for a pair that was never scored — score the pairs "
                "`citation_f1` collects, not a subset."
            ) from None

    return phi
