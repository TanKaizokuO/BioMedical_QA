"""Every knob, in one place, hashed into the run manifest.

Two things this module exists to make impossible:

1. **An untraceable number.** Every result carries the hash of the config that produced it. A table
   cell whose config hash is not in `runs/` is not a result, it is a memory.
2. **A stale index that looks fresh.** `index_fingerprint()` is a content hash of
   `(corpus_id, chunker, encoder)`. The base pipeline decided index freshness with
   `collection.count() == 1000`, and because the *wrong* collection also held 1,000 documents, the
   check passed and preserved a broken index — which is how the original bug survived long enough
   to reach a results file (`docs/harvest/README.md`). Cardinality is never evidence that an index
   is the index you meant.

Configs are declared as frozen dataclasses and composed by `replace()`, so an ablation row is a
*diff* against the base — `config_diff()` renders exactly that, and it is what goes in a table
caption rather than a wall of settings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .prompts import MAX_CLAIM_WORDS

#: 1.1.0 adds ScoringConfig (ADR-0010). 1.2.0 adds RetrievalConfig.corpus_fingerprint, so that the
#: drawn ID list reaches index_fingerprint() as ADR-0012 §1 requires. 1.3.0 adds
#: RetrievalConfig.title_segment, which ADR-0014 §3 says is part of the index's identity and which
#: index_fingerprint() did not previously see. 1.4.0 adds the non-termination controls:
#: ScoringConfig.max_claim_words (the parse-side guard, a re-scorable rule) and
#: GenerationConfig.frequency_penalty / stop (the generation-side cause). 1.5.0 sets
#: GenerationConfig.frequency_penalty to 0.5 from the A4000 sweep, which changes every
#: RunConfig.hash() — deliberately, because a run's decoding is part of its identity — while
#: SCHEMA_VERSION is unaffected because this versions knobs, not records.
CONFIG_VERSION = "1.5.0"


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Passage granularity. hit@5 is only defined per `(chunker, τ)` pair (Lesson 2), so this is
    part of the index identity, not a downstream preference."""

    strategy: str = "abstract"     # "abstract" | "section" | "sentence_window" | "fixed_width"
    window_sentences: int = 3
    stride_sentences: int = 1
    max_chars: int = 2000
    keep_section_labels: bool = True   # PubMedQA's BACKGROUND/METHODS/... are natural boundaries


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """The cascade in Table 1. Each stage can be ablated by setting its flag false."""

    corpus_id: str = "pubmed-2m-v1"          # ADR-0003; the corpus's *name*, identical at every seed
    #: The drawn ID list's content hash — `data/corpus/corpus_manifest.json`'s `fingerprint`, which
    #: `CorpusDraw` computes over the 2M PMIDs themselves. ADR-0012 §1 requires the list, not the
    #: name, in the index fingerprint: `corpus_id` is a claim about which corpus this is, and two
    #: different draws make the same claim. A redraw changes this line, and the test that pins it to
    #: the committed manifest is what fails if it does not.
    corpus_fingerprint: str = "93321598f3f1"
    bm25: bool = True                        # bm25s, not rank_bm25 (borderline at 2M, §3)
    dense: bool = True
    dense_encoder: str = "NCBI/MedCPT-Article-Encoder"
    query_encoder: str = "NCBI/MedCPT-Query-Encoder"   # MedCPT is asymmetric — two encoders
    #: ADR-0014 §3 — how the article encoder is called on a title-free passage: ``"empty"`` for
    #: ``tok("", abstract)`` or ``"single"`` for ``tok(abstract)``. Two indices built from the same
    #: corpus, chunker and encoder under different conventions hold **different vectors** (max abs
    #: component diff 0.0349), so this belongs in the fingerprint. `encode_corpus.py`'s resume guard
    #: already refused to mix the two; until now the fingerprint could not tell them apart, and its
    #: comment claiming "every knob here is inside index_fingerprint()" was false for this one.
    title_segment: str = "empty"
    rrf: bool = True
    rrf_k: int = 60
    rerank: bool = True
    reranker: str = "NCBI/MedCPT-Cross-Encoder"
    pool_size: int = 100                     # cheap high-recall pool before the expensive rerank
    top_k: int = 5                           # scoring may threshold at any k; this is generation's


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Identical across joint / post-hoc / vanilla except where the system differs by definition.

    `max_citations` in particular must be the same for all three: an unequal cap makes C2's gap an
    artifact of citation budget rather than of joint grounding (`CONTEXT.md`).
    """

    backend: str = "vllm"                    # "vllm" | "anthropic" — the W8 deferral (ADR-0004)
    model: str = ""                          # chosen at G0, on citation-format compliance
    max_citations: int = 3
    # Raised from 768 after the W4 live smoke truncated joint mid-CLAIM: joint interleaves CITE
    # lines with verbatim quotes, so it needs the most completion tokens of the three, and a budget
    # that only joint hits is a budget that hands C2 a gap for free. Shared by all three systems,
    # like max_citations. Worst measured prompt is 4568 tokens (post_hoc_cite,
    # docs/harvest/prompt_drafts.json), so 8192 - 1536 = 6656 still clears it.
    max_tokens: int = 1536
    temperature: float = 0.0
    # Non-termination controls, shared by all three systems for the same reason max_citations is
    # (CONFIG_VERSION 1.4.0). `temperature: 0.0` is greedy, and greedy decoding has no escape from a
    # repetition loop: joint `21074975` walked one to 731 words in `parity_iter1b`.
    #
    # **`frequency_penalty`, never `repetition_penalty`.** Verified against vLLM source
    # (`vllm/model_executor/layers/utils.py::apply_penalties`): repetition_penalty is applied over
    # `prompt_mask | output_mask`, so it penalises every token that appears **in the prompt** — which
    # is precisely the tokens a citation has to copy verbatim for `locate_quote` to find the span.
    # It would suppress the measurement to treat a decoding defect. frequency_penalty and
    # presence_penalty are computed from `output_tokens_tensor` alone; frequency_penalty scales with
    # how often a token was already generated, which is what a runaway loop does. Neither is
    # bypassed at temperature 0.0 — the penalty runs in `Sampler.forward` before `greedy_sample`,
    # gated on `no_penalties` (value != default), not on temperature.
    #
    # Set to 0.5 based on the A4000 sweep (`docs/harvest/generate_fp_sweep.md`). The measurement
    # brings chain and over-length claims to zero and reduces the longest joint claim from 731w to
    # 27w, while quote_not_found falls rather than rises (joint 8 -> 0, post_hoc 92 -> 15), inverting
    # the risk anticipated above. The value 0.5 sits on a plateau (claims/query 5.33 at 0.3 vs 5.42 at
    # 0.5) well short of the fp=1.0 total-claims collapse in `docs/harvest/decompose_smoke_fp_sweep.md`,
    # and matches the C7 decomposer path so the pipeline now has one value. Note that the read was
    # n=12 on a slice enriched for the pathology, so it chose a value and did not measure a rate.
    frequency_penalty: float = 0.5
    #: Hard textual backstop, excluded from the returned text by vLLM unless
    #: `include_stop_str_in_output` is set (which this harness never sets). Empty by default: the
    #: observed loop grows *across* CLAIM lines, so no fixed string delimits it, and a stop
    #: sequence that clipped a legitimate reply would be charged to the system.
    stop: tuple[str, ...] = ()
    seeds: tuple[int, ...] = (0, 1, 2)       # ≥3 seeds; only implementable locally (ADR-0004)
    granularity: str = "decontextualized_atomic"
    guided_decoding: bool = False


@dataclass(frozen=True, slots=True)
class VerifierConfig:
    """Thresholds are NOT here. They are swept in scoring — see `schema.VerifierScore`."""

    model: str = "lytang/MiniCheck-Flan-T5-Large"
    also_alignscore: bool = False
    judge_model: str = "claude-opus-5"       # the expensive baseline C5 is measured against


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Rules applied *after* a run, recorded so a number always states which rule produced it.

    Nothing here changes what is written to `records.jsonl` — that is the whole point. A revised
    rule re-*scores* existing records rather than forcing a re-*run*, which only works while the
    rule lives in config and `scoring/` rather than in the schema (ADR-0010).
    """

    abstention_rule_version: str = "1.0.0"   # `scoring.abstention.ABSTENTION_RULE_VERSION`
    # Citation-F1 is reported on both denominators, always — abstention-excluded as primary and
    # abstention-included alongside it. No threshold gates the pair (ADR-0010): both are pure
    # recomputations over identical records, so there is nothing to tune and nothing to defend.
    report_both_abstention_denominators: bool = True
    # Every interval clusters on the question, never the claim (ADR-0011). Claims within a question
    # share passages, answer and topic; resampling them as independent units returns CIs that are
    # too narrow. Flipping this to False exists only to reproduce the pre-ADR-0011 width in the W4
    # dry-run — it is not a legitimate setting for any reported number.
    bootstrap_cluster_on: str = "question"
    # The parse-side non-termination guard (`prompts.MAX_CLAIM_WORDS`), imported rather than
    # retyped so a re-scored G2 cannot disagree with the run log that produced it. It belongs here
    # and not in GenerationConfig because `parse_response`'s errors are re-derived from
    # `raw_generation` at scoring time: revising this number re-scores, it never forces a re-run.
    max_claim_words: int = MAX_CLAIM_WORDS


@dataclass(frozen=True, slots=True)
class RunConfig:
    """The whole knob surface for one run."""

    name: str = "base"
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    split: str = "dev"                       # "dev" | "test" — test runs late, once per seed
    config_version: str = CONFIG_VERSION

    def hash(self) -> str:
        return canonical_hash(asdict(self))

    def index_fingerprint(self) -> str:
        """Identity of the retrieval index. Changing the corpus, the chunker, the encoder, or how
        the encoder is *called* means a different index — anything else does not.

        The corpus enters as the **draw's content hash**, not only its name. `corpus_id` is
        `"pubmed-2m-v1"` for every draw at every seed, so a fingerprint built on the name alone
        cannot tell two corpora apart — which is the ADR-0007 failure this module's docstring cites,
        in the one place it would do the most damage.

        `title_segment` is here for the same reason one step down: `dense_encoder` names the weights,
        not the call. `tok("", abstract)` and `tok(abstract)` run the same checkpoint over the same
        text and produce different vectors, so on the encoder name alone two genuinely different
        indices hash identically (ADR-0014 §3).
        """
        return canonical_hash(
            {
                "corpus_id": self.retrieval.corpus_id,
                "corpus_fingerprint": self.retrieval.corpus_fingerprint,
                "chunk": asdict(self.chunk),
                "encoder": self.retrieval.dense_encoder,
                "title_segment": self.retrieval.title_segment,
            }
        )

    def ablate(self, name: str, **overrides: Any) -> "RunConfig":
        """Derive an ablation row. Nested overrides use dotted keys: `ablate("no-rerank",
        **{"retrieval.rerank": False})`."""
        nested: dict[str, dict[str, Any]] = {}
        flat: dict[str, Any] = {}
        for key, value in overrides.items():
            if "." in key:
                section, attr = key.split(".", 1)
                nested.setdefault(section, {})[attr] = value
            else:
                flat[key] = value
        for section, attrs in nested.items():
            flat[section] = replace(getattr(self, section), **attrs)
        return replace(self, name=name, **flat)


def canonical_hash(obj: Any) -> str:
    """Stable 12-hex-char digest. Sorted keys so that field order never changes a hash — the same
    primitive as `08_6_reproducible_eval_harness.ipynb`."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def config_diff(base: RunConfig, other: RunConfig) -> dict[str, tuple[Any, Any]]:
    """`{dotted.key: (base, other)}` for every differing leaf. This is what a table caption says,
    instead of reprinting the whole config."""

    def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flatten(v, f"{key}."))
            else:
                out[key] = v
        return out

    a, b = flatten(asdict(base)), flatten(asdict(other))
    return {k: (a.get(k), b.get(k)) for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
