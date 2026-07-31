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

CONFIG_VERSION = "1.0.0"


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

    corpus_id: str = "pubmed-2m-v1"          # ADR-0003; part of the index fingerprint
    bm25: bool = True                        # bm25s, not rank_bm25 (borderline at 2M, §3)
    dense: bool = True
    dense_encoder: str = "NCBI/MedCPT-Article-Encoder"
    query_encoder: str = "NCBI/MedCPT-Query-Encoder"   # MedCPT is asymmetric — two encoders
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
    max_tokens: int = 768
    temperature: float = 0.0
    seeds: tuple[int, ...] = (0, 1, 2)       # ≥3 seeds; only implementable locally (ADR-0004)
    granularity: str = "decontextualized_atomic"


@dataclass(frozen=True, slots=True)
class VerifierConfig:
    """Thresholds are NOT here. They are swept in scoring — see `schema.VerifierScore`."""

    model: str = "lytang/MiniCheck-Flan-T5-Large"
    also_alignscore: bool = False
    judge_model: str = "claude-opus-5"       # the expensive baseline C5 is measured against


@dataclass(frozen=True, slots=True)
class RunConfig:
    """The whole knob surface for one run."""

    name: str = "base"
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    split: str = "dev"                       # "dev" | "test" — test runs late, once per seed
    config_version: str = CONFIG_VERSION

    def hash(self) -> str:
        return canonical_hash(asdict(self))

    def index_fingerprint(self) -> str:
        """Identity of the retrieval index. Changing the corpus, the chunker, or the encoder means
        a different index — anything else does not."""
        return canonical_hash(
            {
                "corpus_id": self.retrieval.corpus_id,
                "chunk": asdict(self.chunk),
                "encoder": self.retrieval.dense_encoder,
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
