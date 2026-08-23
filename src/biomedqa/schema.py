"""THE FROZEN OUTPUT SCHEMA — the single most important artifact in this repo.

Everything a run produces is written through these types, and every number in the paper is computed
from them by pure functions in `scoring/`. The types implement the four units frozen in
`CONTEXT.md`: claim, claim validity, citation, support label.

The one rule that governs every field here — the **least-processed-value rule**:

    Store `phi_score: 0.83`, never `supported: true`.
    Store char offsets and the ranked passage list, never a precomputed hit@5.
    Store the 4-way `support_label`, never a collapsed boolean.

Binarizing at write time destroys the AUROC sweep and the calibration bins irrecoverably, and turns
re-chunking into a re-*run* rather than a re-*score*. The rule extends hardest to human labels: an
annotator cannot be re-run (`CONTEXT.md`).

A corollary that is easy to get wrong: **this schema never silently repairs a violation.** A model
that cites four passages when the cap is three has told you something important about itself, and
`validate()` reports it while the record keeps all four citations. Normalising on write would delete
the measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterator, Sequence

SCHEMA_VERSION = "1.0.0"

# The fairness control from CONTEXT.md, not a formatting detail: uncapped, citation recall is
# trivially gamed by citing every retrieved passage. It must be identical in the prompt for every
# system — ours, post-hoc, and vanilla — or C2's gap is an artifact of citation budget.
MAX_CITATIONS = 3


class SupportLabel(str, Enum):
    """Human annotation of a (claim, cited span) pair. Judge attribution, not truth."""

    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"

    @property
    def is_supporting(self) -> bool:
        """The binary collapse `(SUPPORTED | PARTIAL) vs (NOT_SUPPORTED | CONTRADICTED)`.

        **Derived at scoring time, never stored.** G4's Krippendorff α gates on this collapse
        because it is the quantity C4 consumes; the 4-way ordinal α is reported alongside it. The
        4-way label is what lives on disk — collapsing on write would destroy the failure-mode
        analysis (negation, numerics, scope are exactly where a span asserts the opposite rather
        than nothing).
        """
        return self in (SupportLabel.SUPPORTED, SupportLabel.PARTIAL)


class Granularity(str, Enum):
    """Decomposition unit. `DECONTEXTUALIZED_ATOMIC` is the headline configuration (ADR-0005);
    the others exist only as C7 ablation rows."""

    DECONTEXTUALIZED_ATOMIC = "decontextualized_atomic"
    ATOMIC = "atomic"
    SENTENCE = "sentence"


class System(str, Enum):
    """The three systems compared in C2. Same retriever, same generator, same citation cap."""

    JOINT = "joint"          # ours: claim-grounded generation
    POST_HOC = "post_hoc"    # generate, then attach citations
    VANILLA = "vanilla"      # retrieve → generate, no attribution


@dataclass(slots=True)
class Citation:
    """A character span in a retrieved passage — `{passage_id, char_start, char_end}`.

    `quoted_text` is stored despite being derivable: the corpus is addressable, but a re-chunk
    changes passage ids, and the annotators judge the span they were shown. Without the text, a
    gold label made in September cannot be re-read in November.
    """

    passage_id: str
    char_start: int
    char_end: int
    quoted_text: str | None = None

    def __post_init__(self) -> None:
        if self.char_start < 0 or self.char_end < self.char_start:
            raise ValueError(f"invalid span [{self.char_start}, {self.char_end}) in {self.passage_id}")


@dataclass(slots=True)
class VerifierScore:
    """A raw, continuous verifier output. **Never thresholded on write.**

    The threshold is swept in `scoring/` to produce AUROC, ECE, and the calibration bins; a stored
    boolean would fix one operating point and silently discard the sweep. `name` distinguishes
    MiniCheck from AlignScore from the Opus 5 judge, which are compared in Table 3.
    """

    name: str
    score: float
    latency_s: float | None = None


@dataclass(slots=True)
class HumanLabel:
    """One annotator's judgement of one (claim, cited span) pair. Four-way, never collapsed."""

    annotator_id: str
    support_label: SupportLabel
    claim_validity: bool
    citation_index: int | None = None   # which citation of the claim; None = union judgement
    notes: str | None = None


@dataclass(slots=True)
class Claim:
    """A decontextualized atomic claim — the attribution unit (`CONTEXT.md`, ADR-0005)."""

    claim_id: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    granularity: Granularity = Granularity.DECONTEXTUALIZED_ATOMIC
    verifier_scores: list[VerifierScore] = field(default_factory=list)
    human_labels: list[HumanLabel] = field(default_factory=list)
    # Char span of this claim within the raw generation, so a claim can always be traced back to
    # the text that produced it — decomposition is an upstream confound and must stay auditable.
    source_start: int | None = None
    source_end: int | None = None

    @property
    def exceeds_cap(self) -> bool:
        return len(self.citations) > MAX_CITATIONS


@dataclass(slots=True)
class RetrievedPassage:
    """One passage in the ranked list, at the stage named by `retriever`."""

    passage_id: str
    rank: int          # 1-indexed
    score: float
    retriever: str     # "bm25" | "dense" | "rrf" | "rerank"
    text: str | None = None


@dataclass(slots=True)
class QueryRecord:
    """One (question, system, seed) run — the unit of `records.jsonl`.

    On gold tracking: this stores the **full ranked list** and the **gold passage id set**, and
    nothing derived from them. `gold_rank` is a function of the two (`scoring/retrieval.py`), and
    hit@k is a function of `gold_rank` and a `k` chosen at scoring time. The base pipeline stored
    `gold_retrieved: bool` at a fixed k=5, which would make G1's hit@10 escalation a re-run rather
    than a re-score (`docs/harvest/gold-passage-tracking.md`).
    """

    run_id: str
    query_id: str                 # PubMedQA pubid, as a string
    question: str
    system: System
    seed: int
    retrieved: list[RetrievedPassage] = field(default_factory=list)
    gold_passage_ids: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    raw_generation: str = ""      # verbatim model output, before any parsing
    final_decision: str | None = None      # yes/no/maybe, secondary accuracy (C6)
    gold_final_decision: str | None = None
    latency_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> list[str]:
        """Report contract violations; never repair them. An empty list means the record is clean.

        Violations are measurements — the G0 bake-off ranks candidate models on exactly these —
        so they are reported and kept, not normalised away.
        """
        problems: list[str] = []
        valid_ids = {p.passage_id for p in self.retrieved}
        seen: set[str] = set()

        for i, claim in enumerate(self.claims):
            if claim.claim_id in seen:
                problems.append(f"claim[{i}]: duplicate claim_id {claim.claim_id!r}")
            seen.add(claim.claim_id)
            if claim.exceeds_cap:
                problems.append(
                    f"claim[{i}]: {len(claim.citations)} citations exceeds the cap of {MAX_CITATIONS}"
                )
            for j, c in enumerate(claim.citations):
                if valid_ids and c.passage_id not in valid_ids:
                    problems.append(
                        f"claim[{i}].citation[{j}]: cites {c.passage_id!r}, not in the retrieved set"
                    )
                if c.quoted_text is not None and len(c.quoted_text) != c.char_end - c.char_start:
                    problems.append(
                        f"claim[{i}].citation[{j}]: quoted_text length disagrees with the span"
                    )

        ranks = [p.rank for p in self.retrieved]
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            problems.append("retrieved: ranks are not a 1-indexed contiguous sequence")
        if self.system is System.VANILLA and any(c.citations for c in self.claims):
            problems.append("vanilla baseline carries citations — it is defined as having none")
        return problems


@dataclass(slots=True)
class CostRecord:
    """One billable or timed unit of work — the row type of `costs.jsonl`.

    Table 4's columns (input tokens / output tokens / $ per query / wall-clock) are fixed here, in
    W0, precisely so that W5 instruments for them instead of discovering the gap in October.
    """

    run_id: str
    query_id: str | None
    component: str                # "generate" | "decompose" | "decompose_cite" | "verify" | "judge" | "rerank" | "encode"
    backend: str                  # "vllm:<model>" | "anthropic:<model>"
    input_tokens: int | None = None
    output_tokens: int | None = None
    usd: float | None = None
    wall_s: float | None = None
    is_retry: bool = False
    attempt: int = 1
    temperature: float | None = None

# ---------------------------------------------------------------------------------------------
# Serialisation. JSONL in, JSONL out, byte-stable — the round-trip is what makes the schema frozen
# rather than merely declared, and `tests/test_schema_roundtrip.py` is what keeps it that way.
# ---------------------------------------------------------------------------------------------

_ENUMS: dict[str, type[Enum]] = {
    "system": System,
    "granularity": Granularity,
    "support_label": SupportLabel,
}


def to_dict(record: Any) -> dict:
    """Dataclass → plain dict, enums flattened to their string values."""

    def _clean(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_clean(v) for v in value]
        return value

    return _clean(asdict(record))


def query_record_from_dict(d: dict) -> QueryRecord:
    """Plain dict → `QueryRecord`, rebuilding every nested type.

    Unknown keys raise rather than being dropped. A field that appears in a file and vanishes on
    read is exactly the silent data loss the frozen schema exists to prevent — if the schema needs
    a new field, `SCHEMA_VERSION` moves and the change is deliberate.
    """
    d = dict(d)
    d["system"] = System(d["system"])
    d["retrieved"] = [RetrievedPassage(**p) for p in d.get("retrieved", [])]
    claims = []
    for c in d.get("claims", []):
        c = dict(c)
        c["citations"] = [Citation(**x) for x in c.get("citations", [])]
        c["granularity"] = Granularity(c.get("granularity", Granularity.DECONTEXTUALIZED_ATOMIC.value))
        c["verifier_scores"] = [VerifierScore(**v) for v in c.get("verifier_scores", [])]
        c["human_labels"] = [
            HumanLabel(**{**h, "support_label": SupportLabel(h["support_label"])})
            for h in c.get("human_labels", [])
        ]
        claims.append(Claim(**c))
    d["claims"] = claims
    return QueryRecord(**d)


def write_jsonl(path, records: Sequence[Any]) -> None:
    """Append-safe, one record per line, sorted keys so a diff between runs is readable."""
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(to_dict(r), sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def read_query_records(path) -> Iterator[QueryRecord]:
    for d in read_jsonl(path):
        yield query_record_from_dict(d)
