"""Decontextualized atomic claims — **Table 2** (granularity is a config knob, i.e. the C7 rows).

**Off the C2 headline path.** ADR-0009's 2026-08-13 amendment settled that: the shipped systems emit
`CLAIM` lines directly and there is no prose→decompose step in front of the headline number. This
module re-cuts an answer that has *already been generated* into a different unit, so that C7 can
report what the headline number would have been at `sentence` and at bare `atomic`. It is therefore
allowed to exist only now — writing it in W4 would have rebuilt the abandoned architecture and, per
ADR-0009 §4, fixed by default the granularity the parity loop existed to decide. The loop closed
2026-08-14 (`prompts.PARITY_LOOP_CLOSED`), so the unit this decomposer targets is frozen input.

What is frozen, and what it costs to implement:

- The unit is `CONTEXT.md` and ADR-0005: **atomic** (one assertion) and **decontextualized** (every
  pronoun, definite description, and implicit subject resolved). A bare atomic claim with an
  unresolved pronoun is unverifiable in isolation — the verifier takes `premise = cited span,
  hypothesis = claim`, and a non-self-contained hypothesis makes "does the span entail it?"
  indeterminate. An annotator instructed to use no outside knowledge is equally stuck.
- `sentence` and bare `atomic` remain available as `Granularity` settings, but **only as ablation
  rows** — the headline configuration is decontextualized atomic. The three rows differ *only* in
  the unit rule handed to the decomposer, which is why `prompts.DECONTEXTUALIZATION_RULE` and
  `prompts.ATOMICITY_RULE` are separate constants: `atomic` must stay bare, or the ablation compares
  a row against itself.
- Every claim records its char span in the raw generation (`Claim.source_start/source_end`).
  Decomposition quality is an upstream confound on every headline number; `claim_validity` converts
  it into a reportable decomposition-error rate, and the offsets keep it auditable.

Three decisions that the docstring above did not settle, made here:

1. **A claim's span is the span of the sentence it came from, and it is exact.** Atomic and
   decontextualized claims are *rewritten* text — "Metformin reduces mortality" is nowhere in "It
   reduces mortality" — so there is no honest substring to point at. The decomposer therefore works
   from a numbered sentence list and every claim carries the index of its source sentence, whose
   offsets are exact. `Claim.source_start/source_end` answers "which text produced this claim",
   which is the question a decomposition post-mortem asks; it never claims to be a quotation.
2. **The sentences are cut out of the answer's `CLAIM` lines when it has any.** A C7 row's input is
   a `QueryRecord.raw_generation`, which carries `DECISION:` and `CITE:` lines; sentence-splitting
   those would make "DECISION: yes" a unit of the `sentence` row. Prose input (a vanilla answer, a
   pasted paragraph) is split whole. Post-hoc's two stages are **not** accepted joined: both carry
   `CLAIM` lines, so a joined string doubles the denominator of the row silently, and that is a
   `ValueError` rather than a returned error.
3. **Costs and parse failures survive, exactly as in `generate.py`.** Decomposition is a real extra
   model call for the C7 rows and would otherwise be missing from Table 4, and G2's ≥95% valid-parse
   bar cannot be computed from claims that a parser dropped on the floor. Cost rows are stamped
   `component="decompose"` — a C7 run's `costs.jsonl` therefore does not follow
   `scoring.granularity.CALL_ORDER`, which describes a generation run.

**Citations are not this module's business.** A re-cut claim has no citations: the spans belonged to
units that no longer exist. Re-attaching them is a generation-stage question — `generate.cite_claims`
re-runs post-hoc's cite-stage prompt over the new units, positionally, at the cost of one call per
row (chosen over mapping the old citations onto the new claim boundaries; `HANDOFF.md`).

`notebooks/04_3_decompose_then_verify.ipynb` promoted here in shape only — it splits toy sentences on
punctuation and never decontextualizes, which is the hard part and the one this module pays a model
call for.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from . import backends
from .chunk import sentence_spans
from .config import GenerationConfig
from .generate import STAGE_SEPARATOR, Completer
from .prompts import ATOMICITY_RULE, DECONTEXTUALIZATION_RULE, MAX_CLAIM_WORDS
from .schema import Claim, CostRecord, Granularity

#: A `CLAIM` header in generated text, as `parse_response` reads it: the head is case-insensitive
#: and the number is required. Only the header is matched — the body runs to end of line, and its
#: offsets are what a claim's span is measured in.
_CLAIM_HEAD = re.compile(r"^[ \t]*CLAIM[ \t]+\d+[ \t]*:[ \t]*", re.IGNORECASE | re.MULTILINE)

#: Maximum number of sentences sent to the model per `decompose()` LLM call. Long answers (>4
#: sentences) cause an 8B model under frequency penalty to suffer format collapse (syntax drift and
#: dropped sentences). Chunking sentences into batches of at most 4 avoids long generation outputs
#: and guarantees complete coverage.
MAX_SENTENCES_PER_CHUNK = 4

#: The decomposer's own output header: `CLAIM <n> FROM <sentence>`. The claim number is parsed and
#: then **ignored** — ids are assigned in output order. Same lesson as `parse_response`: two live
#: runs showed an 8B model repurposing the number it was asked to write, and reading it as an
#: identity mis-attributed the claim.
_DECOMPOSED_HEAD = re.compile(r"^CLAIM\s+(\d+)\s+FROM\s+(\d+)$", re.IGNORECASE)
_LENIENT_DECOMPOSED_HEAD = re.compile(
    r"^CLAIM\s*[\[\(]?\s*(?:S|[A-Z])?\s*(\d+(?:\.\d+)?|[A-Z])\s*[\]\)]?\s*FROM\s*[\[\(]?\s*(?:SENTENCE|S|implied by sentence)?\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?.*$",
    re.IGNORECASE,
)

#: Bare atomic is a *withholding*, and the model has to be told so explicitly: asked only to split,
#: an instruction-tuned model tidies pronouns away on its own, which would silently turn the
#: `atomic` C7 row into a second `decontextualized_atomic` row.
BARE_ATOMIC_RULE = """Keep the answer's own wording. Do not resolve pronouns, do not replace
"this"/"these" with what they refer to, and do not supply a subject the sentence leaves implicit —
copy the words as they stand, splitting only where the sentence asserts more than one thing."""

DECOMPOSE_TEMPLATE = """You are splitting an answer that is already written into separate claims.

{question_block}Answer, one sentence per line:
{sentences}

Split the answer into claims. Every output line MUST start with 'CLAIM <n> FROM <sentence>:' where <n> is the claim number (1, 2, 3...) and <sentence> is the source sentence number. Use only what the answer says: do not add a fact the answer does not state, do not judge whether the answer is right, and do not leave a sentence out.

{unit_rules}

{format_block}"""

FORMAT_BLOCK = """Reply in exactly this format and add nothing else:

CLAIM 1 FROM 1: the first claim taken from sentence 1
CLAIM 2 FROM 1: the second claim taken from sentence 2
CLAIM 3 FROM 2: the first claim taken from sentence 2

Number the CLAIM lines from 1, in order.
Rules:
- YOU MUST START EVERY LINE WITH "CLAIM <n> FROM <sentence>:" (for example: "CLAIM 1 FROM 1: ..."). NEVER omit "FROM <sentence>".
- Include spaces between keywords and numbers (e.g. write "CLAIM 7 FROM 4", NEVER "CLAIM7FROM4").
- After FROM, write ONLY the plain integer sentence number (e.g. "FROM 4", NEVER "FROM S4", "FROM S4.1", or "FROM (4)").
- Process EVERY numbered sentence from 1 through the last sentence. Do not stop early or leave out trailing sentences.
- A sentence that asserts one thing yields exactly one claim."""


#: Everything the Sep 3 decomposer freeze (ADR-0009 §8) covers, in one fixed order. Not the
#: `Granularity.SENTENCE` row: it makes no model call, so it is a deterministic cut with nothing to
#: freeze — see `unit_rules`.
_FROZEN_FRAGMENTS = (
    DECOMPOSE_TEMPLATE, FORMAT_BLOCK, BARE_ATOMIC_RULE, DECONTEXTUALIZATION_RULE, ATOMICITY_RULE,
)


def decompose_template_digest() -> str:
    """SHA-256 of `_FROZEN_FRAGMENTS`, joined on a byte no fragment contains.

    Mirrors `prompts.post_hoc_answer_template_digest()`: a digest rather than a second copy of the
    text, so the only question ever asked of it is whether the frozen prompt still matches the
    value a test pinned it to.

    **What "frozen" means before Sep 3.** `tests/test_decompose.py` pins this digest today,
    2026-08-15, so an accidental edit is caught immediately rather than discovered in October — the
    same protection `post_hoc_answer_template_digest` gives a template that had already stopped
    moving. It is not yet ADR-0009 §8's freeze: the pinned value is provisional until
    `scripts/decompose_smoke.py` runs on the A4000 and either confirms these fragments or forces a
    dated edit before Sep 3. If Sep 3 arrives with the pin unchanged, that pin *is* the freeze.
    """
    return hashlib.sha256("\x00".join(_FROZEN_FRAGMENTS).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Decomposition:
    """One answer, re-cut. Mirrors `generate.Generation`: the units, what they cost, what failed.

    `errors` is data rather than an exception for the reason `ParsedResponse.errors` is — G2 gates
    on ≥95% valid parse, and a parser that raises on the first malformed line reports one failure
    per answer and hides the rest.
    """

    claims: tuple[Claim, ...]
    costs: tuple[CostRecord, ...]
    errors: tuple[str, ...]


def unit_rules(granularity: Granularity) -> str:
    """The one thing that differs between the C7 rows.

    `SENTENCE` has no rule because it has no model call: a sentence is a deterministic cut, and
    paying a model to reproduce one would put an unnecessary decomposition error into the row that
    exists as the *control* for decomposition error.
    """
    if granularity is Granularity.DECONTEXTUALIZED_ATOMIC:
        return f"{DECONTEXTUALIZATION_RULE} {ATOMICITY_RULE}"
    if granularity is Granularity.ATOMIC:
        return f"{ATOMICITY_RULE}\n\n{BARE_ATOMIC_RULE}"
    raise ValueError(f"{granularity.value} is not decomposed by a model")


def answer_spans(answer: str) -> list[tuple[int, int]]:
    """The regions of `answer` that carry answer text, as offsets into it.

    `CLAIM` line bodies when the response grammar is present, the whole string when it is not — see
    decision 2 in the module docstring.
    """
    heads = list(_CLAIM_HEAD.finditer(answer))
    if not heads:
        return [(0, len(answer))]
    spans: list[tuple[int, int]] = []
    for head in heads:
        stop = answer.find("\n", head.end())
        if stop == -1:
            stop = len(answer)
        if answer[head.end():stop].strip():
            spans.append((head.end(), stop))
    return spans


def sentence_units(answer: str) -> list[tuple[int, int]]:
    """The numbered sentences the decomposer is shown, as exact spans of `answer`.

    This is also the `sentence` granularity's output, which is why it is one function: the ablation
    row and the decomposer's input have to be the same cut, or the rows are not comparable.
    """
    units: list[tuple[int, int]] = []
    for start, end in answer_spans(answer):
        units.extend(_tighten(answer, s, e) for s, e in sentence_spans(answer, start, end))
    return [(s, e) for s, e in units if e > s]


def _tighten(answer: str, start: int, end: int) -> tuple[int, int]:
    """Drop surrounding whitespace from a span, so `answer[start:end] == claim.text` holds exactly.

    The offsets are the audit trail (ADR-0005's decomposition-error rate is read against them), and
    a span whose text does not match the claim is an audit trail that lies about one character.
    """
    while start < end and answer[start].isspace():
        start += 1
    while end > start and answer[end - 1].isspace():
        end -= 1
    return start, end


def build_prompt(
    answer: str,
    granularity: Granularity,
    question: str | None = None,
    units: list[tuple[int, int]] | None = None,
    start_index: int = 1,
) -> str:
    """Render the decomposer prompt for one answer (or a sentence chunk of an answer). Exposed
    because a prompt nobody can print is a prompt nobody can review, and this one is frozen on
    Sep 3 with the rest of the decomposer."""
    if units is None:
        units = sentence_units(answer)
    if not units:
        raise ValueError("answer carries no text to decompose")
    return DECOMPOSE_TEMPLATE.format(
        # The question is optional and, when present, is context only: "it reduces mortality" can
        # need the question to name its subject, and withholding it would push the decomposer into
        # guessing. It is never a source of claims — the format block says so, and the sentence
        # index a claim must carry cannot point at it.
        question_block=f"Question the answer responds to: {question}\n\n" if question else "",
        sentences="\n".join(
            f"{i}. {answer[s:e]}" for i, (s, e) in enumerate(units, start=start_index)
        ),
        unit_rules=unit_rules(granularity),
        format_block=FORMAT_BLOCK,
    )

def parse_decomposition(
    raw: str,
    units: list[tuple[int, int]],
    granularity: Granularity,
    *,
    unit_start_index: int = 1,
    max_claim_words: int = MAX_CLAIM_WORDS,
    check_coverage: bool = True,
) -> tuple[list[Claim], list[str]]:
    """The decomposer's line grammar → claims, plus everything that did not parse.

    Nothing is dropped for being wrong-shaped once it is recognisably a claim: an over-length claim
    is flagged and kept (`MAX_CLAIM_WORDS` is a non-termination detector, and truncating it would
    hide the defect it detects), and a claim whose sentence index is unusable is kept with no span
    rather than deleted, because deleting it would shrink the denominator of the very rate that is
    supposed to report decomposition failure.

    **A repeated claim is the same defect wearing a different shape.** `MAX_CLAIM_WORDS` catches a
    non-terminating generation that grows one claim without bound (`21074975`'s 731 words,
    `parity_iter1b.md`); a live run of this decomposer (`docs/harvest/decompose_smoke.summary.json`,
    2026-08-15) found greedy decoding taking the *other* escape from a repetition loop — re-emitting
    an already-written claim verbatim instead of advancing, sometimes dozens of times, with no upper
    bound but the output cap. Exact-text repetition within one decomposition is therefore also
    flagged (not deduplicated — collapsing it would hide the defect it was added to measure, same
    reasoning as `MAX_CLAIM_WORDS`).
    """
    claims: list[Claim] = []
    errors: list[str] = []
    text_seen: dict[str, str] = {}

    for lineno, line in enumerate(raw.splitlines(), start=1):
        head, sep, text = line.strip().partition(":")
        if not sep:
            continue  # prose the model wrapped around the format; not itself a claim failure
        head, text = head.strip(), text.strip()
        matched = _DECOMPOSED_HEAD.match(head)
        if matched is None:
            lenient_matched = _LENIENT_DECOMPOSED_HEAD.match(head)
            if lenient_matched is None:
                if head.upper().startswith("CLAIM"):
                    errors.append(f"line {lineno}: {head!r} is not 'CLAIM <n> FROM <sentence>'")
                continue
            index = int(float(lenient_matched.group(2)))
        else:
            index = int(matched.group(2))
        if not text:
            errors.append(f"line {lineno}: claim is empty")
            continue
        span: tuple[int, int] | None = None
        rel_global = index - unit_start_index
        rel_chunk = index - 1
        if 0 <= rel_global < len(units):
            span = units[rel_global]
        elif 0 <= rel_chunk < len(units):
            span = units[rel_chunk]
        else:
            errors.append(
                f"line {lineno}: sentence {index} does not exist (the answer has {len(units)})"
            )

        words = len(text.split())
        if words > max_claim_words:
            errors.append(
                f"c{len(claims) + 1}: {words} words exceeds the max claim length of "
                f"{max_claim_words} (non-terminating generation)"
            )
        claims.append(
            Claim(
                claim_id=f"c{len(claims) + 1}",
                text=text,
                granularity=granularity,
                source_start=span[0] if span else None,
                source_end=span[1] if span else None,
            )
        )
        norm = " ".join(text.split()).lower()
        if norm in text_seen:
            errors.append(
                f"{claims[-1].claim_id}: repeats {text_seen[norm]}'s claim text verbatim "
                "(non-terminating generation)"
            )
        else:
            text_seen[norm] = claims[-1].claim_id

    if not claims:
        errors.append("no CLAIM lines")
    elif check_coverage:
        covered = {c.source_start for c in claims}
        missed = [i for i, (s, _) in enumerate(units, start=1) if s not in covered]
        if missed:
            errors.append(
                f"sentences {missed} produced no claim — the answer was dropped, not decomposed"
            )
    return claims, errors


def decompose(
    answer: str,
    config: GenerationConfig,
    *,
    question: str | None = None,
    completer: Completer = backends.complete,
    seed: int = 0,
    run_id: str = "",
    query_id: str | None = None,
    max_claim_words: int = MAX_CLAIM_WORDS,
    max_sentences_per_chunk: int = MAX_SENTENCES_PER_CHUNK,
) -> Decomposition:
    """Re-cut a generated answer into claims at `config.granularity`.

    `answer` is one stage's text — for post-hoc, `generate.split_stages(raw)[-1]`, never the joined
    `raw_generation`. `completer` is injected for the same reason `generate.py` injects it: the box
    is copy-paste only, and a bug found against a live server costs a GPU run.
    """
    if STAGE_SEPARATOR in answer:
        raise ValueError(
            "answer carries both post-hoc stages; pass one stage (generate.split_stages), or the "
            "cite stage's claims are counted twice"
        )
    granularity = Granularity(config.granularity)
    units = sentence_units(answer)

    if granularity is Granularity.SENTENCE:
        return Decomposition(
            claims=tuple(
                Claim(
                    claim_id=f"c{i}",
                    text=answer[s:e],
                    granularity=granularity,
                    source_start=s,
                    source_end=e,
                )
                for i, (s, e) in enumerate(units, start=1)
            ),
            costs=(),
            errors=() if units else ("no sentences",),
        )

    if not units:
        return Decomposition(tuple(), (), ("no sentences",))

    chunks = [
        units[i : i + max_sentences_per_chunk]
        for i in range(0, len(units), max_sentences_per_chunk)
    ]

    all_claims: list[Claim] = []
    costs: list[CostRecord] = []
    errors: list[str] = []
    for chunk_idx, chunk_units in enumerate(chunks):
        chunk_start_index = chunk_idx * max_sentences_per_chunk + 1
        chunk_query_id = (
            f"{query_id}:chunk{chunk_idx}" if len(chunks) > 1 and query_id else query_id
        )
        prompt = build_prompt(
            answer, granularity, question, units=chunk_units, start_index=chunk_start_index
        )
        raw, cost = completer(
            prompt, config, seed=seed, run_id=run_id, query_id=chunk_query_id
        )
        # Table 4 must be able to tell a C7 row's decomposition call from the generation it re-cuts;
        # `backends` stamps every call "generate" because generation is all it has ever been asked for.
        cost.component = "decompose"
        costs.append(cost)

        chunk_claims, chunk_errors = parse_decomposition(
            raw,
            chunk_units,
            granularity,
            unit_start_index=chunk_start_index,
            max_claim_words=max_claim_words,
            check_coverage=len(chunks) == 1,
        )
        errors.extend(chunk_errors)

        for claim in chunk_claims:
            claim_id = f"c{len(all_claims) + 1}"
            all_claims.append(
                Claim(
                    claim_id=claim_id,
                    text=claim.text,
                    granularity=claim.granularity,
                    source_start=claim.source_start,
                    source_end=claim.source_end,
                )
            )

    if len(chunks) > 1:
        if not all_claims:
            errors.append("no CLAIM lines")
        else:
            covered = {c.source_start for c in all_claims if c.source_start is not None}
            missed = [i for i, (s, _) in enumerate(units, start=1) if s not in covered]
            if missed:
                errors.append(
                    f"sentences {missed} produced no claim — the answer was dropped, not decomposed"
                )

    return Decomposition(tuple(all_claims), tuple(costs), tuple(errors))
