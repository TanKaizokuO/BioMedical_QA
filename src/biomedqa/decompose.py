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
   one sentence at a time: the call names the sentence, so the claim's span is that sentence's, by
   construction rather than by an index the model writes and can get wrong.
   `Claim.source_start/source_end` answers "which text produced this claim", which is the question a
   decomposition post-mortem asks; it never claims to be a quotation.
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

#: The decomposer's own output header: `CLAIM <n>`. The number is parsed and then **ignored** — ids
#: are assigned in output order. Same lesson as `parse_response`: two live runs showed an 8B model
#: repurposing the number it was asked to write, and reading it as an identity mis-attributed the
#: claim.
#:
#: There is deliberately **no `FROM <sentence>` field any more.** It was the decomposer's single
#: largest source of error: a live A4000 run (2026-08-16) produced `CLAIM7FROM4`, `FROM S4.1`,
#: `FROM (6)` and, worse than any syntax drift, indices that pointed at the wrong sentence — the
#: claim "CEA use is not associated with stroke mortality" arriving stamped `FROM 1`, whose sentence
#: is about 14-fold variation. A mis-stamped index is invisible in the rates and silently corrupts
#: `Claim.source_start/source_end`, which the decomposition post-mortem reads as ground truth. One
#: call per sentence removes the field and the failure mode with it: the source sentence is what the
#: call was about, so the span is exact by construction rather than by the model's bookkeeping.
_DECOMPOSED_HEAD = re.compile(r"^CLAIM\s+(\d+)$", re.IGNORECASE)

#: The same header after the drift an 8B model applies to it: `CLAIM1`, `- CLAIM 1`, `**CLAIM 1**`,
#: `CLAIM S1`. Accepted so that a well-formed claim is not thrown away over its bullet, and counted
#: as drift by the caller rather than silently normalised away.
_LENIENT_DECOMPOSED_HEAD = re.compile(
    r"^[-*\u2022\s]*\**\s*CLAIM\s*[\[\(]?\s*(?:S|SENTENCE)?\s*(\d+(?:\.\d+)?)?\s*[\]\)]?\s*\**$",
    re.IGNORECASE,
)

#: Bare atomic is a *withholding*, and the model has to be told so explicitly: asked only to split,
#: an instruction-tuned model tidies pronouns away on its own, which would silently turn the
#: `atomic` C7 row into a second `decontextualized_atomic` row.
BARE_ATOMIC_RULE = """Keep the answer's own wording. Do not resolve pronouns, do not replace
"this"/"these" with what they refer to, and do not supply a subject the sentence leaves implicit —
copy the words as they stand, splitting only where the sentence asserts more than one thing."""

DECOMPOSE_TEMPLATE = """You are splitting one sentence of an answer that is already written into separate claims.

{question_block}The whole answer, one sentence per line:
{sentences}

Split sentence {target}, and only sentence {target}, into claims. Use only what sentence {target}
says: do not add a fact it does not state, and do not judge whether it is right. The other
sentences are shown for one reason only — so you can tell what a pronoun or a "this" in sentence
{target} refers to. Never write a claim for any other sentence.

{unit_rules}

{format_block}"""

FORMAT_BLOCK = """Reply in exactly this format and add nothing else:

CLAIM 1: the first claim the sentence makes
CLAIM 2: the second claim the sentence makes

Rules:
- Every line starts with "CLAIM <n>:", numbered from 1. Write nothing before the first CLAIM line
  and nothing after the last one.
- A sentence that asserts one thing yields exactly one claim: write that single line and stop.
- Never write the same claim twice, and never restate in different words a claim you have already
  written. Most sentences yield one or two claims; more than three is almost always repetition.
- Stop as soon as the sentence you were asked to split has been covered."""


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
    target: int = 1,
) -> str:
    """Render the decomposer prompt for **one sentence** of an answer. `target` is that sentence's
    1-based position in `units`; the rest of the answer is still shown, because a decontextualizing
    decomposer cannot resolve "it" from the target sentence alone.

    Exposed because a prompt nobody can print is a prompt nobody can review, and this one is frozen
    on Sep 3 with the rest of the decomposer."""
    if units is None:
        units = sentence_units(answer)
    if not units:
        raise ValueError("answer carries no text to decompose")
    if not 1 <= target <= len(units):
        raise ValueError(f"target sentence {target} is outside the answer's {len(units)} sentences")
    return DECOMPOSE_TEMPLATE.format(
        # The question is optional and, when present, is context only: "it reduces mortality" can
        # need the question to name its subject, and withholding it would push the decomposer into
        # guessing. It is never a source of claims — the prompt names one target sentence, and the
        # question is not one of the numbered lines.
        question_block=f"Question the answer responds to: {question}\n\n" if question else "",
        sentences="\n".join(f"{i}. {answer[s:e]}" for i, (s, e) in enumerate(units, start=1)),
        target=target,
        unit_rules=unit_rules(granularity),
        format_block=FORMAT_BLOCK,
    )


def parse_decomposition(
    raw: str,
    unit: tuple[int, int],
    granularity: Granularity,
    *,
    max_claim_words: int = MAX_CLAIM_WORDS,
) -> tuple[list[Claim], list[str]]:
    """One sentence's reply → its claims, plus everything that did not parse.

    `unit` is the span of the sentence the call was about, and it is what every claim from this
    reply points at. There is no index to resolve and none to get wrong: that is the whole reason
    the decomposer asks for one sentence at a time (`_DECOMPOSED_HEAD`).

    Nothing is dropped for being wrong-shaped once it is recognisably a claim: an over-length claim
    is flagged and kept (`MAX_CLAIM_WORDS` is a non-termination detector, and truncating it would
    hide the defect it detects).

    **A repeated claim is the same defect wearing a different shape.** `MAX_CLAIM_WORDS` catches a
    non-terminating generation that grows one claim without bound (`21074975`'s 731 words,
    `parity_iter1b.md`); live runs of this decomposer found greedy decoding taking the *other*
    escape from a repetition loop — re-emitting an already-written claim verbatim instead of
    advancing. Exact-text repetition within one sentence's reply is therefore also flagged (not
    deduplicated — collapsing it would hide the defect it was added to measure, same reasoning as
    `MAX_CLAIM_WORDS`).
    """
    claims: list[Claim] = []
    errors: list[str] = []
    text_seen: dict[str, str] = {}

    for lineno, line in enumerate(raw.splitlines(), start=1):
        head, sep, text = line.strip().partition(":")
        if not sep:
            continue  # prose the model wrapped around the format; not itself a claim failure
        head, text = head.strip(), text.strip()
        if _DECOMPOSED_HEAD.match(head) is None:
            if _LENIENT_DECOMPOSED_HEAD.match(head) is None:
                if "CLAIM" in head.upper():
                    errors.append(f"line {lineno}: {head!r} is not 'CLAIM <n>'")
                continue
        if not text:
            errors.append(f"line {lineno}: claim is empty")
            continue

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
                source_start=unit[0],
                source_end=unit[1],
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
) -> Decomposition:
    """Re-cut a generated answer into claims at `config.granularity`.

    `answer` is one stage's text — for post-hoc, `generate.split_stages(raw)[-1]`, never the joined
    `raw_generation`. `completer` is injected for the same reason `generate.py` injects it: the box
    is copy-paste only, and a bug found against a live server costs a GPU run.

    **One model call per sentence.** The earlier design sent the whole answer, then chunks of four
    sentences, and asked the model to stamp each claim with the sentence it came from. Both
    collapsed on the same hardware for the same reason: the longer the reply, the further greedy
    decoding drifts, and a live A4000 run (2026-08-16) measured an 18-sentence answer coming back as
    52 claims — repetition loops, sentence indices pointing at the wrong sentence, and dropped
    trailing sentences. A per-sentence call bounds the reply to what one sentence can say, gives
    every sentence its own chance to be covered, and makes the claim's span exact by construction.
    The whole answer still appears in each prompt as context, or the decontextualized row could not
    resolve a pronoun.
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

    all_claims: list[Claim] = []
    costs: list[CostRecord] = []
    errors: list[str] = []
    #: Claim text -> (claim id, source sentence index), to tell a decoding loop that spans calls
    #: from an answer that repeats itself. They are different defects with different owners.
    seen_across: dict[str, tuple[str, int]] = {}

    for target, unit in enumerate(units, start=1):
        sentence_query_id = (
            f"{query_id}:s{target}" if len(units) > 1 and query_id else query_id
        )
        prompt = build_prompt(answer, granularity, question, units=units, target=target)
        raw, cost = completer(
            prompt, config, seed=seed, run_id=run_id, query_id=sentence_query_id
        )
        # Table 4 must be able to tell a C7 row's decomposition call from the generation it re-cuts;
        # `backends` stamps every call "generate" because generation is all it has ever been asked for.
        cost.component = "decompose"
        costs.append(cost)

        sentence_claims, sentence_errors = parse_decomposition(
            raw, unit, granularity, max_claim_words=max_claim_words
        )
        errors.extend(sentence_errors)
        if not sentence_claims:
            errors.append(
                f"sentence {target} produced no claim — the answer was dropped, not decomposed"
            )

        for claim in sentence_claims:
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
            norm = " ".join(claim.text.split()).lower()
            first = seen_across.get(norm)
            if first is None:
                seen_across[norm] = (claim_id, target)
            elif first[1] == target:
                pass  # already reported by `parse_decomposition`, which owns one reply
            elif answer[units[first[1] - 1][0] : units[first[1] - 1][1]] == answer[unit[0] : unit[1]]:
                # The answer says the same sentence twice, so the same claim twice is the honest
                # reading of it. An upstream defect, and not the decomposer's to be charged with.
                errors.append(
                    f"{claim_id}: sentence {target} repeats sentence {first[1]} verbatim in the "
                    "answer being decomposed"
                )
            else:
                errors.append(
                    f"{claim_id}: repeats {first[0]}'s claim text verbatim across sentences "
                    f"{first[1]} and {target} (non-terminating generation)"
                )

    if not all_claims:
        errors.append("no CLAIM lines")

    return Decomposition(tuple(all_claims), tuple(costs), tuple(errors))
