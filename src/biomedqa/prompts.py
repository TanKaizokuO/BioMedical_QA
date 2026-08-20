"""Prompts for joint / post-hoc / vanilla, and the equal-effort ledger that keeps them fair.

**W3 deliverable.** `generate.py` (W4) renders these and calls the backend; nothing here talks to a
model. Kept separate on purpose: the prompt text is the experimental treatment in C2, so it wants
its own file, its own tests, and its own revision history rather than being buried in a call site.

Three decisions are load-bearing.

**1. Citations are emitted as verbatim quotes, not character offsets.**
`Citation` is `{passage_id, char_start, char_end, quoted_text}`, and `QueryRecord.validate()`
requires `len(quoted_text) == char_end - char_start`. A 4-bit 8B model cannot count characters, and
asking it to would convert a generation problem into an arithmetic one it fails at. It can copy.
So the prompt asks for a quote, and `locate_quote()` recovers the span by exact search. The offsets
are then correct by construction rather than by the model's good intentions.

A quote that is not found verbatim in the cited passage is **not repaired** — no fuzzy match, no
nearest-substring. It is returned as a parse error and counted. `schema.validate()` sets the
precedent ("report contract violations; never repair them"), and G2 gates on ≥95% valid claim
parse, which is only a real gate if the failures are allowed to appear.

**2. The response format is line-oriented, not JSON.**
Citations quote biomedical prose containing commas, colons, brackets, percent signs, and — often —
double quotes. Asking a small quantised model for nested JSON puts the dominant failure mode
(escaping a quote inside a quoted string) directly on top of the field the whole experiment
measures. `||` separates the passage id from the quote, and only the first occurrence splits, so a
quote containing `||` still parses. The id is written in square brackets, the same surface form
`render_context` gives it: an example that spells the id differently from the context block teaches
the model a syntax the parser then rejects, and the loss is booked against the system instead of
the harness. `parse_response` strips the brackets before looking the id up.

**3. The citation cap is rendered from `GenerationConfig.max_citations`, never typed into the
prose.** ADR-0005 and `CONTEXT.md` both make an unequal cap the thing that would turn C2's gap into
a budget artifact. One source, three prompts, so a C7 ablation that changes the cap changes it
everywhere or not at all.

**4. A CITE line supports the CLAIM line above it, and carries no number.** The first two live
smoke runs both showed the 8B generator numbering CITE lines 1..k *within* a claim rather than
after the claim they support. Read as a claim id, every claim's first citation was attributed to
c1 — which both corrupted the citation-to-claim mapping that C2 measures and produced cap
violations for claims that had cited once. The number was redundant with line order from the
start, so the grammar drops it rather than asking a 4-bit 8B model to maintain a counter it has
twice failed to maintain. `parse_response` ignores a number if one is written anyway.

Context depth is 10 passages, following ADR-0015: G1 is gated at hit@10, so the passage set the
generator sees is the set the gate certified. Drafting against 5 would grade a context the pipeline
does not produce.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

from .schema import Citation, Claim, Granularity, RetrievedPassage, System

#: ADR-0015 moved G1's k from 5 to 10, so the certified context is 10 passages deep.
CONTEXT_DEPTH = 10

#: PubMedQA's label set. No abstain token: ADR-0010 derives abstention at scoring time, and giving
#: the model an explicit escape hatch would manufacture the behaviour the analysis is measuring.
DECISIONS = ("yes", "no", "maybe")

#: The largest claim, in whitespace-delimited words, that `parse_response` accepts without flagging
#: it. Not a style rule — a non-termination detector. Greedy decoding (`temperature: 0.0`) has no
#: way out of a repetition loop, and joint `21074975` in `parity_iter1b` walked one to 731 words: 13
#: CLAIM lines where each re-emitted its predecessor plus one more clause. Every such claim scored
#: 0.0 recall, so the decoder's failure was being read as the system's failure to ground.
#:
#: 50 rather than 30 because claim-length p95 is 29 / 29 / 34 words for joint / post-hoc / vanilla:
#: a guard at 30 flags 4.73% / 3.06% / 9.43% of claims, which breaks G2's >=95% valid-parse bar on
#: vanilla on its own and taxes the three arms at three different rates, moving C2's gap by
#: instrument. At 50 the cost is 2.78% / 0.24% / 0.25% — joint's excess is its own degeneracy.
#:
#: A **scoring** rule, not a generation knob (ADR-0010): parse errors are re-derived from
#: `raw_generation`, so revising this re-scores existing records and never forces a re-run.
#: `ScoringConfig.max_claim_words` defaults to it; this is the single copy.
MAX_CLAIM_WORDS = 50

def claim_stem(text: str) -> str:
    """Whitespace-collapsed, lowercased, trailing sentence punctuation stripped."""
    return " ".join(text.split()).lower().rstrip(" .,;:")


#: Minimum nested-extension chain length charged as a non-terminating generation.
#:
#: On `docs/harvest/parity_iter1b.records.jsonl`, the chain-length distribution is joint
#: `{2: 22, 3: 1, 10: 1, 11: 1}`, post_hoc `{2: 2, 5: 1}`, vanilla `{2: 1}` -- a large x2 mode
#: plus a sparse loop tail, the same population split ADR-0019 §1 used to justify its x3 threshold,
#: which is why `RUNAWAY_CHAIN_MIN` is 3 and a chain of 2 is `recovered` rather than `errors`.
#:
#: The guard is not redundant with `MAX_CLAIM_WORDS`: joint `17578985` c15/c16/c17 is a 3-chain at
#: 19/26/32 words, wholly invisible to the 50-word guard, and 9 joint + 3 post_hoc chained claims
#: are under 50 words. The detector fires on post_hoc as well as joint, so it is symmetric across
#: the C2 arms and cannot flatter either.
RUNAWAY_CHAIN_MIN = 3


def runaway_chains(texts: Sequence[str], *, min_length: int = 2) -> list[tuple[int, int]]:
    """`(start_index, length)` for each maximal run of consecutive claims where each claim's
    `claim_stem` strictly extends its predecessor's at a non-alphanumeric boundary.
    """
    stems = [claim_stem(t) for t in texts]
    chains: list[tuple[int, int]] = []
    n = len(stems)
    i = 0
    while i < n - 1:
        j = i
        while j < n - 1:
            a = stems[j]
            b = stems[j + 1]
            if len(b) > len(a) and b.startswith(a) and not b[len(a)].isalnum():
                j += 1
            else:
                break
        length = j - i + 1
        if length >= min_length:
            chains.append((i, length))
        i = j + 1
    return chains

_CITATION_SEP = "||"


@dataclass(frozen=True, slots=True)
class Iteration:
    """One round of the equal-effort loop: read outputs, change the prompt, re-run.

    The unit is a *cycle*, not a file edit. A post-hoc cycle may touch both of its stages while a
    joint cycle touches one, and counting edits would charge post-hoc double for the same amount of
    thinking. What ADR-0002's fairness requirement is about is attention spent, and a cycle is the
    honest quantum of that.
    """

    n: int
    date: str
    rationale: str
    stages_touched: tuple[str, ...]


#: **The equal-effort ledger.** `generate.py` reads it, `scripts/prompt_iterations.py` reports it,
#: and `tests/test_prompts.py` fails the build when JOINT and POST_HOC drift apart. The pre-emptive
#: answer to "your post-hoc baseline is a straw man" is a number in the paper, and a number nobody
#: is forced to maintain is a number that will be reconstructed from memory in October.
PROMPT_ITERATIONS: dict[System, tuple[Iteration, ...]] = {
    System.JOINT: (
        Iteration(
            n=1,
            date="2026-08-10",
            rationale="Initial draft against dev retrievals; quote-based citations, 10-passage "
            "context per ADR-0015.",
            stages_touched=("joint",),
        ),
        Iteration(
            n=2,
            date="2026-08-10",
            rationale="W4 live smoke: 0/3 clean parses. Bracketed the passage id in the format "
            "example to match render_context, and named the two compliance failures the 8B model "
            "actually made — dropping '||' and restarting CITE numbering at 1 under each claim.",
            stages_touched=("joint",),
        ),
        Iteration(
            n=3,
            date="2026-08-10",
            rationale="Second live smoke: citations recovered (0 -> 5/11/4) but the model still "
            "numbered CITE lines 1..k within each claim, so every claim's first citation landed on "
            "c1 and manufactured cap violations. Dropped the number from the grammar — a CITE line "
            "now supports the CLAIM above it — and told the model to copy the ':N' chunk suffix.",
            stages_touched=("joint",),
        ),
        Iteration(
            n=4,
            date="2026-08-10",
            rationale="Third live smoke: attribution and cap violations gone, quotes still the "
            "binding failure. Scoped the decontextualization rule to CLAIM lines — unscoped, the "
            "model applied it to quotes and composed standalone sentences the passage does not "
            "contain — and told it to keep the passage's own spelling of numbers.",
            stages_touched=("joint",),
        ),
    ),
    System.POST_HOC: (
        Iteration(
            n=1,
            date="2026-08-10",
            rationale="Initial draft. Same context, same cap, same output grammar as joint; the "
            "only difference is that the answer is written before any passage is cited.",
            stages_touched=("answer", "cite"),
        ),
        Iteration(
            n=2,
            date="2026-08-10",
            rationale="Same format-block fix as joint, and only the cite stage sees it. The answer "
            "stage is untouched, so the pass that writes claims still knows nothing about "
            "citations.",
            stages_touched=("cite",),
        ),
        Iteration(
            n=3,
            date="2026-08-10",
            rationale="Same positional-CITE grammar as joint, cite stage only. The answer stage "
            "still never hears about citations.",
            stages_touched=("cite",),
        ),
        Iteration(
            n=4,
            date="2026-08-10",
            rationale="Same quote-scoping fix as joint. The cite stage carries it; the answer "
            "stage sees only the CLAIM-line wording change every system got.",
            stages_touched=("cite",),
        ),
    ),
    System.VANILLA: (
        Iteration(
            n=1,
            date="2026-08-10",
            rationale="Initial draft. Same context and same claim grammar as joint, minus "
            "citations, which it lacks by construction (schema.py:73).",
            stages_touched=("answer",),
        ),
        Iteration(
            n=2,
            date="2026-08-10",
            rationale="Claim rules now say 'each CLAIM line' rather than 'each claim', so the "
            "shared claim unit stays word-for-word identical across all three systems. No "
            "citation wording reaches vanilla, which still cites nothing by construction.",
            stages_touched=("answer",),
        ),
    ),
}


#: **The parity ledger — ADR-0009 §7's third line, charged to neither system.**
#:
#: The blind granularity-parity loop tunes `POST_HOC_ANSWER_TEMPLATE` and nothing else (§4), so its
#: cycles are a fairness-control cost rather than method development. Booking them to POST_HOC would
#: report the baseline as more heavily engineered than it was; booking a matching cycle to JOINT to
#: keep `effort_is_matched()` true would spend cycles on a prompt §4 puts out of bounds, and inflate
#: the effort number the paper prints. So they are counted here, separately, and `iteration_counts()`
#: never sees them.
#:
#: Bounded by §5: **a hard 10, or Aug 30, whichever comes first.** Not "~10".
PARITY_ITERATIONS: tuple[Iteration, ...] = (
    Iteration(
        n=1,
        date="2026-08-14",
        rationale="Iteration 0 (docs/harvest/parity_iter0.md) measured joint 16 vs post-hoc 20 "
        "median words/claim, +25% against a ±15% tolerance. The diagnostic said the excess is "
        "mostly verbosity, not compounding: restricted to claims carrying no compound marker at "
        "all, the gap is still +21% (14 vs 17), and coordination by 'and' is already identical "
        "across the arms (28.4% vs 26.9%) because _claim_rules() splits it for both. So this "
        "cycle adds a length target, promotes trailing qualifier clauses to their own claims — "
        "post-hoc's subordinate-clause rate is 5.6% against joint's 0.9% — and bans study-framing "
        "preamble ('The study found no evidence that X'), which spends words on the study rather "
        "than the assertion. Says nothing about citing: a first pass that knows quotes are coming "
        "is doing joint grounding, and the gap would close for the wrong reason "
        "(tests/test_prompts.py::test_post_hoc_first_pass_never_mentions_citing).",
        stages_touched=("answer",),
    ),
)

#: §5's hard bound on the loop, kept next to the ledger it bounds.
PARITY_ITERATION_LIMIT = 10


@dataclass(frozen=True, slots=True)
class ParityTermination:
    """§5's stopping event, recorded as data because it is also §6's unblinding event.

    Under ADR-0009 the parity freeze and the first citation-F1 computation are **the same event**, so
    "the loop is closed" cannot be a note in a markdown file — something has to be able to *check*
    it before F1 is computed. `scoring.citation.citation_f1` refuses to run while this is `None`.

    `post_hoc_answer_template_sha256` is the freeze itself: the digest of the template as it stood on
    the terminating run. Editing `POST_HOC_ANSWER_TEMPLATE` after termination breaks
    `tests/test_prompts.py`, which is the only kind of freeze that survives a future session that has
    forgotten this ADR.
    """

    date: str
    #: Artifact prefix the verdict was taken on, under `docs/harvest/`.
    run: str
    iterations_used: int
    #: The gated basis, and the two medians behind the gap.
    basis: str
    joint_median_words_per_claim: float
    post_hoc_median_words_per_claim: float
    gap: float
    #: `gap_bootstrap_ci` on the gated basis: 4000 draws, queries resampled, seed 0.
    interval: tuple[float, float]
    #: §5's one-sided fallback. True keeps the W9 stratified robustness check mandatory.
    residual_favours_c2: bool
    post_hoc_answer_template_sha256: str
    rationale: str


#: **§5's termination and §6's unblinding, in one record.** `None` means the loop is still open and no
#: citation-F1 may be computed on any split, in any form.
#:
#: Closed on iteration 1 of 10, five days inside the Aug 30 drop-dead, because the gate passed on
#: **every** basis and the loop's own measurements showed the residual is below what the gate can
#: resolve. Reopening this is a decision about the paper's methods section, not a code change: the
#: unblinding has happened, so a further iteration would be tuning post-hoc's prompt with citation-F1
#: known — the one thing §6 exists to prevent.
PARITY_LOOP_CLOSED: ParityTermination | None = ParityTermination(
    date="2026-08-14",
    run="parity_iter1b",
    iterations_used=1,
    basis="all records",
    joint_median_words_per_claim=15.0,
    post_hoc_median_words_per_claim=17.0,
    gap=2 / 15,
    interval=(0.0, 1 / 7),
    residual_favours_c2=True,
    post_hoc_answer_template_sha256=(
        "91bc7dddd62db4d6d37c26a91f05f938b22dafcca7a6e5aed4509c714f25ac1a"
    ),
    rationale="The gate passes on all three bases at a 3584 cap — all records +13.3% (joint 15 vs "
    "post-hoc 17), untruncated per arm +14.3%, and untruncated on the same 78 queries in both arms "
    "+6.7% — where the baseline of record (parity_iter0b) failed all three (+25.0% / +42.9% / "
    "+37.9%). The loop stops one iteration in, not because the budget ran out, but because it has "
    "run out of resolution: parity_iter1 and parity_iter1b ran the SAME post-hoc prompt at 2560 and "
    "3584 and read +0.0% and +13.3% on the same basis, the gated statistic being an integer median "
    "of 14-20 words where one word is ~6.7% and the tolerance is two words wide. The query-level "
    "bootstrap separates the movement from the residual: [+18.8%, +40.0%] at the baseline against "
    "[+0.0%, +14.3%] here, non-overlapping, so the edit did close a real gap — while the residual "
    "is one grid step and a further iteration would be fitting run-to-run noise. The residual "
    "favours C2 on every basis, so §5's W9 stratified robustness check stays mandatory; a passing "
    "iteration does not retract a pre-registered check. Full argument: "
    "docs/harvest/parity_iter1b.md.",
)


def iteration_counts() -> dict[str, int]:
    """Cycles spent per system — the number the paper reports.

    Deliberately blind to `PARITY_ITERATIONS`: see that ledger's note.
    """
    return {s.value: len(v) for s, v in PROMPT_ITERATIONS.items()}


def parity_iteration_count() -> int:
    """Cycles spent on the ADR-0009 parity loop, reported as its own line."""
    return len(PARITY_ITERATIONS)


def parity_budget_remains() -> bool:
    """Is there a §5 iteration left? The loop stops on `False` whether or not parity was reached."""
    return parity_iteration_count() < PARITY_ITERATION_LIMIT


def parity_loop_is_open() -> bool:
    """Is the ADR-0009 blind still on? While this is `True`, citation-F1 must not be computed on any
    split, in any form (§6) — `scoring.citation.citation_f1` enforces exactly that."""
    return PARITY_LOOP_CLOSED is None


def post_hoc_answer_template_digest() -> str:
    """SHA-256 of `POST_HOC_ANSWER_TEMPLATE`, the quantity the §8 freeze is checked against.

    A digest rather than a copy of the text: a copy is a second source of truth that drifts, and the
    only question ever asked of it is whether the template still matches the run the loop terminated
    on.
    """
    return hashlib.sha256(POST_HOC_ANSWER_TEMPLATE.encode("utf-8")).hexdigest()


def effort_is_matched() -> bool:
    """Are the two systems C2 compares on equal footing?

    Vanilla is excluded deliberately: it contributes nothing to citation-F1 (ADR-0010), so holding
    it to the same cycle count would be effort spent to protect a comparison nobody makes.
    """
    counts = iteration_counts()
    return counts[System.JOINT.value] == counts[System.POST_HOC.value]


def render_context(passages: list[RetrievedPassage], depth: int = CONTEXT_DEPTH) -> str:
    """The passage block, identical in all three prompts.

    Raises rather than silently truncating on missing text: a passage rendered as an empty body is
    a passage the model cannot cite, and it would show up as a citation-recall loss attributed to
    the system instead of to the harness. `retrieve.py`'s `_rerank` refuses the same way.
    """
    selected = sorted(passages, key=lambda p: p.rank)[:depth]
    missing = [p.passage_id for p in selected if not p.text]
    if missing:
        raise ValueError(
            f"context needs passage text; {len(missing)} of {len(selected)} have none "
            f"(first: {missing[0]}). Load the index with passage_texts."
        )
    return "\n\n".join(f"[{p.passage_id}]\n{p.text}" for p in selected)


#: The two halves of ADR-0005's unit, as separate constants so that `decompose.py` can ask for one
#: without the other: the `atomic` C7 ablation row is *bare* atomic, and a decomposer that resolved
#: pronouns anyway would make the `atomic` and `decontextualized_atomic` rows the same row.
#: `_claim_rules()` joins them with a single space and is byte-identical to what it was before the
#: split — the three systems' prompts are unchanged, and `tests/test_prompts.py` pins the fragments.
DECONTEXTUALIZATION_RULE = """Write each CLAIM line so it stands alone. Resolve every pronoun, every "this"/"these",
and every implied subject, so that a reader who sees the claim and nothing else knows exactly what
it asserts."""

ATOMICITY_RULE = """Each claim states exactly one thing; split anything joined by "and" into separate
claims when the parts could be true or false independently."""


def _claim_rules() -> str:
    """Decontextualization and atomicity — ADR-0005's unit. Shared by all three systems, because
    a baseline whose claims are shaped differently is being compared on the wrong axis.

    Scoped to CLAIM lines explicitly. Unscoped, it read as advice about the whole reply, and the
    live smokes showed the model dutifully decontextualizing its *quotes* as well.
    """
    return f"{DECONTEXTUALIZATION_RULE} {ATOMICITY_RULE}"


def _citation_rules(max_citations: int) -> str:
    """The cap and the verbatim requirement. Withheld from post-hoc's first pass on purpose: a
    pass that knows it will be cited is already grounding jointly.

    The scoping sentence is load-bearing, and it is here because of measured failures rather than
    taste. Live smokes 2 and 3 showed the model composing tidy standalone quotes — "no association
    between utilisation rates for CEA and admission rates for stroke" out of a passage reading
    "... and district stroke mortality (r=-0.06 ...) or admission rates for stroke (r=0.17 ...)" —
    and normalising "Fourteen" to "14". Both yield text the passage does not contain, which
    `locate_quote` refuses, so the citation is lost to a formatting habit rather than to bad
    grounding.
    """
    return f"""Cite at most {max_citations} passages per claim. A quote is a span copied out of one
passage, not a sentence you compose. Copy it character for character: the same capitalisation, the
same punctuation, the same parentheses, and numbers spelled the way the passage spells them — if
the passage says "Fourteen", the quote says "Fourteen", not "14". A quote may begin or end in the
middle of a sentence, and that is correct; do not tidy it into a standalone sentence, and do not
apply the CLAIM rules to it. Never join text that is separated in the passage, and never join text
from two passages, into one quote. Cite more than one passage only when the claim genuinely needs
them together — for instance when one gives the dose and another the outcome."""


def _format_block(max_citations: int, cite: bool, decision: bool = True) -> str:
    """`decision=False` is C7's re-citation only. That stage re-cites claims cut out of an answer
    that already produced its decision, so asking for one again buys nothing and costs a failure
    mode: a live A4000 run logged `no DECISION line` on replies that were otherwise perfect."""
    decision_line = f"DECISION: one of {', '.join(DECISIONS)}\n" if decision else ""
    if not cite:
        return f"""Reply in exactly this format and add nothing else:

{decision_line}CLAIM 1: the first claim
CLAIM 2: the second claim"""
    return f"""Reply in exactly this format and add nothing else:

{decision_line}CLAIM 1: the first claim
CITE: [passage_id] {_CITATION_SEP} exact quote from that passage
CLAIM 2: the second claim
CITE: [passage_id] {_CITATION_SEP} exact quote from that passage
CITE: [passage_id] {_CITATION_SEP} a second exact quote, if the claim needs it

A CITE line supports the CLAIM line directly above it. Write CITE with no number after it.
Copy the passage id exactly as it appears in the brackets above the passage text. Every id ends in
a colon and a number; copy that too, so [name:0] is cited as [name:0] and never as [name].
Put {_CITATION_SEP} between the passage id and the quote. A CITE line with no {_CITATION_SEP} is discarded.
Write the quote as bare text after {_CITATION_SEP}, and do not wrap it in quotation marks.
A claim may have up to {max_citations} CITE lines.
Every CITE line must name one of the passage ids listed above."""


JOINT_TEMPLATE = """You are answering a biomedical research question using only the passages below.

{context}

Question: {question}

Answer the question using only what the passages say. Do not use outside knowledge, and do not
state anything the passages do not support. Write the answer as a list of claims, and support each
claim with a quotation as you write it.

{claim_rules}

{format_block}"""


#: `generate_fp05_n100_guided_both` (2026-08-20) put 11/100 joint calls into an xgrammar
#: "death loop": json_schema's default `[ \t\n]*` whitespace grammar let greedy decoding walk
#: into an unbounded run of indentation tokens inside a claim's `text`/`quote` string, burning the
#: full completion cap on tabs and returning truncated, invalid JSON. Every one of the 11 traces
#: shares a precursor: the model, unprompted, writes a claim for *every* context passage, including
#: ones the question has nothing to do with ("[No claim about X can be made from passage Y].") --
#: e.g. joint 17578985 wrote four such claims back to back before the loop started on the fifth.
#: Nothing in this template asks for one claim per passage; `max_claims=30` in
#: `build_citation_response_format` just gives the model enough headroom to try. A server-side fix
#: (`--structured-outputs-config disable_any_whitespace`) was tried first and reverted -- it crashes
#: vLLM 0.26.0's xgrammar backend on every startup on this box (`structural_tag.py`
#: `SequenceFormat.model_rebuild()` raises a pydantic-core `SchemaError`), verified on 12+ attempts.
#:
#: Two prompt-level mitigations below reduce the loop's incidence but do not eliminate it (measured
#: worse on the full dev set than the 11-query subset they were tuned against: 11/100 -> 15/100, a
#: *different* 15, not a subset -- greedy decoding is chaotic near this defect, and prompt text
#: that changes shifts which trajectories fall into it rather than uniformly reducing them). The
#: real backstop is `generate.py`'s System.JOINT guided branch, which retries a death-loop reply up
#: to twice at rising nonzero temperature; see its comment for the measurement. Net result on
#: `generate_fp05_n100_guided_v4`: 97/100 clean parses, clearing G2's >=95% bar.
#:
#: A third instruction (length target) is unrelated to the death-loop and fixes a different
#: measured defect: the first two mitigations, run together on the full 100, left joint's median
#: claim length at 13 words against post-hoc's 17 (W9
#: `docs/harvest/w9_stratified_parity_both_guided.md` gap widened from +21.4% to +30.8%). Nothing
#: here told the model to write *short* claims; the "format compactly" instruction was about JSON
#: whitespace, and evidently over-generalised to claim prose too. The length-target sentence names
#: the target explicitly and says compactness is about the JSON only, not the claim.
#: This also carries the `claim_rules` and JSON-reply instruction; see `JOINT_TEMPLATE` above for
#: the unguided fallback these lines mirror.
#:
#: Deliberately not booked to `PROMPT_ITERATIONS[JOINT]`: it fixes a guided-JSON decoding defect,
#: the same class of change `POST_HOC_RECITE_JSON_TEMPLATE`'s comment above excludes from the
#: ledger for the post-hoc side. Booking one side and not the other would fail
#: `test_effort_is_matched`; booking both would spend a post-hoc cycle on a stage post-hoc doesn't
#: have.
JOINT_JSON_TEMPLATE = """You are answering a biomedical research question using only the passages below.

{context}

Question: {question}

Answer the question using only what the passages say. Do not use outside knowledge, and do not
state anything the passages do not support. Write the answer as a list of claims, and support each
claim with a quotation from the passages.

{claim_rules}

Only write a claim that says something the passages actually support. If a passage has nothing to
do with the question, leave it out rather than writing a claim that says so.

Write each claim with enough detail to stand on its own: around fifteen to twenty words is a
reasonable target, and a claim under ten words is usually missing a qualifying detail the passage
gives it — the population, the comparison, or the size of the effect.

Format the JSON compactly: a single space after each colon, no extra indentation, and no blank
lines between claims. This is about the JSON's own whitespace only, not about how much a claim
says.

Reply with a single JSON object and nothing else."""

POST_HOC_ANSWER_TEMPLATE = """You are answering a biomedical research question using only the passages below.

{context}

Question: {question}

Answer the question using only what the passages say. Do not use outside knowledge, and do not
state anything the passages do not support. Write the answer as a list of claims.

{claim_rules}

Keep each claim short. A claim carries one assertion and stops; around fifteen words is usually
enough, and a claim running past twenty is usually two claims. When a claim trails off into a
qualifying clause — one starting "which", "because", "although", "despite", or "while" — that
clause is a separate assertion, so write it as its own claim or leave it out. State what the
passages establish about the world, not what a study did or did not find: write "Vitamin D does not
reduce fracture risk", never "The trial found no evidence that vitamin D reduces fracture risk".

{format_block}"""


POST_HOC_CITE_TEMPLATE = """You are attaching supporting quotations to an answer that is already written.

{context}

Question: {question}

Answer already given:
{answer}

For each claim in the answer, find the passages above that support it and quote them. Do not
reword a claim, do not add a claim, and do not drop a claim — reproduce every claim exactly as it
is written above, in the same order. If no passage supports a claim, write the claim with no CITE
line under it.

{claim_rules}

{format_block}"""

#: C7's re-citation pass (`generate.cite_claims`). Deliberately **not** `POST_HOC_CITE_TEMPLATE`:
#: that template carries `_claim_rules()`, which tells the model to resolve pronouns and to split
#: anything joined by "and". For post-hoc's own second stage that is harmless — the claims it cites
#: were written under those same rules one call earlier. For C7 it is a contradiction, and a live
#: A4000 probe (2026-08-16) showed exactly what the model does with it: asked to reproduce five
#: already-cut claims *and* to reshape them, it cited three, then emitted twelve empty `CLAIM` lines
#: and a note explaining that the rest "were not present in the original answer". Every such reply
#: is a `cite stage returned N CLAIM lines for M claims sent` error, which is what pinned
#: `clean_cite_rate` at 0.0 for both model rows.
#:
#: The grammar is unchanged (same `_format_block`, same `_citation_rules`) — only the instruction to
#: reshape a claim is withheld, because at this point the claim is already frozen by `decompose.py`.
POST_HOC_RECITE_TEMPLATE = """You are attaching supporting quotations to claims that are already written.

{context}

Question: {question}

The {claim_count} claims to cite:
{answer}

Copy each of these {claim_count} claims back exactly as it is written above, in the same order, and
put the passages that support it underneath it. The claims are final: do not reword one, do not
split one, do not merge two, do not add one, and do not drop one. Write exactly {claim_count} CLAIM
lines, numbered 1 to {claim_count}, each carrying the full text of the claim above — never an empty
CLAIM line and never a CLAIM line past {claim_count}. If no passage supports a claim, still write
the claim, with no CITE line under it.

{claim_rules}

{format_block}"""


#: The re-citation stage as a **constrained** decode (`generate.cite_claims`, `guided_decoding=True`).
#:
#: `POST_HOC_RECITE_TEMPLATE` above asks the model to transcribe a quote and hope; a live n=100 run
#: put `quote_located_rate` at 0.74, which is `clean_cite_rate` 0.00 once a query ANDs ~30 claims
#: together. Prose could not fix it — `_citation_rules` already spells the copy rule out three
#: failure modes deep. So this stage stops asking. `build_citation_response_format` compiles the
#: passages into a JSON schema whose `quote` is an `enum` of spans taken *out of* the passage text,
#: paired by `const` with the id of the passage they came from, so a quote the passage does not
#: contain is not a reply the decoder can emit. Transcription drift stops being a failure mode
#: instead of being detected as one.
#:
#: The prompt is therefore thin on purpose: the schema carries the grammar, the cap, and the
#: verbatim rule, and repeating them in prose would be three copies of one contract. What prose
#: still has to carry is the part the schema cannot express — that the claims are final, and that an
#: unsupported claim gets an empty `citations` array rather than a borrowed quote.
#:
#: Like `POST_HOC_RECITE_TEMPLATE`, this is C7 decomposition machinery and is **not** booked to
#: `PROMPT_ITERATIONS[POST_HOC]`: the post-hoc baseline never re-cites a decomposed answer, and
#: charging C7's stage to its ledger would report the baseline as more engineered than it was and
#: break `effort_is_matched()`.
POST_HOC_RECITE_JSON_TEMPLATE = """You are attaching supporting quotations to claims that are already written.

{context}

Question: {question}

The {claim_count} claims to cite:
{answer}

For each of these {claim_count} claims, in order, find the passages above that support it and quote
them. Return one entry per claim, with `claim_index` counting 1 to {claim_count}. The claims are
final: do not reword, split, merge, add, or drop one. If no passage supports a claim, return that
claim with an empty `citations` array — never a quote borrowed from a claim it does not support.

Reply with a single JSON object and nothing else."""


VANILLA_TEMPLATE = """You are answering a biomedical research question using only the passages below.

{context}

Question: {question}

Answer the question using only what the passages say. Do not use outside knowledge, and do not
state anything the passages do not support. Write the answer as a list of claims.

{claim_rules}

{format_block}"""


def build_prompt(
    system: System,
    question: str,
    passages: list[RetrievedPassage],
    max_citations: int,
    stage: str = "answer",
    answer: str | None = None,
    depth: int = CONTEXT_DEPTH,
    claim_count: int | None = None,
) -> str:
    """Render one prompt. `stage` is `"cite"` for post-hoc's second pass, `"recite"` for C7's
    re-citation of an already-decomposed answer (`generate.cite_claims`), and `"recite_json"` for
    the same stage under the constrained decode. `"recite"` needs `claim_count`: the model is told
    how many CLAIM lines the reply must carry, because a reply with a different number is
    unmatchable positionally. `"recite_json"` needs it too, but there the count is also compiled
    into the schema's `minItems`/`maxItems`, so the prose is a reminder rather than the guarantee."""
    context = render_context(passages, depth)

    def rules(cite: bool, shape_claims: bool = True, decision: bool = True) -> dict[str, str]:
        block = _claim_rules() if shape_claims else ""
        if cite:
            block = f"{block}\n\n{_citation_rules(max_citations)}" if block else _citation_rules(max_citations)
        return {
            "context": context,
            "question": question,
            "claim_rules": block,
            "format_block": _format_block(max_citations, cite=cite, decision=decision),
        }

    if system is System.JOINT:
        if stage == "joint_json":
            return JOINT_JSON_TEMPLATE.format(
                context=context, question=question, claim_rules=_claim_rules()
            )
        return JOINT_TEMPLATE.format(**rules(cite=True))
    if system is System.VANILLA:
        # Vanilla still gets the passages — it is "retrieve → generate, no attribution"
        # (schema.py:73), so it isolates attribution rather than retrieval.
        return VANILLA_TEMPLATE.format(**rules(cite=False))
    if stage == "recite":
        if answer is None:
            raise ValueError("the re-citation stage needs the claims it is citing")
        if not claim_count:
            raise ValueError("the re-citation stage needs the number of claims it is citing")
        return POST_HOC_RECITE_TEMPLATE.format(
            answer=answer, claim_count=claim_count,
            **rules(cite=True, shape_claims=False, decision=False),
        )
    if stage == "recite_json":
        if answer is None:
            raise ValueError("the re-citation stage needs the claims it is citing")
        if not claim_count:
            raise ValueError("the re-citation stage needs the number of claims it is citing")
        # No `claim_rules` and no `format_block`: `build_citation_response_format` is the grammar
        # here, and a prose copy of a contract the decoder already enforces can only drift from it.
        return POST_HOC_RECITE_JSON_TEMPLATE.format(
            context=context, question=question, answer=answer, claim_count=claim_count,
        )
    if stage == "cite":
        if answer is None:
            raise ValueError("post-hoc's cite stage needs the answer from its first stage")
        return POST_HOC_CITE_TEMPLATE.format(answer=answer, **rules(cite=True))
    # The post-hoc answer stage must not mention citations at all: a first pass that knows it will
    # be cited later is already doing joint grounding, and C2's gap would shrink for a reason that
    # has nothing to do with the systems.
    return POST_HOC_ANSWER_TEMPLATE.format(**rules(cite=False))

def build_citation_response_format(
    passages: Sequence[RetrievedPassage],
    claim_count: int | None = None,
    max_citations: int = 3,
    *,
    min_claims: int = 1,
    max_claims: int = 30,
    is_joint: bool = False,
) -> dict[str, Any] | None:
    """Compile the passages into the JSON schema that makes a citation verbatim by construction.

    One `anyOf` branch per passage, each pairing a `const` passage id with an `enum` of spans cut
    out of *that* passage's text. Two properties follow from the shape rather than from asking:
    a quote the passage does not contain is unreachable, and a quote cannot be filed under a
    different passage's id — the pairing is in the branch, not in the model's discipline.

    Candidates are whole sentences, sentences minus trailing punctuation, and `;`/`:`-delimited
    clauses. That is coarser than the free-form span the unconstrained stage could produce: a quote
    beginning mid-sentence is legal there (`_citation_rules`) and is not offered here. The trade is
    deliberate — a coarser span that is real beats a precise one that is invented — but it does mean
    `char_start`/`char_end` widths are not comparable across the two modes.

    Returns `None` when no passage yields a candidate, which is the caller's signal to fall back to
    the prose stage. The earlier draft padded the enum with `"N/A"` instead; that put a string the
    passage does not contain inside the one structure whose whole purpose is that it cannot happen,
    and `locate_quote` would have booked the model's only legal choice as a quote-not-found error.
    """
    citation_schemas: list[dict[str, Any]] = []

    for p in passages:
        text = p.text or ""
        quotes: set[str] = set()
        for s in (s.strip() for s in re.split(r"(?<=[.!?])\s+", text)):
            if len(s) < 10:
                continue
            # Every candidate is checked back against `text`: the split and the strips are string
            # surgery, and a span that no longer occurs in the passage would be a fabricated quote
            # with a schema's authority behind it.
            for candidate in (s, s.rstrip(".,;:!?()")):
                if len(candidate) >= 10 and candidate in text:
                    quotes.add(candidate)
            for c in re.split(r"[;:]", s):
                c = c.strip(" \t\r\n.,;:!?()")
                if len(c) >= 15 and c in text:
                    quotes.add(c)
        if not quotes:
            continue
        citation_schemas.append({
            "type": "object",
            "properties": {
                "passage_id": {"const": p.passage_id},
                "quote": {"type": "string", "enum": sorted(quotes)},
            },
            "required": ["passage_id", "quote"],
        })

    if not citation_schemas:
        return None

    if claim_count is not None and not is_joint:
        schema = {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    # Exactly `claim_count` entries, because `cite_claims` matches the reply back to the
                    # batch positionally. The count the prose asks for is the count the decoder enforces.
                    "minItems": claim_count,
                    "maxItems": claim_count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_index": {"type": "integer", "minimum": 1, "maximum": claim_count},
                            # The cap is `ScoringConfig.max_citations`, not a literal: a schema with its
                            # own idea of the cap would silently enforce a different fairness contract
                            # than `QueryRecord.validate()` reports on.
                            "citations": {
                                "type": "array",
                                "maxItems": max_citations,
                                "items": {"anyOf": citation_schemas},
                            },
                        },
                        "required": ["claim_index", "citations"],
                    },
                }
            },
            "required": ["claims"],
        }
        return {
            "type": "json_schema",
            "json_schema": {"name": "recitation_response", "schema": schema},
        }

    schema = {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": list(DECISIONS)},
            "claims": {
                "type": "array",
                "minItems": min_claims,
                "maxItems": max_claims,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_index": {"type": "integer", "minimum": 1, "maximum": max_claims},
                        "text": {"type": "string"},
                        "citations": {
                            "type": "array",
                            "maxItems": max_citations,
                            "items": {"anyOf": citation_schemas},
                        },
                    },
                    "required": ["claim_index", "text", "citations"],
                },
            },
        },
        "required": ["decision", "claims"],
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "joint_response", "schema": schema},
    }
def locate_quote(quote: str, passage_id: str, text: str) -> Citation | None:
    """Turn a quoted string into a span, or `None` if the model did not copy it exactly.

    Exact search first, then two widenings that **find** a span the passage really has rather than
    inventing one: outer quote delimiters are stripped, runs of whitespace are allowed to differ,
    and finally case is allowed to differ. A fuzzy match would fabricate `char_start`/`char_end`
    for text the passage does not contain; these do not — every returned `quoted_text` is copied
    back out of `text`, so the span is exact even when the model's transcription was not.

    A widened match is still a defect in the reply. `parse_response` notices (the returned
    `quoted_text` differs from what the model wrote) and records it under `ParsedResponse.recovered`
    so the drift keeps being counted instead of disappearing into a clean rate.
    """
    q = quote.strip()
    for qmark in ('"', "'", '“', '”', '‘', '’'):
        if q.startswith(qmark) and q.endswith(qmark) and len(q) > 1:
            q = q[1:-1].strip()

    start = text.find(q)
    if start >= 0:
        return Citation(
            passage_id=passage_id,
            char_start=start,
            char_end=start + len(q),
            quoted_text=q,
        )

    # Edge punctuation is where the model tidies: a span it copied out of the middle of a sentence
    # comes back finished with a full stop, or a passage's trailing comma is dropped. The words in
    # between still have to match in order, so this finds the span the model meant without letting
    # it invent one. Interior drift — "HR, 1.85" for "HR: 1.85", or two separated spans spliced
    # into one — still fails, which is the point: those are attribution errors, not typography.
    for candidate in (q, q.strip(" \t\r\n.,;:!?")):
        words = candidate.split()
        if not words:
            continue
        pattern_str = r"\s+".join(re.escape(w) for w in words)
        for flags in (0, re.IGNORECASE):
            match = re.compile(pattern_str, flags).search(text)
            if match:
                s, e = match.span()
                return Citation(
                    passage_id=passage_id,
                    char_start=s,
                    char_end=e,
                    quoted_text=text[s:e],
                )

    return None


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    """What came back, plus everything that did not parse. Errors are data, not exceptions.

    G2 gates on ≥95% valid claim parse, so the failures have to survive to be counted. A parser
    that raises on the first malformed line reports one error per generation and hides the rest.

    `recovered` is the third category, between clean and broken: a line whose meaning was
    unambiguous but whose transcription drifted — a quote that differs only in case or whitespace,
    a passage id that dropped its chunk index when only one chunk of that document is in context.
    These are read, not rejected, because refusing them would throw away a citation the passage
    genuinely supports. They are listed anyway, so widening acceptance can never quietly become
    "the defect stopped happening".
    """

    decision: str | None
    claims: list[Claim]
    errors: list[str]
    recovered: list[str] = field(default_factory=list)


def parse_response(
    raw: str,
    passages: list[RetrievedPassage],
    max_citations: int,
    *,
    max_claim_words: int = MAX_CLAIM_WORDS,
    require_decision: bool = True,
) -> ParsedResponse:
    """Parse the line grammar into claims with located citation spans.

    Over-cap citations are **kept**, not trimmed. `QueryRecord.validate()` already reports
    `exceeds_cap`, and silently dropping the fourth citation would erase the evidence that a system
    ignored a cap the fairness argument depends on.

    An over-length claim is kept for the same reason: it is flagged, never truncated and never
    dropped. `max_claim_words` is the largest acceptable claim, inclusive — see `MAX_CLAIM_WORDS`
    for why the number is 50 and why it is a scoring rule rather than a generation knob.
    """
    text_by_id = {p.passage_id: (p.text or "") for p in passages}
    # The constrained re-citation stage (`build_citation_response_format`) replies with a JSON
    # object rather than the line grammar. It is the same contract read out of a different shape:
    # every quote still goes through `locate_quote`, every id still has to be in the context, and a
    # failure is still an entry in `errors` rather than an exception.
    stripped = raw.strip()
    if stripped.startswith("{"):
        res_obj = None
        try:
            res_obj = json.loads(stripped)
        except json.JSONDecodeError:
            # Try appending closing braces/brackets for truncated JSON generations
            for suffix in ("}", "]}", '"}]}', '"]}]}', '""}]}', 'null}]}', '""}]}}'):
                try:
                    res_obj = json.loads(stripped + suffix)
                    break
                except json.JSONDecodeError:
                    pass
        if res_obj is None:
            try:
                # Fallback error reporting
                json.loads(stripped)
            except json.JSONDecodeError as exc:
                return ParsedResponse(None, [], [f"reply is malformed JSON: {exc}"], [])
        decision: str | None = None
        raw_decision = res_obj.get("decision")
        if isinstance(raw_decision, str):
            token = raw_decision.lower()
            if token in DECISIONS:
                decision = token
            else:
                errs.append(f"decision {raw_decision!r} is not one of {DECISIONS}")
        elif raw_decision is not None:
            errs.append(f"decision {raw_decision!r} is not one of {DECISIONS}")
        elif require_decision:
            errs.append("no DECISION line")

        claims_dict: dict[str, Claim] = {}
        order: list[str] = []
        errs: list[str] = []
        recov: list[str] = []
        raw_claims = res_obj.get("claims")
        if not isinstance(raw_claims, list):
            return ParsedResponse(decision, [], ["JSON reply has no 'claims' array"], [])
        for pos, item in enumerate(raw_claims, start=1):
            if not isinstance(item, dict):
                errs.append(f"claim entry {pos} is not an object")
                continue
            c_idx = item.get("claim_index", pos)
            if not isinstance(c_idx, int) or not 1 <= c_idx <= len(raw_claims):
                errs.append(f"claim entry {pos} has claim_index {c_idx!r}, which is out of range")
                continue
            cid = f"c{c_idx}"
            text_val = item.get("text", "")
            if not isinstance(text_val, str):
                text_val = str(text_val)
            claims_dict[cid] = Claim(
                claim_id=cid,
                text=text_val,
                citations=[],
                granularity=Granularity.DECONTEXTUALIZED_ATOMIC,
            )
            if cid not in order:
                order.append(cid)
            citations = item.get("citations")
            if not isinstance(citations, list):
                errs.append(f"claim {cid} has no 'citations' array")
                continue
            for cit in citations:
                if not isinstance(cit, dict):
                    errs.append(f"claim {cid} has a citation that is not an object")
                    continue
                pid, quote = cit.get("passage_id"), cit.get("quote")
                if not pid or not quote:
                    errs.append(f"claim {cid} has a citation missing its passage id or quote")
                    continue
                if pid not in text_by_id:
                    errs.append(f"claim {cid} cites {pid!r}, which is not in the context")
                    continue
                citation = locate_quote(quote, pid, text_by_id[pid])
                if citation is None:
                    errs.append(f"quote not found verbatim in {pid} ({quote[:60]!r})")
                    continue
                if citation.quoted_text != quote.strip():
                    recov.append(f"quote in {pid} matched only after normalising")
                claims_dict[cid].citations.append(citation)

        for cid in order:
            c_text = claims_dict[cid].text
            if c_text:
                words = len(c_text.split())
                if words > max_claim_words:
                    errs.append(
                        f"{cid}: {words} words exceeds the max claim length of {max_claim_words} "
                        "(non-terminating generation)"
                    )
            if len(claims_dict[cid].citations) > max_citations:
                errs.append(
                    f"{cid}: {len(claims_dict[cid].citations)} citations exceeds the cap of {max_citations}"
                )
        claim_texts = [claims_dict[cid].text for cid in order if claims_dict[cid].text]
        if claim_texts:
            for start_idx, length in runaway_chains(claim_texts, min_length=2):
                first_cid = order[start_idx]
                cid = order[start_idx + length - 1]
                if length >= RUNAWAY_CHAIN_MIN:
                    errs.append(
                        f"{cid}: extends {first_cid}'s claim text through {length} nested claims "
                        "(non-terminating generation)"
                    )
                else:
                    sec_text = claims_dict[cid].text
                    recov.append(
                        f"{cid}: extends {first_cid}'s claim text ({sec_text[:60]!r})"
                    )
        if not order:
            errs.append("no CLAIM lines")

        return ParsedResponse(decision, [claims_dict[c] for c in order], errs, recov)
    decision: str | None = None
    claims: dict[str, Claim] = {}
    order: list[str] = []
    errors: list[str] = []
    recovered: list[str] = []

    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        head, sep, rest = line.partition(":")
        if not sep:
            continue  # prose the model added around the format; not itself a claim failure
        head, rest = head.strip().upper(), rest.strip()

        if head == "DECISION":
            token = rest.lower()
            if token not in DECISIONS:
                errors.append(f"line {lineno}: decision {rest!r} is not one of {DECISIONS}")
            else:
                decision = token
            continue

        kind, _, number = head.partition(" ")
        if kind not in ("CLAIM", "CITE"):
            continue

        if kind == "CLAIM":
            if not number.strip().isdigit():
                errors.append(f"line {lineno}: CLAIM line carries no claim number")
                continue
            cid = f"c{int(number)}"
            if cid in claims:
                errors.append(f"line {lineno}: claim {cid} declared twice")
                continue
            if not rest:
                # A live A4000 probe caught the model padding a short reply with bare `CLAIM 4:`
                # lines up to a count it had been given. Keeping them would let a padded reply
                # satisfy the positional match with empty text, so the padding is reported and the
                # line is not counted as a claim.
                errors.append(f"line {lineno}: claim {cid} is empty")
                continue
            claims[cid] = Claim(
                claim_id=cid,
                text=rest,
                citations=[],
                granularity=Granularity.DECONTEXTUALIZED_ATOMIC,
            )
            order.append(cid)
            continue

        # A CITE line supports the claim above it (module docstring, §4). Any number the model
        # writes is ignored: the grammar no longer has one, and two live runs showed an 8B model
        # using it as a within-claim citation index. Reading it as a claim id sent every claim's
        # first citation to c1, which mis-attributed evidence and invented cap violations.
        if not order:
            errors.append(f"line {lineno}: CITE line precedes any CLAIM")
            continue
        cid = order[-1]
        pid, psep, quote = rest.partition(_CITATION_SEP)
        if not psep:
            errors.append(f"line {lineno}: CITE line has no {_CITATION_SEP!r} separator")
            continue
        pid, quote = pid.strip(), quote.strip()
        # render_context writes ids as "[id]", so the model is taught to echo the brackets. They
        # are delimiters, not part of the id; stripping them is reading the grammar, not repairing
        # a wrong answer. A quote that does not match is still an error (module docstring, §1).
        if len(pid) > 1 and pid[0] == "[" and pid[-1] == "]":
            pid = pid[1:-1].strip()
        if pid not in text_by_id:
            # The model routinely drops the chunk index off an id ("[pubmed23n0263_2785:]" for
            # "[pubmed23n0263_2785:0]"). When exactly one passage in the context comes from that
            # document there is only one span it can mean, so the citation is read and recorded as
            # drift. When two chunks of the same document are in context the id is genuinely
            # ambiguous and stays an error — guessing would attribute evidence to the wrong chunk.
            base = pid.split(":")[0]
            candidates = [k for k in text_by_id if k.split(":")[0] == base]
            if len(candidates) == 1:
                recovered.append(f"line {lineno}: cites {pid!r}, read as {candidates[0]!r}")
                pid = candidates[0]
            else:
                errors.append(f"line {lineno}: cites {pid!r}, which is not in the context")
                continue
        citation = locate_quote(quote, pid, text_by_id[pid])
        if citation is None:
            errors.append(
                f"line {lineno}: quote not found verbatim in {pid} ({quote[:60]!r})"
            )
            continue
        if citation.quoted_text != quote.strip():
            recovered.append(
                f"line {lineno}: quote in {pid} matched only after normalising "
                f"delimiters, whitespace or case ({quote[:60]!r})"
            )
        claims[cid].citations.append(citation)

    if decision is None and require_decision:
        errors.append("no DECISION line")
    for cid in order:
        words = len(claims[cid].text.split())
        if words > max_claim_words:
            errors.append(
                f"{cid}: {words} words exceeds the max claim length of {max_claim_words} "
                "(non-terminating generation)"
            )
        if len(claims[cid].citations) > max_citations:
            errors.append(
                f"{cid}: {len(claims[cid].citations)} citations exceeds the cap of {max_citations}"
            )
    claim_texts = [claims[cid].text for cid in order]
    for start_idx, length in runaway_chains(claim_texts, min_length=2):
        first_cid = order[start_idx]
        cid = order[start_idx + length - 1]
        if length >= RUNAWAY_CHAIN_MIN:
            errors.append(
                f"{cid}: extends {first_cid}'s claim text through {length} nested claims "
                "(non-terminating generation)"
            )
        else:
            sec_text = claims[cid].text
            recovered.append(
                f"{cid}: extends {first_cid}'s claim text ({sec_text[:60]!r})"
            )
    if not order:
        errors.append("no CLAIM lines")

    return ParsedResponse(decision, [claims[c] for c in order], errors, recovered)
