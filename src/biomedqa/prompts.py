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

from dataclasses import dataclass

from .schema import Citation, Claim, Granularity, RetrievedPassage, System

#: ADR-0015 moved G1's k from 5 to 10, so the certified context is 10 passages deep.
CONTEXT_DEPTH = 10

#: PubMedQA's label set. No abstain token: ADR-0010 derives abstention at scoring time, and giving
#: the model an explicit escape hatch would manufacture the behaviour the analysis is measuring.
DECISIONS = ("yes", "no", "maybe")

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


def _claim_rules() -> str:
    """Decontextualization and atomicity — ADR-0005's unit. Shared by all three systems, because
    a baseline whose claims are shaped differently is being compared on the wrong axis.

    Scoped to CLAIM lines explicitly. Unscoped, it read as advice about the whole reply, and the
    live smokes showed the model dutifully decontextualizing its *quotes* as well.
    """
    return """Write each CLAIM line so it stands alone. Resolve every pronoun, every "this"/"these",
and every implied subject, so that a reader who sees the claim and nothing else knows exactly what
it asserts. Each claim states exactly one thing; split anything joined by "and" into separate
claims when the parts could be true or false independently."""


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


def _format_block(max_citations: int, cite: bool) -> str:
    if not cite:
        return f"""Reply in exactly this format and add nothing else:

DECISION: one of {", ".join(DECISIONS)}
CLAIM 1: the first claim
CLAIM 2: the second claim"""
    return f"""Reply in exactly this format and add nothing else:

DECISION: one of {", ".join(DECISIONS)}
CLAIM 1: the first claim
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
) -> str:
    """Render one prompt. `stage` is `"cite"` only for post-hoc's second pass."""
    context = render_context(passages, depth)

    def rules(cite: bool) -> dict[str, str]:
        block = _claim_rules()
        if cite:
            block += "\n\n" + _citation_rules(max_citations)
        return {
            "context": context,
            "question": question,
            "claim_rules": block,
            "format_block": _format_block(max_citations, cite=cite),
        }

    if system is System.JOINT:
        return JOINT_TEMPLATE.format(**rules(cite=True))
    if system is System.VANILLA:
        # Vanilla still gets the passages — it is "retrieve → generate, no attribution"
        # (schema.py:73), so it isolates attribution rather than retrieval.
        return VANILLA_TEMPLATE.format(**rules(cite=False))
    if stage == "cite":
        if answer is None:
            raise ValueError("post-hoc's cite stage needs the answer from its first stage")
        return POST_HOC_CITE_TEMPLATE.format(answer=answer, **rules(cite=True))
    # The post-hoc answer stage must not mention citations at all: a first pass that knows it will
    # be cited later is already doing joint grounding, and C2's gap would shrink for a reason that
    # has nothing to do with the systems.
    return POST_HOC_ANSWER_TEMPLATE.format(**rules(cite=False))


def locate_quote(quote: str, passage_id: str, text: str) -> Citation | None:
    """Turn a quoted string into a span, or `None` if the model did not copy it exactly.

    Exact search only. A fuzzy match here would fabricate `char_start`/`char_end` for text the
    passage does not contain, and every downstream verifier reads that span as ground truth.
    """
    start = text.find(quote)
    if start < 0:
        return None
    return Citation(
        passage_id=passage_id,
        char_start=start,
        char_end=start + len(quote),
        quoted_text=quote,
    )


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    """What came back, plus everything that did not parse. Errors are data, not exceptions.

    G2 gates on ≥95% valid claim parse, so the failures have to survive to be counted. A parser
    that raises on the first malformed line reports one error per generation and hides the rest.
    """

    decision: str | None
    claims: list[Claim]
    errors: list[str]


def parse_response(
    raw: str, passages: list[RetrievedPassage], max_citations: int
) -> ParsedResponse:
    """Parse the line grammar into claims with located citation spans.

    Over-cap citations are **kept**, not trimmed. `QueryRecord.validate()` already reports
    `exceeds_cap`, and silently dropping the fourth citation would erase the evidence that a system
    ignored a cap the fairness argument depends on.
    """
    text_by_id = {p.passage_id: (p.text or "") for p in passages}
    decision: str | None = None
    claims: dict[str, Claim] = {}
    order: list[str] = []
    errors: list[str] = []

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
            errors.append(f"line {lineno}: cites {pid!r}, which is not in the context")
            continue
        citation = locate_quote(quote, pid, text_by_id[pid])
        if citation is None:
            errors.append(
                f"line {lineno}: quote not found verbatim in {pid} ({quote[:60]!r})"
            )
            continue
        claims[cid].citations.append(citation)

    if decision is None:
        errors.append("no DECISION line")
    for cid in order:
        if len(claims[cid].citations) > max_citations:
            errors.append(
                f"{cid}: {len(claims[cid].citations)} citations exceeds the cap of {max_citations}"
            )
    if not order:
        errors.append("no CLAIM lines")

    return ParsedResponse(decision, [claims[c] for c in order], errors)
