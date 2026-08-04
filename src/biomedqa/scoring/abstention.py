"""Abstention detection — derived here, never stored (ADR-0010).

A model that says *"the passages do not answer this"* has done the right thing. The G0 bake-off's
`score_compliance` counted that as an uncited claim, which would have made Table 2 penalise a system
for declining to answer and reward one that confabulates a citation — inverting the property this
paper exists to measure (issue #9).

**Why this is a scoring module and not a schema field.** A stored `is_abstention` boolean is a value
derived by a rule and frozen into the record, which is exactly what `schema.py`'s
least-processed-value rule forbids. It is also less expressive than the raw material: G0 showed
abstention arriving in two shapes — Llama emitted it *as a claim* in the list, Qwen emitted it as
prose **outside** the list, where no field on `Claim` could have held it. `Claim.text`,
`Claim.citations` and `QueryRecord.raw_generation` carry both, so `SCHEMA_VERSION` stays at 1.0.0
and the rule lives here, versioned, where revising it is a re-*score* rather than a re-*run*.

**What the rule is for.** Exactly one thing: excluding abstention claims from the **citation-recall
denominator**. Precision is untouched *by construction* — an abstention claim carries no citations,
and precision's denominator is the number of citations. Getting that scope wrong in either direction
inverts the metric, so `answered_claims()` is the only intended consumer.

**The rule is deliberately conservative.** It requires a claim to be uncited *and* to say both that
something cannot be established *and* that the source material is what it cannot be established
from. A false positive silently shrinks a denominator, which is the expensive direction to be wrong
in; a false negative merely reproduces today's known bug on one claim.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..schema import Claim, QueryRecord

#: Bump on any change to the patterns below. Recorded in `RunConfig.scoring.abstention_rule_version`
#: so a run always states which rule produced its numbers.
ABSTENTION_RULE_VERSION = "1.0.0"

#: An abstention says the answer *cannot be established*. Adverbs are tolerated between the verb and
#: its participle ("cannot be **directly** answered") because G0 produced exactly that.
_INCAPACITY = re.compile(
    r"""
    \b(?:
        (?:can|could)\s*not\s+be\s+(?:\w+\s+){0,2}(?:answer|address|determin|establish|conclud|assess)
      | (?:is|are|was|were)\s+not\s+(?:\w+\s+){0,2}(?:answered|addressed|discussed|mentioned|reported|provided|specified)
      | do(?:es)?\s+not\s+(?:\w+\s+){0,2}(?:answer|address|discuss|mention|report|provide|specify|contain|state)
      | (?:no|insufficient|not\s+enough)\s+(?:\w+\s+){0,2}(?:information|evidence|data|detail)
      | (?:cannot|can\s*not)\s+(?:be\s+)?(?:\w+\s+){0,2}(?:answered|determined|established|assessed)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: ...and names the retrieved material as the thing it cannot be established from. Without this,
#: "the mechanism is not well understood" — a substantive, citable claim — would match.
_SOURCE = re.compile(
    r"\b(?:passage|context|excerpt|abstract|document|source|text|article|provided\s+information)s?\b",
    re.IGNORECASE,
)


def is_abstention(claim: Claim) -> bool:
    """Does this claim decline to answer rather than assert something?

    Three conditions, all required:

    1. **No citations.** An abstention has nothing to cite by construction. A claim that cites a
       passage is making an assertion about it, whatever its wording.
    2. **It states an incapacity** — something cannot be answered, addressed, determined.
    3. **It names the source material** as what the incapacity is about. This is what separates
       *"the passages do not report mortality"* (an abstention) from *"metformin does not reduce
       mortality"* (a substantive negative claim) and from *"the mechanism is not well understood"*
       (a substantive hedge). Both of the latter must keep their place in the recall denominator.
    """
    if claim.citations:
        return False
    text = claim.text
    return bool(_INCAPACITY.search(text) and _SOURCE.search(text))


def abstention_claims(record: QueryRecord) -> list[Claim]:
    """The claims in this record that decline to answer."""
    return [c for c in record.claims if is_abstention(c)]


def answered_claims(record: QueryRecord) -> list[Claim]:
    """The citation-recall denominator: claims that ought to carry a citation.

    **This is the fix to issue #9.** `citation.py` computes corpus-level recall over these, not over
    `record.claims`. Precision is computed over citations and needs no equivalent — abstention
    claims contribute none.
    """
    return [c for c in record.claims if not is_abstention(c)]


def is_full_abstention(record: QueryRecord) -> bool:
    """Did the system decline entirely — zero substantive claims?

    Never observed in G0, where both abstentions were *partial* (substantive cited claims plus a
    statement of the gap). ADR-0010 reports the full-set citation-F1 as a robustness check only if
    this ever returns true, so the branch appears only if reality produces it.
    """
    return bool(record.claims) and not answered_claims(record)


def abstention_rate(records: Iterable[QueryRecord]) -> float:
    """Fraction of claims that are abstentions — a **Table 2 column**, per ADR-0010.

    It is a disclosure, not a guard. Excluding abstentions from the recall denominator rewards a
    system that abstains on hard claims, exactly as the unfixed scorer punishes one that abstains
    correctly. What actually guards the number is that citation-F1 is reported on **both**
    denominators, always, and that ADR-0009's parity loop is blind to F1 so prompts cannot be tuned
    toward abstention.
    """
    total = seen = 0
    for record in records:
        for claim in record.claims:
            total += 1
            seen += is_abstention(claim)
    return seen / total if total else 0.0
