"""Drive one (question, system, seed) through the generator and return a `QueryRecord`.

This is the layer between `prompts.py` (what to ask) and `backends.py` (how to ask it). It owns
four things and deliberately nothing else:

1. **Stage orchestration.** Joint and vanilla are one call. Post-hoc is two — answer, then cite —
   and the first call must not know the second exists (`prompts._citation_rules` is withheld from
   it). Getting that wrong turns the baseline into joint grounding under another name and closes
   C2's gap for a non-reason, so the two-call shape lives here rather than in a caller.

2. **Cost accounting that survives the join.** Every stage emits its own `CostRecord`, kept
   separately for Table 4, while the record carries the per-query totals. Post-hoc costs two
   completions per query; hiding that would understate the baseline's price in exactly the table
   that compares prices.

3. **Lossless raw text.** `QueryRecord.raw_generation` holds *both* post-hoc stages, joined by
   `STAGE_SEPARATOR` and recoverable with `split_stages`. Storing only the parsed claims, or only
   the stage that happened to parse, would make a decomposition-error post-mortem impossible.

4. **The contract check, run and reported.** `QueryRecord.validate()` is where the ≤3-citation cap
   and "vanilla carries no citations" are enforced, and both are fairness controls G2's gap rests
   on rather than decoration. A control whose only caller is a smoke test is not a control, so the
   generation path runs it every query and returns the problems verbatim. It never repairs them:
   a violation is a measurement, and silently trimming a fourth citation would hide the thing the
   cap exists to expose.

**Parse errors are not stored.** `parse_response` derives them from `raw_generation`, `retrieved`,
and `max_citations`, all of which the record already carries, so re-deriving them at scoring time
is exact and storing them would be a second copy that can go stale (`CONTEXT.md`, least-processed
value). `generate_one` returns them alongside the record for the run log.

**`retrieved` is the context the model actually saw**, sliced to `depth`, not the 100-deep pool
Table 1 stores. Citation validity is defined against the passages the prompt listed: a citation
naming rank 47 is a hallucinated passage id, and `QueryRecord.validate()` must be able to say so.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from . import backends
from .config import GenerationConfig
from .prompts import CONTEXT_DEPTH, build_prompt, parse_response
from .schema import CostRecord, QueryRecord, RetrievedPassage, System

#: Marks the boundary between post-hoc's two stages inside `raw_generation`. Chosen to be something
#: no model emits: it is not part of the response grammar and carries the project namespace.
STAGE_SEPARATOR = "\n\n===== biomedqa:stage=cite =====\n\n"

#: A completion function with `backends.complete`'s signature. Injected so the orchestration can be
#: exercised without a vLLM server; the box is copy-paste only and a bug found here costs a GPU run.
Completer = Callable[..., tuple[str, CostRecord]]


def split_stages(raw: str) -> tuple[str, ...]:
    """Recover the per-stage text from `raw_generation`. One element for joint and vanilla, two for
    post-hoc, in the order the stages ran."""
    return tuple(raw.split(STAGE_SEPARATOR))


@dataclass(frozen=True, slots=True)
class Generation:
    """One query's output: the record, the per-stage cost rows, what failed to parse, and what the
    record's own contract says about it.

    `errors` and `violations` are different questions. `errors` is "the grammar did not parse" —
    G2's valid-parse rate. `violations` is `QueryRecord.validate()`: the fairness controls G2 rests
    on, chiefly the ≤3-citation cap and vanilla carrying no citations at all. Both are reported,
    neither is repaired, and neither raises: a record that breaks its contract still has to reach
    the denominator or the rate it feeds is measured on the survivors.
    """

    record: QueryRecord
    costs: tuple[CostRecord, ...]
    errors: tuple[str, ...]
    violations: tuple[str, ...]


def generate_one(
    question: str,
    passages: Sequence[RetrievedPassage],
    gold_passage_ids: Sequence[str],
    *,
    system: System,
    config: GenerationConfig,
    seed: int,
    run_id: str,
    query_id: str,
    gold_final_decision: str | None = None,
    depth: int = CONTEXT_DEPTH,
    complete: Completer = backends.complete,
) -> Generation:
    """Run one system on one question and assemble its `QueryRecord`.

    The record is returned whatever the model emitted — an unparseable response yields a record
    with no claims and a populated `errors`, because G2 gates on the valid-parse *rate* and a
    failure that raises is a failure that never reaches the denominator.
    """
    context = list(passages[:depth])
    if not context:
        raise ValueError(f"{query_id}: no passages to ground on; retrieval must run first")

    texts: list[str] = []
    costs: list[CostRecord] = []

    def call(prompt: str) -> str:
        text, cost = complete(
            prompt, config, seed=seed, run_id=run_id, query_id=query_id
        )
        texts.append(text)
        costs.append(cost)
        return text

    if system is System.POST_HOC:
        answer = call(
            build_prompt(
                system, question, context, config.max_citations, stage="answer", depth=depth
            )
        )
        parsed_from = call(
            build_prompt(
                system,
                question,
                context,
                config.max_citations,
                stage="cite",
                answer=answer,
                depth=depth,
            )
        )
    else:
        parsed_from = call(
            build_prompt(system, question, context, config.max_citations, depth=depth)
        )

    parsed = parse_response(parsed_from, context, config.max_citations)

    record = QueryRecord(
        run_id=run_id,
        query_id=query_id,
        question=question,
        system=system,
        seed=seed,
        retrieved=context,
        gold_passage_ids=list(gold_passage_ids),
        claims=parsed.claims,
        raw_generation=STAGE_SEPARATOR.join(texts),
        final_decision=parsed.decision,
        gold_final_decision=gold_final_decision,
        latency_s=_total(c.wall_s for c in costs),
        prompt_tokens=_total(c.input_tokens for c in costs),
        completion_tokens=_total(c.output_tokens for c in costs),
    )
    return Generation(record, tuple(costs), tuple(parsed.errors), tuple(record.validate()))


def _total(values: Iterable[float | None]) -> float | None:
    """Sum across stages, or `None` if any stage did not report — a partial total would read as a
    cheap query in Table 4 rather than as missing instrumentation."""
    seen = list(values)
    if any(v is None for v in seen):
        return None
    return sum(seen)  # type: ignore[arg-type]
