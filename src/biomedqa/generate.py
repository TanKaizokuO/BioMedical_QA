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

**A fifth thing lives here for C7 only: `cite_claims`.** `decompose.py`'s own docstring puts
re-attaching citations to a re-cut answer out of its scope — "a re-cut claim has no citations: the
spans belonged to units that no longer exist" — and names this module as where the re-run belongs.
It reuses `POST_HOC_CITE_TEMPLATE` unchanged rather than inventing a second citation grammar: the
template already says "attach quotations to an answer that is already written" and does not care
which system, or which granularity, wrote that answer.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

import httpx

from . import backends
from .config import GenerationConfig
from .prompts import (
    CONTEXT_DEPTH,
    ParsedResponse,
    build_citation_response_format,
    build_prompt,
    parse_response,
)
from .schema import Claim, CostRecord, QueryRecord, RetrievedPassage, System
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

    `recovered` captures transcription drift that was read rather than rejected — see
    `prompts.ParsedResponse`. The field was dropped when `Generation` was constructed in `generate_one`,
    so `parse_response`'s recovered notes — bracket-stripped ids, ids read past a dropped chunk
    index, quotes matched only after normalising delimiters/whitespace/case, and length-2
    nested-claim chains — never reached a smoke summary, which ADR-0019's Consequences forbid.
    """

    record: QueryRecord
    costs: tuple[CostRecord, ...]
    errors: tuple[str, ...]
    violations: tuple[str, ...]
    #: Transcription drift that was read rather than rejected — see `prompts.ParsedResponse`.
    recovered: tuple[str, ...] = ()


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

    A rejected call (e.g. HTTPStatusError or TransportError) is recorded as a failure rather
    than propagating a traceback. On vLLM with `--max-model-len 8192`, post-hoc's stage 2 prompt
    embeds stage 1's answer; a runaway stage 1 generation can inflate the prompt so that
    prompt_tokens + max_tokens exceeds 8192, triggering a 400 rejection (measured twice during
    the live A4000 sweep on query 21074975). The fix records the failure in `call_failures`
    and `Generation.errors` rather than clamping or retrying, because a per-query effective cap
    cannot be expressed by the manifest's single `generation.max_tokens`.
    """
    context = list(passages[:depth])
    if not context:
        raise ValueError(f"{query_id}: no passages to ground on; retrieval must run first")

    texts: list[str] = []
    costs: list[CostRecord] = []
    call_failures: list[str] = []

    def call(prompt: str, response_format: dict[str, Any] | None = None) -> str:
        stage_num = len(texts) + 1
        t0 = time.perf_counter()
        try:
            complete_kw = {"response_format": response_format} if response_format is not None else {}
            text, cost = complete(
                prompt, config, seed=seed, run_id=run_id, query_id=query_id, **complete_kw
            )
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            wall_s = time.perf_counter() - t0
            exc_str = str(exc)
            if len(exc_str) > 400:
                exc_str = exc_str[:400] + "..."
            call_failures.append(f"call {stage_num} rejected: {exc_str}")
            texts.append("")
            costs.append(
                CostRecord(
                    run_id=run_id,
                    query_id=query_id,
                    component="generate",
                    backend=f"vllm:{config.model}",
                    input_tokens=None,
                    output_tokens=None,
                    wall_s=wall_s,
                )
            )
            return ""
        else:
            texts.append(text)
            costs.append(cost)
            return text

    if system is System.POST_HOC:
        answer = call(
            build_prompt(
                system, question, context, config.max_citations, stage="answer", depth=depth
            )
        )
        stage1_parsed = parse_response(answer, context, config.max_citations, require_decision=False)
        claim_count = len(stage1_parsed.claims)
        guided_format = None
        if config.guided_decoding and claim_count > 0 and config.backend in ("vllm", "anthropic"):
            guided_format = build_citation_response_format(context, claim_count, config.max_citations)

        if guided_format is not None:
            batches = [
                list(stage1_parsed.claims[i : i + MAX_CLAIMS_PER_CITE_CALL])
                for i in range(0, claim_count, MAX_CLAIMS_PER_CITE_CALL)
            ]
            final_claims: list[Claim] = []
            guided_errors: list[str] = []
            guided_recovered: list[str] = []

            for batch in batches:
                rendered_claims = "\n".join(
                    f"CLAIM {i+1}: {c.text}" for i, c in enumerate(batch)
                )
                batch_format = build_citation_response_format(
                    context, len(batch), config.max_citations
                )
                prompt_stage2 = build_prompt(
                    system,
                    question,
                    context,
                    config.max_citations,
                    stage="recite_json",
                    answer=rendered_claims,
                    depth=depth,
                    claim_count=len(batch),
                )
                parsed_from = call(prompt_stage2, response_format=batch_format)
                if parsed_from.startswith('"'):
                    try:
                        unwrapped = json.loads(parsed_from)
                    except json.JSONDecodeError:
                        pass
                    else:
                        if isinstance(unwrapped, str):
                            parsed_from = unwrapped
                parsed_json = parse_response(
                    parsed_from, context, config.max_citations, require_decision=False
                )
                guided_errors.extend(parsed_json.errors)
                guided_recovered.extend(parsed_json.recovered)

                if len(parsed_json.claims) != len(batch):
                    guided_errors.append(
                        f"cite stage returned {len(parsed_json.claims)} CLAIM lines for {len(batch)} claims sent"
                    )

                for i, orig_c in enumerate(batch):
                    if i < len(parsed_json.claims):
                        final_claims.append(
                            replace(orig_c, citations=parsed_json.claims[i].citations)
                        )
                    else:
                        guided_errors.append(
                            f"{orig_c.claim_id}: no matching CLAIM line in the cite-stage reply"
                        )
                        final_claims.append(replace(orig_c, citations=[]))

            parsed = ParsedResponse(
                decision=stage1_parsed.decision,
                claims=final_claims,
                errors=guided_errors,
                recovered=guided_recovered,
            )
        else:
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
            parsed = parse_response(parsed_from, context, config.max_citations)
    elif system is System.JOINT:
        guided_format = None
        if config.guided_decoding and config.backend in ("vllm", "anthropic"):
            guided_format = build_citation_response_format(
                context, claim_count=None, max_citations=config.max_citations, is_joint=True
            )
        if guided_format is not None:
            prompt_joint = build_prompt(
                system,
                question,
                context,
                config.max_citations,
                stage="joint_json",
                depth=depth,
            )
            parsed_from = call(prompt_joint, response_format=guided_format)
            if parsed_from.startswith('"'):
                try:
                    unwrapped = json.loads(parsed_from)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(unwrapped, str):
                        parsed_from = unwrapped
            parsed = parse_response(
                parsed_from, context, config.max_citations, require_decision=True
            )
        else:
            parsed_from = call(
                build_prompt(system, question, context, config.max_citations, depth=depth)
            )
            parsed = parse_response(parsed_from, context, config.max_citations)
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
    return Generation(
        record,
        tuple(costs),
        tuple((*call_failures, *parsed.errors)),
        tuple(record.validate()),
        tuple(parsed.recovered),
    )


def _total(values: Iterable[float | None]) -> float | None:
    """Sum across stages, or `None` if any stage did not report — a partial total would read as a
    cheap query in Table 4 rather than as missing instrumentation."""
    seen = list(values)
    if any(v is None for v in seen):
        return None
    return sum(seen)  # type: ignore[arg-type]


#: Maximum claims sent to the model in one `cite_claims` call. Measured, not chosen: a live A4000
#: run (`docs/harvest/decompose_smoke.summary.json`, 2026-08-16) sent whole re-cut answers — 20 to
#: 25 atomic claims — in a single call and the 8B model stopped reproducing after 4 to 7 of them,
#: so `cite stage returned N CLAIM lines for M claims sent` fired on 8 to 9 of every 10 queries and
#: pinned `clean_cite_rate` at exactly 0.0. Batching bounds what one reply has to copy; the cost is
#: one extra call per five claims, which C7 pays once per row.
MAX_CLAIMS_PER_CITE_CALL = 5


@dataclass(frozen=True, slots=True)
class Recitation:
    """C7's citation re-run: `decompose.py`'s claims, now carrying citations against `passages`.

    Mirrors `Generation`'s error handling — a claim the reply dropped is kept with no citations
    rather than discarded, so the denominator `errors` is read against never shrinks silently.

    `costs` is a tuple because the re-run is batched (`MAX_CLAIMS_PER_CITE_CALL`): Table 4 has to
    see every call the row actually paid for, not just the last one.
    """

    claims: tuple[Claim, ...]
    costs: tuple[CostRecord, ...]
    errors: tuple[str, ...]
    #: Transcription drift that was read rather than rejected — see `prompts.ParsedResponse`.
    recovered: tuple[str, ...] = ()


def cite_claims(
    claims: Sequence[Claim],
    question: str,
    passages: Sequence[RetrievedPassage],
    config: GenerationConfig,
    *,
    complete: Completer = backends.complete,
    seed: int = 0,
    run_id: str = "",
    query_id: str | None = None,
    depth: int = CONTEXT_DEPTH,
    max_claims_per_call: int = MAX_CLAIMS_PER_CITE_CALL,
    guided_decoding: bool = True,
) -> Recitation:
    """Attach fresh citations to an already re-cut answer — chosen as Option A over mapping the
    original citations onto the new claim boundaries (HANDOFF.md): the old citations were located
    against units that no longer exist, so a real citation-F1 for C7 needs a real second pass, at
    the cost of one extra call per row.

    `claims` is one `Decomposition.claims` — any granularity, any originating system. The reply is
    matched back to `claims` **positionally**: the cite stage is told to reproduce every claim in
    order and never told a claim's id, so an id is not a channel this function can rely on to
    re-align a dropped or added line. A count mismatch is `errors`, not a raised exception — same
    reason `decompose.parse_decomposition` never drops a claim for being wrong-shaped.

    Claims are sent in batches of `max_claims_per_call`, each batch numbered from 1, for the reason
    `decompose.decompose` calls the model once per sentence: positional matching is only as good as
    the model's willingness to reproduce every line, and past a handful of claims it stops. Batching
    narrows the positional window; it does not repair a batch that still comes back short, which is
    still counted per claim.
    """
    context = list(passages[:depth])
    if not context:
        raise ValueError(f"{query_id}: no passages to cite against; retrieval must run first")
    if not claims:
        raise ValueError("no claims to cite")
    if max_claims_per_call < 1:
        raise ValueError("max_claims_per_call must be at least 1")

    batches = [
        list(claims[i : i + max_claims_per_call])
        for i in range(0, len(claims), max_claims_per_call)
    ]
    cited: list[Claim] = []
    costs: list[CostRecord] = []
    errors: list[str] = []
    recovered: list[str] = []

    for batch_idx, batch in enumerate(batches):
        rendered = "\n".join(f"CLAIM {i}: {c.text}" for i, c in enumerate(batch, start=1))
        # A schema is only buildable against vLLM's structured-output engine, and only when the
        # passages actually yield candidate spans. `None` on either count falls back to the prose
        # stage rather than silently citing nothing.
        guided_format = (
            build_citation_response_format(context, len(batch), config.max_citations)
            if guided_decoding and config.backend == "vllm"
            else None
        )
        prompt = build_prompt(
            System.POST_HOC, question, context, config.max_citations,
            stage="recite_json" if guided_format else "recite",
            answer=rendered, depth=depth, claim_count=len(batch),
        )
        batch_query_id = (
            f"{query_id}:cite{batch_idx}" if len(batches) > 1 and query_id else query_id
        )
        complete_kw = {"response_format": guided_format} if guided_format is not None else {}
        raw, cost = complete(
            prompt, config, seed=seed, run_id=run_id, query_id=batch_query_id,
            **complete_kw,
        )
        if guided_format and raw.startswith('"'):
            # vLLM occasionally returns the structured reply as a JSON *string* holding the object
            # rather than the object itself. Unwrapping one level of encoding is reading the reply,
            # not repairing it: anything that does not decode to a string is left exactly as it came
            # back, so `parse_response` still gets to call it malformed.
            try:
                unwrapped = json.loads(raw)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(unwrapped, str):
                    raw = unwrapped
        cost.component = "decompose_cite"
        costs.append(cost)
        parsed = parse_response(
            raw, context, config.max_citations, require_decision=False
        )

        errors.extend(parsed.errors)
        recovered.extend(parsed.recovered)
        if len(parsed.claims) != len(batch):
            errors.append(
                f"cite stage returned {len(parsed.claims)} CLAIM lines for {len(batch)} claims sent"
            )
        for i, original in enumerate(batch):
            if i < len(parsed.claims):
                citations = parsed.claims[i].citations
            else:
                citations = []
                errors.append(
                    f"{original.claim_id}: no matching CLAIM line in the cite-stage reply"
                )
            cited.append(replace(original, citations=citations))

    return Recitation(tuple(cited), tuple(costs), tuple(errors), tuple(recovered))
