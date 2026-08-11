"""Annotation task construction and the offline labelling form (ADR-0016, ADR-0006).

Three annotators independently label the full gold set. This module owes them exactly two
things that ADR-0016 made load-bearing, and nothing else:

1. **One seeded question order, shared by all three** (§2). Any common prefix of the three
   passes is then an unbiased random subsample of the gold set, whatever point anyone stops
   at. `question_order()` is a pure function of the question ids and the seed, so the three
   task files are generated from the same call and are checked byte-identical in their order.
2. **Blinding** (§4). The annotator's file carries the question, the claim and the cited
   spans. System, model and run identity live in a separate keyfile, joined at scoring time.
   The primary annotator is the author; without this, α measures the author's memory.

Everything else is deliberately absent. `render_form()` emits one self-contained HTML file per
annotator that stores progress in `localStorage` and downloads JSONL; it needs no server to be
usable. ADR-0016 says a static form suffices and that this is "not a licence to build a tool";
a *dashboard* would let one annotator see another's judgements before finishing, which §4 forbids.

What the form does have, when `collector_url` is passed, is a **write-mostly sidecar**: every save
is also POSTed to a LAN collector (`scripts/annotation_collect.py`) so a cleared cache or a dead
laptop is recoverable and the maintainer can count progress without asking. It is strictly
additive — the form keeps `localStorage` as the primary store and keeps working when the collector
is down. Each form carries only its own annotator's token, so no form can read another's state,
and the keyfile never goes near the collector.

Claims with no citations (every `VANILLA` claim, by definition) carry no attribution judgement
and are not annotation units. They are dropped here, not silently labelled `NOT_SUPPORTED`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .config import canonical_hash
from .schema import Claim, HumanLabel, QueryRecord, SupportLabel

# The date the annotation window opens (ROADMAP.md). Seeds in this repo record *when* a draw
# became immutable, the same convention as `data.SPLIT_SEED`.
ANNOTATION_SEED = 20260907

TEMPLATE_PATH = Path(__file__).resolve().parent / "annotate.html"

#: Row types in an annotator's exported JSONL. Question rows are what make a partial pass
#: separable from a half-finished question (ADR-0016 §2).
LABEL_ROW = "label"
QUESTION_ROW = "question"


@dataclass(slots=True)
class SpanTask:
    """One cited span, shown as a highlight inside the passage it was taken from."""

    citation_index: int
    passage_text: str
    char_start: int
    char_end: int


@dataclass(slots=True)
class ClaimTask:
    """One annotation unit: a claim and its cited spans, stripped of provenance."""

    unit_id: str
    text: str
    spans: list[SpanTask] = field(default_factory=list)


@dataclass(slots=True)
class QuestionTask:
    """One question's worth of work — the unit of ordering, stopping and completion."""

    question_uid: str
    order_index: int
    question: str
    claims: list[ClaimTask] = field(default_factory=list)


def question_order(question_ids: Iterable[str], *, seed: int = ANNOTATION_SEED) -> list[str]:
    """The one shared randomized question order (ADR-0016 §2).

    Pure in its inputs: the same ids and seed give the same order for every annotator, on every
    machine, on any day. Ids are sorted before the shuffle so that the caller's iteration order
    cannot leak into the result.
    """
    ids = sorted(set(question_ids))
    random.Random(seed).shuffle(ids)
    return ids


def _blind_id(kind: str, *parts: str, seed: int) -> str:
    """An opaque, stable id. Opaque because `joint::12345::c2` would announce the system."""
    return f"{kind}_{canonical_hash([kind, seed, *parts])}"


def build_tasks(
    records: Sequence[QueryRecord], *, seed: int = ANNOTATION_SEED
) -> tuple[list[QuestionTask], list[dict]]:
    """Blinded task list in the shared order, plus the keyfile that undoes the blinding.

    Claims for one question are pooled across systems and shuffled together, so consecutive
    units are not one system's output in a block (ADR-0016 §4). The returned keyfile maps each
    `unit_id` back to `(system, run_id, query_id, claim_id)` and must be written somewhere the
    annotators do not read.
    """
    by_question: dict[str, list[tuple[QueryRecord, Claim]]] = {}
    for record in records:
        for claim in record.claims:
            if not claim.citations:
                continue  # no cited span, no attribution judgement — see the module docstring
            by_question.setdefault(record.query_id, []).append((record, claim))

    order = question_order(by_question, seed=seed)
    tasks: list[QuestionTask] = []
    keyfile: list[dict] = []

    for index, query_id in enumerate(order):
        # Sorted into a canonical order first, so the caller's record order cannot survive the
        # shuffle; the key must be total, for the same reason `unit_id` is built from all of it.
        pairs = sorted(
            by_question[query_id],
            key=lambda p: (p[0].run_id, p[0].system.value, p[0].seed, p[1].claim_id),
        )
        random.Random(canonical_hash([seed, query_id])).shuffle(pairs)

        question_uid = _blind_id("q", query_id, seed=seed)
        claims: list[ClaimTask] = []
        for record, claim in pairs:
            # `claim_id` is only unique inside its record — the same question under two systems
            # both produce `c1`. The unit key is the whole provenance tuple, hashed.
            unit_id = _blind_id(
                "u",
                record.run_id,
                record.system.value,
                record.query_id,
                str(record.seed),
                claim.claim_id,
                seed=seed,
            )
            passages = {p.passage_id: p.text for p in record.retrieved}
            spans = [
                SpanTask(
                    citation_index=i,
                    # A citation whose passage text did not survive the run still has its quoted
                    # span; show that alone rather than dropping the unit.
                    passage_text=passages.get(c.passage_id) or (c.quoted_text or ""),
                    char_start=c.char_start if passages.get(c.passage_id) else 0,
                    char_end=c.char_end if passages.get(c.passage_id) else len(c.quoted_text or ""),
                )
                for i, c in enumerate(claim.citations)
            ]
            claims.append(ClaimTask(unit_id=unit_id, text=claim.text, spans=spans))
            keyfile.append(
                {
                    "unit_id": unit_id,
                    "question_uid": question_uid,
                    "order_index": index,
                    "claim_id": claim.claim_id,
                    "query_id": record.query_id,
                    "run_id": record.run_id,
                    "system": record.system.value,
                    "seed": record.seed,
                }
            )

        tasks.append(
            QuestionTask(
                question_uid=question_uid,
                order_index=index,
                question=pairs[0][0].question,
                claims=claims,
            )
        )

    units = [row["unit_id"] for row in keyfile]
    if len(set(units)) != len(units):
        raise ValueError(
            "unit_id collision: two claims share one annotation unit, which would silently make "
            "them one label in the form and one row in the keyfile"
        )

    return tasks, keyfile


def collector_token(annotator_id: str, *, seed: int = ANNOTATION_SEED) -> str:
    """The one secret in an annotator's form: it authorises writes and reads for *that* id only.

    Derived, not random, so `annotation_collect.py` and `annotation_status.py` can recompute it
    from the seed instead of carrying a secrets file around. It is a LAN convenience, not a lock:
    anyone who can read the collector's snapshot directory can read every annotator's labels.
    """
    return canonical_hash(["collector_token", seed, annotator_id])


def snapshot_summary(state: dict, total_questions: int) -> dict:
    """Progress counts for one saved state — the whole of what the maintainer gets to see.

    Deliberately not a per-label view: the point of the collector is a burn-down, and reading
    one annotator's judgements while another is still labelling is what ADR-0016 §4 forbids.
    """
    questions = state.get("questions") or {}
    answers = state.get("answers") or {}
    complete = sum(1 for q in questions.values() if q.get("completed_at"))
    active_s = sum(float(q.get("active_s") or 0.0) for q in questions.values())
    claims = sum(1 for a in answers.values() if a.get("validity") is not None and a.get("union"))
    per_question = active_s / complete if complete else 0.0
    return {
        "questions_started": len(questions),
        "questions_complete": complete,
        "questions_total": total_questions,
        "claims_labeled": claims,
        "active_s": round(active_s, 1),
        "projected_h": round(per_question * total_questions / 3600.0, 2),
    }


def tasks_to_payload(
    tasks: Sequence[QuestionTask],
    annotator_id: str,
    *,
    seed: int,
    collector_url: str | None = None,
) -> dict:
    """The JSON the form is built around. Contains no system, model or run identity."""
    return {
        "annotator_id": annotator_id,
        "seed": seed,
        "order_hash": canonical_hash([t.question_uid for t in tasks]),
        # Only this annotator's token, so one form can never read another's state (§4).
        "collector": (
            None
            if not collector_url
            else {
                "url": collector_url.rstrip("/"),
                "token": collector_token(annotator_id, seed=seed),
            }
        ),
        "questions": [
            {
                "question_uid": t.question_uid,
                "order_index": t.order_index,
                "question": t.question,
                "claims": [
                    {
                        "unit_id": c.unit_id,
                        "text": c.text,
                        "spans": [
                            {
                                "citation_index": s.citation_index,
                                "passage_text": s.passage_text,
                                "char_start": s.char_start,
                                "char_end": s.char_end,
                            }
                            for s in c.spans
                        ],
                    }
                    for c in t.claims
                ],
            }
            for t in tasks
        ],
    }


def render_form(
    tasks: Sequence[QuestionTask],
    annotator_id: str,
    *,
    seed: int = ANNOTATION_SEED,
    collector_url: str | None = None,
) -> str:
    """One self-contained HTML file: open it, label, download JSONL.

    Usable with no network at all. With `collector_url`, saves are additionally mirrored to the
    LAN collector and the form grows a Restore control; a failed mirror is reported in the header
    and never blocks labelling.
    """
    payload = tasks_to_payload(tasks, annotator_id, seed=seed, collector_url=collector_url)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    # `</script` inside a string literal would close the tag early; nothing else can escape it.
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return template.replace("__TASKS_JSON__", blob)


def read_labels(path) -> tuple[list[dict], list[dict]]:
    """Split one annotator's exported JSONL into label rows and question-completion rows."""
    labels: list[dict] = []
    questions: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            (questions if row.get("type") == QUESTION_ROW else labels).append(row)
    return labels, questions


def human_labels(rows: Sequence[dict]) -> dict[str, list[HumanLabel]]:
    """Label rows → `HumanLabel`s keyed by `unit_id`, ready to join through the keyfile.

    `citation_index is None` is the per-claim union judgement (`CONTEXT.md`, *Annotation
    record*), which is the quantity citation recall is scored on.
    """
    out: dict[str, list[HumanLabel]] = {}
    for row in rows:
        out.setdefault(row["unit_id"], []).append(
            HumanLabel(
                annotator_id=row["annotator_id"],
                support_label=SupportLabel(row["support_label"]),
                claim_validity=bool(row["claim_validity"]),
                citation_index=row.get("citation_index"),
                notes=row.get("notes") or None,
            )
        )
    return out


def common_prefix(*passes: Sequence[dict]) -> list[str]:
    """The triple-labeled common prefix — G4's population, and the Sep 20 tripwire (ADR-0016).

    Takes each annotator's *question* rows. A question counts only if that annotator marked it
    complete, so a half-finished question never enters the prefix. The prefix stops at the first
    `order_index` any annotator has not completed; it is a prefix of the shared order, never the
    set intersection, because only a prefix is a random subsample of questions.
    """
    if not passes:
        return []
    done = [
        {r["order_index"]: r["question_uid"] for r in rows if r.get("completed_at")}
        for rows in passes
    ]
    prefix: list[str] = []
    index = 0
    while all(index in d for d in done):
        uids = {d[index] for d in done}
        if len(uids) != 1:
            raise ValueError(
                f"order_index {index} names different questions across annotators ({sorted(uids)}) "
                "— the shared order was violated and ADR-0016 §2 does not hold"
            )
        prefix.append(uids.pop())
        index += 1
    return prefix
