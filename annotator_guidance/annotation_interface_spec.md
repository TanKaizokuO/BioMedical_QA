# Annotation interface — interface specification

**Status:** normative for any annotation interface used in this project · **Written:** 2026-08-13
**Governed by:** ADR-0006, ADR-0011 §2, ADR-0016, ADR-0017 · **Vocabulary:** `CONTEXT.md`

This document specifies the **annotation interface** as a replaceable component: what it is given,
what it must return, and the properties that gate **G4** depends on. It exists because the
interface is being built separately from this repository.

`src/biomedqa/annotate.html` is the reference implementation. This specification is the contract;
where the two disagree, this document is what a replacement must satisfy.

**Terms** (`CONTEXT.md` is authoritative): **claim**, **claim validity**, **citation**,
**support label**. One **annotation unit** is one claim with its cited spans, identified by
`unit_id`. An **annotator** produces one **pass** over the gold set.

---

## 1. What the interface is for

Three annotators independently label the full gold set. For every claim, an annotator gives:

* one **claim validity** flag — is this a well-formed, self-contained claim at all;
* one **support label** for each cited span of the claim;
* one **support label** for all cited spans **together** — the union judgement, which is the
  quantity citation recall is scored on.

Two properties make those judgements usable, and both land on this interface rather than on
scoring. Neither is a courtesy:

1. **One shared, seeded question order, worked as a prefix** (ADR-0016 §2). Any common prefix
   across the three passes is then an unbiased random subsample of the gold set, whatever point
   anyone stops at. This holds only if all three orders are identical, nobody skips ahead, and a
   question is never half-labelled.
2. **Blinding** (ADR-0016 §4). The annotator sees the question, the claim and the cited spans.
   System, model and run identity are not in the annotator's artifact. The primary annotator is the
   author of the method; without blinding, α measures the author's memory.

Everything else about the interface is free.

---

## 2. Input

### 2.1 The task payload

The build step in this repository produces one payload per annotator. It is the interface's only
input. Reference producer: `biomedqa.annotate.tasks_to_payload`, embedded by `render_form` into
`<script id="tasks" type="application/json">`.

```jsonc
{
  "annotator_id": "a1",
  "seed": 20260907,
  "order_hash": "9f1c2ab30d44",        // 12 hex chars; identical in all three payloads
  "collector": {                       // null when built without a collector
    "url": "http://a4000.local:8811",
    "token": "…"                       // this annotator's token only
  },
  "questions": [                       // ALREADY in the shared order; do not sort
    {
      "question_uid": "q_1a2b3c4d5e6f",
      "order_index": 0,
      "question": "Does prophylaxis reduce mortality in …?",
      "claims": [
        {
          "unit_id": "u_0011aabbccdd",
          "text": "Metformin reduces all-cause mortality in patients with type 2 diabetes.",
          "spans": [
            {
              "citation_index": 0,
              "passage_text": "…full passage text…",
              "char_start": 13,
              "char_end": 56
            }
          ]
        }
      ]
    }
  ]
}
```

### 2.2 Input rules

| Rule | Reason |
|---|---|
| `questions` is **already ordered**. Render in array order. Never sort, shuffle or re-key. | The order *is* the shared order |
| `claims` within a question is **already shuffled** across systems. Never sort. | Consecutive units must not be one system's output in a block (§4) |
| `unit_id`, `question_uid`, `order_hash` are **opaque**. Never parse, never derive anything from them. | They are hashes; their structure is not a contract |
| `char_start`/`char_end` are offsets into `passage_text`, and may be equal or out of range. Clamp for display; never mutate. | A citation whose passage text did not survive the run is shown as its quoted span alone |
| A claim has **at least one** span. Claims with no citations are not annotation units and never reach the interface. | Module docstring, `annotate.py`; ADR-0010 |
| A claim has **at most three** spans. | The 3-citation cap, `CONTEXT.md` |
| The payload must not be enriched with anything from outside it. | Blinding |

### 2.3 If the interface also generates ids

Consuming the payload above is the supported seam and is strongly preferred: the ids then match
`annotation/keyfile.jsonl` by construction. An interface that generates its own ids must reproduce
these exactly, or the read-back join fails:

* `canonical_hash(obj)` = first 12 hex chars of `sha256(json.dumps(obj, sort_keys=True, default=str))`.
* `order_hash` = `canonical_hash([question_uid, …])` over the questions in order.
* `unit_id` = `"u_" + canonical_hash(["u", seed, run_id, system, query_id, str(record_seed), claim_id])`.
* `question_uid` = `"q_" + canonical_hash(["q", seed, query_id])`.
* `collector_token(a)` = `canonical_hash(["collector_token", seed, a])`.

`claim_id` is unique only inside one record. The same question under two systems both emit `c1`;
the id must be built from the whole provenance tuple or two claims collapse into one unit.

---

## 3. Output

### 3.1 The export

One JSONL file per annotator, UTF-8, one JSON object per line, `\n`-terminated. Two row types.
This file is the deliverable; everything else the interface stores is private to it.

**Question row** — one per question the annotator *started*:

```jsonc
{
  "type": "question",
  "annotator_id": "a1",
  "question_uid": "q_1a2b3c4d5e6f",
  "order_index": 0,
  "order_hash": "9f1c2ab30d44",
  "started_at": "2026-09-08T09:14:02.318Z",   // ISO-8601 UTC
  "completed_at": "2026-09-08T09:31:44.907Z", // null if not marked complete
  "active_s": 1062.6                          // seconds of active work, see §5
}
```

**Label row** — one per span judgement, plus one per claim for the union judgement:

```jsonc
{
  "type": "label",
  "annotator_id": "a1",
  "unit_id": "u_0011aabbccdd",
  "question_uid": "q_1a2b3c4d5e6f",
  "citation_index": 0,          // integer = that cited span; null = the union judgement
  "support_label": "SUPPORTED", // SUPPORTED | PARTIAL | NOT_SUPPORTED | CONTRADICTED
  "claim_validity": true,       // per claim, repeated on every row of that claim
  "notes": null                 // free text or null
}
```

### 3.2 Output rules

| Rule | Reason |
|---|---|
| Emit a label row **only** when both `support_label` and `claim_validity` are set. | A half-answered claim must be absent, not present with nulls |
| Emit exactly one union row per answered claim, `citation_index: null`. | It is the citation-recall quantity, and a separate population from the span rows |
| `(annotator_id, unit_id, citation_index)` is **unique** across the file. | A duplicate means two passes were merged by hand; read-back refuses |
| Store the **4-way** label. Never write a collapsed or binarized label. | `CONTEXT.md`: binarizing on write destroys the AUROC sweep and the calibration bins irrecoverably |
| `order_hash` on every question row. | A pass from a rebuilt order is caught on read, not after α |
| Question rows for started-but-incomplete questions are **required**. | They are how a partial pass is separated from a half-finished question (ADR-0016 §2) |
| Timestamps ISO-8601 with an explicit UTC offset. | Three machines, three time zones possible |
| No system, model, run or query identity anywhere in the file. | Blinding survives into the artifact |

### 3.3 What the export is joined to

Read-back joins `unit_id` against `annotation/keyfile.jsonl`, which the annotators never receive:

```jsonc
{"unit_id":"u_0011aabbccdd","question_uid":"q_…","order_index":0,
 "claim_id":"c2","query_id":"…","run_id":"…","system":"joint","seed":0}
```

`common_prefix()` then takes the question rows of all three passes and returns the questions every
annotator marked complete, stopping at the first `order_index` any of them did not complete. That
prefix is G4's population. It is a prefix, never a set intersection.

---

## 4. Ordering and completion

1. Present questions in payload order.
2. A question is **complete** only when every claim in it has: a claim validity flag, a support
   label for every span, and a union label. Do not allow completion before that.
3. Do not let the annotator open a question beyond the first incomplete one. The prefix property
   is void if anyone skips ahead.
4. Re-visiting and changing an earlier, already-complete question is allowed. `completed_at` keeps
   its first value.
5. `order_index` in the export must be the payload's value, never a display position.

---

## 5. The clock

`active_s` funds the cost model and the 10–16 h projection, so it must measure work, not wall
clock. Per question:

* accumulate time only while the question is displayed and the interface is visible;
* fold accumulated time in **before** any re-render, so a redraw never discards the interval
  before it;
* stop the clock when the annotator navigates away, when the document becomes hidden, and after a
  fixed idle period with no input (the reference implementation uses 120 s);
* restart it on input.

Report seconds, one decimal. `active_s` accumulates across sessions.

---

## 6. Storage, backup and restore

Requirements, not a design:

1. **Primary store is local to the annotator's machine and survives a browser restart.** The task
   runs 10–16 h over three weeks.
2. **Storage is keyed by `annotator_id` + `order_hash`.** A rebuilt order must not silently adopt
   old progress.
3. **No shared live store.** No annotator may see another's judgements, or any aggregate over
   them, before all passes are finished. This is the one hard prohibition on the interface.
4. **Export on demand, at any time**, including mid-question.
5. **Restore is explicit and never merges.** Offer each recoverable copy with its completion count
   and save time; the chosen copy *replaces* local state. An automatic merge across two partial
   passes can leave a claim labelled by nobody and nothing surfaces it until α is wrong.

### 6.1 Optional backup sidecar (ADR-0017)

If the interface mirrors progress, `scripts/annotation_collect.py` is the existing endpoint. It is
additive: the local store stays primary and a failed mirror is reported, never blocking.

| Route | Behaviour |
|---|---|
| `POST /state/<annotator>?token=…` | Body `{annotator_id, order_hash, saved_at, state}` as `text/plain;charset=UTF-8` (CORS-simple, so a `file://` page needs no preflight). Stores a new timestamped file; never overwrites |
| `GET /state/<annotator>/restore?token=…` | Returns the **furthest-along** stored snapshot, `404` when none |
| `GET /health` | Liveness |

Responses: `403` bad token, `409` `order_hash` disagrees with the built forms, `400` malformed,
`413` body over 8 MB. Coalesce writes — one POST per click buries the burn-down. Retry a network
failure with backoff; do not retry a `409`, which means the order is wrong and mirroring cannot fix
it. `state` is opaque to the collector, so its internal shape is the interface's own choice, but
`state.questions[*].completed_at` and `active_s` must be present for the burn-down and restore
ranking to work.

---

## 7. What the annotator must be shown

Content requirements only; presentation is free.

* The question, the claim text, and every cited span **highlighted inside its passage** — the
  surrounding passage is context, and the judgement is about the highlighted span.
* The four support labels named exactly `SUPPORTED`, `PARTIAL`, `NOT_SUPPORTED`, `CONTRADICTED`,
  never renamed, reordered arbitrarily or reduced.
* The claim validity question, asked separately, and asked even when the claim is malformed.
* The union judgement, presented as a **separate** judgement over all spans together, not as a
  summary of the per-span answers.
* Progress: position in the order, questions complete, and when the last save happened.
* The short guidelines, in the interface. The full guide is `annotator_guidance/ANNOTATOR_GUIDE.md`.

Not shown, ever: another annotator's work, any aggregate over annotators, any system or model
identity, any score, and any indication of which claims are "expected" to be supported.

---

## 8. Acceptance checks

A replacement interface is acceptable when all of these hold.

**Contract**

1. Three payloads built from one build step yield three exports whose question rows carry the same
   `order_hash`.
2. Every `unit_id` in every export is present in `keyfile.jsonl`, and every annotated unit appears
   with exactly one union row.
3. `(annotator_id, unit_id, citation_index)` has no duplicates.
4. Every export row validates against §3.1; `support_label` is always one of the four.

**Blinding**

5. The artifact given to an annotator contains none of `joint`, `post_hoc`, `vanilla`, `run-`,
   `claim_id`, `system`. The reference test greps the embedded payload for exactly this list
   (`tests/test_annotate.py::test_the_form_carries_no_system_model_or_run_identity`); the
   surrounding prose legitimately contains "jointly necessary", so the check is scoped to the
   payload, not the page.
6. One annotator's artifact contains no other annotator's token.

**Order**

7. A question cannot be completed with any answer missing.
8. A question after the first incomplete one cannot be opened.
9. `common_prefix()` over the three exports returns a prefix, and raises if two annotators name
   different questions at one `order_index`.

**Durability**

10. Progress survives a browser restart.
11. Export, clear the local store, restore from that export: state matches, including
    `completed_at` and `active_s`.
12. With the collector unreachable, labelling continues and nothing is lost; when it returns, the
    backup catches up without the annotator doing anything.

**Clock**

13. Ten answers spread over ten minutes accumulate roughly ten minutes, not the interval since the
    last answer.
14. An idle or hidden interface accumulates no time.

---

## 9. Prohibitions

Each of these silently invalidates a published number:

1. **Do not sort or regenerate the question order**, or build the three passes from separate
   randomizations. ADR-0016 §2 dies quietly.
2. **Do not binarize, collapse or drop a label on write.** `PARTIAL` is what makes citation
   precision honest; `CONTRADICTED` is the payload of the biomedical failure-mode analysis, and an
   annotator cannot be re-run.
3. **Do not merge two partial passes.**
4. **Do not adjudicate, resolve or hint at disagreement.** α is computed on raw per-annotator
   labels; adjudication is a separate downstream artifact.
5. **Do not show an annotator anything derived from another annotator.**
6. **Do not put the keyfile, or any provenance, on the annotators' machines or on the collector.**
7. **Do not let the interface omit question rows for incomplete questions** — the prefix cannot be
   computed from label rows alone.

---

## 10. Reference implementation and fixtures

| Artifact | What it gives you |
|---|---|
| `src/biomedqa/annotate.py` | Payload construction, ids, `render_form`, `read_labels`, `human_labels`, `common_prefix` |
| `src/biomedqa/annotate.html` | The reference interface |
| `scripts/build_annotation_ui.py` | Builds three payloads plus the keyfile from a records JSONL |
| `scripts/annotation_collect.py` | The backup endpoint in §6.1 |
| `scripts/annotation_status.py` | Burn-down over snapshots — counts only, never labels |
| `tests/test_annotate.py` | The blinding, ordering and id-collision checks in §8 |
| `annotator_guidance/ANNOTATOR_GUIDE.md` | What the annotators are told; the interface must not contradict it |

Generate a payload to develop against:

```bash
uv run python scripts/build_annotation_ui.py --records <records.jsonl> --out /tmp/annotation
```

It writes `annotate_a1.html`, `annotate_a2.html`, `annotate_a3.html` and `keyfile.jsonl`, and
prints the shared `order_hash`. The payload is the JSON inside `<script id="tasks">` of any form.

---

## 11. Post-Annotation Gate G3 Verdict Execution

For Gate G3 execution procedures, required input artifacts, cost evaluation specs, and full CLI options, see the canonical [Gate G3 Operator Runbook](../docs/harvest/runbooks/g3_runbook.md).
