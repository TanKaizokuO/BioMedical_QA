"""The two properties ADR-0016 made load-bearing: one shared order, and blinding.

Everything else in the annotation path is a form; these are the parts that, if they silently
break, invalidate G4 after 30–49 annotator-hours have already been spent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomedqa.annotate import (
    build_tasks,
    collector_token,
    common_prefix,
    human_labels,
    ingest_annotations,
    question_order,
    render_form,
    snapshot_summary,
    tasks_to_payload,
)
from biomedqa.schema import Citation, Claim, QueryRecord, RetrievedPassage, SupportLabel, System

PASSAGE = "Metformin reduced HbA1c by 1.2% at 12 weeks. No effect on weight was observed."


def record(query_id: str, system: System, n_claims: int = 2) -> QueryRecord:
    return QueryRecord(
        run_id="run-1",   # one run_id across systems, as `generate_smoke` writes it
        query_id=query_id,
        question=f"Does drug X work in cohort {query_id}?",
        system=system,
        seed=0,
        retrieved=[RetrievedPassage(passage_id="p1", rank=1, score=1.0, retriever="rerank", text=PASSAGE)],
        claims=[
            Claim(
                claim_id=f"c{i}",   # per-record ids, which collide across systems and questions
                text=f"Claim {i} about {query_id}.",
                citations=[Citation(passage_id="p1", char_start=0, char_end=44)],
            )
            for i in range(n_claims)
        ],
    )


def corpus(n_questions: int = 6) -> list[QueryRecord]:
    return [
        record(f"q{i}", system)
        for i in range(n_questions)
        for system in (System.JOINT, System.POST_HOC)
    ]


def test_question_order_is_shared_and_seed_stable():
    ids = [f"q{i}" for i in range(20)]
    assert question_order(ids) == question_order(list(reversed(ids)))
    assert question_order(ids) != ids                      # it is actually shuffled
    assert question_order(ids, seed=1) != question_order(ids, seed=2)
    assert sorted(question_order(ids)) == sorted(ids)      # and it is a permutation


def test_all_annotators_get_the_same_order():
    tasks, _ = build_tasks(corpus())
    payloads = [tasks_to_payload(tasks, a, seed=1) for a in ("a1", "a2", "a3")]
    hashes = {p["order_hash"] for p in payloads}
    assert len(hashes) == 1
    uids = [[q["question_uid"] for q in p["questions"]] for p in payloads]
    assert uids[0] == uids[1] == uids[2]
    assert [q["order_index"] for q in payloads[0]["questions"]] == list(range(len(tasks)))


def test_the_form_carries_no_system_model_or_run_identity():
    records = corpus()
    tasks, keyfile = build_tasks(records)
    # The data the form is built around, as it is actually embedded in the delivered file. The
    # surrounding page is prose ("jointly necessary citations"), which is why this is scoped to
    # the payload rather than grepped over the whole HTML.
    html = render_form(tasks, "a1")
    blob = html.split('type="application/json">')[1].split("</script>")[0]
    for leak in ("joint", "post_hoc", "vanilla", "run-", "claim_id", "system"):
        assert leak not in blob, f"{leak!r} leaked into the annotator's file"
    # The keyfile is what carries it, and it covers every unit exactly once.
    units = {c.unit_id for t in tasks for c in t.claims}
    assert {row["unit_id"] for row in keyfile} == units
    assert len(keyfile) == len(units)
    assert {row["system"] for row in keyfile} == {"joint", "post_hoc"}


def test_unit_ids_are_unique_across_systems_and_questions():
    # `claim_id` is per-record: the same question under two systems both emit `c0`. A collision
    # would fuse two claims into one form control and one keyfile row — found in the browser,
    # kept here.
    tasks, keyfile = build_tasks(corpus(4))
    units = [c.unit_id for t in tasks for c in t.claims]
    assert len(units) == len(set(units)) == len(keyfile) == 4 * 2 * 2


def test_claims_of_one_question_interleave_systems():
    tasks, keyfile = build_tasks(corpus(), seed=7)
    key = {row["unit_id"]: row["system"] for row in keyfile}
    blocked = [
        [key[c.unit_id] for c in t.claims] == sorted([key[c.unit_id] for c in t.claims])
        for t in tasks
    ]
    # Every question pools both systems; not every question can interleave by chance, but a
    # seeded shuffle that never does is a shuffle that is not happening.
    assert not all(blocked)


def test_uncited_claims_are_not_annotation_units():
    r = record("q0", System.VANILLA)
    for claim in r.claims:
        claim.citations = []
    tasks, keyfile = build_tasks([r])
    assert tasks == [] and keyfile == []


def test_citation_without_passage_text_falls_back_to_the_quote():
    r = record("q0", System.JOINT, n_claims=1)
    r.retrieved[0].text = None
    r.claims[0].citations = [
        Citation(passage_id="p1", char_start=10, char_end=54, quoted_text="a" * 44)
    ]
    tasks, _ = build_tasks([r])
    span = tasks[0].claims[0].spans[0]
    assert (span.passage_text, span.char_start, span.char_end) == ("a" * 44, 0, 44)


def test_human_labels_round_trip_the_union_judgement(tmp_path):
    path = tmp_path / "labels_a1.jsonl"
    rows = [
        {"type": "label", "annotator_id": "a1", "unit_id": "u_x", "citation_index": None,
         "support_label": "PARTIAL", "claim_validity": True, "notes": None},
        {"type": "label", "annotator_id": "a1", "unit_id": "u_x", "citation_index": 0,
         "support_label": "SUPPORTED", "claim_validity": True, "notes": ""},
        {"type": "question", "annotator_id": "a1", "question_uid": "q_x", "order_index": 0,
         "started_at": "2026-09-08T10:00:00Z", "completed_at": "2026-09-08T10:04:00Z",
         "active_s": 240.0},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    from biomedqa.annotate import read_labels

    labels, questions = read_labels(path)
    assert len(labels) == 2 and len(questions) == 1
    by_unit = human_labels(labels)
    union = [x for x in by_unit["u_x"] if x.citation_index is None]
    assert len(union) == 1 and union[0].support_label is SupportLabel.PARTIAL
    assert union[0].support_label.is_supporting         # the binary collapse G4 gates on
    assert by_unit["u_x"][1].notes is None              # "" is absence, not a note


def q(index: int, uid: str, completed: bool = True) -> dict:
    return {
        "order_index": index,
        "question_uid": uid,
        "completed_at": "2026-09-08T10:00:00Z" if completed else None,
    }


def test_common_prefix_stops_at_the_first_gap():
    a1 = [q(0, "a"), q(1, "b"), q(2, "c"), q(3, "d")]
    a2 = [q(0, "a"), q(1, "b"), q(2, "c")]
    a3 = [q(0, "a"), q(1, "b")]
    assert common_prefix(a1, a2, a3) == ["a", "b"]


def test_a_half_finished_question_does_not_enter_the_prefix():
    a1 = [q(0, "a"), q(1, "b")]
    a2 = [q(0, "a"), q(1, "b")]
    a3 = [q(0, "a"), q(1, "b", completed=False)]
    assert common_prefix(a1, a2, a3) == ["a"]


def test_divergent_orders_are_an_error_not_a_smaller_prefix():
    a1 = [q(0, "a"), q(1, "b")]
    a2 = [q(0, "a"), q(1, "c")]
    with pytest.raises(ValueError, match="shared order was violated"):
        common_prefix(a1, a2)


# --------------------------------------------------------------- collector sidecar (Q6–Q9)


def test_form_without_a_collector_url_stays_offline():
    tasks, _ = build_tasks(corpus())
    payload = tasks_to_payload(tasks, "a1", seed=1)
    assert payload["collector"] is None
    assert "http" not in render_form(tasks, "a1").split("<script id=\"tasks\"")[1].split("</script>")[0]


def test_each_form_carries_only_its_own_token():
    tasks, _ = build_tasks(corpus())
    url = "http://a4000.local:8811"
    tokens = {a: tasks_to_payload(tasks, a, seed=1, collector_url=url)["collector"]["token"]
              for a in ("a1", "a2", "a3")}
    assert len(set(tokens.values())) == 3          # a1 cannot read a2's state
    for annotator, token in tokens.items():
        assert token == collector_token(annotator, seed=1)
        html = render_form(tasks, annotator, seed=1, collector_url=url)
        assert token in html
        assert not any(t in html for a, t in tokens.items() if a != annotator)


def test_collector_url_trailing_slash_does_not_double_up():
    tasks, _ = build_tasks(corpus())
    payload = tasks_to_payload(tasks, "a1", seed=1, collector_url="http://box:8811/")
    assert payload["collector"]["url"] == "http://box:8811"


def test_snapshot_summary_counts_completion_and_projects_hours():
    state = {
        "questions": {
            "u1": {"started_at": "t", "completed_at": "t", "active_s": 300.0},
            "u2": {"started_at": "t", "completed_at": "t", "active_s": 420.0},
            "u3": {"started_at": "t", "completed_at": None, "active_s": 60.0},
        },
        "answers": {
            "x": {"validity": True, "union": "SUPPORTED", "spans": {}},
            "y": {"validity": True, "union": None, "spans": {}},   # half-answered: not counted
        },
    }
    s = snapshot_summary(state, total_questions=100)
    assert (s["questions_started"], s["questions_complete"], s["claims_labeled"]) == (3, 2, 1)
    assert s["active_s"] == 780.0
    # 390 s per completed question over 100 questions ≈ 10.8 h — the pilot's projection.
    assert s["projected_h"] == pytest.approx(10.83, abs=0.01)


def test_snapshot_summary_of_an_untouched_pass_projects_nothing():
    assert snapshot_summary({}, total_questions=250) == {
        "questions_started": 0, "questions_complete": 0, "questions_total": 250,
        "claims_labeled": 0, "active_s": 0.0, "projected_h": 0.0,
    }

# --- ingest_annotations tests ---------------------------------------------------------------------

def test_ingest_annotations_happy_path_and_provenance(tmp_path):
    keyfile_rows = [
        {
            "unit_id": "u_1",
            "question_uid": "q_1",
            "order_index": 0,
            "claim_id": "c0",
            "query_id": "101",
            "run_id": "r_1",
            "system": "joint",
            "seed": 0,
        }
    ]
    kf_path = tmp_path / "keyfile.jsonl"
    kf_path.write_text("\n".join(json.dumps(r) for r in keyfile_rows) + "\n", encoding="utf-8")

    label_rows = [
        {
            "type": "label",
            "annotator_id": "a1",
            "unit_id": "u_1",
            "citation_index": None,
            "support_label": "SUPPORTED",
            "claim_validity": True,
            "notes": "Good claim",
        }
    ]
    lf_path = tmp_path / "labels_a1.jsonl"
    lf_path.write_text("\n".join(json.dumps(r) for r in label_rows) + "\n", encoding="utf-8")

    rec = record("101", System.JOINT, n_claims=1)
    rec.run_id = "r_1"
    rec.seed = 0

    ingested = ingest_annotations(lf_path, kf_path, records=[rec])

    assert len(ingested) == 1
    row = ingested[0]
    assert row["annotator_id"] == "a1"
    assert row["unit_id"] == "u_1"
    assert row["query_id"] == "101"
    assert row["claim_id"] == "c0"
    assert row["run_id"] == "r_1"
    assert row["system"] == "joint"
    assert row["seed"] == 0

    # Verify attached to record claim
    c1 = rec.claims[0]
    assert len(c1.human_labels) == 1
    assert c1.human_labels[0].annotator_id == "a1"
    assert c1.human_labels[0].support_label is SupportLabel.SUPPORTED


def test_ingest_annotations_rejects_duplicates(tmp_path):
    keyfile = [{"unit_id": "u_1", "claim_id": "c1", "query_id": "q1", "run_id": "r1", "system": "joint", "seed": 0}]
    dup_labels = [
        {"type": "label", "annotator_id": "a1", "unit_id": "u_1", "citation_index": None, "support_label": "SUPPORTED", "claim_validity": True},
        {"type": "label", "annotator_id": "a1", "unit_id": "u_1", "citation_index": None, "support_label": "PARTIAL", "claim_validity": True},
    ]

    with pytest.raises(ValueError, match="Duplicate label for"):
        ingest_annotations([dup_labels], keyfile)


def test_ingest_annotations_rejects_invalid_label():
    keyfile = [{"unit_id": "u_1", "claim_id": "c1", "query_id": "q1", "run_id": "r1", "system": "joint", "seed": 0}]
    invalid_labels = [
        {"type": "label", "annotator_id": "a1", "unit_id": "u_1", "citation_index": None, "support_label": "INVALID_LABEL_VAL", "claim_validity": True}
    ]

    with pytest.raises(ValueError, match="Invalid SupportLabel value"):
        ingest_annotations([invalid_labels], keyfile)


def test_ingest_annotations_rejects_unrecognized_unit_id():
    keyfile = [{"unit_id": "u_1", "claim_id": "c1", "query_id": "q1", "run_id": "r1", "system": "joint", "seed": 0}]
    unrecognized_labels = [
        {"type": "label", "annotator_id": "a1", "unit_id": "u_UNKNOWN", "citation_index": None, "support_label": "SUPPORTED", "claim_validity": True}
    ]

    with pytest.raises(ValueError, match="Unrecognized unit_id"):
        ingest_annotations([unrecognized_labels], keyfile)
