"""Tests for G3 Gate Report Driver (scripts/g3_report.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from biomedqa.schema import (
    Citation,
    Claim,
    HumanLabel,
    QueryRecord,
    SupportLabel,
    System,
    VerifierScore,
    read_query_records,
    write_jsonl,
)

_REPO = Path(__file__).resolve().parents[1]


def make_record(
    qid: str,
    score: float | None = None,
    label: SupportLabel | None = None,
) -> QueryRecord:
    claims = []
    if score is not None or label is not None:
        v_scores = [VerifierScore(name="minicheck", score=score)] if score is not None else []
        h_labels = (
            [HumanLabel(annotator_id="a1", support_label=label, claim_validity=True)]
            if label is not None
            else []
        )
        claims.append(
            Claim(
                claim_id=f"{qid}_c1",
                text=f"Claim text for {qid}",
                citations=[Citation(passage_id="p1", char_start=0, char_end=10)],
                verifier_scores=v_scores,
                human_labels=h_labels,
            )
        )

    return QueryRecord(
        run_id="run_test",
        query_id=qid,
        question=f"Question for {qid}",
        system=System.JOINT,
        seed=20260804,
        claims=claims,
    )


def make_records_file(
    path: Path,
    scores: list[float] | None = None,
    labels: list[SupportLabel] | None = None,
    n: int = 10,
) -> Path:
    records = []
    for i in range(n):
        s = scores[i] if scores is not None and i < len(scores) else None
        lbl = labels[i] if labels is not None and i < len(labels) else None
        records.append(make_record(f"q{i}", score=s, label=lbl))
    write_jsonl(path, records)
    return path


def run_driver(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(_REPO / "scripts/g3_report.py"), *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_passing_case(tmp_path: Path) -> None:
    scores = [0.9] * 5 + [0.1] * 5
    labels = [SupportLabel.SUPPORTED] * 5 + [SupportLabel.NOT_SUPPORTED] * 5
    rec_path = make_records_file(tmp_path / "passing.records.jsonl", scores=scores, labels=labels)
    out_path = tmp_path / "passing.json"

    res = run_driver(["--records", str(rec_path), "--cost-ratio", "15.0", "--out", str(out_path)])
    assert res.returncode == 0
    assert "G3 PASSES: true" in res.stdout

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passes"] is True
    assert data["auroc_passes"] is True
    assert data["cost_passes"] is True
    assert data["auroc"] == 1.0
    assert data["cost_ratio"] == 15.0
    assert data["reason"] == "pass"


def test_auroc_below_threshold(tmp_path: Path) -> None:
    # Inverse scores -> AUROC = 0.0 < 0.75
    scores = [0.1] * 5 + [0.9] * 5
    labels = [SupportLabel.SUPPORTED] * 5 + [SupportLabel.NOT_SUPPORTED] * 5
    rec_path = make_records_file(tmp_path / "low_auroc.records.jsonl", scores=scores, labels=labels)
    out_path = tmp_path / "low_auroc.json"

    res = run_driver(["--records", str(rec_path), "--cost-ratio", "15.0", "--out", str(out_path)])
    assert res.returncode == 0
    assert "G3 PASSES: false" in res.stdout

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passes"] is False
    assert data["auroc_passes"] is False
    assert data["cost_passes"] is True
    assert "auroc_below_threshold" in data["reason"]


def test_no_annotations_input(tmp_path: Path) -> None:
    scores = [0.9] * 5 + [0.1] * 5
    rec_path = make_records_file(tmp_path / "no_ann.records.jsonl", scores=scores, labels=None)
    out_path = tmp_path / "no_ann.json"

    res = run_driver(["--records", str(rec_path), "--cost-ratio", "15.0", "--out", str(out_path)])
    assert res.returncode == 0
    assert "G3 PASSES: false" in res.stdout

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passes"] is False
    assert "missing_human_labels" in data["reason"]
    assert data["diagnostics"]["n_missing_annotations"] == 10


def test_missing_cost_ratio(tmp_path: Path) -> None:
    scores = [0.9] * 5 + [0.1] * 5
    labels = [SupportLabel.SUPPORTED] * 5 + [SupportLabel.NOT_SUPPORTED] * 5
    rec_path = make_records_file(tmp_path / "no_cost.records.jsonl", scores=scores, labels=labels)
    out_path = tmp_path / "no_cost.json"

    res = run_driver(["--records", str(rec_path), "--out", str(out_path)])
    assert res.returncode == 0
    assert "G3 PASSES: false" in res.stdout

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passes"] is False
    assert data["cost_passes"] is False
    assert "cost_ratio_missing" in data["reason"]


def test_provenance_and_threshold_keys(tmp_path: Path) -> None:
    scores = [0.9] * 5 + [0.1] * 5
    labels = [SupportLabel.SUPPORTED] * 5 + [SupportLabel.NOT_SUPPORTED] * 5
    rec_path = make_records_file(tmp_path / "prov.records.jsonl", scores=scores, labels=labels)
    out_path = tmp_path / "prov.json"

    run_driver(["--records", str(rec_path), "--cost-ratio", "12.5", "--out", str(out_path)])

    data = json.loads(out_path.read_text(encoding="utf-8"))
    required_keys = {
        "script",
        "finished_at",
        "records_source",
        "records_sha256",
        "annotations_source",
        "git_commit",
        "verifier",
        "thresholds",
        "diagnostics",
        "gate",
        "auroc",
        "auroc_min",
        "auroc_passes",
        "cost_ratio",
        "cost_ratio_min",
        "cost_passes",
        "passes",
        "reason",
    }
    for k in required_keys:
        assert k in data, f"Missing key {k!r} in emitted JSON"

    assert data["thresholds"]["auroc_min"] == 0.75
    assert data["thresholds"]["cost_ratio_min"] == 10.0

    diag_keys = {
        "n_records",
        "n_claims",
        "n_citations",
        "n_scored",
        "n_missing_scores",
        "n_missing_annotations",
        "n_no_majority",
        "no_majority_rate",
        "n_extra_citations",
    }
    for k in diag_keys:
        assert k in data["diagnostics"], f"Missing key {k!r} in diagnostics"


def test_determinism_identical_runs(tmp_path: Path) -> None:
    scores = [0.9] * 5 + [0.1] * 5
    labels = [SupportLabel.SUPPORTED] * 5 + [SupportLabel.NOT_SUPPORTED] * 5
    rec_path = make_records_file(tmp_path / "det.records.jsonl", scores=scores, labels=labels)
    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"

    run_driver(["--records", str(rec_path), "--cost-ratio", "15.0", "--out", str(out1)])
    run_driver(["--records", str(rec_path), "--cost-ratio", "15.0", "--out", str(out2)])

    d1 = json.loads(out1.read_text(encoding="utf-8"))
    d2 = json.loads(out2.read_text(encoding="utf-8"))

    d1.pop("finished_at")
    d2.pop("finished_at")

    assert d1 == d2


def test_annotations_flag_loading(tmp_path: Path) -> None:
    from biomedqa.annotate import ANNOTATION_SEED, build_tasks, LABEL_ROW
    scores = [0.9] * 5 + [0.1] * 5
    rec_path = make_records_file(tmp_path / "ann_flag.records.jsonl", scores=scores, labels=None)
    records = list(read_query_records(rec_path))
    tasks, keyfile = build_tasks(records, seed=ANNOTATION_SEED)

    key_path = tmp_path / "keyfile.jsonl"
    key_path.write_text("\n".join(json.dumps(r) for r in keyfile) + "\n", encoding="utf-8")

    ann_rows = []
    for meta in keyfile:
        qid = meta["query_id"]
        q_idx = int(qid[1:])
        lbl = "SUPPORTED" if q_idx < 5 else "NOT_SUPPORTED"
        ann_rows.append(
            {
                "type": LABEL_ROW,
                "annotator_id": "a1",
                "unit_id": meta["unit_id"],
                "citation_index": 0,
                "support_label": lbl,
                "claim_validity": True,
            }
        )
    ann_path = tmp_path / "annotations.jsonl"
    ann_path.write_text("\n".join(json.dumps(r) for r in ann_rows) + "\n", encoding="utf-8")

    out_path = tmp_path / "ann_flag.json"

    res = run_driver(
        [
            "--records",
            str(rec_path),
            "--annotations",
            str(ann_path),
            "--keyfile",
            str(key_path),
            "--cost-ratio",
            "15.0",
            "--out",
            str(out_path),
        ]
    )
    assert res.returncode == 0
    assert "G3 PASSES: true" in res.stdout

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passes"] is True
def test_reconciliation_identity(tmp_path: Path) -> None:
    # Test reconciliation identity: citations == scored + missing_scores + n_extra_citations
    c1 = Claim(
        claim_id="c1",
        text="Claim with 2 citations",
        citations=[Citation("p1", 0, 5), Citation("p2", 0, 5)],
        verifier_scores=[VerifierScore("minicheck", 0.9), VerifierScore("minicheck", 0.8)],
        human_labels=[
            HumanLabel("a1", SupportLabel.SUPPORTED, True, citation_index=0),
            HumanLabel("a1", SupportLabel.SUPPORTED, True, citation_index=1),
        ],
    )
    c2 = Claim(
        claim_id="c2",
        text="Claim with 1 citation",
        citations=[Citation("p3", 0, 5)],
        verifier_scores=[VerifierScore("minicheck", 0.3)],
        human_labels=[HumanLabel("a1", SupportLabel.NOT_SUPPORTED, True)],
    )
    rec = QueryRecord("r1", "q1", "Q?", System.JOINT, 0, claims=[c1, c2])
    rec_path = tmp_path / "rec.jsonl"
    write_jsonl(rec_path, [rec])
    out_path = tmp_path / "out.json"

    run_driver(["--records", str(rec_path), "--out", str(out_path)])
    data = json.loads(out_path.read_text(encoding="utf-8"))
    diag = data["diagnostics"]

    # Identity: citations (3) == scored (3) + missing_scores (0) + n_extra_citations (0)
    assert diag["n_citations"] == 3
    assert diag["n_scored"] == 3
    assert diag["n_missing_scores"] == 0
    assert diag["n_extra_citations"] == 0
    assert diag["n_citations"] == diag["n_scored"] + diag["n_missing_scores"] + diag["n_extra_citations"]
    assert data["diagnostics"]["n_missing_annotations"] == 0

def test_nan_serialization_emits_null_and_strict_json(tmp_path: Path) -> None:
    rec_path = make_records_file(tmp_path / "nan.records.jsonl", scores=None, labels=None)
    out_path = tmp_path / "nan_out.json"

    res = run_driver(["--records", str(rec_path), "--out", str(out_path)])
    assert res.returncode == 0

    raw_text = out_path.read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text

    def _reject_constant(val: str) -> float:
        raise ValueError(f"Constant {val!r} not allowed in strict JSON")

    data = json.loads(raw_text, parse_constant=_reject_constant)
    assert data["auroc"] is None
    assert data["gate"]["auroc"] is None
def test_end_to_end_post_annotation_rehearsal(tmp_path: Path) -> None:
    """End-to-end post-annotation G3 rehearsal against real verifier_scored records slice."""
    from biomedqa.annotate import ANNOTATION_SEED, LABEL_ROW, QUESTION_ROW, build_tasks

    real_records_path = _REPO / "docs/harvest/generate_fp05_n100_guided_v4.verifier_scored.records.jsonl"
    all_records = list(read_query_records(real_records_path))
    slice_records = [r for r in all_records if any(c.citations for c in r.claims)][:5]

    tasks, keyfile = build_tasks(slice_records, seed=ANNOTATION_SEED)

    key_score_map = {}
    for meta in keyfile:
        for r in slice_records:
            if r.query_id == meta["query_id"] and r.run_id == meta["run_id"] and r.seed == meta["seed"]:
                for c in r.claims:
                    if c.claim_id == meta["claim_id"]:
                        scores = [v.score for v in c.verifier_scores if v.name == "lytang/MiniCheck-Flan-T5-Large"]
                        key_score_map[meta["unit_id"]] = scores

    all_units = []
    for t in tasks:
        for c in t.claims:
            scores = key_score_map.get(c.unit_id, [])
            for s in c.spans:
                score_val = scores[s.citation_index] if s.citation_index < len(scores) else 0.5
                all_units.append((c.unit_id, s.citation_index, score_val))

    total_units = len(all_units)
    assert total_units >= 2

    ann1_rows = []
    ann2_rows = []
    ann3_rows = []

    for t in tasks:
        q_row = {
            "type": QUESTION_ROW,
            "annotator_id": "ann1",
            "question_uid": t.question_uid,
            "order_index": t.order_index,
            "completed_at": "2026-09-07T12:00:00Z",
            "notes": "SYNTHETIC REHEARSAL FIXTURE - NOT EVIDENCE",
        }
        ann1_rows.append(q_row)
        ann2_rows.append({**q_row, "annotator_id": "ann2"})
        ann3_rows.append({**q_row, "annotator_id": "ann3"})

    for idx, (u_id, cit_idx, score_val) in enumerate(all_units):
        lbl = "SUPPORTED" if score_val >= 0.10 else "NOT_SUPPORTED"
        note = "SYNTHETIC REHEARSAL FIXTURE - NOT EVIDENCE"
        if idx == 0:
            # 2-1 majority unit
            ann1_rows.append({"type": LABEL_ROW, "annotator_id": "ann1", "unit_id": u_id, "citation_index": cit_idx, "support_label": "SUPPORTED", "claim_validity": True, "notes": note})
            ann2_rows.append({"type": LABEL_ROW, "annotator_id": "ann2", "unit_id": u_id, "citation_index": cit_idx, "support_label": "SUPPORTED", "claim_validity": True, "notes": note})
            ann3_rows.append({"type": LABEL_ROW, "annotator_id": "ann3", "unit_id": u_id, "citation_index": cit_idx, "support_label": "NOT_SUPPORTED", "claim_validity": True, "notes": note})
        elif idx == 1:
            # Tie unit (ann1=SUPPORTED vs ann2=NOT_SUPPORTED, ann3 abstains)
            ann1_rows.append({"type": LABEL_ROW, "annotator_id": "ann1", "unit_id": u_id, "citation_index": cit_idx, "support_label": "SUPPORTED", "claim_validity": True, "notes": note})
            ann2_rows.append({"type": LABEL_ROW, "annotator_id": "ann2", "unit_id": u_id, "citation_index": cit_idx, "support_label": "NOT_SUPPORTED", "claim_validity": True, "notes": note})
        else:
            # 3-0 consensus units
            ann1_rows.append({"type": LABEL_ROW, "annotator_id": "ann1", "unit_id": u_id, "citation_index": cit_idx, "support_label": lbl, "claim_validity": True, "notes": note})
            ann2_rows.append({"type": LABEL_ROW, "annotator_id": "ann2", "unit_id": u_id, "citation_index": cit_idx, "support_label": lbl, "claim_validity": True, "notes": note})
            ann3_rows.append({"type": LABEL_ROW, "annotator_id": "ann3", "unit_id": u_id, "citation_index": cit_idx, "support_label": lbl, "claim_validity": True, "notes": note})

    rec_p = tmp_path / "synthetic_rehearsal.records.jsonl"
    key_p = tmp_path / "synthetic_rehearsal.keyfile.jsonl"
    ann1_p = tmp_path / "synthetic_rehearsal_ann1.jsonl"
    ann2_p = tmp_path / "synthetic_rehearsal_ann2.jsonl"
    ann3_p = tmp_path / "synthetic_rehearsal_ann3.jsonl"
    out_pass_p = tmp_path / "verdict_pass.json"
    out_fail_p = tmp_path / "verdict_fail.json"

    write_jsonl(rec_p, slice_records)
    key_p.write_text("\n".join(json.dumps(r) for r in keyfile) + "\n", encoding="utf-8")
    ann1_p.write_text("\n".join(json.dumps(r) for r in ann1_rows) + "\n", encoding="utf-8")
    ann2_p.write_text("\n".join(json.dumps(r) for r in ann2_rows) + "\n", encoding="utf-8")
    ann3_p.write_text("\n".join(json.dumps(r) for r in ann3_rows) + "\n", encoding="utf-8")

    # 1. PASS case: cost-ratio 15.0 (> 10x) and separable scores -> G3 PASSES: true
    res_pass = run_driver([
        "--records", str(rec_p),
        "--annotations", str(ann1_p), str(ann2_p), str(ann3_p),
        "--keyfile", str(key_p),
        "--primary-annotator", "ann1",
        "--cost-ratio", "15.0",
        "--out", str(out_pass_p),
    ])
    assert res_pass.returncode == 0
    assert "G3 PASSES: true" in res_pass.stdout
    data_pass = json.loads(out_pass_p.read_text(encoding="utf-8"))
    assert data_pass["passes"] is True
    assert data_pass["auroc_passes"] is True
    assert data_pass["cost_passes"] is True
    assert data_pass["diagnostics"]["n_missing_annotations"] == 0
    assert data_pass["diagnostics"]["n_no_majority"] == 1
    assert abs(data_pass["diagnostics"]["no_majority_rate"] - (1.0 / total_units)) < 1e-6

    # 2. FAIL case: cost-ratio 5.0 (< 10x) -> G3 PASSES: false (both-clause conjunction)
    res_fail = run_driver([
        "--records", str(rec_p),
        "--annotations", str(ann1_p), str(ann2_p), str(ann3_p),
        "--keyfile", str(key_p),
        "--primary-annotator", "ann1",
        "--cost-ratio", "5.0",
        "--out", str(out_fail_p),
    ])
    assert res_fail.returncode == 0
    assert "G3 PASSES: false" in res_fail.stdout
    data_fail = json.loads(out_fail_p.read_text(encoding="utf-8"))
    assert data_fail["passes"] is False
    assert data_fail["auroc_passes"] is True
    assert data_fail["cost_passes"] is False
    assert "cost_ratio_below_threshold" in data_fail["reason"]

    # 3. Loud rejection: label outside SupportLabel
    bad_lbl_p = tmp_path / "bad_label.jsonl"
    bad_lbl_p.write_text(json.dumps({
        "type": LABEL_ROW,
        "annotator_id": "ann1",
        "unit_id": keyfile[0]["unit_id"],
        "citation_index": 0,
        "support_label": "UNSUPPORTED_INVALID_LABEL",
        "claim_validity": True,
    }) + "\n", encoding="utf-8")
    res_bad_lbl = run_driver([
        "--records", str(rec_p),
        "--annotations", str(bad_lbl_p),
        "--keyfile", str(key_p),
        "--out", str(tmp_path / "bad_lbl.json"),
    ])
    assert res_bad_lbl.returncode == 1
    assert "Invalid SupportLabel" in res_bad_lbl.stderr or "UNSUPPORTED_INVALID_LABEL" in res_bad_lbl.stderr

    # 4. Loud rejection: unit_id absent from keyfile
    absent_unit_p = tmp_path / "absent_unit.jsonl"
    absent_unit_p.write_text(json.dumps({
        "type": LABEL_ROW,
        "annotator_id": "ann1",
        "unit_id": "u_synthetic_rehearsal_absent_unit_9999",
        "citation_index": 0,
        "support_label": "SUPPORTED",
        "claim_validity": True,
    }) + "\n", encoding="utf-8")
    res_absent = run_driver([
        "--records", str(rec_p),
        "--annotations", str(absent_unit_p),
        "--keyfile", str(key_p),
        "--out", str(tmp_path / "absent_unit.json"),
    ])
    assert res_absent.returncode == 1
    assert "Unrecognized unit_id" in res_absent.stderr or "u_synthetic_rehearsal_absent_unit_9999" in res_absent.stderr
