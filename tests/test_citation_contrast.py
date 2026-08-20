"""Tests for scripts/citation_contrast.py (Slice A — Goal 6)."""

import random
from pathlib import Path

import pytest
from biomedqa.schema import (
    Citation,
    Claim,
    CostRecord,
    QueryRecord,
    RetrievedPassage,
    System,
    write_jsonl,
)
from citation_contrast import (
    collect_phi_pairs,
    compute_citation_contrast,
    load_run,
    pair_queries,
)


def make_record(
    query_id: str,
    system: System,
    claims: list[Claim],
    raw_generation: str = "Metformin treats type 2 diabetes.",
    passage_text: str = "Metformin treats type 2 diabetes.",
) -> QueryRecord:
    retrieved = [
        RetrievedPassage(
            passage_id="p1",
            rank=1,
            score=1.0,
            retriever="dense",
            text=passage_text,
        )
    ]
    return QueryRecord(
        run_id="test_run",
        query_id=query_id,
        question="What does metformin treat?",
        system=system,
        seed=0,
        retrieved=retrieved,
        claims=claims,
        raw_generation=raw_generation,
    )


def test_hand_built_fixture_delta():
    # Query 1 ("q1"):
    # Joint: 1 claim, cited p1 span (0..33).
    # Post_hoc: 1 claim, uncited.
    c_joint_1 = Claim(
        claim_id="c1_j",
        text="Metformin treats diabetes.",
        citations=[Citation(passage_id="p1", char_start=0, char_end=33)],
    )
    c_ph_1 = Claim(
        claim_id="c1_ph",
        text="Metformin treats diabetes.",
        citations=[],
    )

    # Query 2 ("q2"):
    # Joint: 1 claim, cited p1 span (0..33).
    # Post_hoc: 1 claim, cited p1 span (0..33).
    c_joint_2 = Claim(
        claim_id="c2_j",
        text="Metformin treats diabetes.",
        citations=[Citation(passage_id="p1", char_start=0, char_end=33)],
    )
    c_ph_2 = Claim(
        claim_id="c2_ph",
        text="Metformin treats diabetes.",
        citations=[Citation(passage_id="p1", char_start=0, char_end=33)],
    )

    j1 = make_record("q1", System.JOINT, [c_joint_1])
    p1 = make_record("q1", System.POST_HOC, [c_ph_1])
    j2 = make_record("q2", System.JOINT, [c_joint_2])
    p2 = make_record("q2", System.POST_HOC, [c_ph_2])

    paired, dropped = pair_queries([j1, p1, j2, p2])
    assert len(paired) == 2
    assert len(dropped) == 0

    # Stub phi returns True for premise == passage and hypothesis == claim text
    def stub_phi(premise: str, hypothesis: str) -> bool:
        return (
            premise == "Metformin treats type 2 diabetes."
            and hypothesis == "Metformin treats diabetes."
        )

    res = compute_citation_contrast(paired, stub_phi, seed=0, n_boot=1000)

    # Joint arm: 2 answered claims, both recalled (2/2 = 1.0), 2 citations, both relevant (2/2 = 1.0)
    # Joint F1 = 1.0
    assert abs(res["joint"]["f1"] - 1.0) < 1e-6
    assert abs(res["joint"]["precision"] - 1.0) < 1e-6
    assert abs(res["joint"]["recall"] - 1.0) < 1e-6

    # Post-hoc arm: 2 answered claims, 1 recalled (1/2 = 0.5), 1 citation (1/1 = 1.0)
    # Post-hoc F1 = 2 * 1.0 * 0.5 / 1.5 = 2/3
    assert abs(res["post_hoc"]["precision"] - 1.0) < 1e-6
    assert abs(res["post_hoc"]["recall"] - 0.5) < 1e-6
    assert abs(res["post_hoc"]["f1"] - (2.0 / 3.0)) < 1e-6

    # Paired delta = Joint F1 - Post_hoc F1 = 1.0 - 2/3 = 1/3
    expected_delta = 1.0 - (2.0 / 3.0)
    assert abs(res["delta"]["point"] - expected_delta) < 1e-6
    assert res["delta"]["n_boot"] == 1000
    assert res["delta"]["seed"] == 0


def test_pairing_shuffled_input_identical_result():
    # Build 4 queries with varied claims/citations
    records = []
    for i in range(1, 5):
        qid = f"q{i}"
        c_j = Claim(
            claim_id=f"cj_{i}",
            text=f"Claim text {i}",
            citations=(
                [Citation(passage_id="p1", char_start=0, char_end=33)]
                if i % 2 == 1
                else []
            ),
        )
        c_p = Claim(
            claim_id=f"cp_{i}",
            text=f"Claim text {i}",
            citations=[Citation(passage_id="p1", char_start=0, char_end=33)],
        )
        records.append(make_record(qid, System.JOINT, [c_j]))
        records.append(make_record(qid, System.POST_HOC, [c_p]))

    paired1, dropped1 = pair_queries(records)

    def stub_phi(premise: str, hypothesis: str) -> bool:
        return True

    res1 = compute_citation_contrast(paired1, stub_phi, seed=0, n_boot=500)

    # Shuffle the input records
    rng = random.Random(12345)
    records_shuffled = records.copy()
    rng.shuffle(records_shuffled)

    paired2, dropped2 = pair_queries(records_shuffled)
    res2 = compute_citation_contrast(paired2, stub_phi, seed=0, n_boot=500)

    # Check identical results
    assert paired1 == paired2
    assert dropped1 == dropped2
    assert res1["delta"]["point"] == res2["delta"]["point"]
    assert res1["delta"]["lower"] == res2["delta"]["lower"]
    assert res1["delta"]["upper"] == res2["delta"]["upper"]
    assert res1["delta"]["width"] == res2["delta"]["width"]


def test_unpaired_queries_dropped_and_counted():
    c_valid = Claim(
        claim_id="c_valid",
        text="Valid claim",
        citations=[Citation(passage_id="p1", char_start=0, char_end=5)],
    )

    # q1: complete pair
    j1 = make_record("q1", System.JOINT, [c_valid])
    p1 = make_record("q1", System.POST_HOC, [c_valid])

    # q2: missing post_hoc arm
    j2 = make_record("q2", System.JOINT, [c_valid])

    # q3: call rejected in joint arm (raw_generation="" and 0 claims)
    j3 = make_record("q3", System.JOINT, [], raw_generation="")
    p3 = make_record("q3", System.POST_HOC, [c_valid])

    # q4: zero claims in post_hoc arm (raw_generation!="" and 0 claims)
    j4 = make_record("q4", System.JOINT, [c_valid])
    p4 = make_record("q4", System.POST_HOC, [], raw_generation="Generated response with no claims")

    records = [j1, p1, j2, j3, p3, j4, p4]
    paired, dropped = pair_queries(records)

    assert len(paired) == 1
    assert paired[0][0].query_id == "q1"

    assert len(dropped) == 3
    dropped_by_qid = {d["query_id"]: d["reason"] for d in dropped}
    assert dropped_by_qid["q2"] == "missing post_hoc arm record"
    assert dropped_by_qid["q3"] == "call rejected in joint arm"
    assert dropped_by_qid["q4"] == "zero claims in post_hoc arm"


def test_null_output_tokens_cost_row_does_not_raise(tmp_path: Path):
    null_cost = CostRecord(
        run_id="run1",
        query_id="q1",
        component="generate",
        backend="vllm:model",
        input_tokens=None,
        output_tokens=None,
        usd=None,
        wall_s=0.5,
    )

    c_valid = Claim(
        claim_id="c_valid",
        text="Valid claim",
        citations=[Citation(passage_id="p1", char_start=0, char_end=5)],
    )
    j1 = make_record("q1", System.JOINT, [c_valid])
    p1 = make_record("q1", System.POST_HOC, [c_valid])

    # Ensure pair_queries with null token cost record works
    paired, dropped = pair_queries([j1, p1], costs=[null_cost])
    assert len(paired) == 1

    # Ensure load_run on written files with null tokens works
    rec_file = tmp_path / "test_run.records.jsonl"
    cost_file = tmp_path / "test_run.costs.jsonl"
    sum_file = tmp_path / "test_run.summary.json"

    write_jsonl(rec_file, [j1, p1])
    write_jsonl(cost_file, [null_cost])
    sum_file.write_text("{}", encoding="utf-8")

    loaded_recs, loaded_costs, loaded_sum = load_run(tmp_path / "test_run")
    assert len(loaded_recs) == 2
    assert len(loaded_costs) == 1
    assert loaded_costs[0].output_tokens is None
    assert loaded_costs[0].input_tokens is None

    paired_loaded, dropped_loaded = pair_queries(loaded_recs, loaded_costs, loaded_sum)
    assert len(paired_loaded) == 1
    assert len(dropped_loaded) == 0


def test_collect_phi_pairs():
    c1 = Claim(
        claim_id="c1",
        text="Metformin treats diabetes.",
        citations=[
            Citation(passage_id="p1", char_start=0, char_end=26),
        ],
    )
    rec = make_record("q1", System.JOINT, [c1], passage_text="Metformin treats diabetes.")
    pairs = collect_phi_pairs([rec])
    assert len(pairs) > 0
    assert ("Metformin treats diabetes.", "Metformin treats diabetes.") in pairs
