"""The schema is frozen only if the round-trip is tested. Otherwise it is a convention.

These tests exist to fail loudly on the two mistakes that are irrecoverable rather than merely
annoying: losing a field on write, and binarizing a value that scoring needs continuous.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from biomedqa.config import GenerationConfig
from biomedqa.generate import generate_one, split_stages
from biomedqa.schema import (
    MAX_CITATIONS,
    Citation,
    Claim,
    CostRecord,
    Granularity,
    HumanLabel,
    QueryRecord,
    RetrievedPassage,
    SupportLabel,
    System,
    VerifierScore,
    query_record_from_dict,
    read_query_records,
    to_dict,
    write_jsonl,
)


def make_record() -> QueryRecord:
    """A record exercising every nested type, including the ones only annotation populates."""
    return QueryRecord(
        run_id="run-abc123",
        query_id="21645374",
        question="Does metformin reduce all-cause mortality in type 2 diabetes?",
        system=System.JOINT,
        seed=0,
        retrieved=[
            RetrievedPassage("21645374:0", 1, 12.5, "rerank", text="Metformin ..."),
            RetrievedPassage("18952324:2", 2, 9.1, "rerank"),
        ],
        gold_passage_ids=["21645374:0", "21645374:1"],
        claims=[
            Claim(
                claim_id="c0",
                text="Metformin reduces all-cause mortality in patients with type 2 diabetes.",
                citations=[Citation("21645374:0", 0, 9, quoted_text="Metformin")],
                granularity=Granularity.DECONTEXTUALIZED_ATOMIC,
                verifier_scores=[VerifierScore("minicheck", 0.83, latency_s=0.04)],
                human_labels=[HumanLabel("ann1", SupportLabel.PARTIAL, True, citation_index=0)],
                source_start=0,
                source_end=70,
            )
        ],
        raw_generation="Metformin reduces all-cause mortality ... [1]",
        final_decision="yes",
        gold_final_decision="yes",
        latency_s=1.23,
        prompt_tokens=812,
        completion_tokens=96,
    )


def test_roundtrip_is_lossless():
    original = make_record()
    restored = query_record_from_dict(to_dict(original))
    assert restored == original, "a field was lost or mutated crossing the serialisation boundary"


def test_roundtrip_survives_jsonl(tmp_path):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [make_record()])
    (restored,) = list(read_query_records(path))
    assert restored == make_record()


def test_enums_serialise_as_their_values():
    d = to_dict(make_record())
    assert d["system"] == "joint"
    assert d["claims"][0]["granularity"] == "decontextualized_atomic"
    assert d["claims"][0]["human_labels"][0]["support_label"] == "PARTIAL"
    json.dumps(d)  # must be plain-JSON serialisable, with no custom encoder


def test_unknown_field_raises_rather_than_being_dropped():
    d = to_dict(make_record())
    d["hit_at_5"] = True  # precisely the kind of precomputed value that must never appear
    with pytest.raises(TypeError):
        query_record_from_dict(d)


def test_verifier_score_stays_continuous():
    """A stored boolean fixes one operating point and discards the AUROC sweep irrecoverably."""
    score = to_dict(make_record())["claims"][0]["verifier_scores"][0]["score"]
    assert isinstance(score, float) and not isinstance(score, bool)
    assert 0.0 < score < 1.0


def test_support_label_is_four_way_on_disk():
    """The binary collapse is derived, never stored."""
    label = to_dict(make_record())["claims"][0]["human_labels"][0]["support_label"]
    assert label in {e.value for e in SupportLabel}
    assert SupportLabel(label).is_supporting is True  # PARTIAL collapses to supporting


def test_citation_span_is_validated_on_construction():
    with pytest.raises(ValueError):
        Citation("21645374:0", 10, 4)


class TestValidate:
    """`validate()` reports contract violations and never repairs them — the violation is a
    measurement, and G0 ranks candidate generators on exactly these."""

    def test_clean_record_has_no_problems(self):
        assert make_record().validate() == []

    def test_cap_violation_is_reported_but_citations_are_kept(self):
        record = make_record()
        record.claims[0].citations = [
            Citation("21645374:0", i, i + 1) for i in range(MAX_CITATIONS + 1)
        ]
        problems = record.validate()
        assert any("exceeds the cap" in p for p in problems)
        assert len(record.claims[0].citations) == MAX_CITATIONS + 1, "citations were silently dropped"

    def test_citation_outside_the_retrieved_set_is_reported(self):
        record = make_record()
        record.claims[0].citations = [Citation("99999999:0", 0, 5)]
        assert any("not in the retrieved set" in p for p in record.validate())

    def test_vanilla_baseline_may_not_carry_citations(self):
        record = make_record()
        record.system = System.VANILLA
        assert any("vanilla" in p for p in record.validate())

    def test_non_contiguous_ranks_are_reported(self):
        record = make_record()
        record.retrieved[1].rank = 5
        assert any("1-indexed contiguous" in p for p in record.validate())


def test_cost_record_carries_table_4_columns():
    """Table 4's caption was written in W0; these are the columns it promises."""
    cost = to_dict(CostRecord("run-abc123", "21645374", "generate", "vllm:model",
                              input_tokens=812, output_tokens=96, usd=0.0, wall_s=1.23))
    for column in ("input_tokens", "output_tokens", "usd", "wall_s"):
        assert column in cost


# ---------------------------------------------------------------------------------------------
# End to end: the round-trip has to hold for records the generator actually produces, on the text
# PubMed actually contains, and it has to hold *byte for byte* — equality of dataclasses would
# still pass if every run rewrote the file with keys in a new order.
# ---------------------------------------------------------------------------------------------


class TestByteStability:
    def test_write_read_write_is_byte_identical(self, tmp_path):
        """Reading a records file and writing it back must not change a byte. That is what makes
        two runs diffable: float repr drift, a change in escaping, or a lost field all show up
        here as a whole-file diff instead of as a number that quietly moved.

        It does *not* pin `sort_keys=True` — `asdict` already emits fields in declaration order,
        so dropping it changes nothing today. The flag is insurance against a future `to_dict`
        that builds its dict some other way, and nothing here can prove it.
        """
        first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        write_jsonl(first, [make_record()])
        write_jsonl(second, list(read_query_records(first)))
        assert first.read_bytes() == second.read_bytes()


class TestHostilePubMedText:
    """PubMed abstracts carry line separators that are not `\\n`. The corpus is 2M of them, so
    "rare" means "certain"."""

    SEPARATORS = "a\u2028b\u2029c\x0bd\u0085e"

    def test_unicode_line_separators_do_not_split_a_record(self, tmp_path):
        """`json.dumps(ensure_ascii=False)` leaves U+2028, U+2029 and U+0085 raw in the line —
        only C0 controls get escaped. File iteration splits on `\\n`/`\\r` alone and survives them;
        `str.splitlines()` shatters this one physical line into several. That is why `read_jsonl`
        iterates the handle and must never be "simplified" to `splitlines()`."""
        record = make_record()
        record.raw_generation = self.SEPARATORS
        record.retrieved[0].text = self.SEPARATORS
        path = tmp_path / "records.jsonl"
        write_jsonl(path, [record])

        text = path.read_text(encoding="utf-8")
        assert text.count("\n") == 1, "one record, one physical line"
        assert len(text.splitlines()) > 1, "the hazard is real — splitlines() would over-split"
        restored = list(read_query_records(path))
        assert len(restored) == 1
        assert restored[0].raw_generation == self.SEPARATORS
        assert restored[0].retrieved[0].text == self.SEPARATORS

    def test_non_ascii_is_written_verbatim_rather_than_escaped(self, tmp_path):
        """`ensure_ascii=False` keeps a Greek letter readable in a diff and a grep."""
        record = make_record()
        record.question = "Does α-tocopherol supplementation reduce mortality (≥65 years)?"
        path = tmp_path / "records.jsonl"
        write_jsonl(path, [record])
        assert "α-tocopherol" in path.read_text(encoding="utf-8")
        assert list(read_query_records(path))[0].question == record.question


def test_schema_version_is_preserved_not_restamped():
    """A record written under an older schema must read back saying so. Re-stamping it with the
    current version would make a file claim a shape it does not have."""
    d = to_dict(make_record())
    d["schema_version"] = "0.9.0-historical"
    assert query_record_from_dict(d).schema_version == "0.9.0-historical"


def test_every_query_record_field_reaches_disk():
    """A field added to the dataclass and forgotten in the serialiser is silent data loss. This
    fails on the commit that adds it, not in October when the column is empty."""
    on_disk = set(to_dict(make_record()))
    declared = {f.name for f in dataclasses.fields(QueryRecord)}
    assert declared == on_disk


class TestGeneratedRecords:
    """The round-trip is only proven end to end if it holds for what `generate.py` emits."""

    @staticmethod
    def _completion(text: str):
        def complete(prompt, config, *, seed, run_id, query_id):
            return text, CostRecord(run_id, query_id, "generate", "stub", 100, 20, wall_s=0.1)

        return complete

    @staticmethod
    def _passages():
        return [
            RetrievedPassage("21645374:0", 1, 12.5, "rerank", text="Metformin reduced mortality."),
            RetrievedPassage("18952324:2", 2, 9.1, "rerank", text="No difference was observed."),
        ]

    _CITED = (
        "DECISION: yes\n"
        "CLAIM 1: Metformin reduced mortality.\n"
        "CITE 1: 21645374:0 || Metformin reduced mortality.\n"
    )

    def test_a_generated_record_survives_the_file(self, tmp_path):
        gen = generate_one(
            "Does metformin reduce mortality?",
            self._passages(),
            ["21645374:0"],
            system=System.JOINT,
            config=GenerationConfig(),
            seed=0,
            run_id="run-e2e",
            query_id="21645374",
            complete=self._completion(self._CITED),
        )
        assert gen.errors == (), gen.errors
        path = tmp_path / "records.jsonl"
        write_jsonl(path, [gen.record])
        assert list(read_query_records(path)) == [gen.record]

    def test_post_hoc_stage_boundary_survives_the_file(self, tmp_path):
        """`raw_generation` holds both stages joined by `STAGE_SEPARATOR`. If the separator did not
        survive serialisation, `split_stages` would silently return one stage and a
        decomposition-error post-mortem would be reading the cite pass as the whole answer."""
        gen = generate_one(
            "Does metformin reduce mortality?",
            self._passages(),
            ["21645374:0"],
            system=System.POST_HOC,
            config=GenerationConfig(),
            seed=0,
            run_id="run-e2e",
            query_id="21645374",
            complete=self._completion(self._CITED),
        )
        path = tmp_path / "records.jsonl"
        write_jsonl(path, [gen.record])
        restored = list(read_query_records(path))[0]
        assert len(split_stages(restored.raw_generation)) == 2
        assert split_stages(restored.raw_generation) == split_stages(gen.record.raw_generation)
