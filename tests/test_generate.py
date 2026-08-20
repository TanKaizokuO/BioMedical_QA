"""Stage orchestration for the three systems.

The failure this file exists to catch is the one that is invisible in the output: post-hoc's first
pass being told that citations are coming. That prompt is a *baseline*, and if it grounds jointly
then C2's headline gap shrinks for a reason that has nothing to do with the systems being compared.
No inspection of the generated text can reveal it after the fact, so it is asserted on the prompt.
"""

from __future__ import annotations

import json
import math

import httpx
import pytest

from biomedqa.config import GenerationConfig
from biomedqa.generate import MAX_CLAIMS_PER_CITE_CALL, STAGE_SEPARATOR, generate_one, split_stages
from biomedqa.schema import CostRecord, QueryRecord, RetrievedPassage, System

_TEXT = {
    "p1": "Metformin reduced all-cause mortality by 21% over five years in the treatment arm.",
    "p2": "No difference in cardiovascular events was observed between the two groups.",
    "p3": "The cohort comprised 4,102 adults with type 2 diabetes recruited across nine centres.",
}


def _passages(n: int = 3) -> list[RetrievedPassage]:
    return [
        RetrievedPassage(passage_id=pid, rank=i, score=1.0 / i, retriever="rerank", text=text)
        for i, (pid, text) in enumerate(_TEXT.items(), start=1)
    ][:n]


def _cited_response() -> str:
    return (
        "DECISION: yes\n"
        "CLAIM 1: Metformin reduced all-cause mortality in adults with type 2 diabetes.\n"
        "CITE 1: p1 || Metformin reduced all-cause mortality by 21%\n"
        "CLAIM 2: Metformin did not change cardiovascular event rates.\n"
        "CITE 2: p2 || No difference in cardiovascular events was observed\n"
    )


def _uncited_response() -> str:
    return (
        "DECISION: maybe\n"
        "CLAIM 1: Metformin reduced all-cause mortality in adults with type 2 diabetes.\n"
    )


class _Recorder:
    """A stand-in `backends.complete` that logs prompts and replays scripted responses."""

    def __init__(self, *responses: str, tokens: tuple[int | None, int | None] = (100, 20)):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.response_formats: list[dict | None] = []
        self.tokens = tokens

    def __call__(self, prompt, config, *, seed, run_id, query_id, response_format=None):
        self.prompts.append(prompt)
        self.response_formats.append(response_format)
        text = self.responses[len(self.prompts) - 1]
        return text, CostRecord(
            run_id=run_id,
            query_id=query_id,
            component="generate",
            backend="stub",
            input_tokens=self.tokens[0],
            output_tokens=self.tokens[1],
            wall_s=0.5,
        )
def _run(system: System, *responses: str, **kwargs):
    stub = _Recorder(*responses)
    config = kwargs.pop("config", GenerationConfig(model="stub"))
    gen = generate_one(
        "Does metformin reduce mortality?",
        _passages(),
        ["p1"],
        system=system,
        config=config,
        seed=0,
        run_id="run-1",
        query_id="21645374",
        complete=stub,
        **kwargs,
    )
    return gen, stub


class TestJoint:
    def test_one_call_and_a_clean_record(self):
        gen, stub = _run(System.JOINT, _cited_response())
        assert len(stub.prompts) == 1
        assert gen.record.validate() == []
        assert gen.errors == ()
        assert gen.record.final_decision == "yes"
        assert [c.claim_id for c in gen.record.claims] == ["c1", "c2"]

    def test_citations_are_located_as_exact_spans(self):
        gen, _ = _run(System.JOINT, _cited_response())
        cite = gen.record.claims[0].citations[0]
        assert cite.passage_id == "p1"
        assert _TEXT["p1"][cite.char_start : cite.char_end] == cite.quoted_text

    def test_raw_generation_is_the_verbatim_completion(self):
        gen, _ = _run(System.JOINT, _cited_response())
        assert gen.record.raw_generation == _cited_response()
        assert split_stages(gen.record.raw_generation) == (_cited_response(),)

        assert split_stages(gen.record.raw_generation) == (_cited_response(),)

    def test_joint_guided_path_emits_response_format(self):
        """When guided_decoding=True, joint arm passes response_format and uses joint_json stage."""
        joint_json = json.dumps({
            "decision": "yes",
            "claims": [
                {
                    "claim_index": 1,
                    "text": "Metformin reduced all-cause mortality in adults with type 2 diabetes.",
                    "citations": [{"passage_id": "p1", "quote": "Metformin reduced all-cause mortality by 21%"}],
                }
            ],
        })
        gen, stub = _run(
            System.JOINT,
            joint_json,
            config=GenerationConfig(model="stub", guided_decoding=True),
        )
        assert len(stub.prompts) == 1
        assert stub.response_formats[0] is not None
        assert stub.response_formats[0]["type"] == "json_schema"
        assert stub.response_formats[0]["json_schema"]["name"] == "joint_response"
        assert "Reply with a single JSON object" in stub.prompts[0]
        assert gen.record.final_decision == "yes"
        assert len(gen.record.claims) == 1
        assert gen.record.claims[0].text == "Metformin reduced all-cause mortality in adults with type 2 diabetes."
        assert len(gen.record.claims[0].citations) == 1

    def test_joint_unguided_path_is_unchanged(self):
        """When guided_decoding=False, joint arm passes no response_format and uses line grammar."""
        gen, stub = _run(
            System.JOINT,
            _cited_response(),
            config=GenerationConfig(model="stub", guided_decoding=False),
        )
        assert len(stub.prompts) == 1
        assert stub.response_formats[0] is None
        assert "Reply in exactly this format" in stub.prompts[0]
        assert gen.record.final_decision == "yes"
        assert len(gen.record.claims) == 2
class TestPostHoc:
    def test_the_answer_stage_is_never_told_citations_are_coming(self):
        """The whole baseline rests on this. See the module docstring."""
        _, stub = _run(System.POST_HOC, "Metformin reduces mortality.", _cited_response())
        answer_prompt, cite_prompt = stub.prompts
        for forbidden in ("CITE", "quote", "citation"):
            assert forbidden.lower() not in answer_prompt.lower(), (
                f"post-hoc's first pass leaked {forbidden!r}; it would be grounding jointly"
            )
        assert "CITE" in cite_prompt

    def test_the_cite_stage_receives_the_first_pass_answer(self):
        _, stub = _run(System.POST_HOC, "Metformin reduces mortality.", _cited_response())
        assert "Metformin reduces mortality." in stub.prompts[1]

    def test_both_stages_are_kept_and_recoverable(self):
        gen, _ = _run(System.POST_HOC, "Metformin reduces mortality.", _cited_response())
        assert split_stages(gen.record.raw_generation) == (
            "Metformin reduces mortality.",
            _cited_response(),
        )
        assert STAGE_SEPARATOR in gen.record.raw_generation

    def test_claims_come_from_the_cite_stage_not_the_answer_stage(self):
        gen, _ = _run(System.POST_HOC, "Metformin reduces mortality.", _cited_response())
        assert len(gen.record.claims) == 2
        assert gen.record.claims[0].citations

    def test_two_completions_are_billed_not_one(self):
        """Post-hoc costs two calls per query; Table 4 compares prices and must see both."""
        gen, _ = _run(System.POST_HOC, "Metformin reduces mortality.", _cited_response())
        assert len(gen.costs) == 2
        assert gen.record.prompt_tokens == 200
        assert gen.record.completion_tokens == 40
        assert gen.record.latency_s == pytest.approx(1.0)

    def test_guided_decoding_passes_response_format_and_uses_recite_json(self):
        """When guided_decoding=True, post-hoc stage 2 receives a JSON schema and recite_json template."""
        stage1_resp = "DECISION: yes\nCLAIM 1: Metformin reduced all-cause mortality.\n"
        stage2_json = (
            '{"claims": [{"claim_index": 1, "citations": [{"passage_id": "p1", '
            '"quote": "Metformin reduced all-cause mortality by 21%"}]}]}'
        )
        gen, stub = _run(
            System.POST_HOC,
            stage1_resp,
            stage2_json,
            config=GenerationConfig(model="stub", guided_decoding=True),
        )
        assert len(stub.prompts) == 2
        assert stub.response_formats[0] is None
        assert stub.response_formats[1] is not None
        assert stub.response_formats[1]["type"] == "json_schema"
        assert "Return one entry per claim" in stub.prompts[1]
        assert gen.record.claims[0].citations[0].passage_id == "p1"
        assert gen.record.claims[0].text == "Metformin reduced all-cause mortality."

    def test_guided_decoding_handles_malformed_json_without_crashing(self):
        """Malformed JSON from guided stage is reported in errors rather than raising an exception."""
        stage1_resp = "DECISION: yes\nCLAIM 1: Metformin reduced all-cause mortality.\n"
        stage2_json = "NOT VALID JSON"
        gen, stub = _run(
            System.POST_HOC,
            stage1_resp,
            stage2_json,
            config=GenerationConfig(model="stub", guided_decoding=True),
        )
        assert len(gen.errors) > 0
        assert any("malformed" in e or "no CLAIM" in e for e in gen.errors)

    def test_guided_decoding_batches_large_claim_sets(self):
        """When stage 1 produces > MAX_CLAIMS_PER_CITE_CALL claims, guided decoding splits stage 2 calls."""
        stage1_resp = "DECISION: yes\n" + "\n".join(
            f"CLAIM {i}: Metformin claim {i}." for i in range(1, 8)
        )
        batch0_json = json.dumps({
            "claims": [
                {"claim_index": i, "citations": [{"passage_id": "p1", "quote": "Metformin reduced all-cause mortality by 21%"}]}
                for i in range(1, 6)
            ]
        })
        batch1_json = json.dumps({
            "claims": [
                {"claim_index": i, "citations": [{"passage_id": "p2", "quote": "No difference in cardiovascular events was observed"}]}
                for i in range(1, 3)
            ]
        })

        gen, stub = _run(
            System.POST_HOC,
            stage1_resp,
            batch0_json,
            batch1_json,
            config=GenerationConfig(model="stub", guided_decoding=True),
        )

        expected_stage2_calls = math.ceil(7 / MAX_CLAIMS_PER_CITE_CALL)
        assert len(stub.prompts) == 1 + expected_stage2_calls
        assert len(gen.record.claims) == 7

        # Check schemas match batch size
        assert stub.response_formats[1]["json_schema"]["schema"]["properties"]["claims"]["minItems"] == 5
        assert stub.response_formats[2]["json_schema"]["schema"]["properties"]["claims"]["minItems"] == 2

        # Check claims order and text preserved
        for i, claim in enumerate(gen.record.claims, start=1):
            assert claim.claim_id == f"c{i}"
            assert claim.text == f"Metformin claim {i}."

        # Check citation grounding by batch (no cross-batch mixup)
        for c in gen.record.claims[:5]:
            assert len(c.citations) == 1
            assert c.citations[0].passage_id == "p1"

        for c in gen.record.claims[5:]:
            assert len(c.citations) == 1
            assert c.citations[0].passage_id == "p2"

    def test_guided_decoding_batching_handles_short_or_malformed_batch_reply(self):
        """A malformed/short reply in one batch reports errors for that batch's claims while sibling batch parses cleanly."""
        stage1_resp = "DECISION: yes\n" + "\n".join(
            f"CLAIM {i}: Metformin claim {i}." for i in range(1, 8)
        )
        batch0_json = json.dumps({
            "claims": [
                {"claim_index": i, "citations": [{"passage_id": "p1", "quote": "Metformin reduced all-cause mortality by 21%"}]}
                for i in range(1, 6)
            ]
        })
        # Batch 1 returns JSON with only 1 claim instead of 2
        batch1_short_json = json.dumps({
            "claims": [
                {"claim_index": 1, "citations": [{"passage_id": "p2", "quote": "No difference in cardiovascular events was observed"}]}
            ]
        })

        gen, stub = _run(
            System.POST_HOC,
            stage1_resp,
            batch0_json,
            batch1_short_json,
            config=GenerationConfig(model="stub", guided_decoding=True),
        )

        assert len(gen.record.claims) == 7
        # Batch 0 claims (1-5) got their citations cleanly
        for c in gen.record.claims[:5]:
            assert len(c.citations) == 1
            assert c.citations[0].passage_id == "p1"

        # Batch 1 claim 6 got citation, claim 7 missing from reply
        assert len(gen.record.claims[5].citations) == 1
        assert gen.record.claims[5].citations[0].passage_id == "p2"

        assert gen.record.claims[6].citations == []

        # Error reported for count mismatch and missing claim
        assert any("cite stage returned 1 CLAIM lines for 2 claims sent" in e for e in gen.errors)
        assert any("c7: no matching CLAIM line" in e for e in gen.errors)

class TestVanilla:
    def test_carries_no_citations_and_still_validates(self):
        gen, _ = _run(System.VANILLA, _uncited_response())
        assert gen.record.validate() == []
        assert all(not c.citations for c in gen.record.claims)

    def test_a_vanilla_completion_that_cites_is_reported_not_repaired(self):
        """Issue #3: vanilla carries no citations by definition, and `validate()` is what enforces
        it. `parse_response` is system-blind, so if `generate_one` did not run the check nothing on
        the generation path would notice the baseline quietly acquiring attribution."""
        gen, _ = _run(System.VANILLA, _cited_response())
        assert gen.errors == (), "the grammar parsed; this is a contract breach, not a parse error"
        assert any("vanilla" in v for v in gen.violations)
        assert any(c.citations for c in gen.record.claims), "reported, never silently stripped"

    def test_receives_the_same_passages_as_the_other_systems(self):
        """It isolates attribution, not retrieval (schema.py:73)."""
        _, stub = _run(System.VANILLA, _uncited_response())
        for pid in _TEXT:
            assert f"[{pid}]" in stub.prompts[0]


class TestContractCheck:
    def test_a_clean_record_reports_no_violations(self):
        gen, _ = _run(System.JOINT, _cited_response())
        assert gen.violations == ()

    def test_violations_match_the_record_s_own_verdict(self):
        """`violations` is `validate()`, not a second opinion that can drift from it."""
        gen, _ = _run(System.VANILLA, _cited_response())
        assert gen.violations == tuple(gen.record.validate())


class TestFailureIsData:
    def test_an_unparseable_response_still_produces_a_record(self):
        """G2 gates on the valid-parse rate; a raise would keep the failure out of the denominator."""
        gen, _ = _run(System.JOINT, "I cannot answer this question.")
        assert gen.record.claims == []
        assert gen.errors, "a response with no DECISION line must be reported, not silently empty"
        assert gen.record.raw_generation == "I cannot answer this question."

    def test_errors_are_derivable_from_the_record_rather_than_stored(self):
        from biomedqa.prompts import parse_response

        gen, _ = _run(System.JOINT, "I cannot answer this question.")
        again = parse_response(gen.record.raw_generation, gen.record.retrieved, 3)
        assert tuple(again.errors) == gen.errors


class TestContext:
    def test_the_record_stores_exactly_the_passages_the_prompt_listed(self):
        gen, stub = _run(System.JOINT, _cited_response(), depth=2)
        assert [p.passage_id for p in gen.record.retrieved] == ["p1", "p2"]
        assert "[p3]" not in stub.prompts[0]
        assert gen.record.validate() == [], "ranks must stay contiguous after slicing"

    def test_no_passages_is_a_refusal_not_an_ungrounded_answer(self):
        with pytest.raises(ValueError, match="retrieval must run first"):
            generate_one(
                "q",
                [],
                ["p1"],
                system=System.JOINT,
                config=GenerationConfig(model="stub"),
                seed=0,
                run_id="run-1",
                query_id="1",
                complete=_Recorder(_cited_response()),
            )

    def test_a_stage_with_no_usage_reported_leaves_the_total_missing(self):
        """A partial sum would read as a cheap query in Table 4 rather than as broken instrumentation."""
        stub = _Recorder("Metformin reduces mortality.", _cited_response(), tokens=(None, None))
        gen = generate_one(
            "q",
            _passages(),
            ["p1"],
            system=System.POST_HOC,
            config=GenerationConfig(model="stub"),
            seed=0,
            run_id="run-1",
            query_id="1",
            complete=stub,
        )
        assert gen.record.prompt_tokens is None
        assert gen.record.completion_tokens is None


class TestRecovered:
    def test_generate_one_propagates_recovered_notes(self):
        """Recovered notes (e.g. passage id missing its chunk index) must reach Generation."""
        p = RetrievedPassage(
            passage_id="p1:0", rank=1, score=1.0, retriever="rerank", text=_TEXT["p1"]
        )
        response = (
            "DECISION: yes\n"
            "CLAIM 1: Metformin reduced all-cause mortality in adults with type 2 diabetes.\n"
            "CITE 1: p1 || Metformin reduced all-cause mortality by 21%\n"
        )
        stub = _Recorder(response)
        gen = generate_one(
            "Does metformin reduce mortality?",
            [p],
            ["p1:0"],
            system=System.JOINT,
            config=GenerationConfig(model="stub"),
            seed=0,
            run_id="run-1",
            query_id="21645374",
            complete=stub,
        )
        assert gen.recovered
        assert any("cites 'p1', read as 'p1:0'" in r for r in gen.recovered)


class TestCallFailures:
    def test_first_call_http_status_error_yields_generation(self):
        req = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
        resp_text = (
            "This model's maximum context length is 8192 tokens. However, you requested "
            "3072 output tokens and your prompt contains at least 5121 input tokens, "
            "for a total of at least 8193 tokens."
        )
        resp = httpx.Response(400, request=req, text=resp_text)
        exc = httpx.HTTPStatusError(
            f"vLLM returned 400 for /v1/chat/completions: {resp_text}",
            request=req,
            response=resp,
        )
        def failing_completer(prompt, config, *, seed, run_id, query_id):
            raise exc

        gen = generate_one(
            "q",
            _passages(),
            ["p1"],
            system=System.JOINT,
            config=GenerationConfig(model="stub"),
            seed=0,
            run_id="run-1",
            query_id="21074975",
            complete=failing_completer,
        )
        assert isinstance(gen.record, QueryRecord)
        assert gen.record.claims == []
        assert len(gen.errors) > 0
        assert any("call 1 rejected:" in e for e in gen.errors)
        assert any("maximum context length is 8192" in e for e in gen.errors)
        assert gen.record.prompt_tokens is None
        assert gen.record.completion_tokens is None
        assert isinstance(gen.record.latency_s, float)

    def test_post_hoc_second_call_http_status_error_yields_two_stage_record(self):
        req = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
        resp = httpx.Response(400, request=req, text="Context length exceeded")
        exc = httpx.HTTPStatusError("400 Bad Request", request=req, response=resp)

        call_count = 0

        def second_call_failing(prompt, config, *, seed, run_id, query_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Metformin reduces mortality.", CostRecord(
                    run_id=run_id,
                    query_id=query_id,
                    component="generate",
                    backend="stub",
                    input_tokens=100,
                    output_tokens=20,
                    wall_s=0.5,
                )
            raise exc

        gen = generate_one(
            "q",
            _passages(),
            ["p1"],
            system=System.POST_HOC,
            config=GenerationConfig(model="stub"),
            seed=0,
            run_id="run-1",
            query_id="21074975",
            complete=second_call_failing,
        )
        assert len(split_stages(gen.record.raw_generation)) == 2
        assert split_stages(gen.record.raw_generation)[0] == "Metformin reduces mortality."
        assert split_stages(gen.record.raw_generation)[1] == ""
        assert any("call 2 rejected:" in e for e in gen.errors)
        assert gen.record.prompt_tokens is None
        assert gen.record.completion_tokens is None
        assert isinstance(gen.record.latency_s, float)

    def test_transport_error_yields_generation(self):
        req = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
        exc = httpx.ConnectError("Connection refused", request=req)

        def failing_completer(prompt, config, *, seed, run_id, query_id):
            raise exc

        gen = generate_one(
            "q",
            _passages(),
            ["p1"],
            system=System.JOINT,
            config=GenerationConfig(model="stub"),
            seed=0,
            run_id="run-1",
            query_id="1",
            complete=failing_completer,
        )
        assert any("call 1 rejected: Connection refused" in e for e in gen.errors)
        assert gen.record.prompt_tokens is None
        assert gen.record.completion_tokens is None
        assert isinstance(gen.record.latency_s, float)

    def test_value_error_propagates(self):
        def failing_completer(prompt, config, *, seed, run_id, query_id):
            raise ValueError("programming error")

        with pytest.raises(ValueError, match="programming error"):
            generate_one(
                "q",
                _passages(),
                ["p1"],
                system=System.JOINT,
                config=GenerationConfig(model="stub"),
                seed=0,
                run_id="run-1",
                query_id="1",
                complete=failing_completer,
            )


class TestStageCountRule:
    def test_stage_count_rule_passes_on_batched_counts_and_fails_on_fault(self):
        def evaluate(per_system: dict) -> bool:
            joint_ok = per_system[System.JOINT.value]["stages_seen"] == [1]
            vanilla_ok = per_system[System.VANILLA.value]["stages_seen"] == [1]
            ph_seen = per_system[System.POST_HOC.value]["stages_seen"]
            ph_ok = bool(ph_seen) and all(s >= 2 for s in ph_seen)
            return joint_ok and vanilla_ok and ph_ok

        batched_valid = {
            "joint": {"stages_seen": [1]},
            "vanilla": {"stages_seen": [1]},
            "post_hoc": {"stages_seen": [2, 3, 4]},
        }
        assert evaluate(batched_valid) is True

        post_hoc_collapsed_fault = {
            "joint": {"stages_seen": [1]},
            "vanilla": {"stages_seen": [1]},
            "post_hoc": {"stages_seen": [1]},
        }
        assert evaluate(post_hoc_collapsed_fault) is False

        vanilla_two_calls_fault = {
            "joint": {"stages_seen": [1]},
            "vanilla": {"stages_seen": [2]},
            "post_hoc": {"stages_seen": [2]},
        }
        assert evaluate(vanilla_two_calls_fault) is False

        joint_two_calls_fault = {
            "joint": {"stages_seen": [2]},
            "vanilla": {"stages_seen": [1]},
            "post_hoc": {"stages_seen": [2]},
        }
        assert evaluate(joint_two_calls_fault) is False
