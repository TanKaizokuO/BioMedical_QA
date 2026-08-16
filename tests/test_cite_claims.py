"""C7's citation re-run — `generate.cite_claims` attaches fresh citations to a re-cut answer.

`decompose.py` deliberately does not do this (its own docstring says so): a re-cut claim's span
points at its source sentence, not at a quotable region of the new claim text, so citations belong
to a second model call. This is Option A from `HANDOFF.md` — re-run the cite stage for real
citation-F1, rather than mapping the old citations onto claim boundaries that no longer exist.
"""

from __future__ import annotations

import pytest

from biomedqa.config import GenerationConfig
from biomedqa.generate import cite_claims
from biomedqa.schema import Claim, CostRecord, Granularity, RetrievedPassage

_TEXT = {
    "p1": "Metformin reduced all-cause mortality by 21% over five years in the treatment arm.",
    "p2": "No difference in cardiovascular events was observed between the two groups.",
}


def _passages() -> list[RetrievedPassage]:
    return [
        RetrievedPassage(passage_id=pid, rank=i, score=1.0 / i, retriever="rerank", text=text)
        for i, (pid, text) in enumerate(_TEXT.items(), start=1)
    ]


def _claims() -> list[Claim]:
    return [
        Claim(claim_id="c1", text="Metformin reduces mortality.",
              granularity=Granularity.ATOMIC, source_start=0, source_end=29),
        Claim(claim_id="c2", text="It does not change cardiovascular events.",
              granularity=Granularity.ATOMIC, source_start=30, source_end=73),
    ]


class _Recorder:
    """A stand-in `backends.complete` that logs prompts and replays one scripted response."""

    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt, config, *, seed, run_id, query_id):
        self.prompts.append(prompt)
        return self.response, CostRecord(
            run_id=run_id, query_id=query_id, component="generate", backend="stub",
            input_tokens=100, output_tokens=20, wall_s=0.5,
        )


class _Scripted(_Recorder):
    """`_Recorder` for a batched run: one scripted reply per call, in call order."""

    def __init__(self, responses: list[str]):
        super().__init__("")
        self.responses = list(responses)

    def __call__(self, prompt, config, *, seed=0, run_id="", query_id=None):
        self.response = self.responses[len(self.prompts)]
        return super().__call__(prompt, config, seed=seed, run_id=run_id, query_id=query_id)


def _cited_response() -> str:
    return (
        "DECISION: yes\n"
        "CLAIM 1: Metformin reduces mortality.\n"
        "CITE: [p1] || Metformin reduced all-cause mortality by 21%\n"
        "CLAIM 2: It does not change cardiovascular events.\n"
        "CITE: [p2] || No difference in cardiovascular events was observed\n"
    )


class TestHappyPath:
    def test_citations_attach_positionally_and_claim_identity_survives(self):
        stub = _Recorder(_cited_response())
        result = cite_claims(
            _claims(), "Does metformin reduce mortality?", _passages(),
            GenerationConfig(model="stub"), complete=stub, run_id="run-1", query_id="q1",
        )
        assert result.errors == ()
        assert [c.claim_id for c in result.claims] == ["c1", "c2"]
        assert [c.granularity for c in result.claims] == [Granularity.ATOMIC, Granularity.ATOMIC]
        assert [(c.source_start, c.source_end) for c in result.claims] == [(0, 29), (30, 73)]
        assert [len(c.citations) for c in result.claims] == [1, 1]
        assert result.claims[0].citations[0].passage_id == "p1"

    def test_the_cite_stage_prompt_reproduces_claims_in_order_untouched(self):
        stub = _Recorder(_cited_response())
        cite_claims(
            _claims(), "Does metformin reduce mortality?", _passages(),
            GenerationConfig(model="stub"), complete=stub,
        )
        assert "CLAIM 1: Metformin reduces mortality." in stub.prompts[0]
        assert "CLAIM 2: It does not change cardiovascular events." in stub.prompts[0]

    def test_the_cost_row_is_stamped_decompose_cite(self):
        stub = _Recorder(_cited_response())
        result = cite_claims(
            _claims(), "Does metformin reduce mortality?", _passages(),
            GenerationConfig(model="stub"), complete=stub,
        )
        assert [c.component for c in result.costs] == ["decompose_cite"]


class TestCountMismatchIsData:
    def test_a_dropped_claim_keeps_its_identity_with_no_citations_and_an_error(self):
        short_reply = (
            "DECISION: yes\n"
            "CLAIM 1: Metformin reduces mortality.\n"
            "CITE: [p1] || Metformin reduced all-cause mortality by 21%\n"
        )
        stub = _Recorder(short_reply)
        result = cite_claims(
            _claims(), "Does metformin reduce mortality?", _passages(),
            GenerationConfig(model="stub"), complete=stub,
        )
        assert [c.claim_id for c in result.claims] == ["c1", "c2"]
        assert result.claims[1].citations == []
        assert any("no matching CLAIM line" in e for e in result.errors)
        assert any("returned 1 CLAIM lines for 2 claims" in e for e in result.errors)


class TestBatching:
    """One call per `max_claims_per_call` claims. A live A4000 run showed the 8B model reproducing
    only the first handful of a 20-25 claim answer, so the whole-answer call could never come back
    complete; batching is what makes positional matching survive a long re-cut answer."""

    def _four_claims(self) -> list[Claim]:
        return [
            Claim(claim_id=f"c{i}", text=f"Claim number {i}.", granularity=Granularity.ATOMIC)
            for i in range(1, 5)
        ]

    def test_claims_are_sent_in_batches_each_renumbered_from_one(self):
        stub = _Scripted([
            "DECISION: yes\nCLAIM 1: Claim number 1.\nCLAIM 2: Claim number 2.\n",
            "DECISION: yes\nCLAIM 1: Claim number 3.\n"
            "CITE: [p1] || Metformin reduced all-cause mortality by 21%\n"
            "CLAIM 2: Claim number 4.\n",
        ])
        result = cite_claims(
            self._four_claims(), "q", _passages(), GenerationConfig(model="stub"),
            complete=stub, query_id="q1", max_claims_per_call=2,
        )
        assert len(stub.prompts) == 2
        assert "CLAIM 1: Claim number 1." in stub.prompts[0]
        assert "CLAIM 2: Claim number 2." in stub.prompts[0]
        assert "Claim number 3." not in stub.prompts[0]
        # The second batch restarts at 1: the reply is matched positionally within its own batch.
        assert "CLAIM 1: Claim number 3." in stub.prompts[1]
        assert "CLAIM 2: Claim number 4." in stub.prompts[1]

    def test_every_batch_keeps_its_claim_identities_costs_and_citations(self):
        stub = _Scripted([
            "DECISION: yes\nCLAIM 1: Claim number 1.\nCLAIM 2: Claim number 2.\n",
            "DECISION: yes\nCLAIM 1: Claim number 3.\n"
            "CITE: [p1] || Metformin reduced all-cause mortality by 21%\n"
            "CLAIM 2: Claim number 4.\n",
        ])
        result = cite_claims(
            self._four_claims(), "q", _passages(), GenerationConfig(model="stub"),
            complete=stub, query_id="q1", max_claims_per_call=2,
        )
        assert result.errors == ()
        assert [c.claim_id for c in result.claims] == ["c1", "c2", "c3", "c4"]
        # The citation belongs to the third claim overall, not to the first of its batch.
        assert [len(c.citations) for c in result.claims] == [0, 0, 1, 0]
        assert [c.component for c in result.costs] == ["decompose_cite"] * 2
        assert [c.query_id for c in result.costs] == ["q1:cite0", "q1:cite1"]

    def test_a_short_batch_is_counted_and_does_not_shift_later_batches(self):
        stub = _Scripted([
            "DECISION: yes\nCLAIM 1: Claim number 1.\n",  # batch 1 drops its second claim
            "DECISION: yes\nCLAIM 1: Claim number 3.\nCLAIM 2: Claim number 4.\n",
        ])
        result = cite_claims(
            self._four_claims(), "q", _passages(), GenerationConfig(model="stub"),
            complete=stub, max_claims_per_call=2,
        )
        assert [c.claim_id for c in result.claims] == ["c1", "c2", "c3", "c4"]
        assert any("returned 1 CLAIM lines for 2 claims" in e for e in result.errors)
        assert any("c2: no matching CLAIM line" in e for e in result.errors)

    def test_a_single_batch_keeps_the_unsuffixed_query_id(self):
        stub = _Scripted([_cited_response()])
        result = cite_claims(
            _claims(), "q", _passages(), GenerationConfig(model="stub"),
            complete=stub, query_id="q1", max_claims_per_call=5,
        )
        assert [c.query_id for c in result.costs] == ["q1"]


class TestGuardRails:
    def test_no_claims_raises(self):
        with pytest.raises(ValueError, match="no claims"):
            cite_claims([], "q", _passages(), GenerationConfig(model="stub"), complete=_Recorder(""))

    def test_no_passages_raises(self):
        with pytest.raises(ValueError, match="no passages"):
            cite_claims(_claims(), "q", [], GenerationConfig(model="stub"), complete=_Recorder(""))
