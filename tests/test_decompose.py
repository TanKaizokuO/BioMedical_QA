"""Re-cutting a generated answer into C7's units.

Two properties are load-bearing, and every test here is one of them or a way of breaking one:

1. **The offsets do not lie.** `answer[claim.source_start:claim.source_end]` is the text the claim
   came from, exactly — that span is the audit trail ADR-0005's decomposition-error rate is read
   against. `test_every_sentence_unit_of_every_real_generation_is_its_own_span` checks it over all
   300 records of `parity_iter1b` rather than a fixture, because the fixture is the case that was
   thought of and the runaway 731-word joint claim was not.
2. **The three C7 rows are three different rows.** `atomic` is *bare* atomic; a decomposer that
   resolves pronouns anyway silently makes it a second copy of the headline row, and the ablation
   would then report that granularity does not matter because it was never varied.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from biomedqa.config import GenerationConfig
from biomedqa.decompose import (
    BARE_ATOMIC_RULE,
    answer_spans,
    build_prompt,
    decompose,
    decompose_template_digest,
    sentence_units,
    unit_rules,
)
from biomedqa.generate import STAGE_SEPARATOR, split_stages
from biomedqa.prompts import DECONTEXTUALIZATION_RULE
from biomedqa.schema import CostRecord, Granularity, read_query_records

_HARVEST = Path(__file__).resolve().parents[1] / "docs" / "harvest"

_GENERATION = """DECISION: yes
CLAIM 1: Metformin reduces all-cause mortality. It was not observed in the elderly.
CITE: [p1:0] || metformin reduced all-cause mortality
CLAIM 2: This effect is dose dependent."""


def _config(granularity: str) -> GenerationConfig:
    return GenerationConfig(model="m", granularity=granularity)


class _Replay:
    """A stand-in `backends.complete` that logs prompts and replays one scripted response."""

    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt, config, *, seed=0, run_id="", query_id=None):
        self.prompts.append(prompt)
        return self.response, CostRecord(
            run_id=run_id,
            query_id=query_id,
            component="generate",
            backend=f"vllm:{config.model}",
            input_tokens=11,
            output_tokens=22,
        )


def _explode(prompt, config, *, seed=0, run_id="", query_id=None):
    raise AssertionError("the sentence row must not call a model")


class TestSegmentation:
    def test_units_are_cut_from_claim_bodies_not_from_the_response_grammar(self):
        """A C7 row's input is a `raw_generation`. Sentence-splitting it whole would make
        "DECISION: yes" a unit of the `sentence` row and a quote a unit of every row."""
        assert [_GENERATION[s:e] for s, e in sentence_units(_GENERATION)] == [
            "Metformin reduces all-cause mortality.",
            "It was not observed in the elderly.",
            "This effect is dose dependent.",
        ]

    def test_prose_with_no_claim_lines_is_split_whole(self):
        """Vanilla answers and pasted paragraphs have no grammar to strip."""
        prose = "Metformin lowers HbA1c. It does not prevent fractures."
        assert answer_spans(prose) == [(0, len(prose))]
        assert [prose[s:e] for s, e in sentence_units(prose)] == [
            "Metformin lowers HbA1c.",
            "It does not prevent fractures.",
        ]

    def test_the_two_post_hoc_stages_are_refused_rather_than_counted_twice(self):
        """Both stages carry CLAIM lines, so a joined `raw_generation` doubles the row's
        denominator — silently, and in the direction that makes post-hoc look finer-grained."""
        joined = _GENERATION + STAGE_SEPARATOR + _GENERATION
        with pytest.raises(ValueError, match="both post-hoc stages"):
            decompose(joined, _config("sentence"))

    def test_every_sentence_unit_of_every_real_generation_is_its_own_span(self):
        """The offsets are the audit trail; a span that does not reproduce its claim is an audit
        trail that is wrong by however many characters it is wrong by."""
        records = list(read_query_records(_HARVEST / "parity_iter1b.records.jsonl"))
        assert len(records) == 300

        checked = 0
        for record in records:
            stage = split_stages(record.raw_generation)[-1]
            for claim in decompose(stage, _config("sentence")).claims:
                assert stage[claim.source_start:claim.source_end] == claim.text
                assert claim.text == claim.text.strip() != ""
                checked += 1
        assert checked > 3000


class TestTheThreeRowsDiffer:
    def test_bare_atomic_withholds_the_decontextualization_rule(self):
        """Asked only to split, an instruction-tuned model resolves pronouns on its own — so the
        `atomic` row has to say not to, or it is the headline row under another name."""
        atomic = build_prompt(_GENERATION, Granularity.ATOMIC)
        headline = build_prompt(_GENERATION, Granularity.DECONTEXTUALIZED_ATOMIC)

        assert "Resolve every pronoun" not in atomic
        assert BARE_ATOMIC_RULE in atomic
        assert DECONTEXTUALIZATION_RULE in headline
        assert BARE_ATOMIC_RULE not in headline
        assert all("states exactly one thing" in p for p in (atomic, headline))

    def test_the_sentence_row_pays_for_no_decomposition_error(self):
        """It is the control for decomposition error. A model call would put some in."""
        with pytest.raises(ValueError, match="not decomposed by a model"):
            unit_rules(Granularity.SENTENCE)
        claims = decompose(_GENERATION, _config("sentence"), completer=_explode).claims
        assert [c.granularity for c in claims] == [Granularity.SENTENCE] * 3

    def test_the_numbered_sentences_are_what_the_model_is_shown(self):
        prompt = build_prompt(_GENERATION, Granularity.ATOMIC, question="Does metformin help?")
        assert "Question the answer responds to: Does metformin help?" in prompt
        assert "1. Metformin reduces all-cause mortality." in prompt
        assert "DECISION" not in prompt
        assert "Question the answer" not in build_prompt(_GENERATION, Granularity.ATOMIC)


def _target_of(prompt: str) -> int:
    """Which sentence this prompt asked for. The decomposer makes one call per sentence, so a stub
    that ignores the target would answer every call with the same claims."""
    return int(re.search(r"Split sentence (\d+)", prompt).group(1))


class _PerSentence:
    """Replays one scripted reply per target sentence, keyed by the sentence the prompt names."""

    def __init__(self, by_target: dict[int, str]):
        self.by_target = by_target
        self.prompts: list[str] = []
        self.targets: list[int] = []

    def __call__(self, prompt, config, *, seed=0, run_id="", query_id=None):
        self.prompts.append(prompt)
        target = _target_of(prompt)
        self.targets.append(target)
        return self.by_target.get(target, ""), CostRecord(
            run_id=run_id, query_id=query_id, component="generate",
            backend=f"vllm:{config.model}", input_tokens=11, output_tokens=22,
        )


class TestParsing:
    def test_ids_follow_output_order_and_the_models_numbering_is_ignored(self):
        """`parse_response` learned this the expensive way: an 8B model repurposes the number it
        was asked to write, and reading it as an identity mis-attributes the claim."""
        stub = _PerSentence({
            1: "CLAIM 4: Metformin reduces all-cause mortality.\n",
            2: "CLAIM 4: Metformin's benefit was not seen in the elderly.\n",
            3: "CLAIM 9: Metformin's mortality benefit is dose dependent.\n",
        })
        result = decompose(_GENERATION, _config("decontextualized_atomic"), completer=stub)

        assert [c.claim_id for c in result.claims] == ["c1", "c2", "c3"]
        assert [_GENERATION[c.source_start:c.source_end] for c in result.claims] == [
            "Metformin reduces all-cause mortality.",
            "It was not observed in the elderly.",
            "This effect is dose dependent.",
        ]
        assert result.errors == ()

    def test_a_claims_span_is_the_sentence_its_call_was_about(self):
        """The span is no longer a number the model writes and can get wrong: it is the sentence
        the call asked about. A live run mis-stamped claims onto sentences they did not come from,
        which is invisible in the rates and corrupts the decomposition audit trail."""
        stub = _PerSentence({
            1: "CLAIM 1: Metformin reduces all-cause mortality.\n",
            # The model tries to answer for another sentence anyway; it cannot — there is no field
            # left to say so with, so the claim belongs to the sentence that was asked about.
            2: "CLAIM 1: This effect is dose dependent.\n",
            3: "CLAIM 1: This effect is dose dependent.\n",
        })
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert [_GENERATION[c.source_start:c.source_end] for c in result.claims] == [
            "Metformin reduces all-cause mortality.",
            "It was not observed in the elderly.",
            "This effect is dose dependent.",
        ]

    def test_an_over_length_claim_is_flagged_and_kept(self):
        """`MAX_CLAIM_WORDS` is a non-termination detector, not a style rule: truncating the claim
        would hide the defect the detector exists to surface."""
        runaway = " ".join(["and metformin reduces mortality"] * 20)
        stub = _PerSentence({
            1: f"CLAIM 1: {runaway}\n",
            2: "CLAIM 1: Metformin's benefit was not seen in the elderly.\n",
            3: "CLAIM 1: Metformin's mortality benefit is dose dependent.\n",
        })
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert result.claims[0].text == runaway
        assert any("exceeds the max claim length" in e for e in result.errors)

    def test_a_dropped_sentence_is_an_error_not_a_shorter_answer(self):
        """A decomposer that skips a sentence deletes an assertion from the answer it is re-cutting,
        which moves the C7 row for a reason that is not granularity. One call per sentence makes
        this exact: the sentence whose own call came back empty is the one named."""
        stub = _PerSentence({1: "CLAIM 1: Metformin reduces all-cause mortality.\n"})
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert len(result.claims) == 1
        assert any("sentence 2 produced no claim" in e for e in result.errors)
        assert any("sentence 3 produced no claim" in e for e in result.errors)

    def test_malformed_and_empty_lines_are_reported_rather_than_raised(self):
        """G2 gates on >=95% valid parse, so every failure has to survive to be counted; a parser
        that raises reports one failure per answer and hides the rest."""
        stub = _PerSentence({
            1: "Here are the claims:\nCLAIM ONE AND A HALF: Metformin reduces mortality.\n",
            2: "CLAIM 1:\n",
            3: "CLAIM 1: Metformin's mortality benefit is dose dependent.\n",
        })
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert [c.text for c in result.claims] == [
            "Metformin's mortality benefit is dose dependent."
        ]
        assert any("is not 'CLAIM <n>'" in e for e in result.errors)
        assert any("claim is empty" in e for e in result.errors)

    def test_a_response_with_no_claims_is_an_error_and_no_claims(self):
        result = decompose(_GENERATION, _config("atomic"), completer=_Replay("I cannot help."))
        assert result.claims == ()
        assert "no CLAIM lines" in result.errors

    def test_same_reply_x2_repeat_collapses_and_is_recovered(self):
        """A x2 repeat within one sentence's reply collapses into its first occurrence:
        claims stay dense (c1, c2, ...), no error is produced, and a recovered note is logged."""
        stub = _PerSentence({
            1: "CLAIM 1: Metformin reduces all-cause mortality.\n"
               "CLAIM 2: Metformin reduces all-cause mortality.\n",
            2: "CLAIM 1: Metformin's benefit was not seen in the elderly.\n",
            3: "CLAIM 1: Metformin's mortality benefit is dose dependent.\n",
        })
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert [c.claim_id for c in result.claims] == ["c1", "c2", "c3"]
        assert result.errors == ()
        assert len(result.recovered) == 1
        assert "collapsed repeat of c1's claim text verbatim" in result.recovered[0]

    def test_same_reply_x3_repeat_triggers_loop_guard_error(self):
        """A x3 repeat within one reply collapses the claims but triggers the loop guard error
        for non-terminating generation."""
        stub = _PerSentence({
            1: "CLAIM 1: Metformin reduces all-cause mortality.\n"
               "CLAIM 2: Metformin reduces all-cause mortality.\n"
               "CLAIM 3:   Metformin reduces   all-cause mortality.  \n",
            2: "CLAIM 1: Metformin's benefit was not seen in the elderly.\n",
            3: "CLAIM 1: Metformin's mortality benefit is dose dependent.\n",
        })
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert [c.claim_id for c in result.claims] == ["c1", "c2", "c3"]
        assert len(result.recovered) == 2
        assert len(result.errors) == 1
        assert "c1: repeats claim text verbatim 3 times in one reply" in result.errors[0]
        assert "non-terminating generation" in result.errors[0]

    def test_a_claim_repeated_across_sentences_names_both_sentences(self):
        """Claims from different source sentences are both kept (c1, c2), with distinct spans.
        The duplicate moves to recovered (Case B decontextualisation) and produces no error."""
        stub = _PerSentence({
            1: "CLAIM 1: Metformin reduces all-cause mortality.\n",
            2: "CLAIM 1: Metformin reduces all-cause mortality.\n",
            3: "CLAIM 1: This effect is dose dependent.\n",
        })
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert [c.claim_id for c in result.claims] == ["c1", "c2", "c3"]
        assert result.errors == ()
        assert len(result.recovered) == 1
        assert "c2: repeats c1's claim text verbatim across sentences 1 and 2" in result.recovered[0]

    def test_a_repeat_of_a_sentence_the_answer_itself_repeats_is_not_the_decomposers_fault(self):
        """14 of 100 live post-hoc answers repeat a sentence verbatim. Charging the decomposer with
        a repetition loop for faithfully re-cutting a repeated sentence would blame it for an
        upstream defect, so it is recorded in recovered without charging an error."""
        answer = "Metformin reduces mortality. Metformin reduces mortality."
        stub = _PerSentence({
            1: "CLAIM 1: Metformin reduces mortality.\n",
            2: "CLAIM 1: Metformin reduces mortality.\n",
        })
        result = decompose(answer, _config("atomic"), completer=stub)

        assert len(result.claims) == 2
        assert result.errors == ()
        assert len(result.recovered) == 1
        assert "c2: sentence 2 repeats sentence 1 verbatim in the answer" in result.recovered[0]

    def test_cross_sentence_duplicate_spanning_3_sentences_errors(self):
        """A duplicate claim text spanning 3 or more distinct sentences triggers the loop guard
        and stays an errors entry."""
        stub = _PerSentence({
            1: "CLAIM 1: Metformin reduces all-cause mortality.\n",
            2: "CLAIM 1: Metformin reduces all-cause mortality.\n",
            3: "CLAIM 1: Metformin reduces all-cause mortality.\n",
        })
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert [c.claim_id for c in result.claims] == ["c1", "c2", "c3"]
        assert len(result.errors) == 1
        assert "c3: repeats c1's claim text verbatim across 3 sentences" in result.errors[0]

    def test_duplicate_claim_count_unchanged_in_magnitude(self):
        """duplicate_claim_count in decompose_smoke matches duplicates from BOTH errors and
        recovered, so moving a x2 duplicate to recovered does not decrease the count."""
        stub = _PerSentence({
            1: "CLAIM 1: Metformin reduces all-cause mortality.\n",
            2: "CLAIM 1: Metformin reduces all-cause mortality.\n",
        })
        decomp = decompose(_GENERATION, _config("atomic"), completer=stub)

        count_from_both = sum(
            1 for problem in (*decomp.errors, *decomp.recovered)
            if "claim text verbatim" in problem or "repeats sentence" in problem
        )
        assert count_from_both == 1
    def test_drift_variants_are_parsed_leniently_and_logged(self):
        """Bullets, missing spaces and bold markers around the head are drift, not a broken claim:
        the line still carries exactly one claim, so throwing it away would understate the row."""
        stub = _PerSentence({
            1: "CLAIM1: Metformin reduces all-cause mortality.\n",
            2: "- **CLAIM 2**: Metformin's benefit was not seen in the elderly.\n",
            3: "CLAIM: Metformin's mortality benefit is dose dependent.\n",
        })
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert [c.text for c in result.claims] == [
            "Metformin reduces all-cause mortality.",
            "Metformin's benefit was not seen in the elderly.",
            "Metformin's mortality benefit is dose dependent.",
        ]
        assert all(c.source_start is not None for c in result.claims)
        assert result.errors == ()


class TestCost:
    def test_every_sentence_call_is_billed_to_decompose(self):
        """Table 4 has to be able to tell a C7 row's extra calls from the generation it re-cuts;
        `backends` stamps everything "generate" because generation is all it has been asked for.
        There is one call per sentence, and every one of them is a cost row."""
        stub = _PerSentence({i: f"CLAIM 1: claim {i}.\n" for i in (1, 2, 3)})
        result = decompose(
            _GENERATION, _config("atomic"), completer=stub, run_id="c7", query_id="q1"
        )

        assert len(result.costs) == 3
        assert {c.component for c in result.costs} == {"decompose"}
        assert [c.query_id for c in result.costs] == ["q1:s1", "q1:s2", "q1:s3"]
        assert {c.run_id for c in result.costs} == {"c7"}
        assert decompose(_GENERATION, _config("sentence")).costs == ()


class TestPerSentenceCalls:
    def test_one_call_per_sentence_each_naming_its_own_target(self):
        answer = ". ".join(f"Sentence {i} asserts fact {i}" for i in range(1, 16)) + "."

        def completer(prompt, config, *, seed=0, run_id="", query_id=None):
            target = _target_of(prompt)
            return f"CLAIM 1: sentence {target} fact.", CostRecord(
                run_id=run_id, query_id=query_id, component="decompose", backend="test",
                input_tokens=10, output_tokens=10,
            )

        result = decompose(answer, _config("atomic"), completer=completer, query_id="q_multi")

        assert len(result.claims) == 15
        assert (result.claims[0].claim_id, result.claims[14].claim_id) == ("c1", "c15")
        assert result.errors == ()

    def test_every_prompt_carries_the_whole_answer_so_pronouns_can_be_resolved(self):
        """The target sentence alone cannot decontextualize "it" — the rest of the answer is the
        only place the referent lives."""
        stub = _PerSentence({i: f"CLAIM 1: claim {i}.\n" for i in (1, 2, 3)})
        decompose(_GENERATION, _config("decontextualized_atomic"), completer=stub)

        assert stub.targets == [1, 2, 3]
        for prompt in stub.prompts:
            assert "1. Metformin reduces all-cause mortality." in prompt
            assert "3. This effect is dose dependent." in prompt

    def test_a_target_outside_the_answer_is_refused_rather_than_silently_clamped(self):
        with pytest.raises(ValueError, match="outside the answer"):
            build_prompt(_GENERATION, Granularity.ATOMIC, target=4)


class TestFreeze:
    def test_the_decomposer_prompt_is_pinned_ahead_of_the_sep_3_freeze(self):
        """ADR-0009 §8 freezes the decomposer prompt Sep 3. Pinned now as a tripwire, same
        reasoning as `test_prompts.py`'s `post_hoc_answer_template_digest` pin: an edit to
        `DECOMPOSE_TEMPLATE`, `FORMAT_BLOCK`, `BARE_ATOMIC_RULE`, or either unit rule after this
        point must be a deliberate, dated re-pin — never a silent drift discovered in October.

        Re-pinned 2026-08-16: the decomposer now makes one call per sentence and its grammar lost
        the `FROM <sentence>` field, which a live A4000 run showed the model both mis-spelling and,
        worse, mis-pointing — see `decompose._DECOMPOSED_HEAD`.
        """
        assert (
            decompose_template_digest()
            == "4129a884c7a1b4854739ffe2e3900a1db39626db7fd1bf076d6b549027a7d737"
        )