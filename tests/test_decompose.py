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


class TestParsing:
    def test_ids_follow_output_order_and_the_models_numbering_is_ignored(self):
        """`parse_response` learned this the expensive way: an 8B model repurposes the number it
        was asked to write, and reading it as an identity mis-attributes the claim."""
        stub = _Replay(
            "CLAIM 4 FROM 2: Metformin's benefit was not seen in the elderly.\n"
            "CLAIM 4 FROM 1: Metformin reduces all-cause mortality.\n"
            "CLAIM 9 FROM 3: Metformin's mortality benefit is dose dependent.\n"
        )
        result = decompose(_GENERATION, _config("decontextualized_atomic"), completer=stub)

        assert [c.claim_id for c in result.claims] == ["c1", "c2", "c3"]
        assert [_GENERATION[c.source_start:c.source_end] for c in result.claims] == [
            "It was not observed in the elderly.",
            "Metformin reduces all-cause mortality.",
            "This effect is dose dependent.",
        ]
        assert result.errors == ()

    def test_an_unusable_sentence_index_keeps_the_claim_without_a_span(self):
        """Dropping it would shrink the denominator of the rate that is supposed to report
        decomposition failure — the failure would erase its own evidence."""
        stub = _Replay(
            "CLAIM 1 FROM 1: Metformin reduces all-cause mortality.\n"
            "CLAIM 2 FROM 2: Metformin's benefit was not seen in the elderly.\n"
            "CLAIM 3 FROM 9: Metformin's mortality benefit is dose dependent.\n"
        )
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert len(result.claims) == 3
        assert (result.claims[2].source_start, result.claims[2].source_end) == (None, None)
        assert any("sentence 9 does not exist" in e for e in result.errors)

    def test_an_over_length_claim_is_flagged_and_kept(self):
        """`MAX_CLAIM_WORDS` is a non-termination detector, not a style rule: truncating the claim
        would hide the defect the detector exists to surface."""
        runaway = " ".join(["and metformin reduces mortality"] * 20)
        stub = _Replay(
            f"CLAIM 1 FROM 1: {runaway}\n"
            "CLAIM 2 FROM 2: Metformin's benefit was not seen in the elderly.\n"
            "CLAIM 3 FROM 3: Metformin's mortality benefit is dose dependent.\n"
        )
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert result.claims[0].text == runaway
        assert any("exceeds the max claim length" in e for e in result.errors)

    def test_a_dropped_sentence_is_an_error_not_a_shorter_answer(self):
        """A decomposer that skips a sentence deletes an assertion from the answer it is re-cutting,
        which moves the C7 row for a reason that is not granularity."""
        stub = _Replay("CLAIM 1 FROM 1: Metformin reduces all-cause mortality.\n")
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert len(result.claims) == 1
        assert any("produced no claim" in e for e in result.errors)

    def test_malformed_and_empty_lines_are_reported_rather_than_raised(self):
        """G2 gates on >=95% valid parse, so every failure has to survive to be counted; a parser
        that raises reports one failure per answer and hides the rest."""
        stub = _Replay(
            "Here are the claims:\n"
            "CLAIM 1: Metformin reduces all-cause mortality.\n"
            "CLAIM 2 FROM 2:\n"
            "CLAIM 3 FROM 3: Metformin's mortality benefit is dose dependent.\n"
        )
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert [c.text for c in result.claims] == [
            "Metformin's mortality benefit is dose dependent."
        ]
        assert any("is not 'CLAIM <n> FROM <sentence>'" in e for e in result.errors)
        assert any("claim is empty" in e for e in result.errors)

    def test_a_response_with_no_claims_is_an_error_and_no_claims(self):
        result = decompose(_GENERATION, _config("atomic"), completer=_Replay("I cannot help."))
        assert result.claims == ()
        assert "no CLAIM lines" in result.errors

    def test_a_repeated_claim_is_flagged_not_deduplicated(self):
        """The live 2026-08-15 A4000 run found greedy decoding taking the repetition escape
        `MAX_CLAIM_WORDS` does not cover: re-emitting an already-written claim instead of a new
        one. It is kept, same reasoning as an over-length claim — collapsing duplicates would hide
        the defect the flag exists to surface, and would understate `total_claims` too."""
        stub = _Replay(
            "CLAIM 1 FROM 1: Metformin reduces all-cause mortality.\n"
            "CLAIM 2 FROM 1: Metformin reduces all-cause mortality.\n"
            "CLAIM 3 FROM 1:   Metformin reduces   all-cause mortality.  \n"
            "CLAIM 4 FROM 2: Metformin's benefit was not seen in the elderly.\n"
        )
        result = decompose(_GENERATION, _config("atomic"), completer=stub)

        assert len(result.claims) == 4
        assert sum(1 for e in result.errors if "repeats" in e and "verbatim" in e) == 2
        assert any("c1" in e for e in result.errors)


class TestCost:
    def test_the_decomposition_call_is_billed_to_decompose(self):
        """Table 4 has to be able to tell a C7 row's extra call from the generation it re-cuts;
        `backends` stamps everything "generate" because generation is all it has been asked for."""
        stub = _Replay(
            "CLAIM 1 FROM 1: Metformin reduces all-cause mortality.\n"
            "CLAIM 2 FROM 2: Metformin's benefit was not seen in the elderly.\n"
            "CLAIM 3 FROM 3: Metformin's mortality benefit is dose dependent.\n"
        )
        result = decompose(
            _GENERATION, _config("atomic"), completer=stub, run_id="c7", query_id="q1"
        )

        assert len(result.costs) == 1
        assert result.costs[0].component == "decompose"
        assert (result.costs[0].run_id, result.costs[0].query_id) == ("c7", "q1")
        assert decompose(_GENERATION, _config("sentence")).costs == ()


class TestChunking:
    def test_multi_chunk_decomposition(self):
        answer = ". ".join(f"Sentence {i} asserts fact {i}" for i in range(1, 16)) + "."
        prompts_seen = []

        def chunk_completer(prompt, config, *, seed=0, run_id="", query_id=None):
            prompts_seen.append(prompt)
            lines = []
            for line in prompt.splitlines():
                if line and line[0].isdigit() and ". Sentence " in line:
                    num = line.split(".")[0]
                    lines.append(f"CLAIM {num} FROM {num}: sentence {num} fact.")
            return "\n".join(lines), CostRecord(run_id=run_id, query_id=query_id, component="decompose", backend="test", input_tokens=10, output_tokens=10)

        result = decompose(
            answer,
            _config("atomic"),
            completer=chunk_completer,
            max_sentences_per_chunk=5,
            run_id="c7_test",
            query_id="q_multi",
        )

        assert len(prompts_seen) == 3
        assert len(result.claims) == 15
        assert result.claims[0].claim_id == "c1"
        assert result.claims[14].claim_id == "c15"
        assert len(result.costs) == 3
        assert result.costs[0].query_id == "q_multi:chunk0"
        assert result.costs[1].query_id == "q_multi:chunk1"
        assert result.costs[2].query_id == "q_multi:chunk2"
        assert result.errors == ()


class TestFreeze:
    def test_the_decomposer_prompt_is_pinned_ahead_of_the_sep_3_freeze(self):
        """ADR-0009 §8 freezes the decomposer prompt Sep 3. Pinned now (2026-08-15) as a tripwire,
        same reasoning as `test_prompts.py`'s `post_hoc_answer_template_digest` pin: an edit to
        `DECOMPOSE_TEMPLATE`, `FORMAT_BLOCK`, `BARE_ATOMIC_RULE`, or either unit rule after this
        point must be a deliberate, dated re-pin — never a silent drift discovered in October.

        This value is provisional until `scripts/decompose_smoke.py` runs on the A4000 (still
        pending — the box is not reachable from this environment). If that run forces a prompt
        edit before Sep 3, this test's expected digest is repinned in the same commit as the edit,
        with a note of what changed and why. If Sep 3 arrives with the pin unchanged, this pin is
        the freeze ADR-0009 §8 names.
        """
        assert (
            decompose_template_digest()
            == "da7e4a4c8e808a39f263a3e6cd02bd7014c95080dc79216608325cd4dbf11150"
        )