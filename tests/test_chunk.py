"""Passage granularity — the four chunkers, and the offsets citations are defined in.

The load-bearing property is not "the text looks right", it is
`abstract_text[chunk.char_start:chunk.char_end] == chunk.text` for every chunk of every strategy.
A citation is `{passage_id, char_start, char_end}` (`CONTEXT.md`), so a chunker that gets the text
right and the offsets wrong produces citations that point at the wrong words while every string in
the record reads correctly — and `schema.Citation` cannot catch it, because the span it validates is
into the passage, not into the abstract.

`Instance.abstract_text` joins sections with a single space, and `data.py` advances its cursor by
`len(text) + 1` to match. Every test here uses a real join rather than a hand-built offset, so the
two cannot drift apart silently.
"""

from __future__ import annotations

import pytest

from biomedqa.chunk import chunk_instance
from biomedqa.config import ChunkConfig
from biomedqa.data import GoldPassage, Instance


def instance(*sections: tuple[str, str], pubid: str = "21645374") -> Instance:
    """An `Instance` built the way `load_instances` builds one, cursor arithmetic included."""
    passages, cursor = [], 0
    for i, (label, text) in enumerate(sections):
        passages.append(GoldPassage(
            passage_id=f"{pubid}:{i}", pubid=pubid, index=i, label=label, text=text,
            char_start=cursor, char_end=cursor + len(text),
        ))
        cursor += len(text) + 1
    return Instance(pubid=pubid, question="Is chunking load-bearing?", passages=passages)


ABSTRACT = instance(
    ("BACKGROUND", "Programmed cell death is a regulated process. It is not passive."),
    ("METHODS", "We assayed forty samples. Each was run in triplicate."),
    ("RESULTS", "Caspase activity rose sharply."),
)


def assert_offsets_round_trip(inst: Instance, chunks) -> None:
    """The one assertion every strategy has to satisfy, factored out so no strategy can skip it."""
    text = inst.abstract_text
    assert chunks, "a chunker that emits nothing makes its abstract unretrievable"
    for c in chunks:
        assert text[c.char_start:c.char_end] == c.text, (
            f"chunk {c.passage_id} claims [{c.char_start}, {c.char_end}) but that span holds "
            f"{text[c.char_start:c.char_end]!r}, not {c.text!r} — every citation into this passage "
            "would point at the wrong words"
        )


@pytest.mark.parametrize("strategy", ["abstract", "section", "sentence_window", "fixed_width"])
def test_every_chunker_emits_offsets_that_index_the_abstract(strategy: str) -> None:
    """Parametrised over all four so a strategy added later cannot quietly skip the invariant."""
    chunks = chunk_instance(ABSTRACT, ChunkConfig(strategy=strategy, max_chars=40))
    assert_offsets_round_trip(ABSTRACT, chunks)


@pytest.mark.parametrize("strategy", ["abstract", "section", "sentence_window", "fixed_width"])
def test_no_text_is_left_unretrievable(strategy: str) -> None:
    """Every non-space character has to sit in at least one chunk.

    A chunker that drops a span makes that text unretrievable while every number downstream still
    computes: hit@5 falls a little, and it reads as the retriever being weak rather than as the
    index missing text. Whitespace is exempt — the join space between sections belongs to no
    section, and skipping it is what keeps a chunk from starting with a space.
    """
    text = ABSTRACT.abstract_text
    chunks = chunk_instance(ABSTRACT, ChunkConfig(strategy=strategy, max_chars=40))
    covered = set()
    for c in chunks:
        covered.update(range(c.char_start, c.char_end))
    missing = [i for i, ch in enumerate(text) if i not in covered and not ch.isspace()]
    assert not missing, f"{len(missing)} characters are in no chunk, starting at {missing[:3]}"


@pytest.mark.parametrize("window,stride", [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (5, 4)])
def test_no_text_is_left_unretrievable_at_any_valid_window_and_stride(window, stride) -> None:
    """The coverage test above ran only the default stride, and that is how
    `stride_sentences > window_sentences` got past it — the gap it opens is invisible at stride 1.
    Parametrised over the whole legal range so the next gap-opening combination cannot hide the
    same way. Illegal combinations are refused outright; see the stride test below.
    """
    text = ABSTRACT.abstract_text
    chunks = chunk_instance(ABSTRACT, ChunkConfig(
        strategy="sentence_window", window_sentences=window, stride_sentences=stride))
    covered = set()
    for c in chunks:
        covered.update(range(c.char_start, c.char_end))
    missing = [i for i, ch in enumerate(text) if i not in covered and not ch.isspace()]
    assert not missing, f"window={window} stride={stride} leaves {len(missing)} chars unchunked"


def test_the_section_chunker_reproduces_the_gold_passage_ids() -> None:
    """`Instance.gold_passage_ids` is the set hit@5 and `gold_rank` are defined over. Under the
    section strategy the chunker is re-deriving exactly what `data.py` already produced, so the ids
    must agree — otherwise the gold set and the index disagree about what a gold passage is, and
    every hit@5 under this chunker is measured against the wrong denominator.
    """
    chunks = chunk_instance(ABSTRACT, ChunkConfig(strategy="section"))
    assert [c.passage_id for c in chunks] == ABSTRACT.gold_passage_ids
    assert [c.text for c in chunks] == [p.text for p in ABSTRACT.passages]
    assert [c.label for c in chunks] == ["BACKGROUND", "METHODS", "RESULTS"]


def test_the_abstract_chunker_emits_one_chunk_holding_everything() -> None:
    """The ADR-0003 baseline: retrieval unit = whole abstract. It is the row every other chunker in
    Table 1 is compared against, so it has to be exactly one passage, not one-per-section."""
    chunks = chunk_instance(ABSTRACT, ChunkConfig(strategy="abstract"))
    assert len(chunks) == 1
    assert chunks[0].text == ABSTRACT.abstract_text


def test_sentence_windows_overlap_when_the_stride_is_shorter_than_the_window() -> None:
    """The point of a sentence window is that a claim spanning a boundary still sits whole inside
    some window. With stride < window that requires consecutive windows to share sentences; a
    chunker that silently strode by the window size would tile instead, and the failure shows up
    only as a few unretrievable boundary-spanning claims.
    """
    chunks = chunk_instance(ABSTRACT, ChunkConfig(strategy="sentence_window",
                                                  window_sentences=2, stride_sentences=1))
    assert len(chunks) > 1
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.char_start < earlier.char_end, "consecutive windows do not overlap"


def test_the_last_sentence_window_is_emitted_once() -> None:
    """Striding past the end would emit a run of windows that are all suffixes of each other,
    indexing the abstract's tail several times and weighting it in the retrieval scores."""
    chunks = chunk_instance(ABSTRACT, ChunkConfig(strategy="sentence_window",
                                                  window_sentences=3, stride_sentences=1))
    ends = [c.char_end for c in chunks]
    assert ends.count(len(ABSTRACT.abstract_text)) == 1


def test_max_chars_caps_every_strategy_not_just_fixed_width() -> None:
    """MedCPT truncates at 512 tokens. A chunk longer than the cap is a chunk whose stored offsets
    cover text the encoder never saw, so the citation span and the indexed vector disagree."""
    for strategy in ["abstract", "section", "sentence_window", "fixed_width"]:
        chunks = chunk_instance(ABSTRACT, ChunkConfig(strategy=strategy, max_chars=30))
        assert max(len(c.text) for c in chunks) <= 30, strategy


def test_dropping_section_labels_is_a_config_choice_that_reaches_the_chunks() -> None:
    """`keep_section_labels` is in the index fingerprint, so it has to change the index."""
    kept = chunk_instance(ABSTRACT, ChunkConfig(strategy="section", keep_section_labels=True))
    dropped = chunk_instance(ABSTRACT, ChunkConfig(strategy="section", keep_section_labels=False))
    assert [c.label for c in kept] == ["BACKGROUND", "METHODS", "RESULTS"]
    assert [c.label for c in dropped] == [None, None, None]


def test_a_stride_wider_than_the_window_is_refused() -> None:
    """`window_sentences=2, stride_sentences=3` tiles with a gap: over five sentences it emits the
    first two and the last two, and the middle sentence is **in no chunk at all**. Nothing
    downstream can see that. The abstract is still retrievable, hit@5 just drops a little, and it
    reads as a weak retriever rather than as an index missing text — the same shape as ADR-0007.

    Both numbers are in the index fingerprint, so this is a config that would name a real, built,
    quietly lossy index. Refused rather than clamped: silently changing someone's stride would make
    the fingerprint describe a chunker they did not ask for.
    """
    with pytest.raises(ValueError, match="stride"):
        chunk_instance(ABSTRACT, ChunkConfig(strategy="sentence_window",
                                             window_sentences=2, stride_sentences=3))


@pytest.mark.parametrize("bad", [
    ChunkConfig(strategy="sentence_window", window_sentences=0),
    ChunkConfig(strategy="sentence_window", stride_sentences=0),
    ChunkConfig(strategy="fixed_width", max_chars=0),
    ChunkConfig(strategy="abstract", max_chars=-1),
])
def test_degenerate_chunk_sizes_are_refused(bad: ChunkConfig) -> None:
    """`max_chars=0` is the one that matters: `_enforce_max_chars` advances its cursor by
    `max_chars`, so zero does not raise — it **hangs**, and it hangs inside a 2M-row encode on the
    box rather than here. A crash is recoverable from a traceback; a hang costs the run.
    """
    with pytest.raises(ValueError):
        chunk_instance(ABSTRACT, bad)


def test_an_unknown_strategy_is_refused() -> None:
    """The strategy is part of the index fingerprint, so a typo would name an index never built."""
    with pytest.raises(ValueError, match="unknown chunk strategy"):
        chunk_instance(ABSTRACT, ChunkConfig(strategy="sliding"))


def test_a_distractor_has_no_sections_so_the_section_chunker_falls_back_to_the_abstract() -> None:
    """MedRAG rows carry no BACKGROUND/METHODS labels. This is the one gold/distractor asymmetry
    that cannot be designed away — under `"section"` gold is cut into sections and distractors are
    not — so it is pinned here rather than discovered while reading a hit@5 table.
    """
    from biomedqa.chunk import chunk_text

    prose = "Bisabolol reduces peptic activity. The effect depends on dosage."
    chunks = chunk_text(prose, "21", ChunkConfig(strategy="section"))
    assert len(chunks) == 1
    assert chunks[0].text == prose
    assert chunks[0].passage_id == "21:0"
