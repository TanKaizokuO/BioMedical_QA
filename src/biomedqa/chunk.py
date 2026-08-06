"""Passage granularity — Table 1 rows, one per `(chunker, τ)`.

Design constraints that were settled before this was written, so that implementing it was not a
design exercise:

- **hit@5 is only defined per `(chunker, τ)` pair** (Lesson 2). A chunker is therefore part of the
  index identity, which is why `ChunkConfig` feeds `RunConfig.index_fingerprint()`.
- **Every emitted chunk carries its char offsets** into `Instance.abstract_text`. Citations are
  `{passage_id, char_start, char_end}` (`CONTEXT.md`); a chunker that loses offsets makes the
  attribution unit unrepresentable.
- **Distractors are abstract prose with no title**, read through `corpus.passage_text` and never by
  indexing a MedRAG field directly. Gold has no title — PubMedQA carries none — so a titled
  distractor is a gold/distractor format difference in the space hit@5 is measured in. The gold
  articles' *real* titles are not the repair: a PubMedQA question is its article's title verbatim
  (measured, `corpus.py`), so titled gold turns G1 into a string match.
- **Section labels survive.** `data.py` keeps PubMedQA's BACKGROUND/METHODS/RESULTS boundaries
  precisely so the section and sentence-window strategies can use them.
- Promoted from `notebooks/02_1_chunking_granularity.ipynb`, which validates against a toy corpus
  using `all-MiniLM-L6-v2` as a stand-in for MedCPT — neither assumption survives 2M abstracts
  (`research_roadmap.md` §0, promotion table).

**One splitter cuts gold and distractors both.** `chunk_instance` and `chunk_text` are two callers
of the same `span` functions, and that is deliberate rather than tidy: ADR-0014 §2 rejects any
property that every gold passage shares and no distractor has, and *how the text was cut* is such a
property. Two splitters that agree today would drift, and the drift would sit in exactly the space
hit@5 and ADR-0012 §2's confusability probe are measured in. Distractors have no section labels, so
`"section"` degrades to `"abstract"` for them — the one asymmetry that cannot be designed away, and
the reason the section chunker's dev hit@5 has to be read with it in mind.

**Offsets are always into the string that was passed in.** For gold that is
`Instance.abstract_text`, the coordinate space `data.py` builds and the only one citations are
meaningful in. Nothing here re-joins or normalises text: every span is a slice of the input, so
`text[c.char_start:c.char_end] == c.text` holds by construction and is asserted for all four
strategies in `tests/test_chunk.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import ChunkConfig
from .data import Instance

#: A sentence ends at `.`, `!` or `?` followed by whitespace or end-of-text. Deliberately not an
#: abbreviation-aware splitter: PubMed abstracts are dense with them ("i.v.", "vs.", "et al."), a
#: wrong split costs a slightly odd window boundary, and a dependency here would have to be pinned
#: into the index fingerprint to keep `corpus_id`'s promise that a config reproduces an index.
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")

#: Strategies `ChunkConfig.strategy` may name. Kept here rather than in `config.py` so that the
#: module that implements them is the one that says which exist.
STRATEGIES = ("abstract", "section", "sentence_window", "fixed_width")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable passage, and where it came from.

    `passage_id` is `f"{source_id}:{index}"` — the same shape `data.py` gives `GoldPassage`, so the
    `"section"` chunker reproduces the gold ids exactly and `Instance.gold_passage_ids` keeps
    meaning what it meant. Under every other strategy the ids are new, which is correct: a re-chunk
    is a re-run, not a re-score (`schema.py`).
    """

    passage_id: str
    source_id: str
    index: int
    label: str | None
    text: str
    char_start: int
    char_end: int


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Sentence boundaries within `[start, end)`, as offsets into `text`.

    Leading whitespace is skipped rather than included, so a chunk never begins with the space
    `Instance.abstract_text` joined its sections on.
    """
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in _SENTENCE_END.finditer(text, start, end):
        stop = match.end()
        if text[cursor:stop].strip():
            spans.append((cursor, stop))
        cursor = stop
        while cursor < end and text[cursor].isspace():
            cursor += 1
    if text[cursor:end].strip():
        spans.append((cursor, end))
    return spans


def _windowed(spans: list[tuple[int, int]], size: int, stride: int) -> list[tuple[int, int]]:
    """Overlapping windows over `spans`, each window collapsed to one span.

    The last window is emitted and then iteration stops, so the tail is covered exactly once rather
    than by a run of windows that are all suffixes of each other.
    """
    if not spans:
        return []
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(spans):
        window = spans[i:i + size]
        out.append((window[0][0], window[-1][1]))
        if i + size >= len(spans):
            break
        i += stride
    return out


def _enforce_max_chars(spans: list[tuple[int, int]], max_chars: int) -> list[tuple[int, int]]:
    """Cut any span longer than `max_chars` into consecutive pieces of at most that.

    Applied to every strategy, not just `"fixed_width"`, because the cap is an encoder input limit
    rather than a granularity preference — MedCPT truncates at 512 tokens, and a chunk it truncates
    is a chunk whose stored offsets cover text that was never encoded.
    """
    out: list[tuple[int, int]] = []
    for start, end in spans:
        if end - start <= max_chars:
            out.append((start, end))
            continue
        cursor = start
        while cursor < end:
            out.append((cursor, min(cursor + max_chars, end)))
            cursor += max_chars
    return out


def _validate(config: ChunkConfig) -> None:
    """Refuse chunker settings that would build a real index quietly missing text — or hang.

    Checked here rather than in `ChunkConfig.__post_init__` because these are this module's
    constraints, not the dataclass's: `config.py` holds the knobs for the whole pipeline and has no
    business knowing that a stride wider than its window leaves gaps.

    **Refused, never clamped.** Every one of these fields is in `index_fingerprint()`, so silently
    repairing a value would leave the fingerprint describing a chunker nobody asked for — the
    fingerprint's whole job is that it cannot do that.
    """
    if config.max_chars < 1:
        raise ValueError(
            f"max_chars must be at least 1, got {config.max_chars}. Zero does not raise on its own "
            "— it makes the cap loop advance by nothing and hang, inside a 2M-row encode rather "
            "than here."
        )
    if config.strategy != "sentence_window":
        return
    if config.window_sentences < 1 or config.stride_sentences < 1:
        raise ValueError(
            f"window_sentences and stride_sentences must be at least 1, got "
            f"{config.window_sentences} and {config.stride_sentences}."
        )
    if config.stride_sentences > config.window_sentences:
        raise ValueError(
            f"stride_sentences {config.stride_sentences} is wider than window_sentences "
            f"{config.window_sentences}, so the windows tile with gaps and the sentences in those "
            "gaps end up in no chunk at all. Nothing downstream can see that: the abstract is "
            "still retrievable, hit@5 just drops, and it reads as a weak retriever rather than as "
            "an index missing text."
        )


def _spans(
    text: str,
    config: ChunkConfig,
    sections: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """The strategy switch, over offsets alone. `sections` is `None` for text that has none."""
    whole = [(0, len(text))]
    if config.strategy == "abstract":
        spans = whole
    elif config.strategy == "section":
        spans = sections if sections else whole
    elif config.strategy == "sentence_window":
        spans = _windowed(
            _sentence_spans(text, 0, len(text)),
            config.window_sentences,
            config.stride_sentences,
        )
    elif config.strategy == "fixed_width":
        spans = whole
    else:
        raise ValueError(
            f"unknown chunk strategy {config.strategy!r}; expected one of {STRATEGIES}. The "
            "strategy is part of the index fingerprint, so a typo here would name an index that "
            "was never built."
        )
    return _enforce_max_chars([s for s in spans if text[s[0]:s[1]].strip()], config.max_chars)


def chunk_text(
    text: str,
    source_id: str,
    config: ChunkConfig,
    sections: list[tuple[int, int, str | None]] | None = None,
) -> list[Chunk]:
    """Chunk one passage of prose. The distractor path, and what `chunk_instance` delegates to.

    `sections` is `(char_start, char_end, label)` triples for text that carries them; MedRAG rows do
    not, so distractors pass `None` and `"section"` degrades to `"abstract"` (module docstring).
    """
    _validate(config)
    bounds = [(s, e) for s, e, _ in sections] if sections else None
    label_at = {(s, e): label for s, e, label in sections} if sections else {}
    out: list[Chunk] = []
    for i, (start, end) in enumerate(_spans(text, config, bounds)):
        label = label_at.get((start, end)) if config.keep_section_labels else None
        out.append(Chunk(
            passage_id=f"{source_id}:{i}",
            source_id=source_id,
            index=i,
            label=label,
            text=text[start:end],
            char_start=start,
            char_end=end,
        ))
    return out


def chunk_instance(instance: Instance, config: ChunkConfig) -> list[Chunk]:
    """Split one gold abstract into retrievable passages, preserving char offsets.

    Offsets index `Instance.abstract_text`, which is the only coordinate space in which a citation
    into a PubMedQA abstract means anything (`data.py`).
    """
    return chunk_text(
        instance.abstract_text,
        instance.pubid,
        config,
        sections=[(p.char_start, p.char_end, p.label) for p in instance.passages],
    )
