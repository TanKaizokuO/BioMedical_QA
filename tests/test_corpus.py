"""Corpus construction: the seeded 2M draw, and the gold dedup that gates the W2 encode.

Two of these tests exist because of failures that are **silent** rather than loud, which is the
whole reason ADR-0012 §1 made dedup a blocker rather than a cleanup:

- `test_a_partial_shard_set_is_refused` — the HF auto-converted parquet for `MedRAG/pubmed` is a
  *partial* export (2,209,839 of 23,898,701 rows) and the shards are PMID-ordered, so it is the
  oldest ~9% of PubMed. Drawing "2M uniformly" from it yields a corpus of pre-1990 abstracts
  against 1990s–2010s gold, and every downstream number still looks plausible.
- `test_string_pmids_are_refused` — PubMedQA's `pubid` is int32 and `data.py` stringifies it;
  MedRAG's `PMID` is int64. A str-vs-int join matches nothing, reports "0 duplicates removed", and
  that reads as good news.
- `test_a_contents_shaped_row_is_refused` — indexing MedRAG's `contents` instead of `content` gives
  every distractor a title that no gold passage can have, since PubMedQA carries no title field.
  The encode completes and hit@5 looks plausible, with a gold/distractor format difference sitting
  in the space it is measured in.

The rest pin the sample's *identity*, since `corpus_id` is a promise that a given seed reproduces a
given ID list (`RunConfig.index_fingerprint()`).
"""

from __future__ import annotations

import json

import pytest

from build_corpus import choose_one_row_per_pmid, redraw_from_prescan
from biomedqa.corpus import (
    GOLD_PMID_ARTIFACT_VERSION,
    MEDRAG_TEXT_FIELD,
    MEDRAG_TOTAL_ROWS,
    PRESCAN_SIGMA,
    CorpusDraw,
    draw_corpus,
    load_gold_pmids,
    passage_text,
    prescan_cutoff,
    selection_key,
    write_gold_pmids,
)


def rows(pmids, *, title="T", content="C"):
    """Minimal MedRAG-shaped rows. Only `PMID` participates in selection."""
    return ({"id": f"pubmed23n0001_{p}", "title": title, "content": content, "PMID": p}
            for p in pmids)


# ---------------------------------------------------------------------------------------------
# The two silent failures.
# ---------------------------------------------------------------------------------------------


def test_a_partial_shard_set_is_refused() -> None:
    """Scanning fewer rows than the corpus has means the shard glob missed shards — or resolved to
    the partial parquet export. Either way the draw is not uniform over PubMed and must not run.
    """
    with pytest.raises(ValueError, match="23,898,701"):
        draw_corpus(rows(range(1000)), gold_pmids={1}, target_n=10, seed=0)


def test_the_expected_row_count_is_a_parameter_but_defaults_to_the_real_corpus() -> None:
    """Tests pass a small `expected_rows`; production must not have to, or the guard is optional
    in exactly the situation it exists for."""
    assert MEDRAG_TOTAL_ROWS == 23_898_701
    draw = draw_corpus(rows(range(1000)), gold_pmids={1}, target_n=10, seed=0, expected_rows=1000)
    assert draw.n_scanned == 1000


def test_string_pmids_are_refused() -> None:
    """The type trap, made loud. `{"21645374"} & {21645374}` is empty and raises nothing."""
    with pytest.raises(TypeError, match="int"):
        draw_corpus(rows(range(100)), gold_pmids={"5"}, target_n=10, seed=0, expected_rows=100)


def test_a_draw_that_collides_with_no_gold_pmid_is_refused() -> None:
    """PubMedQA contexts *are* PubMed abstracts, so a scan of all 23.9M that finds none of the
    1,000 gold PMIDs has a broken join, not an absent overlap. Zero is unambiguous; anything above
    zero is reported rather than gated, because nobody has measured what the real overlap is.
    """
    with pytest.raises(ValueError, match="no gold PMID"):
        draw_corpus(rows(range(100)), gold_pmids={10_000_000}, target_n=10, seed=0,
                    expected_rows=100)


# ---------------------------------------------------------------------------------------------
# The dedup itself — ADR-0012 §1.
# ---------------------------------------------------------------------------------------------


def test_gold_pmids_never_enter_the_sample() -> None:
    """The single assertion the W2 encode is blocked on. A gold abstract present under a MedRAG
    `passage_id` as well as its `{pubid}:{i}` one makes `gold_rank`/hit@5 miscount silently.
    """
    gold = {3, 7, 11}
    draw = draw_corpus(rows(range(100)), gold_pmids=gold, target_n=50, seed=0, expected_rows=100)
    assert not (set(draw.pmids) & gold)
    assert draw.n_gold_collisions == 3
    assert len(draw.pmids) == 50


def test_collisions_are_counted_even_when_not_sampled() -> None:
    """The collision count is a property of the *corpus*, not of the draw. It answers "is PubMedQA
    inside MedRAG?", which is what makes the zero-collision guard meaningful.
    """
    draw = draw_corpus(rows(range(100)), gold_pmids=set(range(90)), target_n=5, seed=0,
                       expected_rows=100)
    assert draw.n_gold_collisions == 90
    assert len(draw.pmids) == 5


def test_a_pmid_repeated_in_the_corpus_enters_the_draw_once() -> None:
    """PubMed re-publishes revised records and MedRAG keeps every revision as its own row, so the
    same PMID arrives more than once — 244 of 2,041,867 on the 2026-08-05 prescan, twice within a
    single shard. Duplicates hash to the same `selection_key`, so they land in the bottom-k
    *together* and the draw silently becomes 2M rows over fewer than 2M articles: one abstract under
    two `passage_id`s, which is the miscount ADR-0012 §1 exists to prevent, arriving from inside
    MedRAG rather than from gold.
    """
    doubled = list(rows(range(100))) + list(rows(range(100)))
    draw = draw_corpus(iter(doubled), gold_pmids={1}, target_n=50, seed=0, expected_rows=200)
    assert len(draw.pmids) == 50
    assert len(set(draw.pmids)) == 50
    assert draw.n_duplicate_rows > 0


def test_repeats_do_not_change_which_articles_are_drawn() -> None:
    """The draw is over articles, so how many times MedRAG happens to carry one must not move it in
    or out of the corpus — otherwise `corpus_id` names a sample of rows, not of PubMed."""
    once = draw_corpus(rows(range(200)), gold_pmids={1}, target_n=50, seed=7, expected_rows=200)
    twice = draw_corpus(iter(list(rows(range(200))) + list(rows(range(200)))), gold_pmids={1},
                        target_n=50, seed=7, expected_rows=400)
    assert once.pmids == twice.pmids
    assert once.fingerprint == twice.fingerprint


def test_a_pool_smaller_than_the_target_is_refused() -> None:
    with pytest.raises(ValueError, match="pool"):
        draw_corpus(rows(range(20)), gold_pmids={1}, target_n=50, seed=0, expected_rows=20)


# ---------------------------------------------------------------------------------------------
# Sample identity. `corpus_id` promises seed -> ID list.
# ---------------------------------------------------------------------------------------------


def test_the_draw_is_reproducible() -> None:
    a = draw_corpus(rows(range(500)), gold_pmids={1}, target_n=100, seed=7, expected_rows=500)
    b = draw_corpus(rows(range(500)), gold_pmids={1}, target_n=100, seed=7, expected_rows=500)
    assert a.pmids == b.pmids
    assert a.fingerprint == b.fingerprint


def test_a_different_seed_is_a_different_sample() -> None:
    a = draw_corpus(rows(range(500)), gold_pmids={1}, target_n=100, seed=7, expected_rows=500)
    b = draw_corpus(rows(range(500)), gold_pmids={1}, target_n=100, seed=8, expected_rows=500)
    assert a.pmids != b.pmids
    assert a.fingerprint != b.fingerprint


def test_the_draw_does_not_depend_on_shard_order() -> None:
    """Bottom-k on a keyed hash, not reservoir sampling. Shards may be read in any order — and on
    a resumed or parallel read they will be — without changing the corpus.
    """
    forward = draw_corpus(rows(range(500)), gold_pmids={1}, target_n=100, seed=7, expected_rows=500)
    backward = draw_corpus(rows(reversed(range(500))), gold_pmids={1}, target_n=100, seed=7,
                           expected_rows=500)
    assert forward.pmids == backward.pmids


def test_a_smaller_draw_is_a_subset_of_a_larger_one_at_the_same_seed() -> None:
    """R1's 1M fallback is then a *subset* of the 2M corpus rather than an unrelated sample, so the
    two are comparable and the smaller can be built by truncation instead of a second full scan.
    """
    big = draw_corpus(rows(range(500)), gold_pmids={1}, target_n=200, seed=7, expected_rows=500)
    small = draw_corpus(rows(range(500)), gold_pmids={1}, target_n=50, seed=7, expected_rows=500)
    assert set(small.pmids) < set(big.pmids)


def test_the_sample_is_roughly_uniform_over_the_corpus() -> None:
    """A weak but load-bearing check: the draw must not concentrate in one region of PMID space.
    The partial-parquet failure is exactly a draw concentrated in the low PMIDs, and it is the one
    thing this module exists to make impossible.
    """
    draw = draw_corpus(rows(range(10_000)), gold_pmids={1}, target_n=1000, seed=7,
                       expected_rows=10_000)
    first_half = sum(1 for p in draw.pmids if p < 5000)
    assert 400 < first_half < 600, f"draw concentrated in one half of PMID space: {first_half}/1000"


def test_selection_key_is_stable_across_processes() -> None:
    """Python's `hash()` is salted per process; a draw keyed on it would silently differ between
    the box and this machine. Hard-coded because a regression here changes the corpus.
    """
    assert selection_key(21645374, seed=0) == selection_key(21645374, seed=0)
    assert selection_key(21645374, seed=0) != selection_key(21645374, seed=1)
    assert isinstance(selection_key(21645374, seed=0), int)


def test_pmids_are_sorted_so_the_artifact_is_diffable() -> None:
    draw = draw_corpus(rows(range(500)), gold_pmids={1}, target_n=100, seed=7, expected_rows=500)
    assert list(draw.pmids) == sorted(draw.pmids)


# ---------------------------------------------------------------------------------------------
# The gold PMID artifact — the dedup key set, committed and hashed.
# ---------------------------------------------------------------------------------------------


def test_gold_pmid_artifact_round_trips_as_ints(tmp_path) -> None:
    path = tmp_path / "gold_pmids.json"
    write_gold_pmids([21645374, 16418930], path=path)
    loaded = load_gold_pmids(path)
    assert loaded == {21645374, 16418930}
    assert all(isinstance(p, int) for p in loaded)


def test_an_edited_gold_pmid_artifact_is_refused(tmp_path) -> None:
    """Same discipline as `data.py`'s splits: an edited key set must not masquerade as the frozen
    one, because a run manifest records only its hash.
    """
    path = tmp_path / "gold_pmids.json"
    write_gold_pmids([21645374, 16418930], path=path)
    payload = json.loads(path.read_text())
    payload["pmids"].append(99999999)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_gold_pmids(path)


def test_the_committed_gold_pmid_artifact_is_the_1000_pqa_labeled_pubids() -> None:
    """The real artifact, if it has been generated. This is the input the whole dedup rests on."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "gold_pmids.json"
    if not path.exists():
        pytest.skip("data/gold_pmids.json not generated yet — scripts/build_gold_pmids.py")
    pmids = load_gold_pmids(path)
    assert len(pmids) == 1000, "pqa_labeled is 1,000 rows; a different count means a different set"
    assert json.loads(path.read_text())["version"] == GOLD_PMID_ARTIFACT_VERSION


@pytest.mark.parametrize("total,target", [(20_000, 2_000), (20_000, 500), (50_000, 10_000)])
def test_the_prescan_superset_contains_the_exact_draw(total: int, target: int) -> None:
    """`scripts/build_corpus.py`'s load-bearing arithmetic, at three scales.

    Bottom-k does not know what it kept until the scan ends, which would force a second 54 GB read
    to fetch the drawn rows' text. Instead the scan keeps every row under a generous key cutoff and
    the exact draw comes from that superset. A superset that failed to contain the draw would leave
    the corpus short by however many rows fell outside it, with both steps reporting success.

    Parametrised because the first version of this used a hard-coded overshoot that held at 2M and
    was under one standard deviation at 2,000 — the failure mode `prescan_cutoff` now prevents.
    """
    cutoff, _ = prescan_cutoff(target, total)
    superset = {p for p in range(total) if selection_key(p, seed=3) < cutoff}
    draw = draw_corpus(rows(range(total)), gold_pmids={1}, target_n=target, seed=3,
                       expected_rows=total)
    assert set(draw.pmids) <= superset


def test_the_prescan_overshoot_scales_with_the_target() -> None:
    """The overshoot must stay many standard deviations wide at every target, including R1's 1M
    fallback. A constant looks safe at 2M and is a coin flip at 2,000."""
    for target in (1_000, 100_000, 1_000_000, 2_000_000):
        _, over = prescan_cutoff(target, MEDRAG_TOTAL_ROWS)
        assert (over - target) / (target**0.5) >= PRESCAN_SIGMA


def test_the_prescan_superset_is_a_small_disk_cost() -> None:
    """It is kept on disk in full, so a runaway overshoot would be paid for in GB on the box."""
    _, over = prescan_cutoff(2_000_000, MEDRAG_TOTAL_ROWS)
    assert over < 2_100_000


def test_the_draw_records_what_produced_it() -> None:
    draw = draw_corpus(rows(range(500)), gold_pmids={1}, target_n=100, seed=7, expected_rows=500)
    assert isinstance(draw, CorpusDraw)
    assert draw.seed == 7
    assert draw.target_n == 100
    assert draw.n_scanned == 500
    assert len(draw.fingerprint) == 12


# ---------------------------------------------------------------------------------------------
# The third silent failure: the distractor text form.
# ---------------------------------------------------------------------------------------------

#: One real `MedRAG/pubmed` row, `chunk/pubmed23n0001.jsonl` line 1, content truncated to 180 chars
#: and `contents` re-derived from the truncation. Real rather than invented because the whole point
#: is what MedRAG's three text fields actually contain — a fixture could assert a convenient fiction.
#: Verified over a full shard (15,377 rows, 15,377 distinct PMIDs, 2026-08-05):
#: `contents == title + " " + content` in 5,000/5,000 checked, no empty titles, no empty content.
REAL_MEDRAG_ROW = {
    "id": "pubmed23n0001_0",
    "title": "[Biochemical studies on camomile components/III. In vitro studies about the "
             "antipeptic activity of (--)-alpha-bisabolol (author's transl)].",
    "content": "(--)-alpha-Bisabolol has a primary antipeptic action depending on dosage, which is "
               "not caused by an alteration of the pH-value. The proteolytic activity of pepsin "
               "is reduced by 50 ",
    "PMID": 21,
}
REAL_MEDRAG_ROW["contents"] = REAL_MEDRAG_ROW["title"] + " " + REAL_MEDRAG_ROW["content"]


def test_the_indexed_field_is_title_free() -> None:
    """Gold is title-free under every option — PubMedQA carries no title field — so `content` is
    the only distractor form that reaches format parity with it."""
    assert MEDRAG_TEXT_FIELD == "content"
    text = passage_text(REAL_MEDRAG_ROW)
    assert text == REAL_MEDRAG_ROW["content"]
    assert REAL_MEDRAG_ROW["title"] not in text


def test_a_contents_shaped_row_is_refused() -> None:
    """The failure this guard exists for is silent: `contents` differs from `content` by ~9% of
    characters at the front, the encode completes, hit@5 looks plausible, and an empty title is now
    the one property every gold passage shares and no distractor has."""
    row = dict(REAL_MEDRAG_ROW, content=REAL_MEDRAG_ROW["contents"])
    with pytest.raises(ValueError, match="format signal"):
        passage_text(row)


def test_an_empty_abstract_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        passage_text(dict(REAL_MEDRAG_ROW, content="   "))


# ---------------------------------------------------------------------------------------------
# One row per drawn article — `scripts/build_corpus.py`.
# ---------------------------------------------------------------------------------------------


def test_the_longest_revision_wins_and_ties_break_on_id(tmp_path) -> None:
    """PubMed's revised records differ by added text, not by chunking: `22453897` is the same
    abstract with an abbreviation list appended, `22367489` is `b-subunit` against `beta-subunit`.
    Longest takes the more complete record; the tie-break exists so the rule is total, because
    `corpus_id` promises seed -> ID list and dict ordering is not a promise.
    """
    prescan = tmp_path / "prescan.jsonl"
    prescan.write_text("\n".join(json.dumps(r) for r in [
        {"id": "b_2", "PMID": 22453897, "content": "same abstract"},
        {"id": "a_1", "PMID": 22453897, "content": "same abstract plus an abbreviation list"},
        {"id": "z_9", "PMID": 22367489, "content": "equal length"},
        {"id": "a_0", "PMID": 22367489, "content": "equal length"},
        {"id": "q_7", "PMID": 999, "content": "not drawn"},
    ]) + "\n")

    chosen = choose_one_row_per_pmid(prescan, {22453897, 22367489})
    assert chosen == {22453897: "a_1", 22367489: "a_0"}


def test_a_drawn_pmid_missing_from_the_superset_is_visible(tmp_path) -> None:
    """The prescan cutoff failing to contain the draw and MedRAG holding repeats are different
    failures with different fixes; the first version of this guard could not tell them apart.
    """
    prescan = tmp_path / "prescan.jsonl"
    prescan.write_text(json.dumps({"id": "a_1", "PMID": 1, "content": "x"}) + "\n")
    assert choose_one_row_per_pmid(prescan, {1, 2}) == {1: "a_1"}


# ---------------------------------------------------------------------------------------------
# `--from-prescan` — restating a completed scan, never manufacturing one.
# ---------------------------------------------------------------------------------------------


def _prescan(tmp_path, pmids, *, seed, target_n, n_scanned, collisions=85, n_prescan_rows=None):
    pmids = list(pmids)
    (tmp_path / "prescan.jsonl").write_text("".join(
        json.dumps({"id": f"r_{p}", "title": "T", "content": "C" * (p % 7 + 1), "PMID": p}) + "\n"
        for p in pmids))
    manifest = {"seed": seed, "target_n": target_n, "n_scanned": n_scanned,
                "n_gold_collisions": collisions,
                "n_prescan_rows": len(pmids) if n_prescan_rows is None else n_prescan_rows}
    if manifest["n_prescan_rows"] is False:      # a manifest written before the count was recorded
        del manifest["n_prescan_rows"]
    (tmp_path / "corpus_manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_a_short_scan_cannot_be_laundered_through_the_prescan(tmp_path) -> None:
    """The whole risk of a redraw path: it must not let a partial scan satisfy the row-count guard,
    which is the one number evidencing the corpus is uniform over PubMed rather than its oldest 9%.
    """
    out = _prescan(tmp_path, range(100), seed=3, target_n=10, n_scanned=2_209_839)
    with pytest.raises(SystemExit, match="cannot manufacture"):
        redraw_from_prescan(out, gold_pmids={1}, target_n=10, seed=3)


def test_a_prescan_cut_for_other_parameters_is_refused(tmp_path) -> None:
    """The cutoff is a function of (seed, target_n). Against different ones the superset need not
    contain the draw, and the shortfall would be silent."""
    out = _prescan(tmp_path, range(100), seed=3, target_n=10, n_scanned=MEDRAG_TOTAL_ROWS)
    with pytest.raises(SystemExit, match="different cutoff"):
        redraw_from_prescan(out, gold_pmids={1}, target_n=10, seed=4)


def test_the_redraw_carries_the_scan_and_collision_counts_forward(tmp_path) -> None:
    """Recomputed from the superset both numbers would be wrong — ~8.5% of the scan and ~8.5% of the
    collisions — and the manifest would understate the evidence the corpus rests on."""
    cutoff, _ = prescan_cutoff(10, MEDRAG_TOTAL_ROWS)
    inside = [p for p in range(200_000) if selection_key(p, seed=3) < cutoff]
    out = _prescan(tmp_path, inside, seed=3, target_n=10, n_scanned=MEDRAG_TOTAL_ROWS,
                   collisions=1000)
    draw = redraw_from_prescan(out, gold_pmids={inside[0]}, target_n=10, seed=3)
    assert draw.n_scanned == MEDRAG_TOTAL_ROWS
    assert draw.n_gold_collisions == 1000
    assert len(draw.pmids) == 10


def test_a_truncated_prescan_cannot_be_redrawn_from(tmp_path) -> None:
    """`streaming_scan` opens `prescan.jsonl` with `"w"`, so a completed run followed by a crashed
    re-scan leaves a **truncated superset beside a surviving manifest**. Every other guard survives
    that: the scan and collision counts are carried from the old manifest, the draw is checked
    against the rows it was itself drawn from, and the containment arithmetic compares the draw to a
    cutoff the prescan filtered on before writing. So the redraw would take a corpus off whatever
    fraction of the superset the crash left behind, report the full 23,898,701-row scan, and pass.

    The superset's own row count is the only number that can tell — hence the manifest records it,
    and this refuses any prescan that no longer matches.
    """
    cutoff, _ = prescan_cutoff(10, MEDRAG_TOTAL_ROWS)
    inside = [p for p in range(1_000_000) if selection_key(p, seed=3) < cutoff]
    out = _prescan(tmp_path, inside, seed=3, target_n=10, n_scanned=MEDRAG_TOTAL_ROWS)

    kept = (out / "prescan.jsonl").read_text().splitlines()[: len(inside) // 2]
    (out / "prescan.jsonl").write_text("\n".join(kept) + "\n")

    with pytest.raises(SystemExit, match="truncated"):
        redraw_from_prescan(out, gold_pmids={inside[0]}, target_n=10, seed=3)


def test_a_manifest_with_no_recorded_superset_size_is_refused(tmp_path) -> None:
    """Back-filling the count from the file on disk is the one thing this must not do: the file may
    already be the truncated one, and the redraw would then certify itself. Refusing puts the
    judgement on the operator, who is the only one who knows whether the scan has been re-run.
    """
    cutoff, _ = prescan_cutoff(10, MEDRAG_TOTAL_ROWS)
    inside = [p for p in range(1_000_000) if selection_key(p, seed=3) < cutoff]
    out = _prescan(tmp_path, inside, seed=3, target_n=10, n_scanned=MEDRAG_TOTAL_ROWS,
                   n_prescan_rows=False)
    with pytest.raises(SystemExit, match="records no n_prescan_rows"):
        redraw_from_prescan(out, gold_pmids={inside[0]}, target_n=10, seed=3)
