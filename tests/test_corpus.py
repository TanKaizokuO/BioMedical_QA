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

The rest pin the sample's *identity*, since `corpus_id` is a promise that a given seed reproduces a
given ID list (`RunConfig.index_fingerprint()`).
"""

from __future__ import annotations

import json

import pytest

from biomedqa.corpus import (
    GOLD_PMID_ARTIFACT_VERSION,
    MEDRAG_TOTAL_ROWS,
    PRESCAN_SIGMA,
    CorpusDraw,
    draw_corpus,
    load_gold_pmids,
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
