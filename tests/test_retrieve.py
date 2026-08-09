"""Dense scoring — the chunked float16 -> float32 matvec.

`dense.npy` is float16 to halve the 2.16M-passage index on disk and in RAM, but the dot
product runs in float32. Promoting the whole matrix per query allocates a second copy the
size of the index (6.6 GB at production scale) on *every* query, so `_dense_scores` promotes
`DENSE_CHUNK_ROWS` at a time.

The defect that hides here is a tail bug. `sims` is `np.empty`, so a chunk loop that fails
to cover the last partial slice returns *uninitialised memory* for those rows rather than
raising — garbage scores that sort into the top-k and read as a retrieval failure. Every
test below therefore uses a row count that `DENSE_CHUNK_ROWS` does not divide evenly, and
checks all N rows against an independently computed reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from biomedqa import retrieve


@pytest.fixture()
def small_chunks(monkeypatch):
    """Force the chunk loop to iterate on test-sized matrices."""
    monkeypatch.setattr(retrieve, "DENSE_CHUNK_ROWS", 7)


def _random_index(n: int, dim: int = 768, seed: int = 0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    query = rng.standard_normal(dim).astype(np.float32)
    query /= np.linalg.norm(query)
    return emb.astype(np.float16), query


def test_covers_every_row_including_a_partial_tail(small_chunks):
    """23 rows at chunk 7 leaves a 2-row tail; an uncovered tail returns np.empty garbage."""
    emb, query = _random_index(23)
    sims = retrieve._dense_scores(emb, query)

    reference = np.array(
        [float(np.dot(emb[i].astype(np.float32), query)) for i in range(len(emb))]
    )
    assert sims.shape == (23,)
    assert sims.dtype == np.float32
    np.testing.assert_allclose(sims, reference, rtol=0, atol=1e-6)


def test_ranking_matches_the_whole_matrix_cast(small_chunks):
    """Chunking may shift a score by a float32 ULP; it must never reorder results."""
    emb, query = _random_index(101, seed=3)

    chunked = retrieve._dense_scores(emb, query)
    whole = emb.astype(np.float32) @ query

    assert np.array_equal(np.argsort(-chunked), np.argsort(-whole))
    assert np.abs(chunked - whole).max() < 1e-6


def test_single_chunk_is_exact():
    """When the matrix fits one chunk the result is bit-identical to the naive cast."""
    emb, query = _random_index(64, seed=5)
    assert len(emb) < retrieve.DENSE_CHUNK_ROWS

    assert np.array_equal(retrieve._dense_scores(emb, query), emb.astype(np.float32) @ query)


def test_unit_vectors_score_one_against_themselves(small_chunks):
    """Cosine sanity: the encoder L2-normalises, so a passage matched to itself scores 1."""
    emb, _ = _random_index(15, seed=11)
    query = emb[9].astype(np.float32)
    query /= np.linalg.norm(query)

    sims = retrieve._dense_scores(emb, query)

    assert int(np.argmax(sims)) == 9
    assert sims[9] == pytest.approx(1.0, abs=1e-3)
