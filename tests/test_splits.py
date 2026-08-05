"""The frozen splits, and the pool the gold-attribution set is drawn from.

These assert against the **real** `data/splits.json` rather than a fixture, for the same reason
`test_abstention.py` asserts against the real G0 records: the file is a frozen artifact every
number in the paper depends on, and a fixture would only prove the code can round-trip itself.

The load-bearing one is `test_gold_pool_is_disjoint_from_test`. If the gold-attribution questions
came from `test`, C4's verifier-vs-human agreement would be measured on the same questions the
paper reports, and the two would stop being independent evidence. 500 of the 1,000 questions are
unassigned, so the independence costs nothing — this test is what stops it being given away later.
"""

from __future__ import annotations

import pytest

from biomedqa.config import canonical_hash
from biomedqa.data import DEV_N, SPLITS_PATH, TEST_N, gold_pool, load_splits

pytestmark = pytest.mark.skipif(
    not SPLITS_PATH.exists(), reason="splits not frozen yet (data.py freeze_splits)"
)


def test_splits_have_the_sizes_section_3_promises():
    s = load_splits()
    assert len(s["dev"]) == DEV_N
    assert len(s["test"]) == TEST_N


def test_dev_and_test_do_not_intersect():
    s = load_splits()
    assert not set(s["dev"]) & set(s["test"]), "the test set is contaminated"


def test_hash_covers_the_membership():
    """A silently edited split file must not be able to masquerade as the frozen one."""
    s = load_splits()
    recorded = s.pop("hash")
    assert canonical_hash(s) == recorded

    s["test"] = s["test"][:-1]  # drop one question
    assert canonical_hash(s) != recorded


def test_load_splits_rejects_a_tampered_file(tmp_path):
    import json

    payload = load_splits()
    payload["dev"] = payload["dev"][:-1]
    p = tmp_path / "splits.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_splits(p)


def test_gold_pool_is_disjoint_from_test():
    """C4's independence, asserted rather than assumed.

    Skipped under the house invocation (`uv run --with pytest …`), which has no `datasets`. Run it
    with `uv run --with pytest --with datasets python -m pytest tests/test_splits.py` — and do run
    it before W6, because this is the assertion that keeps the gold questions out of `test`.
    """
    pytest.importorskip("datasets", reason="gold_pool() loads all 1,000 pqa_labeled rows")
    s = load_splits()
    pool = gold_pool()
    assert not set(pool) & set(s["test"]), "gold-attribution questions leak into the reported set"
    assert not set(pool) & set(s["dev"])
    assert len(pool) == 1000 - DEV_N - TEST_N == 500
