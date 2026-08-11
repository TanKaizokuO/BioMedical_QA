"""The collector's three refusals: wrong token, wrong order, destroyed history.

The sidecar is allowed to be unavailable — the form survives that. It is not allowed to hand one
annotator another's pass (ADR-0016 §4), to accept a snapshot from a rebuilt question order (§2),
or to let a later write destroy an earlier one. Those are the tests.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from annotation_collect import Handler, best_snapshot, latest_snapshot, snapshot_paths  # noqa: E402
from biomedqa.annotate import collector_token  # noqa: E402

SEED = 1
ORDER = "abc123"


def state(complete: int = 1) -> dict:
    return {
        "answers": {"u": {"validity": True, "union": "SUPPORTED", "spans": {"0": "PARTIAL"}}},
        "questions": {
            f"q{i}": {"started_at": "t", "completed_at": "t" if i < complete else None,
                      "active_s": 60.0}
            for i in range(2)
        },
        "saved_at": "2026-09-08T10:00:00Z",
    }


def snapshot(annotator: str = "a1", order_hash: str = ORDER, complete: int = 1) -> dict:
    return {"annotator_id": annotator, "order_hash": order_hash, "state": state(complete)}


@pytest.fixture
def collector(tmp_path):
    Handler.root = tmp_path
    Handler.seed = SEED
    Handler.order_hash = ORDER
    Handler.annotators = ("a1", "a2")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown()
    httpd.server_close()


def call(url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "text/plain"})
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def post(base: str, annotator: str, body: dict, token: str | None = None) -> tuple[int, dict]:
    tok = collector_token(annotator, seed=SEED) if token is None else token
    return call(f"{base}/state/{annotator}?token={tok}", body)


def get_restore(base: str, annotator: str, token: str | None = None) -> tuple[int, dict]:
    tok = collector_token(annotator, seed=SEED) if token is None else token
    return call(f"{base}/state/{annotator}/restore?token={tok}")


def test_round_trip_stores_and_returns_the_pass(collector):
    base, root = collector
    assert post(base, "a1", snapshot())[0] == 200
    status, got = get_restore(base, "a1")
    assert status == 200
    assert got["state"]["answers"]["u"]["union"] == "SUPPORTED"
    assert len(snapshot_paths(root, "a1")) == 1


def test_one_annotators_token_cannot_read_another(collector):
    base, _ = collector
    post(base, "a2", snapshot("a2"))
    stolen = collector_token("a1", seed=SEED)
    assert get_restore(base, "a2", token=stolen)[0] == 403
    assert post(base, "a2", snapshot("a2"), token=stolen)[0] == 403
    assert get_restore(base, "a2", token="")[0] == 403


def test_unknown_annotator_is_refused_even_with_a_derived_token(collector):
    base, _ = collector
    assert post(base, "a9", snapshot("a9"), token=collector_token("a9", seed=SEED))[0] == 403


def test_a_rebuilt_order_is_refused_rather_than_stored(collector):
    base, root = collector
    status, body = post(base, "a1", snapshot(order_hash="different"))
    assert status == 409 and "order_hash" in body["error"]
    assert snapshot_paths(root, "a1") == []


def test_writes_are_append_only_and_a_bad_file_cannot_mask_a_good_one(collector):
    base, root = collector
    post(base, "a1", snapshot(complete=1))
    post(base, "a1", snapshot(complete=2))
    paths = snapshot_paths(root, "a1")
    assert len(paths) == 2                                   # nothing overwritten
    assert latest_snapshot(root, "a1")[1]["state"]["questions"]["q1"]["completed_at"] == "t"
    (paths[-1]).write_text("{truncated", encoding="utf-8")   # a write dies mid-flight
    path, recovered = latest_snapshot(root, "a1")
    assert path == paths[0]
    assert recovered["state"]["questions"]["q1"]["completed_at"] is None


def test_malformed_bodies_are_rejected_with_a_reason(collector):
    base, root = collector
    assert call(f"{base}/state/a1?token={collector_token('a1', seed=SEED)}", [1, 2])[0] == 400
    assert post(base, "a1", snapshot("a2"))[0] == 400        # path and payload disagree
    assert snapshot_paths(root, "a1") == []


def test_no_route_lists_or_joins_annotators(collector):
    base, _ = collector
    post(base, "a1", snapshot())
    assert call(f"{base}/state")[0] == 404
    assert call(f"{base}/state/a1")[0] == 404                # no bare-state route exists at all
    status, health = call(f"{base}/health")
    assert status == 200 and "state" not in health


def test_missing_snapshot_is_a_404_not_an_empty_pass(collector):
    base, _ = collector
    assert get_restore(base, "a1")[0] == 404


def test_a_wiped_browser_cannot_erase_the_restore_candidate(collector):
    """The live-run defect: a cleared cache mirrors an empty pass before anyone notices."""
    base, root = collector
    post(base, "a1", snapshot(complete=2))
    post(base, "a1", snapshot(complete=0))          # localStorage.clear(), then one more save
    assert len(snapshot_paths(root, "a1")) == 2
    assert latest_snapshot(root, "a1")[1]["state"]["questions"]["q0"]["completed_at"] is None
    status, offered = get_restore(base, "a1")
    assert status == 200
    assert sum(1 for q in offered["state"]["questions"].values() if q["completed_at"]) == 2
    assert best_snapshot(root, "a1")[1] == offered
