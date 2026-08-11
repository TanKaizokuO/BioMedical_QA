#!/usr/bin/env python3
"""Append-only backup collector for the three annotators' forms (ADR-0016 §4, Q6–Q9).

Runs on the A4000 box beside the vLLM server, on a different port. Every save in an annotator's
form POSTs a snapshot here; the form keeps `localStorage` as its primary store, so this process
dying costs progress *visibility*, never labels.

Three properties are the point, and each is a refusal:

* **Append-only.** Every POST becomes a new file under `state/<annotator>/`. A corrupt or
  truncated snapshot can therefore never destroy a good earlier one, which is the failure this
  exists to prevent.
* **Write-mostly.** `GET /state/<a>/restore` returns the furthest-along pass stored for *that*
  annotator and no other, and
  the token needed to reach it is baked into that annotator's form alone. No endpoint lists
  labels across annotators, because reading one pass while another is unfinished is what §4
  forbids. This is a convenience, not a lock: anyone who can read this directory reads everything.
* **Order-hash gated.** A snapshot whose `order_hash` disagrees with the forms is refused with
  409 rather than stored, so a rebuilt form is caught at the first save instead of after α.

The keyfile never comes here. Start it (no service manager; start it again by hand after a
reboot):

    uv run python scripts/annotation_collect.py --out annotation --port 8811

stdlib only — `pyproject.toml` gains nothing for this.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.annotate import ANNOTATION_SEED, collector_token  # noqa: E402

MAX_BODY = 8 << 20  # a full 250-claim pass is ~1 MB of JSON; 8 MB is a generous ceiling
_SAFE_ID = re.compile(r"^[a-z0-9_-]{1,32}$")


def state_dir(root: Path, annotator: str) -> Path:
    return root / "state" / annotator


def snapshot_paths(root: Path, annotator: str) -> list[Path]:
    """Stored snapshots for one annotator, oldest first (filenames are UTC timestamps)."""
    d = state_dir(root, annotator)
    return sorted(d.glob("*.json")) if d.is_dir() else []


def latest_snapshot(root: Path, annotator: str) -> tuple[Path, dict] | None:
    """The newest snapshot that still parses. A half-written file must not mask a good one."""
    for path in reversed(snapshot_paths(root, annotator)):
        try:
            return path, json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _completed(snapshot: dict) -> int:
    questions = (snapshot.get("state") or {}).get("questions") or {}
    return sum(1 for q in questions.values() if q.get("completed_at"))


def best_snapshot(root: Path, annotator: str) -> tuple[Path, dict] | None:
    """The *furthest-along* stored pass — what a restore should offer, not merely the newest.

    A cleared browser still mirrors one more time before anyone notices, and that snapshot is
    empty. Offering the newest file would hand the annotator back the very loss they came to
    undo. Completion only ever moves forward within a pass, so "most questions complete, newest
    wins ties" identifies the copy worth keeping without merging anything.
    """
    best: tuple[Path, dict] | None = None
    for path in snapshot_paths(root, annotator):
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if best is None or _completed(snapshot) >= _completed(best[1]):
            best = (path, snapshot)
    return best


def store_snapshot(root: Path, annotator: str, snapshot: dict) -> Path:
    d = state_dir(root, annotator)
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = d / f"{stamp}.json"
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(snapshot, sort_keys=True), encoding="utf-8")
    tmp.rename(path)  # atomic: a reader never sees a partial file under *.json
    return path


class Handler(BaseHTTPRequestHandler):
    server_version = "biomedqa-annotation-collector/1"
    root: Path
    seed: int
    order_hash: str | None
    annotators: tuple[str, ...]

    # -- plumbing ---------------------------------------------------------------------------

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # The forms are opened from file://, so their Origin is "null"; without this the
        # restore GET cannot read the response.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self, annotator: str) -> bool:
        if not _SAFE_ID.match(annotator) or annotator not in self.annotators:
            return False
        want = collector_token(annotator, seed=self.seed)
        got = parse_qs(urlparse(self.path).query).get("token", [""])[0]
        return got == want

    def log_message(self, fmt: str, *args) -> None:  # pragma: no cover - stderr shape only
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    # -- routes -----------------------------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        if path == "/health":
            self._send(200, {"ok": True, "annotators": list(self.annotators)})
            return
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "state" and parts[2] == "restore":
            annotator = parts[1]
            if not self._authorised(annotator):
                self._send(403, {"error": "bad or missing token"})
                return
            found = best_snapshot(self.root, annotator)
            if found is None:
                self._send(404, {"error": "no snapshot stored"})
                return
            self._send(200, found[1])
            return
        self._send(404, {"error": "no such route"})

    def do_POST(self) -> None:  # noqa: N802
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "state":
            self._send(404, {"error": "no such route"})
            return
        annotator = parts[1]
        if not self._authorised(annotator):
            self._send(403, {"error": "bad or missing token"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self._send(413, {"error": f"body must be 1..{MAX_BODY} bytes"})
            return
        try:
            snapshot = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": f"not JSON: {exc}"})
            return
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("state"), dict):
            self._send(400, {"error": "expected {annotator_id, order_hash, state}"})
            return
        if snapshot.get("annotator_id") != annotator:
            self._send(400, {"error": "annotator_id does not match the path"})
            return
        if self.order_hash and snapshot.get("order_hash") != self.order_hash:
            # Refuse rather than store: mixing orders silently invalidates §2.
            self._send(409, {"error": "order_hash does not match the built forms"})
            return
        path = store_snapshot(self.root, annotator, snapshot)
        questions = snapshot["state"].get("questions") or {}
        complete = sum(1 for q in questions.values() if q.get("completed_at"))
        self._send(200, {"ok": True, "stored": path.name, "complete": complete})


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=_REPO / "annotation", help="build/state directory")
    ap.add_argument("--host", default="0.0.0.0", help="bind address (LAN-wide by default)")
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--annotators", nargs="+", default=["a1", "a2", "a3"])
    ap.add_argument("--seed", type=int, default=ANNOTATION_SEED)
    ap.add_argument(
        "--order-hash",
        default=None,
        help="reject snapshots from a different question order; read from a form if omitted",
    )
    args = ap.parse_args()

    order_hash = args.order_hash
    if order_hash is None:
        form = args.out / f"annotate_{args.annotators[0]}.html"
        if form.exists():
            payload = form.read_text(encoding="utf-8")
            match = re.search(r'"order_hash":\s*"([0-9a-f]+)"', payload)
            order_hash = match.group(1) if match else None
    if not order_hash:
        print("warning: no order hash known — snapshots will not be order-gated", file=sys.stderr)

    Handler.root = args.out
    Handler.seed = args.seed
    Handler.order_hash = order_hash
    Handler.annotators = tuple(args.annotators)

    (args.out / "state").mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"collector on http://{args.host}:{args.port}  ->  {args.out / 'state'}")
    print(f"order hash  {order_hash or '(ungated)'}")
    print("annotators  " + ", ".join(args.annotators))
    print("append-only; ^C to stop. The keyfile does not belong on this machine.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
