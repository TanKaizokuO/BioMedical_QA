#!/usr/bin/env python3
"""Burn-down over the three annotators' backed-up passes — counts only, never labels.

Reads the append-only snapshots written by `annotation_collect.py` and prints, per annotator:
questions complete out of the gold set, claims labelled, time actually spent in the form, and the
hours that rate projects to over the whole set.

    uv run python scripts/annotation_status.py --out annotation

The projection is the pilot's answer to "is 10–16 h real?", and it is honest only because the
form measures active time per question rather than wall-clock. It is a number to read, not a
gate — no threshold here decides anything.

Deliberately absent: any view of the labels themselves. Comparing two unfinished passes before
Krippendorff's α is exactly what ADR-0016 §4 rules out, and the keyfile is not on this box.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.annotate import snapshot_summary  # noqa: E402

sys.path.insert(0, str(_REPO / "scripts"))

from annotation_collect import best_snapshot, latest_snapshot, snapshot_paths  # noqa: E402


def total_questions(out: Path, annotators: list[str]) -> int:
    """How long the gold set is, taken from a built form rather than assumed."""
    for annotator in annotators:
        form = out / f"annotate_{annotator}.html"
        if not form.exists():
            continue
        match = re.search(r'<script id="tasks" type="application/json">(.*?)</script>',
                          form.read_text(encoding="utf-8"), re.S)
        if match:
            return len(json.loads(match.group(1).replace("<\\/", "</"))["questions"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_REPO / "annotation")
    ap.add_argument("--annotators", nargs="+", default=["a1", "a2", "a3"])
    args = ap.parse_args()

    total = total_questions(args.out, args.annotators)
    if not total:
        print(f"no built forms in {args.out} — nothing to measure against", file=sys.stderr)
        return 1

    print(f"gold set      {total} questions")
    print(f"{'ann':<5}{'complete':>12}{'claims':>9}{'active':>10}{'projected':>12}   last backup")
    hashes: dict[str, set[str]] = {}
    for annotator in args.annotators:
        # Counts come from the furthest-along pass; the clock comes from the newest write, so a
        # browser that was wiped this morning still shows this morning as its last contact.
        found = best_snapshot(args.out, annotator)
        newest = latest_snapshot(args.out, annotator)
        if found is None:
            print(f"{annotator:<5}{'—':>12}{'—':>9}{'—':>10}{'—':>12}   never")
            continue
        path, snap = found
        s = snapshot_summary(snap.get("state") or {}, total)
        hashes.setdefault(snap.get("order_hash") or "?", set()).add(annotator)
        kept = len(snapshot_paths(args.out, annotator))
        seen = (newest[1].get("saved_at") if newest else None) or path.stem
        print(
            f"{annotator:<5}{s['questions_complete']:>7}/{total:<4}{s['claims_labeled']:>9}"
            f"{s['active_s'] / 3600.0:>9.1f}h{s['projected_h']:>11.1f}h"
            f"   {seen}  ({kept} snapshots)"
        )

    if len(hashes) > 1:
        print("\nORDER MISMATCH — passes are not comparable:", file=sys.stderr)
        for h, who in hashes.items():
            print(f"  {h}  {', '.join(sorted(who))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
