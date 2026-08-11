#!/usr/bin/env python3
"""Build the three annotators' labelling forms and the keyfile that de-blinds them.

One records JSONL in; three self-contained HTML files plus one `keyfile.jsonl` out. The forms
are the *only* artifact the annotators receive, and they contain no system, model or run
identity (ADR-0016 §4) — the keyfile is what joins a `unit_id` back to `(system, run_id,
query_id, claim_id)`, and it stays with the maintainer.

All three forms are generated from a single `build_tasks()` call, so the shared seeded question
order (ADR-0016 §2) is shared by construction rather than by discipline. The `order_hash` is
printed and embedded in every form and in every exported row, so a mismatched pass is caught on
read rather than after α is computed.

    uv run python scripts/build_annotation_ui.py --records docs/harvest/generate_smoke.records.jsonl

Overwriting forms after annotation has begun is refused: a rebuilt form with a different order
silently invalidates §2, and `localStorage` progress is keyed by the order hash. Pass
`--force` only when nobody has started.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.annotate import (  # noqa: E402
    ANNOTATION_SEED,
    build_tasks,
    render_form,
    tasks_to_payload,
)
from biomedqa.schema import read_query_records  # noqa: E402

ANNOTATORS = ("a1", "a2", "a3")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, required=True, help="records JSONL to draw claims from")
    ap.add_argument("--out", type=Path, default=_REPO / "annotation", help="output directory")
    ap.add_argument("--annotators", nargs="+", default=list(ANNOTATORS))
    ap.add_argument("--seed", type=int, default=ANNOTATION_SEED)
    ap.add_argument("--force", action="store_true", help="overwrite existing forms")
    args = ap.parse_args()

    records = list(read_query_records(args.records))
    tasks, keyfile = build_tasks(records, seed=args.seed)
    if not tasks:
        print("no claims with citations in this records file — nothing to annotate", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    forms = {a: args.out / f"annotate_{a}.html" for a in args.annotators}
    existing = [p for p in forms.values() if p.exists()]
    if existing and not args.force:
        print(f"refusing to overwrite {len(existing)} existing form(s); pass --force", file=sys.stderr)
        return 1

    order_hash = tasks_to_payload(tasks, "-", seed=args.seed)["order_hash"]
    for annotator, path in forms.items():
        html = render_form(tasks, annotator, seed=args.seed)
        if order_hash not in html:
            raise SystemExit(f"form for {annotator} does not carry the shared order hash")
        path.write_text(html, encoding="utf-8")

    key_path = args.out / "keyfile.jsonl"
    with open(key_path, "w", encoding="utf-8") as fh:
        for row in keyfile:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    claims = sum(len(t.claims) for t in tasks)
    spans = sum(len(c.spans) for t in tasks for c in t.claims)
    print(f"questions     {len(tasks)}")
    print(f"claims        {claims}  ({claims / len(tasks):.2f} per question)")
    print(f"span labels   {spans}  + {claims} union judgements, per annotator")
    print(f"order hash    {order_hash}  (identical in all {len(forms)} forms)")
    for annotator, path in forms.items():
        print(f"  {annotator}  {path}")
    print(f"keyfile       {key_path}  — do not send this to annotators")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
