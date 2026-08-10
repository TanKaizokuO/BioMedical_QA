#!/usr/bin/env python3
"""Turn Table 1's recorded ranked lists into the cells the paper prints. CPU-only, no GPU.

`table1_baseline.py` ran the cascade once on the A4000 and stored, per query, the **full 100-deep
ranked list and the gold passage id set** — nothing derived (`docs/harvest/gold-passage-tracking.md`).
Every number below is a re-score of that file, so producing Table 1 at a second `k`, or with a
metric that did not exist when the run happened, costs no GPU. That is exactly what ADR-0015 spent:
G1 moved from k=5 to k=10 without re-running retrieval.

Both k are reported. ADR-0015 §3 relaxed G1's gate to hit@10 and requires the failing k=5 reading
to be printed beside it, in Table 1 and in the gate row, so the relaxation is legible as a
relaxation. The thresholds — 0.90 point, 0.85 Wilson lower — are untouched.

    uv run python scripts/table1_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.schema import QueryRecord, query_record_from_dict  # noqa: E402
from biomedqa.scoring.retrieval import (  # noqa: E402
    gate_g1,
    hit_at_k,
    mrr,
    ndcg,
    recall_at_k,
    wilson_interval,
)

#: G1 as written. Only `k` ever moved, once, on the record (ADR-0015 §3).
G1_POINT = 0.90
G1_WILSON_LOWER = 0.85

#: The gate's original k and the relaxed one. Table 1 prints both columns.
GATE_K = 5
RELAXED_K = 10


def load_rows(path: Path) -> dict[int, list[QueryRecord]]:
    """Group the records file by its `table1_row` tag, which is not part of the schema.

    Iterating the file splits on `\\n` alone. `str.splitlines` also breaks on the vertical tabs and
    U+2028 that PubMed abstract text carries, which tears a record in half mid-passage.
    """
    rows: dict[int, list[QueryRecord]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            row = raw.pop("table1_row")
            rows.setdefault(int(row), []).append(query_record_from_dict(raw))
    return rows


def score(records: list[QueryRecord]) -> dict:
    """Every Table 1 cell for one system, plus the G1 reading at both k."""
    cells = {}
    for k in (GATE_K, RELAXED_K):
        hits, n = hit_at_k(records, k)
        point, lower, upper = wilson_interval(hits, n)
        cells[f"hit_at_{k}"] = {
            "hits": hits,
            "n": n,
            "point": point,
            "wilson_lower": lower,
            "wilson_upper": upper,
            "clears_g1": point >= G1_POINT and lower > G1_WILSON_LOWER,
        }
    cells["recall_at_5"] = recall_at_k(records, 5)
    cells["mrr"] = mrr(records)
    cells["ndcg_at_10"] = ndcg(records, 10)
    return cells


def _ci(cell: dict) -> str:
    return f"[{cell['wilson_lower']:.2f}, {cell['wilson_upper']:.2f}]"


def markdown(rows: list[dict]) -> str:
    """The table body, in the column order `paper/skeleton.md` already committed to."""
    lines = [
        "| System | hit@5 | 95% CI (Wilson) | hit@10 | 95% CI (Wilson) | recall@5 | MRR | nDCG@10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        label = f"**{r['label']}**" if r["row"] == 4 else r["label"]
        c = r["cells"]
        lines.append(
            f"| {label} | {c['hit_at_5']['point']:.2f} | {_ci(c['hit_at_5'])} | "
            f"{c['hit_at_10']['point']:.2f} | {_ci(c['hit_at_10'])} | "
            f"{c['recall_at_5']:.2f} | {c['mrr']:.2f} | {c['ndcg_at_10']:.2f} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Score Table 1 from its recorded ranked lists (CPU)")
    ap.add_argument(
        "--records", type=Path, default=Path("docs/harvest/table1_rows_1_4.records.jsonl")
    )
    ap.add_argument("--run", type=Path, default=Path("docs/harvest/table1_rows_1_4.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/harvest/table1_metrics.json"))
    args = ap.parse_args()

    source = json.loads(args.run.read_text(encoding="utf-8"))
    labels = {r["row"]: r["label"] for r in source["rows"]}
    grouped = load_rows(args.records)
    missing = set(labels) - set(grouped)
    if missing:
        print(f"{args.records} has no records for row(s) {sorted(missing)}", file=sys.stderr)
        return 1

    rows = [
        {"row": row, "label": labels[row], "cells": score(grouped[row])} for row in sorted(grouped)
    ]
    table = markdown(rows)
    print(table)

    full = grouped[max(grouped)]
    gate = {str(k): gate_g1(full, k) for k in (GATE_K, RELAXED_K)}
    print(
        f"\nG1 on row {max(grouped)} — "
        f"k={GATE_K}: {gate[str(GATE_K)]['hit_at_k']:.4f} / "
        f"{gate[str(GATE_K)]['wilson_lower']:.4f} "
        f"{'passes' if gate[str(GATE_K)]['passes'] else 'FAILS'}   ·   "
        f"k={RELAXED_K}: {gate[str(RELAXED_K)]['hit_at_k']:.4f} / "
        f"{gate[str(RELAXED_K)]['wilson_lower']:.4f} "
        f"{'passes' if gate[str(RELAXED_K)]['passes'] else 'FAILS'}"
    )

    report = {
        "script": "scripts/table1_report.py",
        "table": "Table 1 — retrieval cascade (C1)",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "records_source": str(args.records),
        "run_source": str(args.run),
        "config": source["config"],
        "gate": {
            "definition": "point >= 0.90 and Wilson lower > 0.85; only k moved (ADR-0015 §3)",
            "original_k": GATE_K,
            "gated_at_k": RELAXED_K,
            "row_4": gate,
        },
        "rows": rows,
        "markdown": table,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
