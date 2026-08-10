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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.config import RunConfig  # noqa: E402
from biomedqa.schema import QueryRecord, query_record_from_dict  # noqa: E402
from biomedqa.scoring.retrieval import gate_g1, mrr, ndcg, recall_at_k  # noqa: E402

#: The k the gate was written at, and the k it is gated at. Only `k` ever moved, once, on the
#: record (ADR-0015 §3); the thresholds live in `retrieval.gate_g1` and are not restated here.
ORIGINAL_K = 5
GATE_K = 10

#: G1 is read on the full cascade and on nothing else (ADR-0015 §3). Scoring whichever row happens
#: to be last in the file would print a gate verdict computed on RRF-only if row 4 were absent.
CASCADE_ROW = 4


def chunk_config_behind(index_fingerprint: str) -> dict | None:
    """The `(chunker, τ)` pair the records were scored under, or `None` if it cannot be named.

    Issue #2: "hit@5 is only defined per `(chunker, τ)` pair — report per pair, never
    marginalised." The records file carries a fingerprint, not the pair, and the fingerprint is a
    one-way hash of it. So the pair is recovered by candidate: `RunConfig()`'s chunker is hashed
    and returned only if it reproduces the recorded fingerprint. A pair that does not hash to the
    index the numbers came from is a caption that lies, which is worse than no caption.
    """
    candidate = RunConfig()
    if candidate.index_fingerprint() != index_fingerprint:
        return None
    return asdict(candidate.chunk)


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
    """Every Table 1 cell for one system, plus the G1 reading at both k.

    The pass/fail comes from `gate_g1`, which owns the 0.90 / 0.85 arithmetic. Re-spelling it here
    would let a threshold edit in `retrieval.py` leave these cells reading the old gate.
    """
    cells: dict = {f"hit_at_{k}": gate_g1(records, k) for k in (ORIGINAL_K, GATE_K)}
    cells["recall_at_5"] = recall_at_k(records, 5)
    cells["mrr"] = mrr(records)
    cells["ndcg_at_10"] = ndcg(records, 10)
    return cells


def _ci(cell: dict) -> str:
    return f"[{cell['wilson_lower']:.2f}, {cell['wilson_upper']:.2f}]"


def markdown(rows: list[dict], cascade_row: int) -> str:
    """The table body, in the column order `paper/skeleton.md` already committed to.

    `cascade_row` is bolded and is the row G1 is read on — one notion, passed in, so the table
    cannot bold one row while the gate block scores another.
    """
    lines = [
        "| System | hit@5 | 95% CI (Wilson) | hit@10 | 95% CI (Wilson) | recall@5 | MRR | nDCG@10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        label = f"**{r['label']}**" if r["row"] == cascade_row else r["label"]
        c = r["cells"]
        lines.append(
            f"| {label} | {c['hit_at_5']['hit_at_k']:.2f} | {_ci(c['hit_at_5'])} | "
            f"{c['hit_at_10']['hit_at_k']:.2f} | {_ci(c['hit_at_10'])} | "
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
    if CASCADE_ROW not in grouped:
        print(
            f"{args.records} carries no row {CASCADE_ROW}; G1 is read on the full cascade and on "
            "nothing else (ADR-0015 §3)",
            file=sys.stderr,
        )
        return 1

    recorded_fingerprint = source["config"]["index_fingerprint"]
    chunk = chunk_config_behind(recorded_fingerprint)
    if chunk is None:
        print(
            f"index_fingerprint {recorded_fingerprint} is not the one RunConfig() describes, so "
            "this script cannot say which (chunker, τ) produced these records",
            file=sys.stderr,
        )
        return 1

    rows = [
        {"row": row, "label": labels[row], "cells": score(grouped[row])} for row in sorted(grouped)
    ]
    table = markdown(rows, CASCADE_ROW)
    print(table)

    gate = {str(k): gate_g1(grouped[CASCADE_ROW], k) for k in (ORIGINAL_K, GATE_K)}
    original, gated = gate[str(ORIGINAL_K)], gate[str(GATE_K)]
    print(
        f"\nG1 on row {CASCADE_ROW} ({labels[CASCADE_ROW]}) — "
        f"k={ORIGINAL_K}: {original['hit_at_k']:.4f} / {original['wilson_lower']:.4f} "
        f"{'passes' if original['passes'] else 'FAILS'}   ·   "
        f"k={GATE_K}: {gated['hit_at_k']:.4f} / {gated['wilson_lower']:.4f} "
        f"{'passes' if gated['passes'] else 'FAILS'}"
    )

    report = {
        "script": "scripts/table1_report.py",
        "table": "Table 1 — retrieval cascade (C1)",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "records_source": str(args.records),
        "run_source": str(args.run),
        # hit@5 is only defined per (chunker, τ) pair (issue #2), so the pair travels with the
        # numbers. It is not asserted: `chunk_config_behind` returns it only when it hashes to the
        # fingerprint the records were scored under.
        "config": {**source["config"], "chunk": chunk},
        "gate": {
            "definition": "point >= 0.90 and Wilson lower > 0.85; only k moved (ADR-0015 §3)",
            "computed_by": "biomedqa.scoring.retrieval.gate_g1",
            "original_k": ORIGINAL_K,
            "gated_at_k": GATE_K,
            "row": CASCADE_ROW,
            "readings": gate,
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
