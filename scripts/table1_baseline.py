#!/usr/bin/env python3
"""Table 1, rows 1–3 — retrieval ablation baseline.

    Row 1: BM25 only  (no dense, no RRF, no rerank)
    Row 2: Dense only (no BM25,  no RRF, no rerank)
    Row 3: BM25 + Dense + RRF  (no rerank)

Each row reports hit@5, Wilson 95% CI, and gold-rank distribution over the dev split.  The cross-
encoder rerank row (row 4) is Week 3.

RUNS ON THE A4000 (dense retrieval needs GPU for the MedCPT query encoder).

    python scripts/table1_baseline.py \\
      --index-dir data/index \\
      --out docs/harvest/table1_rows_1_3.json
"""

from __future__ import annotations

import argparse
import json
import sys
import statistics
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomedqa.config import RetrievalConfig  # noqa: E402
from biomedqa.data import Instance, load_splits, load_instances  # noqa: E402
from biomedqa.retrieve import RetrievalIndex, retrieve  # noqa: E402
from biomedqa.schema import (  # noqa: E402
    QueryRecord,
    RetrievedPassage,
    System,
    SCHEMA_VERSION,
    to_dict,
)
from biomedqa.scoring.retrieval import (  # noqa: E402
    gold_rank,
    hit_at_k,
    wilson_interval,
    gate_g1,
)

# ---------------------------------------------------------------------------
# Row definitions
# ---------------------------------------------------------------------------

# Base config — all features on, top_k=5
_BASE_CONFIG = RetrievalConfig(
    bm25=True,
    dense=True,
    rrf=True,
    rerank=False,   # rerank is Row 4 (Week 3)
    top_k=5,
)

ROWS: list[dict] = [
    {
        "row": 1,
        "label": "BM25 only",
        "config_overrides": {"bm25": True, "dense": False, "rrf": False, "rerank": False},
    },
    {
        "row": 2,
        "label": "Dense only",
        "config_overrides": {"bm25": False, "dense": True, "rrf": False, "rerank": False},
    },
    {
        "row": 3,
        "label": "BM25 + Dense + RRF",
        "config_overrides": {"bm25": True, "dense": True, "rrf": True, "rerank": False},
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _instances_for_split(split_name: str) -> list[Instance]:
    splits = load_splits()
    pubids: set[str] = set(splits[split_name])
    all_instances = load_instances()
    return [i for i in all_instances if i.pubid in pubids]


def _run_retrieval(
    instances: list[Instance],
    config: RetrievalConfig,
    index: RetrievalIndex,
    row_label: str,
) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    for i, inst in enumerate(instances):
        passages = retrieve(inst.question, config, index)
        records.append(
            QueryRecord(
                run_id=f"table1_{row_label.lower().replace(' ', '_').replace('+', 'and')}",
                query_id=inst.pubid,
                question=inst.question,
                system=System.JOINT,
                seed=0,
                retrieved=passages,
                gold_passage_ids=inst.gold_passage_ids,
                schema_version=SCHEMA_VERSION,
            )
        )
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(instances)} …")
    return records


def _gold_rank_stats(records: list[QueryRecord]) -> dict:
    ranks = [gold_rank(r) for r in records]
    retrieved = [r for r in ranks if r is not None]
    not_retrieved = len(ranks) - len(retrieved)

    if not retrieved:
        return {
            "retrieved_count": 0,
            "not_retrieved_count": not_retrieved,
            "rank_mean": None,
            "rank_median": None,
            "rank_min": None,
            "rank_max": None,
            "rank_distribution": {},
        }

    dist: dict[int, int] = {}
    for r in retrieved:
        dist[r] = dist.get(r, 0) + 1

    return {
        "retrieved_count": len(retrieved),
        "not_retrieved_count": not_retrieved,
        "rank_mean": round(statistics.mean(retrieved), 3),
        "rank_median": round(statistics.median(retrieved), 3),
        "rank_min": min(retrieved),
        "rank_max": max(retrieved),
        "rank_distribution": {str(k): v for k, v in sorted(dist.items())},
    }


def _eval_row(
    row_spec: dict,
    index: RetrievalIndex,
    instances: list[Instance],
    k: int,
) -> tuple[dict, list[QueryRecord]]:
    """Evaluate one Table 1 row. Returns the row summary **and its records**.

    The records are returned rather than consumed because the summary is a lossy reduction of
    them: a mean rank cannot be re-thresholded at hit@10 (R2's ladder ends there) and cannot be
    bootstrapped clustered on the question (ADR-0011). The caller persists them.
    """
    label = row_spec["label"]
    config = replace(_BASE_CONFIG, **row_spec["config_overrides"])

    print(f"\n  Row {row_spec['row']}: {label}")
    records = _run_retrieval(instances, config, index, label)

    hits, n = hit_at_k(records, k)
    point, lower, upper = wilson_interval(hits, n)
    g1 = gate_g1(records, k)
    rank_stats = _gold_rank_stats(records)

    row_result = {
        "row": row_spec["row"],
        "label": label,
        "config": {
            "bm25": config.bm25,
            "dense": config.dense,
            "rrf": config.rrf,
            "rerank": config.rerank,
            "rrf_k": config.rrf_k,
            "corpus_id": config.corpus_id,
            "corpus_fingerprint": config.corpus_fingerprint,
            "top_k": config.top_k,
        },
        "n": n,
        "hits": hits,
        f"hit_at_{k}": round(point, 4),
        "wilson_lower": round(lower, 4),
        "wilson_upper": round(upper, 4),
        "g1_passes": g1["passes"],
        # Least-processed value: the per-question rank, keyed so it can be joined back to the
        # question. `gold_rank` below it is a convenience summary, not the record.
        "gold_rank_per_query": [
            {"query_id": r.query_id, "gold_rank": gold_rank(r)} for r in records
        ],
        "gold_rank": rank_stats,
    }

    print(
        f"    hit@{k}={point:.4f}  Wilson [{lower:.4f}, {upper:.4f}]"
        f"  G1={'PASS' if g1['passes'] else 'FAIL'}"
    )
    return row_result, records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Table 1 rows 1–3: BM25-only, Dense-only, BM25+Dense+RRF (no rerank)"
    )
    ap.add_argument(
        "--index-dir",
        required=True,
        type=Path,
        help="Prebuilt index directory (BM25 + dense embeddings)",
    )
    ap.add_argument(
        "--split",
        default="dev",
        choices=["dev", "test"],
        help="Split to evaluate (default: dev; test is run only once per system)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=5,
        help="k for hit@k (default: 5; G1 gates at k=5)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/harvest/table1_rows_1_3.json"),
        help="Output JSON path",
    )
    ap.add_argument(
        "--no-gpu-check",
        action="store_true",
        help="Skip the CUDA guard (dry-run / syntax check only)",
    )
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available() and not args.no_gpu_check:
        print("CUDA not available — dense retrieval needs the A4000 GPU.", file=sys.stderr)
        return 1

    started_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Load instances for the chosen split
    # ------------------------------------------------------------------
    print(f"Loading '{args.split}' split …")
    instances = _instances_for_split(args.split)
    print(f"  {len(instances)} instances")

    # ------------------------------------------------------------------
    # Load index once — shared across all rows
    # ------------------------------------------------------------------
    # Use the most permissive config for loading (BM25 + dense); individual row
    # configs then selectively enable/disable stages at query time
    print(f"\nLoading index from {args.index_dir} …")
    load_config = replace(_BASE_CONFIG, bm25=True, dense=True)
    index = RetrievalIndex.load(args.index_dir, load_config)
    print("  Index loaded")

    # ------------------------------------------------------------------
    # Evaluate each row
    # ------------------------------------------------------------------
    print(f"\nEvaluating Table 1 rows 1–3 on '{args.split}' split (k={args.k}) …")
    row_results: list[dict] = []
    all_records: list[tuple[int, QueryRecord]] = []
    for row_spec in ROWS:
        result, records = _eval_row(row_spec, index, instances, args.k)
        row_results.append(result)
        all_records.extend((row_spec["row"], r) for r in records)

    # ------------------------------------------------------------------
    # Print Table 1 summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Table 1  (dev split, hit@{args.k}):")
    print(f"{'='*60}")
    header = f"{'Row':<4} {'System':<22} {'hit@' + str(args.k):>8} {'Wilson lo':>10} {'Wilson hi':>10} {'G1':>5}"
    print(header)
    print("-" * 60)
    for r in row_results:
        g1_str = "PASS" if r["g1_passes"] else "fail"
        print(
            f"{r['row']:<4} {r['label']:<22} "
            f"{r[f'hit_at_{args.k}']:>8.4f} "
            f"{r['wilson_lower']:>10.4f} "
            f"{r['wilson_upper']:>10.4f} "
            f"{g1_str:>5}"
        )
    print("=" * 60)
    print("(Row 4 — BM25+Dense+RRF+Rerank — is Week 3)")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "script": "scripts/table1_baseline.py",
        "table": "Table 1 rows 1–3",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "index_dir": str(args.index_dir),
            "split": args.split,
            "k": args.k,
            "corpus_id": _BASE_CONFIG.corpus_id,
            "corpus_fingerprint": _BASE_CONFIG.corpus_fingerprint,
        },
        "rows": row_results,
        "note": "Row 4 (cross-encoder rerank) is Week 3. G1 gate requires hit@5 ≥ 0.90 and Wilson lower > 0.85.",
    }
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults written to {args.out}")

    # The ranked lists themselves. Every summary above is recoverable from this file; nothing in
    # this file is recoverable from the summaries. hit@10, a question-clustered bootstrap and
    # "which question missed, and what did it get instead" all need it, and re-deriving it costs
    # another index load and another 3x100 retrievals.
    records_path = args.out.with_suffix(".records.jsonl")
    with records_path.open("w", encoding="utf-8") as fh:
        for row_num, rec in all_records:
            payload = to_dict(rec)
            payload["table1_row"] = row_num
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"Per-query records written to {records_path}  ({len(all_records)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
