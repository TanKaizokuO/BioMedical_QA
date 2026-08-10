#!/usr/bin/env python3
"""Table 1, rows 1–4 — retrieval ablation baseline.

    Row 1: BM25 only  (no dense, no RRF, no rerank)
    Row 2: Dense only (no BM25,  no RRF, no rerank)
    Row 3: BM25 + Dense + RRF  (no rerank)
    Row 4: BM25 + Dense + RRF + cross-encoder rerank  (Gate G1 reads this row)

Each row reports hit@5, Wilson 95% CI, and gold-rank distribution over the dev split.

RUNS ON THE A4000 (dense retrieval and the cross-encoder need GPU).

    uv run python scripts/table1_baseline.py \\
      --index-dir data/index \\
      --out docs/harvest/table1_rows_1_4.json

``--rows`` restricts the run to a subset (e.g. ``--rows 4``); the artifact then holds only those
rows, so keep the default when producing the Table 1 of record.
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

from biomedqa.config import CONFIG_VERSION, RetrievalConfig, RunConfig  # noqa: E402
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
    rerank=False,   # rows 1–3 are pre-rerank; row 4 turns it on via config_overrides
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
    {
        "row": 4,
        "label": "BM25 + Dense + RRF + Rerank",
        "config_overrides": {"bm25": True, "dense": True, "rrf": True, "rerank": True},
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
    gate_k: int,
) -> list[QueryRecord]:
    """Retrieve to the full pool depth, not to ``top_k``.

    ``retrieve()`` builds a ``pool_size``-deep ranking and then returns ``pool[:top_k]``. Recording
    that truncated list censors ``gold_rank`` at ``top_k``: "not retrieved" then means "not in the
    top 5" and cannot distinguish rank 6 from rank 2,000,000 — which are opposite diagnoses, and
    the difference between a rerank that can rescue the query and one that never sees the passage.
    It also makes R2's hit@10 ladder unanswerable without re-running the GPU.

    The deep pool is already computed, so asking for it costs nothing. Passage *text* is dropped
    beyond ``gate_k`` — it is reproducible from ``passage_id`` plus the index, and keeping all 100
    would put tens of megabytes of duplicated abstracts in a public repo.
    """
    deep_config = replace(config, top_k=config.pool_size)

    records: list[QueryRecord] = []
    for i, inst in enumerate(instances):
        passages = retrieve(inst.question, deep_config, index)
        passages = [
            p if p.rank <= gate_k else replace(p, text=None)
            for p in passages
        ]
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


#: hit@k is reported across the whole curve. All of these come free from one deep pool, and the
#: gate reading is worthless without them: hit@5 alone cannot say whether a miss is a near miss.
HIT_AT_K_CURVE = (1, 5, 10, 20, 50, 100)


def _gold_rank_stats(records: list[QueryRecord], pool_size: int) -> dict:
    """Rank distribution over the *pool*, so "not retrieved" means "not in the pool at all"."""
    ranks = [gold_rank(r) for r in records]
    retrieved = [r for r in ranks if r is not None]
    not_in_pool = len(ranks) - len(retrieved)

    stats = {
        "pool_size": pool_size,
        "in_pool_count": len(retrieved),
        "not_in_pool_count": not_in_pool,
        "rank_mean": None,
        "rank_median": None,
        "rank_min": None,
        "rank_max": None,
        "rank_histogram": {},
    }
    if not retrieved:
        return stats

    buckets = [("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-20", 11, 20),
               ("21-50", 21, 50), ("51-100", 51, 100)]
    stats.update({
        "in_pool_count": len(retrieved),
        "rank_mean": round(statistics.mean(retrieved), 3),
        "rank_median": round(statistics.median(retrieved), 3),
        "rank_min": min(retrieved),
        "rank_max": max(retrieved),
        "rank_histogram": {
            name: sum(1 for r in retrieved if lo <= r <= hi) for name, lo, hi in buckets
        },
    })
    return stats


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
    records = _run_retrieval(instances, config, index, label, k)

    hits, n = hit_at_k(records, k)
    point, lower, upper = wilson_interval(hits, n)
    g1 = gate_g1(records, k)
    rank_stats = _gold_rank_stats(records, config.pool_size)

    curve = {}
    for kk in HIT_AT_K_CURVE:
        if kk > config.pool_size:
            continue
        h, nn = hit_at_k(records, kk)
        p, lo, hi = wilson_interval(h, nn)
        curve[f"hit_at_{kk}"] = {
            "point": round(p, 4), "wilson_lower": round(lo, 4), "wilson_upper": round(hi, 4),
            "hits": h,
        }

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
        "hit_at_k_curve": curve,
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
    print(
        "    gold rank in pool: "
        + "  ".join(f"{name}={cnt}" for name, cnt in rank_stats["rank_histogram"].items())
        + f"  not_in_pool={rank_stats['not_in_pool_count']}"
    )
    print(
        "    hit@k curve: "
        + "  ".join(f"@{kk}={curve[f'hit_at_{kk}']['point']:.2f}" for kk in HIT_AT_K_CURVE
                    if f"hit_at_{kk}" in curve)
    )
    return row_result, records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Table 1 rows 1–4: BM25, Dense, RRF fusion, cross-encoder rerank"
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
        "--rows",
        type=int,
        nargs="+",
        choices=[r["row"] for r in ROWS],
        default=[r["row"] for r in ROWS],
        help="Subset of Table 1 rows to evaluate (default: all)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/harvest/table1_rows_1_4.json"),
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
    selected = [r for r in ROWS if r["row"] in set(args.rows)]
    print(
        f"\nEvaluating Table 1 rows {', '.join(str(r['row']) for r in selected)} "
        f"on '{args.split}' split (k={args.k}) …"
    )
    row_results: list[dict] = []
    all_records: list[tuple[int, QueryRecord]] = []
    for row_spec in selected:
        result, records = _eval_row(row_spec, index, instances, args.k)
        row_results.append(result)
        all_records.extend((row_spec["row"], r) for r in records)

    # ------------------------------------------------------------------
    # Print Table 1 summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Table 1  (dev split, hit@{args.k}):")
    print(f"{'='*60}")
    header = f"{'Row':<4} {'System':<28} {'hit@' + str(args.k):>8} {'Wilson lo':>10} {'Wilson hi':>10} {'G1':>5}"
    print(header)
    print("-" * 60)
    for r in row_results:
        g1_str = "PASS" if r["g1_passes"] else "fail"
        print(
            f"{r['row']:<4} {r['label']:<28} "
            f"{r[f'hit_at_{args.k}']:>8.4f} "
            f"{r['wilson_lower']:>10.4f} "
            f"{r['wilson_upper']:>10.4f} "
            f"{g1_str:>5}"
        )
    print("=" * 60)
    print("G1 reads row 4: hit@5 ≥ 0.90 AND Wilson lower > 0.85.")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "script": "scripts/table1_baseline.py",
        "table": "Table 1 rows " + ", ".join(str(r["row"]) for r in row_results),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "index_dir": str(args.index_dir),
            "split": args.split,
            "k": args.k,
            "corpus_id": _BASE_CONFIG.corpus_id,
            "corpus_fingerprint": _BASE_CONFIG.corpus_fingerprint,
            "title_segment": _BASE_CONFIG.title_segment,
            "reranker": _BASE_CONFIG.reranker,
            "pool_size": _BASE_CONFIG.pool_size,
            # The index's identity, not the run's. Without it a row cannot be attributed to an
            # index after the fact, and ADR-0014 §3 made `title_segment` part of that identity.
            "index_fingerprint": RunConfig(
                retrieval=_BASE_CONFIG, split=args.split
            ).index_fingerprint(),
            "config_version": CONFIG_VERSION,
        },
        "rows": row_results,
        "note": "G1 gate requires hit@5 ≥ 0.90 and Wilson lower > 0.85, read off row 4.",
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
