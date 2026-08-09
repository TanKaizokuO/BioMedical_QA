#!/usr/bin/env python3
"""ADR-0014 §3 — Empty title-segment convention comparison.

Measures dev hit@5 under both conventions for how MedCPT-Article-Encoder is called when passages
carry no titles:

    empty  —  tok("", abstract)   — two segments, empty title
    single —  tok(abstract)        — one segment, no title slot

The convention affects how passages were *indexed* (article encoder), so this requires two
separately-built indices.  Query encoding is identical for both.

RUNS ON THE A4000.

    python scripts/title_convention_eval.py \\
      --index-dir-empty  data/index-empty \\
      --index-dir-single data/index-single \\
      --out docs/harvest/title_convention_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomedqa.config import RetrievalConfig  # noqa: E402
from biomedqa.data import Instance, load_splits, load_instances  # noqa: E402
from biomedqa.retrieve import RetrievalIndex, retrieve  # noqa: E402
from biomedqa.schema import QueryRecord, RetrievedPassage, System, SCHEMA_VERSION  # noqa: E402
from biomedqa.scoring.retrieval import hit_at_k, wilson_interval  # noqa: E402


def _instances_for_split(split_name: str) -> list[Instance]:
    splits = load_splits()
    pubids: set[str] = set(splits[split_name])
    all_instances = load_instances()
    return [i for i in all_instances if i.pubid in pubids]


def _make_query_records(
    instances: list[Instance],
    config: RetrievalConfig,
    index: RetrievalIndex,
    convention_label: str,
) -> list[QueryRecord]:
    """Run retrieval for every instance and wrap results in QueryRecord stubs."""
    records: list[QueryRecord] = []
    for i, inst in enumerate(instances):
        passages = retrieve(inst.question, config, index)
        records.append(
            QueryRecord(
                run_id=f"title_conv_eval_{convention_label}",
                query_id=inst.pubid,
                question=inst.question,
                system=System.JOINT,   # system field is irrelevant for retrieval eval
                seed=0,
                retrieved=passages,
                gold_passage_ids=inst.gold_passage_ids,
                schema_version=SCHEMA_VERSION,
            )
        )
        if (i + 1) % 10 == 0:
            print(f"  [{convention_label}] {i + 1}/{len(instances)} …")
    return records


def _eval_convention(
    label: str,
    index_dir: Path,
    instances: list[Instance],
    config: RetrievalConfig,
    k: int,
) -> dict:
    """Load index, retrieve, compute hit@k + Wilson CI."""
    print(f"\nLoading '{label}' index from {index_dir} …")
    index = RetrievalIndex.load(index_dir, config)
    print(f"Running retrieval for {len(instances)} dev questions [{label}] …")
    records = _make_query_records(instances, config, index, label)

    hits, n = hit_at_k(records, k)
    point, lower, upper = wilson_interval(hits, n)

    from biomedqa.scoring.retrieval import gold_rank  # local import to avoid polluting top-level
    gold_ranks: list[int | None] = [gold_rank(r) for r in records]

    retrieved_ranks = [r for r in gold_ranks if r is not None]
    return {
        "convention": label,
        "index_dir": str(index_dir),
        "n": n,
        "hits": hits,
        f"hit_at_{k}": round(point, 4),
        "wilson_lower": round(lower, 4),
        "wilson_upper": round(upper, 4),
        "gold_retrieved_count": len(retrieved_ranks),
        "gold_not_retrieved_count": n - len(retrieved_ranks),
        # Both conventions run the SAME 100 questions, so the comparison that answers ADR-0014 §3
        # is paired — which questions moved, and in which direction. A mean of each side cannot
        # support that test, and the run costs two index builds to reproduce. Keyed by query_id
        # so the two sides join.
        "gold_rank_per_query": [
            {"query_id": r.query_id, "gold_rank": gr} for r, gr in zip(records, gold_ranks)
        ],
        "gold_rank_mean": round(sum(retrieved_ranks) / len(retrieved_ranks), 2) if retrieved_ranks else None,
        "gold_rank_min": min(retrieved_ranks) if retrieved_ranks else None,
        "gold_rank_max": max(retrieved_ranks) if retrieved_ranks else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ADR-0014 §3: compare empty-vs-single title segment convention on dev hit@5"
    )
    ap.add_argument(
        "--index-dir-empty",
        required=True,
        type=Path,
        help="Index built with tok('', abstract) — empty title segment",
    )
    ap.add_argument(
        "--index-dir-single",
        required=True,
        type=Path,
        help="Index built with tok(abstract) — single segment, no title slot",
    )
    ap.add_argument(
        "--split",
        default="dev",
        choices=["dev", "test"],
        help="Split to evaluate on (default: dev)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=5,
        help="k for hit@k (default: 5)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/harvest/title_convention_eval.json"),
        help="Output JSON path",
    )
    ap.add_argument(
        "--no-gpu-check",
        action="store_true",
        help="Skip the CUDA guard (dry-run / syntax check)",
    )
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available() and not args.no_gpu_check:
        print("CUDA not available — this script must run on the A4000.", file=sys.stderr)
        return 1

    started_at = datetime.now(timezone.utc).isoformat()

    # Both conventions use the same retrieval pipeline; only the index differs
    config = RetrievalConfig(
        bm25=True,
        dense=True,
        rrf=True,
        rerank=False,
        top_k=args.k,
    )

    print(f"Loading '{args.split}' split instances …")
    instances = _instances_for_split(args.split)
    print(f"  {len(instances)} instances loaded")

    result_empty = _eval_convention("empty", args.index_dir_empty, instances, config, args.k)
    result_single = _eval_convention("single", args.index_dir_single, instances, config, args.k)

    # ------------------------------------------------------------------
    # Decision aid
    # ------------------------------------------------------------------
    winner_key = f"hit_at_{args.k}"
    if result_empty[winner_key] > result_single[winner_key]:
        recommendation = "empty  (tok('', abstract))"
    elif result_single[winner_key] > result_empty[winner_key]:
        recommendation = "single (tok(abstract))"
    else:
        recommendation = "tie — pick 'empty' for symmetry with trained distribution"

    print(f"\n=== Title-convention comparison (hit@{args.k}) ===")
    for res in [result_empty, result_single]:
        print(
            f"  {res['convention']:6s}: hit@{args.k}={res[winner_key]:.4f} "
            f"  Wilson [{res['wilson_lower']:.4f}, {res['wilson_upper']:.4f}]"
        )
    print(f"  Recommendation: {recommendation}")
    print("  (Record chosen convention in index fingerprint — ADR-0014 §3)")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "script": "scripts/title_convention_eval.py",
        "adr": "ADR-0014 §3",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "split": args.split,
            "k": args.k,
            "retrieval": {
                "bm25": config.bm25,
                "dense": config.dense,
                "rrf": config.rrf,
                "rerank": config.rerank,
                "rrf_k": config.rrf_k,
                "corpus_id": config.corpus_id,
                "corpus_fingerprint": config.corpus_fingerprint,
            },
        },
        "results": {
            "empty": result_empty,
            "single": result_single,
        },
        f"recommended_convention": recommendation,
        "note": (
            "Record the winning convention in the index fingerprint before encoding production indices. "
            "The convention is part of the index's identity (ADR-0014 §3)."
        ),
    }
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
