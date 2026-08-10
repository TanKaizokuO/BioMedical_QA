#!/usr/bin/env python3
"""ADR-0014 §3 — title-segment convention, measured inside the pool the 2M index already produced.

    empty  —  tok("", abstract)   — two segments, empty title   (what `data/index/empty` holds)
    single —  tok(abstract)       — one segment, no title slot

`title_convention_eval.py` answers this by building the whole index both ways: two ~2 h A4000
encodes.  This script answers the **ordering** half of the same question for the price of encoding
~18k passages (about a minute), by reusing the 100-deep pools Table 1 already recorded:

1. take each dev query's dense pool from `table1_rows_1_3.records.jsonl`;
2. score its candidates with the `empty` vectors already in `dense.npy`;
3. re-encode only those candidates as `single`, and score them again;
4. compare gold rank per query, paired.

**What this measures:** whether the convention reorders a fixed candidate set — the precision
question, which is the one Table 1 left open (dense hit@5 0.59 against BM25's 0.71, with RRF pool
recall at 0.97).

**What it cannot measure:** whether `single` pulls into the top 100 some passage `empty` never
retrieved.  The candidate set is fixed to `empty`'s pool, so recall is held constant by construction
and the `single` arm is, strictly, an upper bound conditioned on `empty`'s recall.  A material
ordering effect here is what earns the full two-index build; a null here settles §3 without it.

**Built-in check:** the `empty` arm re-derives Table 1 row 2 from the same vectors, so its hit@k must
reproduce the recorded row.  `--expect-hit5` fails the run if it does not.

RUNS ON THE A4000 (needs the article encoder and `dense.npy`).

    python scripts/title_convention_pool_eval.py \\
      --index-dir data/index/empty \\
      --out docs/harvest/title_convention_pool_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomedqa.config import RetrievalConfig, RunConfig  # noqa: E402
from biomedqa.retrieve import (  # noqa: E402
    RetrievalIndex,
    _encode_query,
    build_dense_index,
)
from biomedqa.scoring.retrieval import wilson_interval  # noqa: E402

HIT_KS = (1, 5, 10, 20, 50, 100)


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test under p=0.5.  Ties are excluded by the caller."""
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))


def load_pools(records_path: Path, run_id: str) -> list[dict]:
    """Per-query candidate pools in recorded rank order, from the Table 1 records."""
    pools: list[dict] = []
    with open(records_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("run_id") != run_id:
                continue
            pools.append(
                {
                    "query_id": rec["query_id"],
                    "question": rec["question"],
                    "gold_passage_ids": list(rec["gold_passage_ids"]),
                    "pool": [p["passage_id"] for p in rec["retrieved"]],
                }
            )
    return pools


def gold_rank_from_order(ordered_ids: list[str], gold: set[str]) -> int | None:
    """1-indexed rank of the best-placed gold passage in *ordered_ids*, or None."""
    for i, pid in enumerate(ordered_ids, start=1):
        if pid in gold:
            return i
    return None


def hit_curve(gold_ranks: list[int | None]) -> dict:
    """hit@k with Wilson intervals over a list of per-query gold ranks."""
    n = len(gold_ranks)
    curve = {}
    for k in HIT_KS:
        hits = sum(1 for r in gold_ranks if r is not None and r <= k)
        point, lower, upper = wilson_interval(hits, n)
        curve[f"hit_at_{k}"] = {
            "hits": hits,
            "point": round(point, 4),
            "wilson_lower": round(lower, 4),
            "wilson_upper": round(upper, 4),
        }
    return curve


def main() -> int:
    ap = argparse.ArgumentParser(description="ADR-0014 §3 pool-restricted convention comparison")
    ap.add_argument("--index-dir", required=True, type=Path, help="The `empty`-convention index")
    ap.add_argument(
        "--records",
        type=Path,
        default=Path("docs/harvest/table1_rows_1_3.records.jsonl"),
        help="Table 1 per-query records (least-processed pools)",
    )
    ap.add_argument(
        "--run-id",
        default="table1_dense_only",
        help="Which Table 1 run's pools to re-rank (default: the dense-only row)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/harvest/title_convention_pool_eval.json"),
    )
    ap.add_argument("--batch-size", type=int, default=32, help="Article-encoder batch size")
    ap.add_argument(
        "--expect-hit5",
        type=float,
        default=None,
        help=(
            "Fail unless the `empty` arm reproduces this hit@5 (the recorded Table 1 row). "
            "The arm re-ranks the same candidates with the same vectors, so a mismatch means the "
            "harness is wrong and the `single` arm cannot be trusted either."
        ),
    )
    ap.add_argument("--no-gpu-check", action="store_true")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available() and not args.no_gpu_check:
        print("CUDA not available — this script must run on the A4000.", file=sys.stderr)
        return 1
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    pools = load_pools(args.records, args.run_id)
    if not pools:
        print(f"No records with run_id={args.run_id!r} in {args.records}", file=sys.stderr)
        return 1
    print(f"{len(pools)} queries, pool depth {min(len(p['pool']) for p in pools)}"
          f"–{max(len(p['pool']) for p in pools)}")

    config = RetrievalConfig(bm25=False, dense=True, rrf=False, rerank=False)
    print(f"Loading index from {args.index_dir} …")
    index = RetrievalIndex.load(args.index_dir, config)
    if index.dense_embeddings is None:
        print(f"{args.index_dir}/dense.npy is missing.", file=sys.stderr)
        return 1

    position = {pid: i for i, pid in enumerate(index.passage_ids)}
    candidates = sorted({pid for p in pools for pid in p["pool"]})
    missing = [pid for pid in candidates if pid not in position]
    if missing:
        print(f"{len(missing)} pooled passage ids are not in the index, e.g. {missing[:3]}",
              file=sys.stderr)
        return 1
    print(f"{len(candidates):,} distinct pooled passages to re-encode")

    rows = [position[pid] for pid in candidates]
    slot = {pid: i for i, pid in enumerate(candidates)}

    # `empty` side: already encoded, already L2-normalised.  Read it, do not recompute — recomputing
    # would measure this script's encoder call rather than the index the numbers came from.
    empty_vecs = index.dense_embeddings[rows].astype(np.float32)

    print("Encoding the same passages as single-segment …")
    single_vecs = build_dense_index(
        [(pid, index.passage_texts[position[pid]]) for pid in candidates],
        config,
        batch_size=args.batch_size,
        empty_title=False,
    ).astype(np.float32)

    diff = float(np.abs(empty_vecs - single_vecs).max())
    cos = float(np.mean(np.sum(empty_vecs * single_vecs, axis=1)))
    print(f"Convention delta: max abs component {diff:.4f}, mean cosine {cos:.4f}")

    per_query: list[dict] = []
    for i, p in enumerate(pools, start=1):
        gold = set(p["gold_passage_ids"])
        idx = np.fromiter((slot[pid] for pid in p["pool"]), dtype=np.int64, count=len(p["pool"]))
        qvec = _encode_query(p["question"], config.query_encoder).astype(np.float32)

        ranked: dict[str, list[str]] = {}
        for name, mat in (("empty", empty_vecs), ("single", single_vecs)):
            scores = mat[idx] @ qvec
            # Descending score; ties broken by the recorded pool order, which is stable.
            order = np.argsort(-scores, kind="stable")
            ranked[name] = [p["pool"][j] for j in order]

        per_query.append(
            {
                "query_id": p["query_id"],
                "pool_depth": len(p["pool"]),
                "gold_in_pool": bool(gold & set(p["pool"])),
                "gold_rank_empty": gold_rank_from_order(ranked["empty"], gold),
                "gold_rank_single": gold_rank_from_order(ranked["single"], gold),
            }
        )
        if i % 20 == 0:
            print(f"  {i}/{len(pools)} queries ranked …")

    empty_ranks = [q["gold_rank_empty"] for q in per_query]
    single_ranks = [q["gold_rank_single"] for q in per_query]

    # Paired on rank, over the queries where gold is in the pool at all — elsewhere both arms are
    # None and the pair carries no information about ordering.
    paired = [q for q in per_query if q["gold_in_pool"]]
    better = sum(1 for q in paired if q["gold_rank_single"] < q["gold_rank_empty"])
    worse = sum(1 for q in paired if q["gold_rank_single"] > q["gold_rank_empty"])

    summary = {
        "n_queries": len(per_query),
        "n_gold_in_pool": len(paired),
        "convention_delta": {"max_abs_component": round(diff, 6), "mean_cosine": round(cos, 6)},
        "empty": hit_curve(empty_ranks),
        "single": hit_curve(single_ranks),
        "paired_gold_rank": {
            "single_better": better,
            "single_worse": worse,
            "unchanged": len(paired) - better - worse,
            "sign_test_p": round(sign_test_p(better, worse), 6),
            "mean_rank_delta": round(
                sum(q["gold_rank_single"] - q["gold_rank_empty"] for q in paired) / len(paired), 4
            )
            if paired
            else None,
        },
    }

    print("\n=== ADR-0014 §3, pool-restricted ===")
    print(json.dumps(summary, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "script": "scripts/title_convention_pool_eval.py",
                "adr": "ADR-0014 §3",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "index_dir": str(args.index_dir),
                    "records": str(args.records),
                    "run_id": args.run_id,
                    "dense_encoder": config.dense_encoder,
                    "query_encoder": config.query_encoder,
                    "index_fingerprint_empty": RunConfig(
                        retrieval=RetrievalConfig(title_segment="empty")
                    ).index_fingerprint(),
                    "index_fingerprint_single": RunConfig(
                        retrieval=RetrievalConfig(title_segment="single")
                    ).index_fingerprint(),
                },
                "summary": summary,
                "per_query": per_query,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nResults written to {args.out}")

    if args.expect_hit5 is not None:
        got = summary["empty"]["hit_at_5"]["point"]
        if abs(got - args.expect_hit5) > 1e-9:
            print(
                f"\nFAIL: the `empty` arm gave hit@5={got}, not the recorded {args.expect_hit5}. "
                "Same candidates, same vectors, same query encoder should reproduce Table 1 row 2 "
                "exactly. The harness is wrong; do not read the `single` arm.",
                file=sys.stderr,
            )
            return 1
        print(f"Check passed: the `empty` arm reproduces Table 1 hit@5={got}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
