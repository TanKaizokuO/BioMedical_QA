#!/usr/bin/env python3
"""Can any chunker rescue G1? — answered inside Table 1's recorded pool, not by seven index builds.

G1 failed at hit@5 (row 4: 0.86, Wilson lower 0.7786). The documented ladder ends at relaxing to
hit@10 *and saying so in the paper*, with the chunker sweep as the expensive rung before it:
`chunker_sweep.py` builds one 2M index per configuration, ~2 h of A4000 each, seven of them.

This script asks the sweep's question the cheap way, exactly as ADR-0014 §3 answered the
title-convention question inside the pools Table 1 had already produced.

What it measures
----------------
For each dev question, take the **abstracts** Table 1 row 4 pooled (100 deep — reranking is a
permutation of the fused pool, so row 3 and row 4 pool the same set), re-chunk those abstracts
under a candidate `ChunkConfig`, score every resulting chunk with the cross-encoder, and ask
whether a chunk of the gold abstract lands in the top 5.

This is sound to run without a GPU-hours index build because **the cross-encoder needs no corpus
statistics**. Unlike BM25's idf — which shifts when re-chunking changes document frequencies across
all 2M rows — a `(query, passage)` cross-encoder score is a function of that pair alone. The
number this produces for the `abstract` configuration is therefore not an approximation of row 4;
it *is* row 4, which is what `--expect-hit5` pins.

Scope, stated plainly
---------------------
Every number here is an **upper bound** on what the corresponding full build would report, for two
reasons, both of which run in the same direction:

1. The real cascade reranks a 100-deep pool of *chunks* chosen by BM25+dense. Here the reranker
   sees every chunk of 100 pooled *abstracts* — several hundred candidates — so the gold chunk is
   guaranteed a look it might not get for real.
2. Candidates from abstracts outside the pool cannot compete. A finer chunker would surface new
   rivals from the other ~2.16M abstracts; here it cannot.

That asymmetry is what makes the cheap version decision-useful. A configuration whose **upper
bound** misses 0.90 cannot pass G1 for real, and its ~2 h build is refused on the evidence. Only a
configuration that clears 0.90 here earns the full build — and its real number will be lower.

`not_in_pool` is inherited and unfixable here: the 3 questions whose gold abstract RRF never
surfaced stay missing under every configuration, capping every arm at 0.97.

A note on `section`: MedRAG corpus rows carry no section labels, so distractors degrade to
`abstract` while gold abstracts split on their real BACKGROUND/METHODS/… boundaries. That
asymmetry is not an artifact of this script — it is how the real index would be built, and
`chunk_text(sections=None)` is the same call `encode_corpus.py` makes.

RUNS ON THE A4000 (cross-encoder; needs the index only for passage text, not the 3.1 GB matrix).

    python scripts/chunker_pool_eval.py --index-dir data/index/empty --out docs/harvest/chunker_pool_eval.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.chunk import Chunk, chunk_instance, chunk_text  # noqa: E402
from biomedqa.config import ChunkConfig, RetrievalConfig, RunConfig  # noqa: E402
from biomedqa.data import Instance, load_instances, load_splits  # noqa: E402
from biomedqa.retrieve import RetrievalIndex, _get_cross_encoder  # noqa: E402
from biomedqa.scoring.retrieval import wilson_interval  # noqa: E402

sys.path.insert(0, str(_REPO / "scripts"))

from chunker_sweep import SWEEP  # noqa: E402  — one grid, defined once

#: Reported for every arm. hit@5 is the gate's k; hit@10 is the rung below it on R2's ladder, and
#: a chunker that only moves hit@10 is answering a question the ladder has already conceded.
HIT_AT_K_CURVE = (1, 5, 10, 20)


def _pool_by_query(records_path: Path, row: int) -> dict[str, list[str]]:
    """`{query_id: [source_id, …]}` — the abstracts one Table 1 row pooled, in rank order."""
    pools: dict[str, list[str]] = {}
    with records_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("table1_row") != row:
                continue
            seen: set[str] = set()
            ordered: list[str] = []
            for p in rec["retrieved"]:
                src = p["passage_id"].split(":")[0]
                if src not in seen:
                    seen.add(src)
                    ordered.append(src)
            pools[rec["query_id"]] = ordered
    return pools


def _rechunk(
    source_ids: list[str],
    texts: dict[str, str],
    instance: Instance,
    config: ChunkConfig,
) -> list[Chunk]:
    """Re-chunk one query's pooled abstracts under *config*.

    The gold abstract goes through `chunk_instance` so its section boundaries and offsets are the
    real ones; distractors go through `chunk_text(sections=None)`, which is the call the corpus
    encoder makes for a MedRAG row.
    """
    out: list[Chunk] = []
    for src in source_ids:
        if src == instance.pubid:
            out.extend(chunk_instance(instance, config))
            continue
        text = texts.get(src)
        if text:
            out.extend(chunk_text(text, src, config))
    return out


def _gold_rank(ranked: list[Chunk], pubid: str) -> int | None:
    """1-indexed rank of the first chunk belonging to the gold abstract.

    Gold membership is a set over the gold abstract's chunks (`data.py`), and every one of them has
    `source_id == pubid`, so source identity is the same predicate one level cheaper.
    """
    for rank, chunk in enumerate(ranked, start=1):
        if chunk.source_id == pubid:
            return rank
    return None


def _evaluate_config(
    name: str,
    config: ChunkConfig,
    instances: list[Instance],
    pools: dict[str, list[str]],
    texts: dict[str, str],
    cross_encoder,
    batch_size: int,
) -> dict:
    ranks: list[tuple[str, int | None]] = []
    chunk_counts: list[int] = []

    for i, inst in enumerate(instances):
        source_ids = pools.get(inst.pubid)
        if not source_ids:
            continue
        chunks = _rechunk(source_ids, texts, inst, config)
        if not chunks:
            continue
        chunk_counts.append(len(chunks))

        scores = cross_encoder.predict(
            [(inst.question, c.text) for c in chunks], batch_size=batch_size
        )
        ranked = [c for _, c in sorted(zip(scores, chunks), key=lambda x: -float(x[0]))]
        ranks.append((inst.pubid, _gold_rank(ranked, inst.pubid)))

        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(instances)} …")

    n = len(ranks)
    curve = {}
    for k in HIT_AT_K_CURVE:
        hits = sum(1 for _, r in ranks if r is not None and r <= k)
        point, lower, upper = wilson_interval(hits, n)
        curve[f"hit_at_{k}"] = {
            "point": round(point, 4),
            "wilson_lower": round(lower, 4),
            "wilson_upper": round(upper, 4),
            "hits": hits,
        }

    return {
        "chunker": name,
        "chunk_config": {
            "strategy": config.strategy,
            "window_sentences": config.window_sentences,
            "stride_sentences": config.stride_sentences,
            "max_chars": config.max_chars,
            "keep_section_labels": config.keep_section_labels,
        },
        "n": n,
        # Upper bound, never "hit@5". Naming it for what it is stops the number being quoted as a
        # gate reading three weeks from now, which is exactly how a bound becomes a claim.
        "hit_at_k_upper_bound": curve,
        "candidates_per_query": {
            "mean": round(statistics.mean(chunk_counts), 1) if chunk_counts else None,
            "max": max(chunk_counts) if chunk_counts else None,
        },
        "gold_rank_per_query": [{"query_id": q, "gold_rank": r} for q, r in ranks],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Upper-bound the chunker sweep inside Table 1's recorded pool (A4000)"
    )
    ap.add_argument("--index-dir", required=True, type=Path, help="Index dir (passage text only)")
    ap.add_argument(
        "--records",
        type=Path,
        default=Path("docs/harvest/table1_rows_1_4.records.jsonl"),
        help="Table 1 records holding the pools to re-chunk",
    )
    ap.add_argument("--row", type=int, default=4, help="Table 1 row whose pool is reused")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument(
        "--configs",
        default=",".join(SWEEP),
        help=f"Comma-separated subset of {','.join(SWEEP)}",
    )
    ap.add_argument(
        "--expect-hit5",
        type=float,
        default=None,
        help=(
            "Harness check: the 'abstract' arm must reproduce this (row 4's 0.86). The cross-"
            "encoder needs no corpus statistics, so that arm is row 4 recomputed, not an estimate "
            "of it — a mismatch means the pool or the gold predicate is being read wrongly and "
            "every other arm is unreadable."
        ),
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--out", type=Path, default=Path("docs/harvest/chunker_pool_eval.json")
    )
    ap.add_argument("--no-gpu-check", action="store_true")
    args = ap.parse_args()

    unknown = [c for c in args.configs.split(",") if c and c not in SWEEP]
    if unknown:
        print(f"Unknown chunker(s): {', '.join(unknown)}", file=sys.stderr)
        return 1
    selected = [c for c in args.configs.split(",") if c]

    if args.expect_hit5 is not None and "abstract" not in selected:
        print(
            "--expect-hit5 checks the 'abstract' arm, which is not in --configs.",
            file=sys.stderr,
        )
        return 1

    import torch

    if not torch.cuda.is_available() and not args.no_gpu_check:
        print("CUDA not available — the cross-encoder needs the A4000.", file=sys.stderr)
        return 1

    started_at = datetime.now(timezone.utc).isoformat()

    pubids = set(load_splits()[args.split])
    instances = [i for i in load_instances() if i.pubid in pubids]
    print(f"{len(instances)} instances in '{args.split}'")

    pools = _pool_by_query(args.records, args.row)
    print(f"{len(pools)} pools from {args.records} (row {args.row})")
    if not pools:
        print(f"No records with table1_row == {args.row}.", file=sys.stderr)
        return 1

    needed = {src for ids in pools.values() for src in ids}
    print(f"{len(needed):,} distinct pooled abstracts")

    # Text only: bm25=False and dense=False keep the 3.1 GB matrix and the BM25 model off the box's
    # 16 GB, exactly as the probe's control mode does.
    text_config = RetrievalConfig(bm25=False, dense=False, rrf=False, rerank=False)
    print(f"Loading passage text from {args.index_dir} …")
    index = RetrievalIndex.load(args.index_dir, text_config)
    texts: dict[str, str] = {}
    for pid, text in zip(index.passage_ids, index.passage_texts):
        src = pid.split(":")[0]
        if src in needed and src not in texts:
            texts[src] = text
    print(f"  resolved text for {len(texts):,}/{len(needed):,} abstracts")

    cross_encoder = _get_cross_encoder(RetrievalConfig().reranker)

    results: list[dict] = []
    for name in selected:
        print(f"\n  {name}")
        results.append(
            _evaluate_config(
                name, SWEEP[name], instances, pools, texts, cross_encoder, args.batch_size
            )
        )
        c = results[-1]["hit_at_k_upper_bound"]
        print(
            f"    upper bound hit@5={c['hit_at_5']['point']:.4f} "
            f"(Wilson lower {c['hit_at_5']['wilson_lower']:.4f})  "
            f"hit@10={c['hit_at_10']['point']:.4f}  "
            f"candidates/query mean {results[-1]['candidates_per_query']['mean']}"
        )

    print(f"\n{'='*72}")
    print(f"{'chunker':<22}{'hit@5 (UB)':>12}{'Wilson lo':>12}{'hit@10 (UB)':>13}{'≥0.90?':>9}")
    print("-" * 72)
    for r in results:
        c = r["hit_at_k_upper_bound"]
        print(
            f"{r['chunker']:<22}{c['hit_at_5']['point']:>12.4f}"
            f"{c['hit_at_5']['wilson_lower']:>12.4f}{c['hit_at_10']['point']:>13.4f}"
            f"{('yes' if c['hit_at_5']['point'] >= 0.90 else 'no'):>9}"
        )
    print("=" * 72)
    print("An arm reading 'no' cannot pass G1 for real; its full build is refused on this evidence.")

    check: dict | None = None
    if args.expect_hit5 is not None:
        got = next(
            r["hit_at_k_upper_bound"]["hit_at_5"]["point"]
            for r in results
            if r["chunker"] == "abstract"
        )
        ok = abs(got - args.expect_hit5) < 1e-9
        check = {"expected_hit5": args.expect_hit5, "abstract_arm_hit5": got, "passed": ok}
        print(f"\nHarness check: abstract arm {got:.4f} vs expected {args.expect_hit5:.4f} — "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            print(
                "The abstract arm must equal row 4 exactly. Not writing an artifact that cannot be "
                "read.",
                file=sys.stderr,
            )
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "script": "scripts/chunker_pool_eval.py",
                "question": "Can a chunker rescue G1's hit@5, before spending ~2 h/config on builds?",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "index_dir": str(args.index_dir),
                    "records": str(args.records),
                    "table1_row": args.row,
                    "split": args.split,
                    "reranker": RetrievalConfig().reranker,
                    "index_fingerprint": RunConfig(split=args.split).index_fingerprint(),
                },
                "scope": (
                    "Upper bound. Candidate abstracts are fixed to the recorded row-4 pool, and "
                    "every chunk of them is reranked rather than a 100-deep chunk pool, so a real "
                    "build reports no more than this. The 'abstract' arm is row 4 recomputed, not "
                    "estimated: the cross-encoder uses no corpus statistics."
                ),
                "harness_check": check,
                "arms": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
