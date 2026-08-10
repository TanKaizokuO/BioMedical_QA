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
all 2M rows — a `(query, passage)` cross-encoder score is a function of that pair alone. Every
candidate row 4 scored is therefore scored identically here, and the `abstract` configuration
recomputes row 4 rather than estimating it.

It does not always *equal* row 4, and `--audit-pool` says which case holds. An abstract longer
than `max_chars` is stored as several chunks, and row 4's top 100 may have surfaced only some of
them; reassembling and re-chunking that abstract puts its remaining siblings back in the candidate
set. Those additions can demote gold or leave it — they can never promote it. So the harness
demands equality when the audit reports no additions, and "no better than row 4, and gold promoted
in zero queries" when it reports additions. The rule is derived from the measured pool, never
loosened after a failure.

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

    uv run python scripts/chunker_pool_eval.py --index-dir data/index/empty --audit-pool
    uv run python scripts/chunker_pool_eval.py --index-dir data/index/empty --expect-hit5 0.86 --out docs/harvest/chunker_pool_eval.json
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


def _pooled_passage_ids(records_path: Path, row: int) -> set[str]:
    """Every passage id one Table 1 row pooled, flattened across queries."""
    out: set[str] = set()
    with records_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("table1_row") == row:
                out.update(p["passage_id"] for p in rec["retrieved"])
    return out


def audit_pool(index_dir: Path, records_path: Path, row: int) -> dict:
    """How many pooled abstracts the index stores as more than one chunk — no GPU, no text load.

    This decides whether the harness check *can* be exact. Re-chunking a pooled abstract yields
    every chunk of it, including siblings the pool never held; each of those is a competitor row 4
    never scored. If no pooled abstract has an unpooled sibling, the abstract arm must reproduce
    row 4 to the digit and any gap is a bug. If some do, the gap has a known, countable cause and
    the check has to be stated against that rule instead of against row 4's raw number.

    Reads `passage_ids.json` only — a few hundred MB of strings, not the 2.5 GB of passage text.
    """
    passage_ids: list[str] = json.loads(
        (Path(index_dir) / "passage_ids.json").read_text(encoding="utf-8")
    )
    pooled = _pooled_passage_ids(records_path, row)
    pooled_sources = {pid.split(":")[0] for pid in pooled}

    chunks_by_source: dict[str, set[str]] = defaultdict(set)
    for pid in passage_ids:
        src = pid.split(":")[0]
        if src in pooled_sources:
            chunks_by_source[src].add(pid)

    multi = {s: ids for s, ids in chunks_by_source.items() if len(ids) > 1}
    with_unpooled_siblings = {s: sorted(ids - pooled) for s, ids in multi.items() if ids - pooled}

    return {
        "pooled_passages": len(pooled),
        "pooled_abstracts": len(pooled_sources),
        "abstracts_resolved_in_index": len(chunks_by_source),
        "abstracts_stored_as_multiple_chunks": len(multi),
        "max_chunks_for_one_abstract": max((len(v) for v in multi.values()), default=1),
        "abstracts_with_unpooled_siblings": len(with_unpooled_siblings),
        "extra_candidates_introduced": sum(len(v) for v in with_unpooled_siblings.values()),
        "harness_check_can_be_exact": not with_unpooled_siblings,
        "examples": dict(list(with_unpooled_siblings.items())[:5]),
    }


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


def _reconstruct_abstracts(
    passage_ids: list[str], passage_texts: list[str], needed: set[str]
) -> dict[str, str]:
    """`{source_id: full abstract text}`, reassembled from however many chunks the index holds.

    An abstract longer than `ChunkConfig.max_chars` is stored as `X:0`, `X:1`, … even under the
    `abstract` strategy, because `_enforce_max_chars` cuts any over-long span. Taking the first
    chunk per source — which this function replaced — silently truncated every long abstract, and
    since those abstracts are the *distractors*, it weakened the competition and inflated gold.
    That defect reads as a chunker win, which is precisely the claim this script exists to make.

    `_enforce_max_chars` cuts consecutive, non-overlapping spans of the source text and inserts no
    separator, so concatenating the pieces in index order reproduces the original exactly.
    """
    pieces: dict[str, dict[int, str]] = defaultdict(dict)
    for pid, text in zip(passage_ids, passage_texts):
        src, _, idx = pid.partition(":")
        if src in needed:
            pieces[src][int(idx) if idx.isdigit() else 0] = text
    return {src: "".join(p[i] for i in sorted(p)) for src, p in pieces.items()}


def _reference_ranks(
    records_path: Path, row: int, instances: list[Instance]
) -> dict[str, int | None]:
    """Row 4's gold rank per query, read from the records the pools came from."""
    gold_by_pubid = {i.pubid: set(i.gold_passage_ids) for i in instances}
    ranks: dict[str, int | None] = {}
    with records_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("table1_row") != row:
                continue
            gold = gold_by_pubid.get(rec["query_id"], set())
            ranks[rec["query_id"]] = next(
                (p["rank"] for p in rec["retrieved"] if p["passage_id"] in gold), None
            )
    return ranks


def _evaluate_config(
    name: str,
    config: ChunkConfig,
    instances: list[Instance],
    pools: dict[str, list[str]],
    texts: dict[str, str],
    cross_encoder,
    batch_size: int,
    reference_ranks: dict[str, int | None] | None = None,
) -> dict:
    per_query: list[dict] = []
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
        rank = _gold_rank(ranked, inst.pubid)

        # Diagnostics carried for every query, not only on failure: they are the only way to tell
        # an arm that beat row 4 on ordering from one that beat it by handing gold extra chunks or
        # a different string to be scored on.
        indexed_gold = texts.get(inst.pubid)
        entry = {
            "query_id": inst.pubid,
            "gold_rank": rank,
            "n_candidates": len(chunks),
            "n_gold_chunks": sum(1 for c in chunks if c.source_id == inst.pubid),
            "gold_in_pool": inst.pubid in source_ids,
            "gold_text_matches_index": (
                None if indexed_gold is None else indexed_gold == inst.abstract_text
            ),
        }
        if reference_ranks is not None:
            entry["reference_gold_rank"] = reference_ranks.get(inst.pubid)
        per_query.append(entry)

        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(instances)} …")

    n = len(per_query)
    curve = {}
    for k in HIT_AT_K_CURVE:
        hits = sum(1 for e in per_query if e["gold_rank"] is not None and e["gold_rank"] <= k)
        point, lower, upper = wilson_interval(hits, n)
        curve[f"hit_at_{k}"] = {
            "point": round(point, 4),
            "wilson_lower": round(lower, 4),
            "wilson_upper": round(upper, 4),
            "hits": hits,
        }

    result = {
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
        "gold_split_into_multiple_chunks": sum(1 for e in per_query if e["n_gold_chunks"] > 1),
        "gold_text_differs_from_index": sum(
            1 for e in per_query if e["gold_text_matches_index"] is False
        ),
        "gold_rank_per_query": per_query,
    }

    if reference_ranks is not None:
        disagreements = [
            e for e in per_query if e["gold_rank"] != e.get("reference_gold_rank")
        ]
        result["disagreements_with_reference"] = {
            "n": len(disagreements),
            "queries": disagreements,
        }
    return result


def _rank_key(rank: int | None) -> float:
    """Gold that was never found sorts after gold that was, so ranks compare with one operator."""
    return float("inf") if rank is None else float(rank)


def _harness_check(abstract_arm: dict, expected: float, exact: bool) -> dict:
    """Is the 'abstract' arm consistent with row 4, given what the pool audit found?

    A cross-encoder score is a function of the `(query, passage)` pair alone, so every candidate
    row 4 scored is scored identically here, and gold's own chunks are unchanged. Re-chunking only
    ever *adds* candidates: the siblings of pooled abstracts that never reached row 4's top 100.
    An added candidate can demote gold or leave it. It cannot promote gold. So:

    * audit says nothing is added -> the arm must equal row 4 exactly;
    * audit says candidates are added -> the arm must be no better than row 4, in aggregate and
      per query. One improved gold rank is impossible under the invariant, so it is a defect —
      a sharper instrument than any tolerance on the aggregate, which would pass a run that
      promoted gold in five queries and demoted it in five others.
    """
    per_query = abstract_arm["gold_rank_per_query"]
    improved = [
        e
        for e in per_query
        if _rank_key(e["gold_rank"]) < _rank_key(e.get("reference_gold_rank"))
    ]
    demoted = [
        e
        for e in per_query
        if _rank_key(e["gold_rank"]) > _rank_key(e.get("reference_gold_rank"))
    ]
    got = abstract_arm["hit_at_k_upper_bound"]["hit_at_5"]["point"]
    if exact:
        rule = "abstract arm == row 4 exactly (audit: re-chunking introduces no new candidate)"
        passed = abs(got - expected) < 1e-9 and not improved and not demoted
    else:
        rule = (
            "abstract arm <= row 4 and no query's gold rank improves (audit: re-chunking adds "
            "candidates row 4 never scored, and those can only demote gold)"
        )
        passed = got <= expected + 1e-9 and not improved
    return {
        "rule": rule,
        "exact_equality_attainable": exact,
        "expected_hit5_row4": expected,
        "abstract_arm_hit5": got,
        "queries_gold_improved": {"n": len(improved), "queries": improved},
        "queries_gold_demoted": {"n": len(demoted), "queries": demoted},
        "passed": passed,
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
            "Row 4's hit@5 (0.86), which the 'abstract' arm is checked against. The cross-encoder "
            "needs no corpus statistics, so that arm recomputes row 4 rather than estimating it. "
            "Whether the check demands equality or only 'no better than, and never promoted' is "
            "decided by the pool audit, which runs first and is recorded in the artifact."
        ),
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--out", type=Path, default=Path("docs/harvest/chunker_pool_eval.json")
    )
    ap.add_argument("--no-gpu-check", action="store_true")
    ap.add_argument(
        "--audit-pool",
        action="store_true",
        help=(
            "Report how the index stores the pooled abstracts and exit. No GPU, no text load, "
            "seconds not minutes. Run this before trusting any arm."
        ),
    )
    args = ap.parse_args()

    if args.audit_pool:
        report = audit_pool(args.index_dir, args.records, args.row)
        print(json.dumps(report, indent=2))
        print(
            "\nharness_check_can_be_exact = "
            f"{report['harness_check_can_be_exact']}: "
            + (
                "re-chunking introduces no candidate row 4 did not score, so the abstract arm "
                "must equal row 4 exactly."
                if report["harness_check_can_be_exact"]
                else f"re-chunking adds {report['extra_candidates_introduced']} competitor(s) "
                "row 4 never scored, so exact equality is not attainable and the check must be "
                "restated."
            )
        )
        return 0

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

    # The audit decides what the harness may demand, so it runs before the GPU does and is
    # recorded beside the arms. Deriving the rule from a measured property of the pool beats
    # asserting equality and then loosening it when equality turns out to be unattainable.
    audit = audit_pool(args.index_dir, args.records, args.row)
    print(
        f"Pool audit: {audit['abstracts_with_unpooled_siblings']:,} abstract(s) have siblings "
        f"outside the pool, adding {audit['extra_candidates_introduced']:,} candidate(s) row "
        f"{args.row} never scored — exact equality attainable: "
        f"{audit['harness_check_can_be_exact']}"
    )

    # Text only: bm25=False and dense=False keep the 3.1 GB matrix and the BM25 model off the box's
    # 16 GB, exactly as the probe's control mode does.
    text_config = RetrievalConfig(bm25=False, dense=False, rrf=False, rerank=False)
    print(f"Loading passage text from {args.index_dir} …")
    index = RetrievalIndex.load(args.index_dir, text_config)
    texts = _reconstruct_abstracts(index.passage_ids, index.passage_texts, needed)
    print(f"  resolved text for {len(texts):,}/{len(needed):,} abstracts")

    cross_encoder = _get_cross_encoder(RetrievalConfig().reranker)

    # Row 4's own per-query ranks. The abstract arm is compared against them query by query, so a
    # harness failure names the queries that moved instead of only the aggregate that shifted.
    reference_ranks = _reference_ranks(args.records, args.row, instances)

    results: list[dict] = []
    for name in selected:
        print(f"\n  {name}")
        results.append(
            _evaluate_config(
                name,
                SWEEP[name],
                instances,
                pools,
                texts,
                cross_encoder,
                args.batch_size,
                reference_ranks=reference_ranks if name == "abstract" else None,
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
        abstract_arm = next(r for r in results if r["chunker"] == "abstract")
        check = _harness_check(
            abstract_arm, args.expect_hit5, audit["harness_check_can_be_exact"]
        )
        print(
            f"\nHarness check — {check['rule']}"
            f"\n  abstract arm {check['abstract_arm_hit5']:.4f} vs row 4 "
            f"{args.expect_hit5:.4f}; gold improved in "
            f"{check['queries_gold_improved']['n']} query(s), demoted in "
            f"{check['queries_gold_demoted']['n']} — "
            f"{'PASS' if check['passed'] else 'FAIL'}"
        )
        if not check["passed"]:
            # Refuse the artifact, but never throw away the run that earned the refusal: a second
            # GPU pass to re-learn what just happened is waste, and the diagnostic is the whole
            # value of a failed harness. The filename cannot be mistaken for a result.
            diag = args.out.with_suffix(".HARNESS_FAILED.json")
            diag.parent.mkdir(parents=True, exist_ok=True)
            diag.write_text(
                json.dumps(
                    {
                        "script": "scripts/chunker_pool_eval.py",
                        "status": "HARNESS CHECK FAILED — no arm here is readable as a bound",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "harness_check": check,
                        "why_it_matters": (
                            "The abstract arm recomputes row 4 rather than estimating it, because "
                            "the cross-encoder uses no corpus statistics. Re-chunking may add "
                            "candidates, which can only demote gold. A better-than-row-4 gold "
                            "rank, or an aggregate above row 4, means the candidate set or the "
                            "gold text differs from what row 4 scored — and the same difference "
                            "is present, unquantified, in every other arm."
                        ),
                        "abstract_arm": abstract_arm,
                        "other_arms_unreadable": [
                            {
                                "chunker": r["chunker"],
                                "hit_at_5": r["hit_at_k_upper_bound"]["hit_at_5"]["point"],
                                "candidates_per_query": r["candidates_per_query"],
                                "gold_split_into_multiple_chunks": r[
                                    "gold_split_into_multiple_chunks"
                                ],
                                "gold_text_differs_from_index": r["gold_text_differs_from_index"],
                            }
                            for r in results
                            if r["chunker"] != "abstract"
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(
                f"The abstract arm must equal row 4 exactly. No result artifact written; "
                f"diagnostics in {diag}",
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
                "pool_audit": audit,
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
