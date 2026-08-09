#!/usr/bin/env python3
"""Chunker sweep — dev hit@5 for each `(chunker, τ)`, which is the only pair it is defined at.

`chunk.py` implements four strategies; this decides which one the G1 index is built with, and it
decides it by measurement. **hit@5 is only meaningful per `(chunker, τ)`** (Lesson 2), so a sweep
is not an optimisation over a scalar — it is the enumeration that makes any single number
interpretable. Table 1 has one row per configuration here.

**Each configuration is a different index**, because `ChunkConfig` is inside
`RunConfig.index_fingerprint()`. There is no shortcut: a chunker that splits an abstract into four
passages is a different set of retrievable units, so the passages must be re-encoded. Budget
~1.6 h of A4000 per configuration at 2M (the G0 measurement), and use `--sample` while iterating.

RUNS ON THE A4000.

    # cheap first pass over a 50k-row sample, all four strategies
    python scripts/chunker_sweep.py --corpus data/corpus/corpus.jsonl \\
        --work data/sweep --sample 50000

    # the real thing, over the configurations that survived
    python scripts/chunker_sweep.py --corpus data/corpus/corpus.jsonl \\
        --work data/sweep --configs abstract,sentence_window_3_1

**The sweep never picks τ to pass G1.** It reports every configuration it ran; the escalation
ladder for a failing gate is in the roadmap and ends at relaxing to hit@10 *and saying so in the
paper*, never at quietly moving a threshold (`retrieve.py`, ADR-0009).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.config import ChunkConfig, RetrievalConfig, RunConfig  # noqa: E402
from biomedqa.data import Instance, load_instances, load_splits  # noqa: E402
from biomedqa.retrieve import RetrievalIndex, retrieve  # noqa: E402
from biomedqa.schema import QueryRecord, System  # noqa: E402
from biomedqa.scoring.retrieval import gold_rank, hit_at_k, wilson_interval  # noqa: E402

#: The sweep grid. Each entry is one index and one Table 1 row.
#:
#: `abstract` is the ADR-0003 baseline — retrieval unit = whole abstract — and is the row every
#: other configuration is read against. The sentence windows vary the overlap: stride < window
#: means a claim spanning a boundary still sits whole inside some passage, which is the entire
#: point of windowing and the reason stride 1 is included next to stride 3.
#: `fixed_width` is the granularity control: it cuts on nothing but character count, so a lift
#: over it is evidence that linguistic boundaries matter rather than passage length.
SWEEP: dict[str, ChunkConfig] = {
    "abstract":             ChunkConfig(strategy="abstract", max_chars=2000),
    "section":              ChunkConfig(strategy="section", max_chars=2000),
    "sentence_window_3_1":  ChunkConfig(strategy="sentence_window", window_sentences=3,
                                        stride_sentences=1, max_chars=2000),
    "sentence_window_3_3":  ChunkConfig(strategy="sentence_window", window_sentences=3,
                                        stride_sentences=3, max_chars=2000),
    "sentence_window_5_2":  ChunkConfig(strategy="sentence_window", window_sentences=5,
                                        stride_sentences=2, max_chars=2000),
    "fixed_width_512":      ChunkConfig(strategy="fixed_width", max_chars=512),
    "fixed_width_1024":     ChunkConfig(strategy="fixed_width", max_chars=1024),
}


def _dev_instances(split: str = "dev") -> list[Instance]:
    pubids = set(load_splits()[split])
    return [i for i in load_instances() if str(i.pubid) in pubids]


def _encode(corpus: Path, index_dir: Path, cfg: ChunkConfig, *, title_convention: str,
            batch_size: int, sample: int | None, resume: bool) -> None:
    """Shell out to `encode_corpus.py` so the sweep encodes through exactly the same path the
    real 2M run uses. A second in-process encoder here would be a second thing to keep correct,
    and the one that is not exercised by the real run is the one that drifts."""
    cmd = [
        sys.executable, str(_REPO / "scripts" / "encode_corpus.py"),
        "--corpus", str(corpus),
        "--out", str(index_dir),
        "--title-convention", title_convention,
        "--strategy", cfg.strategy,
        "--max-chars", str(cfg.max_chars),
        # Passed unconditionally, not only for sentence_window: omitting them would silently
        # collapse sentence_window_3_3 and sentence_window_5_2 onto the encoder's (3, 1) default,
        # and the sweep would report three identical indices under three different names.
        "--window-sentences", str(cfg.window_sentences),
        "--stride-sentences", str(cfg.stride_sentences),
        "--batch-size", str(batch_size),
        "--build-bm25",
    ]
    if resume:
        cmd.append("--resume")
    if sample is not None:
        cmd += ["--limit", str(sample)]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _evaluate(index_dir: Path, instances: list[Instance], rcfg: RetrievalConfig,
              k: int) -> dict:
    """Dev hit@k for one index, plus the gold-rank distribution behind it.

    The ranked list is scored through `scoring/retrieval.py` rather than counted here, so the
    sweep and the gate compute hit@k with the same function — `gold_rank` is a *minimum over the
    gold set*, because one abstract becomes many chunks and gold membership is a set, not an
    identity. A sweep that reimplemented that comparison could rank a chunker highly for getting
    the containment test wrong.
    """
    index = RetrievalIndex.load(index_dir, rcfg)
    records: list[QueryRecord] = []
    t0 = time.perf_counter()
    for inst in instances:
        retrieved = retrieve(inst.question, rcfg, index)
        records.append(QueryRecord(
            run_id="chunker-sweep",
            query_id=str(inst.pubid),
            question=inst.question,
            system=System.VANILLA,
            seed=0,
            gold_passage_ids=inst.gold_passage_ids,
            retrieved=retrieved,
        ))
    wall = time.perf_counter() - t0

    hits, n = hit_at_k(records, k)
    point, lower, upper = wilson_interval(hits, n)
    ranks = [r for r in (gold_rank(rec) for rec in records) if r is not None]
    return {
        "k": k,
        "hits": hits,
        "n": n,
        f"hit_at_{k}": point,
        "wilson_lower": lower,
        "wilson_upper": upper,
        # G1's condition, reported but never used to choose a configuration.
        "passes_g1": point >= 0.90 and lower > 0.85,
        "n_passages_indexed": len(index.passage_ids),
        "gold_found": len(ranks),
        "gold_rank_median": sorted(ranks)[len(ranks) // 2] if ranks else None,
        "gold_rank_mean": (sum(ranks) / len(ranks)) if ranks else None,
        "retrieval_wall_s": round(wall, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep chunkers; report dev hit@k per (chunker, tau).")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus/corpus.jsonl"))
    ap.add_argument("--work", type=Path, default=Path("data/sweep"),
                    help="parent directory; one index subdirectory per configuration")
    ap.add_argument("--configs", default="",
                    help=f"comma-separated subset of {','.join(SWEEP)} (default: all)")
    ap.add_argument("--split", default="dev", choices=["dev", "test"],
                    help="test is run once, late, per ADR-0009 — dev is the sweep's split")
    ap.add_argument("--k", type=int, default=5, help="k for hit@k (G1 gates at 5)")
    ap.add_argument("--sample", type=int, default=None,
                    help="encode only the first N corpus rows (iteration; not a reportable number)")
    ap.add_argument("--title-convention", default="empty", choices=["empty", "single"],
                    help="ADR-0014 §3; hold it fixed across the sweep or the rows are not comparable")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--resume", action="store_true", help="resume each per-config encode")
    ap.add_argument("--skip-encode", action="store_true",
                    help="evaluate indices already under --work")
    ap.add_argument("--out", type=Path, default=Path("docs/harvest/chunker_sweep.json"))
    args = ap.parse_args()

    names = [c.strip() for c in args.configs.split(",") if c.strip()] or list(SWEEP)
    unknown = [n for n in names if n not in SWEEP]
    if unknown:
        raise SystemExit(f"unknown configurations {unknown}; expected from {list(SWEEP)}")

    instances = _dev_instances(args.split)
    print(f"{args.split} split: {len(instances)} questions")
    if args.sample:
        print(f"NOTE: --sample {args.sample:,} — an iteration aid. Numbers from a sampled corpus "
              f"are not Table 1 rows; the reportable sweep runs the full corpus.\n")

    # The retrieval stage is held fixed across the sweep so the only thing varying is the chunker.
    # Reranking is off: it is W3, and ADR-0012 §2's probe reads the pre-rerank pool.
    rcfg = RetrievalConfig(bm25=True, dense=True, rrf=True, rerank=False, pool_size=100,
                           top_k=max(args.k, 10))

    rows: list[dict] = []
    for name in names:
        ccfg = SWEEP[name]
        index_dir = args.work / name
        print(f"\n{'=' * 70}\n{name}  —  {ccfg}\n{'=' * 70}", flush=True)
        if not args.skip_encode:
            index_dir.mkdir(parents=True, exist_ok=True)
            _encode(args.corpus, index_dir, ccfg, title_convention=args.title_convention,
                    batch_size=args.batch_size, sample=args.sample, resume=args.resume)
        elif not index_dir.exists():
            print(f"  skipped: {index_dir} does not exist")
            continue

        result = _evaluate(index_dir, instances, rcfg, args.k)
        # The fingerprint is what makes a row traceable to an index, and the chunker is inside it.
        fingerprint = RunConfig(chunk=ccfg, retrieval=rcfg).index_fingerprint()
        rows.append({"config": name, "chunk": vars(ccfg) if not hasattr(ccfg, "__dataclass_fields__")
                     else {f: getattr(ccfg, f) for f in ccfg.__dataclass_fields__},
                     "index_fingerprint": fingerprint, **result})
        print(f"  hit@{args.k} = {result[f'hit_at_{args.k}']:.3f} "
              f"[{result['wilson_lower']:.3f}, {result['wilson_upper']:.3f}]  "
              f"passages={result['n_passages_indexed']:,}", flush=True)

    rows.sort(key=lambda r: r[f"hit_at_{args.k}"], reverse=True)

    print(f"\n{'=' * 96}")
    print(f"CHUNKER SWEEP — {args.split} hit@{args.k}, Wilson 95% CI, clustered on nothing "
          f"(a proportion over questions)")
    print("=" * 96)
    print(f"{'configuration':<22}{'hit@k':>8}{'wilson lo':>11}{'wilson hi':>11}"
          f"{'passages':>12}{'median rank':>13}{'G1':>6}")
    print("-" * 96)
    for r in rows:
        med = r["gold_rank_median"]
        print(f"{r['config']:<22}{r[f'hit_at_{args.k}']:>8.3f}{r['wilson_lower']:>11.3f}"
              f"{r['wilson_upper']:>11.3f}{r['n_passages_indexed']:>12,}"
              f"{(str(med) if med is not None else '-'):>13}"
              f"{('PASS' if r['passes_g1'] else 'fail'):>6}")
    print("=" * 96)
    if rows:
        best = rows[0]
        print(f"\nBest by point estimate: {best['config']} at hit@{args.k}="
              f"{best[f'hit_at_{args.k}']:.3f}, index_fingerprint={best['index_fingerprint']}.")
        print("Record the chosen configuration in RetrievalConfig/ChunkConfig before the 2M encode "
              "— the fingerprint is the index's identity, and a row whose fingerprint is not in "
              "runs/ is not a result.")
        if args.sample:
            print("This ran on a SAMPLED corpus; re-run without --sample before quoting Table 1.")

    payload = {
        "kind": "chunker_sweep",
        "utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "k": args.k,
        "n_questions": len(instances),
        "sample": args.sample,
        "title_convention": args.title_convention,
        "retrieval": {f: getattr(rcfg, f) for f in rcfg.__dataclass_fields__},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
