#!/usr/bin/env python3
"""Dump the top-N passage texts for a Table 1 row, so prompts can be drafted off the box.

`table1_baseline.py` stores `text` for ranks 1–5 only, which was the right depth while G1 was a
hit@5 gate. ADR-0015 moved the gate to hit@10, so the context the generator actually receives is
10 passages deep and the committed records can no longer render it. This closes that gap without a
GPU pass: the index is opened for **passage text only** — no dense matrix, no BM25 model — exactly
as `chunker_pool_eval.py` and the probe's control mode do.

Least-processed value (`CONTEXT.md`): this writes passage text keyed by id and rank, not a rendered
prompt. Prompt wording is the treatment in C2 and will change; the retrieved context will not.

RUNS ON THE A4000 (needs `data/index/empty` for `passage_texts.jsonl`; no GPU).

    uv run python scripts/dump_contexts.py --index-dir data/index/empty --depth 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.config import RetrievalConfig  # noqa: E402
from biomedqa.data import load_instances, load_splits  # noqa: E402
from biomedqa.prompts import CONTEXT_DEPTH  # noqa: E402
from biomedqa.retrieve import RetrievalIndex  # noqa: E402


def _wanted(records_path: Path, row: int, depth: int) -> dict[str, list[dict]]:
    """`{query_id: [{passage_id, rank, retriever, score}, …]}` for the top *depth* of one row."""
    out: dict[str, list[dict]] = {}
    with records_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("table1_row") != row:
                continue
            top = sorted(rec["retrieved"], key=lambda p: p["rank"])[:depth]
            out[rec["query_id"]] = [
                {
                    "passage_id": p["passage_id"],
                    "rank": p["rank"],
                    "score": p["score"],
                    "retriever": p["retriever"],
                }
                for p in top
            ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump top-N context text for a Table 1 row (A4000)")
    ap.add_argument("--index-dir", required=True, type=Path)
    ap.add_argument(
        "--records", type=Path, default=Path("docs/harvest/table1_rows_1_4.records.jsonl")
    )
    ap.add_argument("--row", type=int, default=4)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--depth", type=int, default=CONTEXT_DEPTH)
    ap.add_argument("--out", type=Path, default=Path("docs/harvest/dev_contexts_top10.jsonl"))
    args = ap.parse_args()

    wanted = _wanted(args.records, args.row, args.depth)
    if not wanted:
        print(f"No records with table1_row == {args.row}.", file=sys.stderr)
        return 1
    needed = {p["passage_id"] for ps in wanted.values() for p in ps}
    print(f"{len(wanted)} queries, {len(needed):,} distinct passages at depth {args.depth}")

    text_config = RetrievalConfig(bm25=False, dense=False, rrf=False, rerank=False)
    print(f"Loading passage text from {args.index_dir} …")
    index = RetrievalIndex.load(args.index_dir, text_config)
    texts = {
        pid: t for pid, t in zip(index.passage_ids, index.passage_texts) if pid in needed
    }
    missing = needed - texts.keys()
    if missing:
        # A passage with no text is a passage the model cannot cite; it would read as a
        # citation-recall loss caused by the system rather than by this dump.
        print(
            f"{len(missing)} passage(s) have no text in the index (first: {sorted(missing)[0]}).",
            file=sys.stderr,
        )
        return 1

    pubids = set(load_splits()[args.split])
    questions = {i.pubid: i.question for i in load_instances() if i.pubid in pubids}
    gold = {
        i.pubid: list(i.gold_passage_ids) for i in load_instances() if i.pubid in pubids
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for qid, ps in sorted(wanted.items()):
            fh.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "question": questions.get(qid, ""),
                        "gold_passage_ids": gold.get(qid, []),
                        "table1_row": args.row,
                        "depth": args.depth,
                        "passages": [{**p, "text": texts[p["passage_id"]]} for p in ps],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
