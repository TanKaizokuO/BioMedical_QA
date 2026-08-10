#!/usr/bin/env python3
"""What do G1's 14 misses have in common? — and a prediction registered before the sweep answers.

Row 4 puts gold in the top 5 for 86 of 100 dev questions. Deciding what to spend next requires
knowing *which* 14 fail and why, because the two candidate levers repair different defects:

* **Finer chunking** repairs *dilution* — the answering sentences sit inside a long abstract whose
  embedding is dominated by unrelated content. Its signature is missed golds being **longer** than
  hit golds.
* Nothing in the retrieval stack repairs an **underspecified query**. Its signature is missed
  questions being **shorter** than hit ones.

Runs anywhere — CPU only, reads the recorded records and the frozen split.

    python scripts/g1_miss_analysis.py --out docs/harvest/g1_miss_analysis.json

The prediction this file registers is checked by `chunker_pool_eval.py`, and it is written down
*first* on purpose: a hypothesis confirmed after seeing the answer is not evidence.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.data import Instance, load_instances, load_splits  # noqa: E402

#: Permutation resamples. Exact enumeration of C(100,14) is not available; 200k draws puts the
#: Monte-Carlo standard error on a p near 0.01 at ~2e-4, which is far below any decision boundary
#: this p is used at.
N_PERMUTATIONS = 200_000


def _gold_rank(record: dict, gold_ids: set[str]) -> int | None:
    for passage in record["retrieved"]:
        if passage["passage_id"] in gold_ids:
            return passage["rank"]
    return None


def permutation_p(
    values: dict[str, float], miss: list[str], hit: list[str], seed: int
) -> tuple[float, float]:
    """Two-sided permutation test on the difference of means (miss − hit).

    A t-test would assume normality of character counts over 14 questions; the label shuffle
    assumes only exchangeability under the null, which is exactly what "miss membership is
    unrelated to length" says.
    """
    observed = statistics.mean([values[q] for q in miss]) - statistics.mean(
        [values[q] for q in hit]
    )
    pool = list(values.values())
    rng = random.Random(seed)
    n_miss = len(miss)
    extreme = 0
    for _ in range(N_PERMUTATIONS):
        rng.shuffle(pool)
        delta = statistics.mean(pool[:n_miss]) - statistics.mean(pool[n_miss:])
        if abs(delta) >= abs(observed):
            extreme += 1
    return observed, (extreme + 1) / (N_PERMUTATIONS + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Characterise the questions G1 misses at k=5")
    ap.add_argument(
        "--records",
        type=Path,
        default=Path("docs/harvest/table1_rows_1_4.records.jsonl"),
    )
    ap.add_argument("--row", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("docs/harvest/g1_miss_analysis.json"))
    args = ap.parse_args()

    pubids = set(load_splits()[args.split])
    instances: dict[str, Instance] = {
        i.pubid: i for i in load_instances() if i.pubid in pubids
    }

    ranks: dict[str, int | None] = {}
    with args.records.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("table1_row") != args.row:
                continue
            inst = instances[record["query_id"]]
            ranks[inst.pubid] = _gold_rank(record, set(inst.gold_passage_ids))

    miss = [q for q, r in ranks.items() if r is None or r > args.k]
    hit = [q for q, r in ranks.items() if r is not None and r <= args.k]

    features = {
        "question_chars": lambda i: float(len(i.question)),
        "gold_abstract_chars": lambda i: float(len(i.abstract_text)),
        "gold_sections": lambda i: float(len(i.passages)),
    }

    contrasts = {}
    for name, fn in features.items():
        values = {q: fn(instances[q]) for q in ranks}
        delta, p = permutation_p(values, miss, hit, args.seed)
        contrasts[name] = {
            "miss_mean": round(statistics.mean([values[q] for q in miss]), 1),
            "hit_mean": round(statistics.mean([values[q] for q in hit]), 1),
            "delta_miss_minus_hit": round(delta, 2),
            "permutation_p": round(p, 5),
        }
        print(f"{name:<22} miss {contrasts[name]['miss_mean']:>8}  hit "
              f"{contrasts[name]['hit_mean']:>8}  delta {delta:>7.1f}  p={p:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "script": "scripts/g1_miss_analysis.py",
                "question": "Is G1's k=5 failure dilution (chunkable) or underspecified queries (not)?",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "records": str(args.records),
                    "table1_row": args.row,
                    "k": args.k,
                    "split": args.split,
                    "seed": args.seed,
                    "n_permutations": N_PERMUTATIONS,
                },
                "n_miss": len(miss),
                "n_hit": len(hit),
                "contrasts": contrasts,
                "registered_prediction": (
                    "Missed golds are not longer than hit golds, so dilution is not the defect and "
                    "no chunker in the sweep lifts the upper bound in chunker_pool_eval.py to 0.90. "
                    "Registered before that script was run. Falsified if any arm's upper bound "
                    "reaches 0.90 — in which case the full ~2 h build for that arm is owed."
                ),
                # Least-processed: the rank, per query, with the features behind the contrast.
                "per_query": [
                    {
                        "query_id": q,
                        "gold_rank": ranks[q],
                        "missed": q in miss,
                        "question_chars": len(instances[q].question),
                        "gold_abstract_chars": len(instances[q].abstract_text),
                        "gold_sections": len(instances[q].passages),
                    }
                    for q in sorted(ranks, key=lambda x: (ranks[x] is None, ranks[x] or 0))
                ],
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
