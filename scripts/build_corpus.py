"""Draw the ~2M distractor corpus from `MedRAG/pubmed`, deduped against gold (ADR-0012 §1).

**Runs on the A4000 box**, not on the dev machine — it streams ~54 GB and feeds an encode that
lives there. Nothing here needs the GPU.

    uv run --with datasets python scripts/build_corpus.py --out data/corpus

Writes two artifacts:

    <out>/corpus_manifest.json   the seed, the guards' counts, the fingerprint, the 2M PMIDs
    <out>/corpus.jsonl           the 2M rows' text, ready for the chunker

**One pass over the network, and exactly 2M rows.** Bottom-k needs the whole scan before it knows
which rows it kept, which would ordinarily force a second 54 GB read. Instead the scan keeps a
generous superset on disk — every row under `corpus.prescan_cutoff`, ~30 sd above the target — and
the exact bottom-2M is taken from that. The scan itself still sees all 23,898,701 rows, so
`draw_corpus`'s guards are unaffected, which matters because the **row-count guard is the only one
that catches the partial-parquet trap**: just 3 of the 1,000 gold PMIDs fall inside the partial
export's range, so the collision guard would see 3 collisions and happily pass.

**Not resumable, deliberately.** A mid-scan failure restarts the read. Resuming would mean either
checkpointing the heap or trusting a partial scan, and a partial scan is precisely the failure the
row-count guard exists to refuse — a resume bug would forge the one number that proves the corpus
is uniform. The cost of being wrong here is an index; the cost of restarting is ~1 h of network on
a box that is otherwise idle. Nothing on the GPU is at risk either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomedqa.corpus import (  # noqa: E402
    CORPUS_SEED,
    MEDRAG_DATA_FILES,
    MEDRAG_REPO,
    MEDRAG_TOTAL_ROWS,
    TARGET_N,
    draw_corpus,
    load_gold_pmids,
    prescan_cutoff,
    selection_key,
)

def streaming_scan(out: Path, seed: int, cutoff: int):
    """Yield every MedRAG row, side-writing the Bernoulli superset to disk as it goes.

    A pass-through generator rather than a filter: `draw_corpus` must see the full 23.9M so its
    row-count guard means something, while only the superset is paid for in disk.
    """
    from datasets import load_dataset

    ds = load_dataset(MEDRAG_REPO, data_files=MEDRAG_DATA_FILES, split="train", streaming=True)
    kept = 0
    with (out / "prescan.jsonl").open("w") as fh:
        for n, row in enumerate(ds, 1):
            if selection_key(row["PMID"], seed=seed) < cutoff:
                fh.write(json.dumps(row) + "\n")
                kept += 1
            if n % 1_000_000 == 0:
                print(f"  scanned {n:>12,} / {MEDRAG_TOTAL_ROWS:,}   kept {kept:>9,}", flush=True)
            yield row


def choose_one_row_per_pmid(prescan: Path, selected: set[int]) -> dict[int, str]:
    """`{PMID: row id}` — the single row each drawn article contributes to the index.

    **PubMed re-publishes revised records, and MedRAG keeps every revision as its own row.** Measured
    on the 2026-08-05 prescan: 244 of 2,041,867 drawn PMIDs carry 2–3 rows, 129 of them with
    differing `content`. The differences are revisions of one whole abstract, not chunks of it —
    `22367489` is `b-subunit` against `beta-subunit`, `22453897` is the same text with an
    abbreviation list appended — so ADR-0014 §2's "a row is an article" holds, and the unstated
    "and appears once" does not.

    **Longest `content`, ties broken by the lexicographically smallest `id`.** Longest because a
    revision that adds text is the more complete record and both observed cases add rather than cut;
    the tie-break exists only so the rule is total, since `corpus_id` promises seed → ID list and a
    rule that leaves 244 rows to dict ordering does not keep that promise.

    Returning the ids rather than the rows keeps ~2M short strings in memory instead of ~2 GB of
    JSON, at the cost of a second pass over a local file.
    """
    best: dict[int, tuple[int, str]] = {}
    with prescan.open() as fh:
        for line in fh:
            row = json.loads(line)
            pmid = row["PMID"]
            if pmid not in selected:
                continue
            cand = (len(row.get("content") or ""), row["id"])
            prev = best.get(pmid)
            if prev is None or cand[0] > prev[0] or (cand[0] == prev[0] and cand[1] < prev[1]):
                best[pmid] = cand
    return {pmid: rid for pmid, (_, rid) in best.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/corpus"))
    ap.add_argument("--seed", type=int, default=CORPUS_SEED)
    ap.add_argument("--target-n", type=int, default=TARGET_N)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gold = load_gold_pmids()
    cutoff, over = prescan_cutoff(args.target_n, MEDRAG_TOTAL_ROWS)
    print(f"gold PMIDs: {len(gold):,}  (dedup key set, ADR-0012 §1)")
    print(f"streaming {MEDRAG_REPO}:{MEDRAG_DATA_FILES} — {MEDRAG_TOTAL_ROWS:,} rows")
    print(f"prescan superset: ~{over:,} rows kept on disk for a {args.target_n:,} draw "
          f"({(over - args.target_n) / args.target_n**0.5:.0f} sd headroom)\n")

    draw = draw_corpus(
        streaming_scan(args.out, args.seed, cutoff),
        gold_pmids=gold,
        target_n=args.target_n,
        seed=args.seed,
    )

    print(f"\nscanned            : {draw.n_scanned:,}")
    print(f"gold collisions    : {draw.n_gold_collisions:,} of {len(gold):,} removed before the draw")
    print(f"drawn              : {len(draw.pmids):,}")
    print(f"duplicate rows     : {draw.n_duplicate_rows:,} suppressed (revised records, one per PMID)")
    print(f"fingerprint        : {draw.fingerprint}")

    (args.out / "corpus_manifest.json").write_text(json.dumps(draw.to_json(), indent=2) + "\n")

    selected = set(draw.pmids)
    chosen = choose_one_row_per_pmid(args.out / "prescan.jsonl", selected)
    if len(chosen) != len(selected):
        raise SystemExit(
            f"the superset holds {len(chosen):,} of the {len(selected):,} drawn PMIDs. The prescan "
            "cutoff did not contain the draw — do not encode this."
        )

    written = 0
    with (args.out / "prescan.jsonl").open() as src, (args.out / "corpus.jsonl").open("w") as dst:
        for line in src:
            row = json.loads(line)
            if chosen.get(row["PMID"]) == row["id"]:
                dst.write(line)
                written += 1
    if written != len(selected):
        raise SystemExit(
            f"wrote {written:,} rows for {len(selected):,} drawn PMIDs — do not encode this."
        )

    print(f"\nwrote {args.out}/corpus.jsonl ({written:,} rows) and corpus_manifest.json")
    print("Paste the fingerprint and the collision count back; they go in the run manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
