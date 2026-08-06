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

`--from-prescan` re-draws from the superset a **completed** scan already left on disk — minutes, no
network. It is not a resume: it refuses to run unless the existing manifest records a full
23,898,701-row scan. See `redraw_from_prescan`.

**Not resumable, deliberately.** A mid-scan failure restarts the read. Resuming would mean either
checkpointing the heap or trusting a partial scan, and a partial scan is precisely the failure the
row-count guard exists to refuse — a resume bug would forge the one number that proves the corpus
is uniform. The cost of being wrong here is an index; the cost of restarting is ~1 h of network on
a box that is otherwise idle. Nothing on the GPU is at risk either way.
"""

from __future__ import annotations

import argparse
import dataclasses
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
    CorpusDraw,
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


def redraw_from_prescan(out: Path, *, gold_pmids: set[int], target_n: int, seed: int) -> CorpusDraw:
    """Re-draw from the superset a **completed** scan left on disk, without re-reading 54 GB.

    This is not the resume path the module docstring refuses. A resume would let a *partial* scan
    satisfy the row-count guard — forging the one number that evidences the corpus is uniform. This
    refuses to run unless an existing `corpus_manifest.json` already records a full 23,898,701-row
    scan, and it carries that count and its gold-collision count forward rather than recomputing
    them from the superset, where both would be wrong. The redraw can only ever restate a scan that
    already passed the guards; it cannot manufacture one.

    **What it checks about the superset is the row count, and it has to be.** An earlier version
    compared every drawn key against the prescan cutoff and called that containment. It was not a
    check at all: `streaming_scan` only ever writes rows whose key is already below the cutoff, so
    the draw is under it by construction and the comparison could not fail. The one thing that
    varies is how much of the superset is still on disk — `streaming_scan` opens `prescan.jsonl`
    with `"w"`, so a completed run followed by a crashed re-scan leaves a truncated superset next to
    a manifest that still describes the full one. Nothing else here can see that: the scan and
    collision counts are carried from that manifest, and `draw_corpus` is handed the surviving rows
    as its own `expected_rows`. So the count is recorded at scan time and matched here.

    **The count is only evidence if it was written before the crash.** A manifest with no
    `n_prescan_rows` is refused rather than back-filled automatically, because back-filling from a
    file that may already be truncated would launder the very failure this exists to catch.

    Used after 2026-08-05: the scan completed, the write step then found MedRAG's repeated PMIDs,
    and only the draw needed redoing.
    """
    manifest_path = out / "corpus_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} is missing — --from-prescan needs a completed scan.")
    prior = json.loads(manifest_path.read_text())
    if prior.get("n_scanned") != MEDRAG_TOTAL_ROWS:
        raise SystemExit(
            f"the manifest records n_scanned={prior.get('n_scanned'):,}, not {MEDRAG_TOTAL_ROWS:,}. "
            "--from-prescan restates a completed scan's guards; it cannot manufacture them. "
            "Re-run the full scan."
        )
    if (prior.get("seed"), prior.get("target_n")) != (seed, target_n):
        raise SystemExit(
            f"the prescan was cut for seed={prior.get('seed')} target_n={prior.get('target_n'):,}, "
            f"and this run asks for seed={seed} target_n={target_n:,}. A different cutoff means the "
            "superset need not contain the draw. Re-run the full scan."
        )

    def prescan_rows():
        with (out / "prescan.jsonl").open() as fh:
            for line in fh:
                yield json.loads(line)

    n_rows = sum(1 for _ in prescan_rows())
    recorded = prior.get("n_prescan_rows")
    if recorded is None:
        raise SystemExit(
            f"{manifest_path} records no n_prescan_rows, so a truncated superset would be "
            f"indistinguishable from an intact one. The file now holds {n_rows:,} rows. If — and "
            "only if — that scan is known not to have been re-run since, add "
            f'"n_prescan_rows": {n_rows} to the manifest and try again. Otherwise re-run the full '
            "scan: the count is only evidence when it is recorded before the crash, not after."
        )
    if n_rows != recorded:
        raise SystemExit(
            f"prescan.jsonl holds {n_rows:,} rows, and the manifest records {recorded:,} — the "
            "superset was truncated after the scan that produced this manifest. streaming_scan "
            "opens it with 'w', so a re-scan that crashed leaves exactly this: a partial superset "
            "beside a surviving manifest. A redraw from it is a corpus over whatever fraction "
            "survived, reported as a full scan. Re-run the full scan."
        )

    draw = draw_corpus(prescan_rows(), gold_pmids=gold_pmids, target_n=target_n, seed=seed,
                       expected_rows=n_rows)

    cutoff, _ = prescan_cutoff(target_n, MEDRAG_TOTAL_ROWS)
    worst = max(selection_key(p, seed=seed) for p in draw.pmids)
    print(f"prescan            : {n_rows:,} rows, matching the manifest; draw sits "
          f"{(cutoff - worst) / cutoff:.2%} inside the cutoff (scan of {MEDRAG_TOTAL_ROWS:,} rows "
          "carried from the manifest)")

    return dataclasses.replace(
        draw,
        n_scanned=MEDRAG_TOTAL_ROWS,
        n_gold_collisions=prior["n_gold_collisions"],
    )


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
    ap.add_argument("--from-prescan", action="store_true",
                    help="re-draw from the superset a completed scan left on disk, no network")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gold = load_gold_pmids()
    print(f"gold PMIDs: {len(gold):,}  (dedup key set, ADR-0012 §1)")

    if args.from_prescan:
        draw = redraw_from_prescan(args.out, gold_pmids=gold, target_n=args.target_n,
                                   seed=args.seed)
    else:
        cutoff, over = prescan_cutoff(args.target_n, MEDRAG_TOTAL_ROWS)
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

    # Counted from the file as it stands at write time, not carried from the scan: the manifest's
    # job is to describe the prescan a later --from-prescan will actually read.
    with (args.out / "prescan.jsonl").open() as fh:
        n_prescan_rows = sum(1 for _ in fh)
    print(f"prescan rows       : {n_prescan_rows:,} recorded, so a later redraw can see truncation")

    (args.out / "corpus_manifest.json").write_text(
        json.dumps(draw.to_json() | {"n_prescan_rows": n_prescan_rows}, indent=2) + "\n")

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
