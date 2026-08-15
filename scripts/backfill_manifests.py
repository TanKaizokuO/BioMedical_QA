#!/usr/bin/env python3
"""Manifest the artifacts that predate `harness.py`, so G5 can read them at all.

Every number in the paper today — Table 1's four rows, the parity iterations, the first citation-F1
pair — was produced before a manifest existed. G5 refuses a cell whose manifest cannot be produced,
so without this those numbers are unquotable, and re-running them costs 2 h of A4000 time for Table 1
and four generation passes for the parity loop.

**A backfilled manifest is worth less than a real one, and has to say so.** What it can and cannot
recover:

- *Recovered* — the knobs the artifact itself wrote down. `table1_baseline.py` recorded the index
  fingerprint, the corpus fingerprint, the title convention, the reranker, the pool size and `k`;
  `generate_smoke.py` recorded the model, `max_tokens`, the temperature, the citation cap, the seed
  and the context depth.
- *Recovered from git, and labelled as such* — `git_sha` is the commit that **added the records**, an
  upper bound on the tree that produced them, and `config_version` is `CONFIG_VERSION` as it stood in
  that commit. Both are checkable by anyone; neither is the run's own tree.
- *Not recoverable* — everything else is a **default**, which means the backfilled `config_hash` is a
  hash over partly-default values and does not identify the run. `harness.verify_run()` prints that
  caveat on every read, forever, and it is why `backfill_manifest()` refuses an empty `unrecovered`.

Idempotent per prefix only with `--overwrite`: a manifest that already exists is left alone, because
overwriting one is how a live manifest would silently become a backfilled one.

    uv run python scripts/backfill_manifests.py            # write the missing ones
    uv run python scripts/backfill_manifests.py --dry-run  # say what would be written
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.config import RunConfig  # noqa: E402
from biomedqa.harness import (  # noqa: E402
    backfill_manifest,
    commit_that_added,
    manifest_path,
    records_path,
    verify_run,
)
from biomedqa.schema import read_jsonl  # noqa: E402

HARVEST = _REPO / "docs" / "harvest"

#: Knobs whose *concept* postdates every artifact here: `CONFIG_VERSION` 1.4.0 added the
#: non-termination controls on 2026-08-14, after the last parity run. Recording them as unrecovered
#: rather than as their defaults is the difference between "the run used 0.0" and "the run predates
#: the knob" — and only the second one is true.
POSTDATES_EVERY_ARTIFACT = ("generation.frequency_penalty", "generation.stop", "scoring.max_claim_words")


@dataclass(frozen=True)
class Spec:
    """One artifact family: where its knobs were written down, and how to read them."""

    prefix: str
    source_suffix: str
    family: str


ARTIFACTS = (
    Spec("table1_rows_1_3", ".json", "retrieval"),
    Spec("table1_rows_1_4", ".json", "retrieval"),
    Spec("generate_smoke", ".summary.json", "generation"),
    Spec("parity_iter0", ".summary.json", "generation"),
    Spec("parity_iter0b", ".summary.json", "generation"),
    Spec("parity_iter1", ".summary.json", "generation"),
    Spec("parity_iter1b", ".summary.json", "generation"),
)


def config_version_at(sha: str) -> str | None:
    """`CONFIG_VERSION` as it stood in the commit that added the records.

    The artifact does not record it (except Table 1's, which does), and today's value is certainly
    wrong: 1.4.0 landed after every run here. The file at that commit is the closest checkable
    answer, and the manifest says that is where it came from.
    """
    if sha == "unknown":
        return None
    try:
        blob = subprocess.run(
            ["git", "show", f"{sha}:src/biomedqa/config.py"],
            cwd=_REPO, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r'^CONFIG_VERSION = "([^"]+)"', blob, re.MULTILINE)
    return match.group(1) if match else None


FLAGS = ("bm25", "dense", "rrf", "rerank")


def arm_id(flags: dict[str, Any]) -> str:
    """The `run_id` `table1_baseline.py` stamps on a row, rebuilt from that row's own flags.

    It derives the id the same way the script does — `f"table1_{label}"` with spaces and `+` mapped —
    so `{"bm25": True}` alone is `table1_bm25_only` and all four is
    `table1_bm25_and_dense_and_rrf_and_rerank`. The ids in the records are what this has to match, so
    a mismatch surfaces immediately as a foreign-record finding rather than as a wrong hash.
    """
    on = [name for name in FLAGS if flags[name]]
    return f"table1_{'_and_'.join(on)}" + ("_only" if len(on) == 1 else "")


def retrieval_knobs(source: dict) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    """`table1_baseline.py`'s run block, plus one arm per Table 1 row.

    The rows are four retrieval configs in one records file, and each row recorded its own flags — so
    the per-arm hashes are recovered, not defaulted. A retrieval-only run means the generation section
    is not "defaulted" but *inapplicable*; it is listed as unrecovered either way, because a reader
    cannot tell those two apart from a hash.
    """
    cfg = source["config"]
    knobs: dict[str, Any] = {
        "split": cfg["split"],
        "retrieval.corpus_id": cfg["corpus_id"],
        "retrieval.corpus_fingerprint": cfg["corpus_fingerprint"],
        "retrieval.top_k": cfg["k"],
    }
    unrecovered = ["chunk.*", "generation.* (no generation in this run)", "verifier.*", "scoring.*"]
    for key, field in (("title_segment", "retrieval.title_segment"),
                       ("reranker", "retrieval.reranker"),
                       ("pool_size", "retrieval.pool_size")):
        if key in cfg:
            knobs[field] = cfg[key]
        else:
            unrecovered.append(field)
    arms = {
        arm_id(row["config"]): {
            **{f"retrieval.{name}": row["config"][name] for name in FLAGS},
            "retrieval.rrf_k": row["config"]["rrf_k"],
            "retrieval.top_k": row["config"]["top_k"],
        }
        for row in source["rows"]
    }
    return knobs, unrecovered, arms


def generation_knobs(source: dict) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    """`generate_smoke.py`'s summary block, as it was before it deferred to the manifest."""
    cfg = source["config"]
    knobs: dict[str, Any] = {
        "generation.backend": "vllm",
        "generation.model": cfg["model"],
        "generation.max_tokens": cfg["max_tokens"],
        "generation.temperature": cfg["temperature"],
        "generation.max_citations": cfg["max_citations"],
        "generation.seeds": (cfg["seed"],),
        "retrieval.top_k": cfg["depth"],
    }
    # The contexts came from `dev_contexts_top10.jsonl`, which was dumped from the `empty` index —
    # so the retrieval identity is the default one. It is still unrecovered: the summary does not say
    # so, and a manifest may only assert what its artifact recorded.
    unrecovered = ["chunk.*", "retrieval.* (except top_k)", "verifier.*", "scoring.*"]
    return knobs, unrecovered, {}


READERS = {"retrieval": retrieval_knobs, "generation": generation_knobs}


def run_id_of(prefix: Path, arms: dict[str, Any]) -> str:
    """The `run_id` the records themselves carry. Not the prefix name: `parity_iter0` through
    `parity_iter1b` all stamp `w4_generate_smoke`, and renaming a run to match its prefix would mean
    editing the records.

    With arms, the records carry one id **per arm** and none for the run as a whole, so the prefix
    name is the only thing left to call it — and `arms` is what keeps the arm ids from reading as
    foreign records. Any id that is neither the prefix nor a declared arm is left for a human.
    """
    ids = {str(row.get("run_id")) for row in read_jsonl(records_path(prefix))}
    if arms:
        unexpected = ids - set(arms)
        if unexpected:
            raise SystemExit(
                f"{records_path(prefix)} carries run_ids {sorted(unexpected)} that no recovered arm "
                f"declares (arms: {sorted(arms)}) — resolve by hand."
            )
        return prefix.name
    if len(ids) != 1:
        raise SystemExit(f"{records_path(prefix)} carries run_ids {sorted(ids)} — resolve by hand.")
    return ids.pop()


def backfill(spec: Spec, *, dry_run: bool) -> list[str]:
    prefix = HARVEST / spec.prefix
    source_path = Path(str(prefix) + spec.source_suffix)
    source = json.loads(source_path.read_text())

    knobs, unrecovered, arms = READERS[spec.family](source)
    sha = commit_that_added(records_path(prefix))
    version = config_version_at(sha)
    recorded_version = source.get("config", {}).get("config_version")
    if recorded_version:
        knobs["config_version"] = recorded_version
    elif version:
        knobs["config_version"] = version
        unrecovered.append(f"config_version (read as {version} from CONFIG_VERSION at {sha[:7]})")
    else:
        unrecovered.append("config_version")
    unrecovered.extend(POSTDATES_EVERY_ARTIFACT)

    if dry_run:
        print(f"{spec.prefix:18s} would recover {len(knobs)} knobs and {len(arms)} arms, "
              f"{len(unrecovered)} unrecovered, sha {sha[:7]}, "
              f"config_version {knobs.get('config_version')}")
        return []

    backfill_manifest(
        RunConfig().ablate(spec.prefix, **knobs),
        prefix,
        run_id=run_id_of(prefix, arms),
        started_at=source.get("started_at"),
        finished_at=source.get("finished_at"),
        source=source_path.name,
        unrecovered=unrecovered,
        arms=arms,
    )
    return verify_run(prefix)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="re-backfill prefixes that already have a manifest (a live manifest would be lost)",
    )
    args = ap.parse_args()

    written = 0
    for spec in ARTIFACTS:
        prefix = HARVEST / spec.prefix
        if not records_path(prefix).exists():
            print(f"{spec.prefix:18s} skipped — no {records_path(prefix).name}")
            continue
        if manifest_path(prefix).exists():
            if not args.overwrite:
                print(f"{spec.prefix:18s} skipped — already manifested")
                continue
            manifest_path(prefix).unlink()
        problems = backfill(spec, dry_run=args.dry_run)
        if not args.dry_run:
            written += 1
            print(f"{spec.prefix:18s} manifested")
            for problem in problems:
                print(f"{'':18s}   caveat: {problem}")

    if not args.dry_run:
        print(f"\n{written} manifest(s) written. Every one is backfilled: verify_run() reports the "
              "caveat, and no table may quote one of these as if it were a live manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
