#!/usr/bin/env python3
"""Populate Claim.verifier_scores on QueryRecords and write output + sidecar coverage JSON."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from biomedqa.schema import read_query_records, write_jsonl
from biomedqa.scoring.verifier_scores import (
    DEFAULT_VERIFIER_NAME,
    load_minicheck_cache,
    populate_verifier_scores,
)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Populate Claim.verifier_scores on QueryRecords from MiniCheck cache."
    )
    parser.add_argument(
        "--records",
        type=str,
        required=True,
        help="Path to input records JSONL",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="docs/harvest/minicheck_cache.json",
        help="Path to MiniCheck cache file (default: docs/harvest/minicheck_cache.json)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to output scored records JSONL (must differ from --records)",
    )
    parser.add_argument(
        "--verifier-name",
        type=str,
        default=DEFAULT_VERIFIER_NAME,
        help=f"Verifier name identifier (default: {DEFAULT_VERIFIER_NAME})",
    )

    args = parser.parse_args()

    records_path = Path(args.records).resolve()
    if not records_path.exists():
        print(f"Error: records file does not exist: {records_path}", file=sys.stderr)
        return 1

    cache_path = Path(args.cache).resolve()
    if not cache_path.exists():
        print(f"Error: cache file does not exist: {cache_path}", file=sys.stderr)
        return 1

    if args.out is not None:
        out_records_path = Path(args.out).resolve()
    else:
        stem = records_path.name
        for suffix in [".verifier_scored.records.jsonl", ".records.jsonl", ".jsonl"]:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        out_records_path = records_path.with_name(f"{stem}.verifier_scored.records.jsonl").resolve()

    if out_records_path == records_path:
        print(
            f"Error: Refusing to write output into the source path: {records_path}",
            file=sys.stderr,
        )
        return 1

    # Form sidecar coverage JSON path
    out_name = out_records_path.name
    if out_name.endswith(".verifier_scored.records.jsonl"):
        cov_name = out_name.removesuffix(".verifier_scored.records.jsonl") + ".verifier_scored.coverage.json"
    elif out_name.endswith(".records.jsonl"):
        cov_name = out_name.removesuffix(".records.jsonl") + ".coverage.json"
    elif out_name.endswith(".jsonl"):
        cov_name = out_name.removesuffix(".jsonl") + ".coverage.json"
    else:
        cov_name = out_records_path.stem + ".coverage.json"
    coverage_path = out_records_path.with_name(cov_name)

    print(f"Loading records from: {records_path}")
    records = list(read_query_records(records_path))

    print(f"Loading cache from: {cache_path}")
    cache = load_minicheck_cache(cache_path)

    cache_bytes = cache_path.read_bytes()
    cache_sha256 = hashlib.sha256(cache_bytes).hexdigest()

    populated_records, coverage = populate_verifier_scores(
        records, cache, verifier_name=args.verifier_name
    )

    provenance = {
        "source_records_path": str(records_path),
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha256,
        "verifier_name": args.verifier_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
    }
    coverage["provenance"] = provenance

    out_records_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing scored records to: {out_records_path}")
    write_jsonl(out_records_path, populated_records)

    print(f"Writing coverage report to: {coverage_path}")
    coverage_path.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"Done: {coverage['n_records']} records, {coverage['n_claims']} claims, "
        f"{coverage['n_citations']} citations ({coverage['n_scored']} scored, "
        f"{coverage['n_missing']} missing, rate={coverage['coverage_rate']:.4f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
