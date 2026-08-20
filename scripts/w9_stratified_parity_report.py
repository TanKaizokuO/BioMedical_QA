#!/usr/bin/env python3
"""ADR-0009 §5 W9 Stratified Robustness Check for Granularity Parity.

Usage:
    uv run python scripts/w9_stratified_parity_report.py docs/harvest/generate_fp05_n100_guided_batched --max-tokens 3584

Reads a run's `records.jsonl` and `costs.jsonl`, computing per-stratum parity statistics
and overall robustness verdicts across three pre-registered stratification schemes derived from
ADR-0009 §5 & §2:
  1. Compound structure (simple claims vs compound claims)
  2. Claim length bands (1-10, 11-15, 16-20, 21-30, 31+ words)
  3. Query claim volume (1-5 claims, 6-10 claims, 11+ claims per query)

Tolerates cost records carrying `output_tokens: null` without raising TypeError.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.harness import costs_path, records_path  # noqa: E402
from biomedqa.schema import CostRecord, read_jsonl, read_query_records  # noqa: E402
from biomedqa.scoring.granularity import (  # noqa: E402
    PARITY_TOLERANCE,
    parity_gate,
    stratified_parity_check,
)


def print_stratified_report(prefix: Path, max_tokens: int, min_queries: int = 5) -> None:
    records = list(read_query_records(records_path(prefix)))
    costs = [CostRecord(**d) for d in read_jsonl(costs_path(prefix))]

    print(f"ADR-0009 §5 W9 Stratified Robustness Check: {prefix.name}")
    print(f"  records: {len(records)}, cost rows: {len(costs)}, max_tokens: {max_tokens}, min_queries: {min_queries}")

    overall = parity_gate(records, basis="all records")
    print(f"\nOverall Parity Gate (all records):")
    print(f"  joint median: {overall.joint.median_words_per_claim:.1f} w/c, post_hoc median: {overall.post_hoc.median_words_per_claim:.1f} w/c")
    print(f"  gap: {overall.gap:+.1%} against ±{PARITY_TOLERANCE:.0%}  ->  {'PASS' if overall.passes else 'FAIL'}")

    schemes = stratified_parity_check(records, min_queries=min_queries)
    all_schemes_pass = True

    for scheme_name, gate in schemes.items():
        print(f"\nStratification Scheme: {scheme_name}")
        print(f"  {'stratum':<15}{'queries':>8}{'j_claims':>9}{'ph_claims':>10}{'j_median':>10}{'ph_median':>11}{'gap':>9}{'status':>14}")
        print("  " + "-" * 86)
        for s in gate.strata:
            if s.underpowered:
                status = "UNDERPOWERED"
                j_m = "-"
                ph_m = "-"
                gap_str = "-"
            else:
                status = "PASS" if s.passes else "FAIL"
                j_m = f"{s.joint_median_words:.1f}"
                ph_m = f"{s.post_hoc_median_words:.1f}"
                gap_str = f"{s.gap:+.1%}"
            print(f"  {s.stratum:<15}{s.n_queries:>8}{s.n_joint_claims:>9}{s.n_post_hoc_claims:>10}{j_m:>10}{ph_m:>11}{gap_str:>9}{status:>14}")

        scheme_verdict = "PASS" if gate.passes else "FAIL"
        if not gate.passes:
            all_schemes_pass = False
        print(f"  Scheme verdict -> {scheme_verdict} (powered strata: {len(gate.powered_strata)}/{len(gate.strata)})")

    print("\n" + "=" * 88)
    overall_verdict = "PASS" if all_schemes_pass else "FAIL"
    print(f"OVERALL W9 STRATIFIED ROBUSTNESS VERDICT: {overall_verdict}")
    print("=" * 88)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("prefix", help="artifact prefix, e.g. docs/harvest/generate_fp05_n100_guided_batched")
    ap.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="the per-call output cap the run was generated under",
    )
    ap.add_argument(
        "--min-queries",
        type=int,
        default=5,
        help="minimum queries required per stratum for power (default: 5)",
    )
    args = ap.parse_args()
    print_stratified_report(Path(args.prefix), max_tokens=args.max_tokens, min_queries=args.min_queries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
