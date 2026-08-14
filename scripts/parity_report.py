#!/usr/bin/env python3
"""The ADR-0009 parity gate for one run, off its committed artifacts. CPU-only, no GPU.

    uv run python scripts/parity_report.py docs/harvest/parity_iter1 --max-tokens 2560

Prints the gate on **both bases** — all records, and untruncated-only — because on `parity_iter0b`
they disagree by 18 points (+25.0% vs +42.9%) and a single number is not an answer to the gate. Also
prints claims/query and the compound profile, which together answer the question the gate cannot ask
itself: *did words/claim fall because the answers got finer, or because they got shorter?* If
claims/query drops alongside words/claim, the model is answering **less** rather than answering
**finer**, and a pass is a pass for the wrong reason.

`--max-tokens` is the **per-call** cap the run was generated under, and it is required: the
untruncated basis cannot be computed without it, and a post-hoc record's `completion_tokens` is the
sum of both its stages, so it cannot be recovered from the records file. Comparing against
`parity_iter0b` means 2560 (see `docs/harvest/parity_iter0.md`).

Blind by construction (ADR-0009 §6): nothing here reads a citation, a verifier score, or a label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.prompts import (  # noqa: E402
    PARITY_ITERATION_LIMIT,
    parity_budget_remains,
    parity_iteration_count,
)
from biomedqa.schema import CostRecord, System, read_jsonl, read_query_records  # noqa: E402
from biomedqa.scoring.granularity import (  # noqa: E402
    CALL_ORDER,
    PARITY_TOLERANCE,
    ArmGranularity,
    ParityGate,
    arm_granularity,
    compound_profile,
    parity_gate,
    stage_output_tokens,
    truncated_queries,
)

#: The baseline of record every iteration is judged against (`docs/harvest/parity_iter0.md`).
BASELINE = {"prefix": "parity_iter0b", "joint": 16, "post_hoc": 20, "gap": 0.250}


def _arm_row(arm: ArmGranularity) -> str:
    return (
        f"  {arm.system:<10}{arm.n_records:>6}{arm.n_claims:>9}"
        f"{arm.median_words_per_claim:>10.1f}{arm.mean_words_per_claim:>9.2f}"
        f"{arm.p25_words_per_claim:>7.0f}{arm.p75_words_per_claim:>7.0f}"
        f"{arm.p90_words_per_claim:>7.0f}{arm.median_claims_per_query:>10.1f}"
    )


def _print_basis(gate: ParityGate) -> None:
    print(f"\n{gate.basis}")
    print(f"  {'system':<10}{'recs':>6}{'claims':>9}{'median':>10}{'mean':>9}"
          f"{'p25':>7}{'p75':>7}{'p90':>7}{'claims/q':>10}")
    print("  " + "-" * 73)
    for arm in (gate.joint, gate.post_hoc):
        print(_arm_row(arm))
    verdict = "PASS" if gate.passes else "FAIL"
    print(f"  gap {gate.gap:+.1%} against ±{PARITY_TOLERANCE:.0%}  ->  {verdict}")
    if gate.requires_w9_robustness_check:
        print("  residual gap favours C2 -> ADR-0009 §5: the W9 stratified check is MANDATORY")
    elif not gate.passes:
        print("  residual gap runs against C2 -> ADR-0009 §5: note it and proceed")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("prefix", help="artifact prefix, e.g. docs/harvest/parity_iter1")
    ap.add_argument("--max-tokens", type=int, required=True,
                    help="the per-call output cap the run was generated under")
    args = ap.parse_args()

    prefix = Path(args.prefix)
    records = list(read_query_records(prefix.with_suffix(".records.jsonl")))
    costs = [CostRecord(**d) for d in read_jsonl(prefix.with_suffix(".costs.jsonl"))]

    print(f"{prefix.name}: {len(records)} records, {len(costs)} cost rows, "
          f"per-call cap {args.max_tokens}")

    stages = stage_output_tokens(records, costs)
    at_cap = {name: sum(1 for s in stages.values() if s[name] >= args.max_tokens)
              for name in CALL_ORDER}
    print("  calls at the cap: " + ", ".join(f"{k} {v}/{len(stages)}" for k, v in at_cap.items()))
    if at_cap["post_hoc_cite"] > at_cap["joint"]:
        print("  NOTE: the cite stage truncates more than joint does. Post-hoc claims are parsed "
              "from\n        that stage, so its truncation drops claims off the record.")

    truncated = truncated_queries(records, costs, args.max_tokens)
    _print_basis(parity_gate(records, basis="all records"))

    joint = arm_granularity(records, System.JOINT, exclude=truncated[System.JOINT.value])
    post_hoc = arm_granularity(records, System.POST_HOC,
                               exclude=truncated[System.POST_HOC.value])
    gap = ((post_hoc.median_words_per_claim - joint.median_words_per_claim)
           / joint.median_words_per_claim)
    _print_basis(ParityGate(basis="untruncated only", joint=joint, post_hoc=post_hoc, gap=gap))

    print("\ncompound profile (all records)")
    print(f"  {'system':<10}{'simple%':>9}{'simple med':>12}{'and%':>8}{'sub%':>7}{'2+comma%':>10}")
    print("  " + "-" * 56)
    for system in (System.JOINT, System.POST_HOC):
        p = compound_profile([c for r in records if r.system is system for c in r.claims])
        print(f"  {system.value:<10}{100 * p['simple_claim_share']:>8.1f}%"
              f"{p['median_words_per_simple_claim']:>12.1f}"
              f"{100 * p['marker_rate']['and']:>7.1f}%"
              f"{100 * p['marker_rate']['subordinate']:>6.1f}%"
              f"{100 * p['marker_rate']['multi_comma']:>9.1f}%")

    every = parity_gate(records, basis="all records")
    print(f"\nvs the baseline of record ({BASELINE['prefix']}: joint {BASELINE['joint']} / "
          f"post-hoc {BASELINE['post_hoc']}, {BASELINE['gap']:+.1%})")
    print(f"  joint    {every.joint.median_words_per_claim:>6.1f} "
          f"({every.joint.median_words_per_claim - BASELINE['joint']:+.1f})")
    print(f"  post_hoc {every.post_hoc.median_words_per_claim:>6.1f} "
          f"({every.post_hoc.median_words_per_claim - BASELINE['post_hoc']:+.1f})")
    print(f"  gap      {every.gap:>6.1%} ({every.gap - BASELINE['gap']:+.1%})")
    if every.post_hoc.median_claims_per_query < 8.0:
        print("  WARNING: post-hoc claims/query fell below the baseline's 8.0. Check that "
              "words/claim\n           fell because claims got finer, not because the answer got "
              "shorter.")

    print(f"\nparity ledger: {parity_iteration_count()} of {PARITY_ITERATION_LIMIT} used, "
          f"budget remains: {parity_budget_remains()}. Drop-dead Aug 30 (ADR-0009 §5).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
