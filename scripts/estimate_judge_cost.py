#!/usr/bin/env python3
"""Estimate judge cost sweep over (claim, cited span) evaluation units without network access.

Produces a strict-JSON estimate artifact with full provenance and sensitivity ranges.
NON-GOAL: Does NOT execute network API calls, does NOT run the judge, does NOT emit CostRecord/costs.jsonl.
"""

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from biomedqa.verify import JUDGE_TEMPLATE

PRICE_SOURCE = "docs/adr/0004-local-generator-frontier-judge.md:75"
OPUS_5_INPUT_PRICE_PER_M = 5.0
OPUS_5_OUTPUT_PRICE_PER_M = 25.0
TOKENIZATION_METHOD = "char_approx_4.0_chars_per_token"


def count_tokens_char_approx(text: str, chars_per_token: float = 4.0) -> int:
    """Deterministic character-based token approximation (4.0 chars / token)."""
    if not text:
        return 0
    return math.ceil(len(text) / chars_per_token)


def get_git_commit() -> str:
    """Return the current HEAD commit hash if in git repo, else 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def estimate_judge_cost(
    records_path: str | Path,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Enumerate (claim, cited span) units from records_path and compute judge cost estimate.

    Does NOT issue network calls or emit CostRecords.
    """
    records_path = Path(records_path)
    units: list[tuple[str, str]] = []
    records_sha256 = ""

    if records_path.exists():
        raw_bytes = records_path.read_bytes()
        records_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        for line in raw_bytes.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for claim in rec.get("claims", []):
                hypothesis = claim.get("text", "")
                for cit in claim.get("citations", []):
                    premise = cit.get("quoted_text", "")
                    units.append((premise, hypothesis))

    n_units = len(units)
    token_counts = [
        count_tokens_char_approx(JUDGE_TEMPLATE.format(premise=p, hypothesis=h))
        for p, h in units
    ]
    total_input_tokens = sum(token_counts)

    # Token assumptions per unit
    out_per_unit_baseline = 2
    out_per_unit_pessimistic = 4
    out_per_unit_max = 8

    total_output_tokens_assumed = n_units * out_per_unit_baseline
    usd_input = (total_input_tokens * OPUS_5_INPUT_PRICE_PER_M) / 1_000_000
    usd_output = (total_output_tokens_assumed * OPUS_5_OUTPUT_PRICE_PER_M) / 1_000_000
    usd_total = usd_input + usd_output
    per_unit_usd = (usd_total / n_units) if n_units > 0 else 0.0

    pessimistic_out_tokens = n_units * out_per_unit_pessimistic
    pessimistic_usd_output = (
        pessimistic_out_tokens * OPUS_5_OUTPUT_PRICE_PER_M
    ) / 1_000_000
    pessimistic_usd_total = usd_input + pessimistic_usd_output

    max_out_tokens = n_units * out_per_unit_max
    max_usd_output = (max_out_tokens * OPUS_5_OUTPUT_PRICE_PER_M) / 1_000_000
    max_usd_total = usd_input + max_usd_output

    artifact: dict[str, Any] = {
        "estimate_only": True,
        "is_measured_evidence": False,
        "n_units": n_units,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens_assumed": total_output_tokens_assumed,
        "usd_input": usd_input,
        "usd_output": usd_output,
        "usd_total": usd_total,
        "per_unit_usd": per_unit_usd,
        "tokenization_method": TOKENIZATION_METHOD,
        "assumptions": [
            "No network access or API calls were performed; pure estimation.",
            "Input tokens counted via character approximation: ceil(len(prompt) / 4.0).",
            "Baseline output token budget per unit is 2 tokens (single integer percentage reply, 0-100).",
            "Pessimistic sensitivity range computed at 2x output tokens (4 tokens/unit) and max budget limit (8 tokens/unit).",
            f"Opus-5 rates derived from {PRICE_SOURCE} (${OPUS_5_INPUT_PRICE_PER_M}/MTok input, ${OPUS_5_OUTPUT_PRICE_PER_M}/MTok output).",
        ],
        "price_source": PRICE_SOURCE,
        "price_input_per_m": OPUS_5_INPUT_PRICE_PER_M,
        "price_output_per_m": OPUS_5_OUTPUT_PRICE_PER_M,
        "records_path": str(records_path),
        "records_sha256": records_sha256,
        "git_commit": get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sensitivity_range": {
            "floor": {
                "output_tokens_per_unit": out_per_unit_baseline,
                "total_output_tokens": total_output_tokens_assumed,
                "usd_output": usd_output,
                "usd_total": usd_total,
            },
            "pessimistic_2x": {
                "output_tokens_per_unit": out_per_unit_pessimistic,
                "total_output_tokens": pessimistic_out_tokens,
                "usd_output": pessimistic_usd_output,
                "usd_total": pessimistic_usd_total,
            },
            "max_budget_8": {
                "output_tokens_per_unit": out_per_unit_max,
                "total_output_tokens": max_out_tokens,
                "usd_output": max_usd_output,
                "usd_total": max_usd_total,
            },
        },
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, allow_nan=False)

    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate Opus-5 judge sweep cost without spending or API calls."
    )
    parser.add_argument(
        "--records",
        type=str,
        default="docs/harvest/generate_fp05_n100_guided_v4.verifier_scored.records.jsonl",
        help="Path to verifier scored records JSONL",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="docs/harvest/runbooks/judge_cost_estimate.json",
        help="Path to output JSON estimate artifact",
    )
    args = parser.parse_args()

    artifact = estimate_judge_cost(args.records, args.out)

    print("--- Opus-5 Judge Cost Estimate Summary ---")
    print(f"Records file:        {args.records}")
    print(f"Evaluation units:    {artifact['n_units']}")
    print(f"Input tokens:        {artifact['total_input_tokens']:,}")
    print(f"Input cost:          ${artifact['usd_input']:.4f}")
    print(f"Baseline output $:   ${artifact['usd_output']:.4f} (2 tokens/unit)")
    print(f"Baseline total $:    ${artifact['usd_total']:.4f}")
    print(
        f"Sensitivity range:   ${artifact['sensitivity_range']['floor']['usd_total']:.4f} (floor) "
        f"to ${artifact['sensitivity_range']['pessimistic_2x']['usd_total']:.4f} (2x output) "
        f"to ${artifact['sensitivity_range']['max_budget_8']['usd_total']:.4f} (max 8 tokens/unit)"
    )
    print(f"Price source:        {artifact['price_source']}")
    print(f"Estimate written to: {args.out}")
    print("==========================================")
    print(
        f"Headline Opus-5 Judge Estimate: ${artifact['usd_total']:.4f} "
        f"(Range: ${artifact['sensitivity_range']['floor']['usd_total']:.4f} - "
        f"${artifact['sensitivity_range']['pessimistic_2x']['usd_total']:.4f})"
    )


if __name__ == "__main__":
    main()
