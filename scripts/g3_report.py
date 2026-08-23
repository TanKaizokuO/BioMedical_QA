#!/usr/bin/env python3
"""G3 Gate Report Driver (ADR-0020).

Consumes verifier-scored QueryRecords and human label annotations to report G3 gate status.

Usage:
    uv run python scripts/g3_report.py --records <path> [--annotations <path...>] [--keyfile <path>] [--primary-annotator <id>] [--cost-ratio <float>] --out <path>

Exit Code Convention:
    Returns 0 when the driver successfully completes and writes the report artifact,
    regardless of whether the G3 gate verdict passes or fails. The verdict pass/fail
    status is recorded in the output JSON `passes` field and printed on stdout as
    `G3 PASSES: true/false`. Returns 1 only for fatal execution errors (e.g. missing records file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.annotate import ingest_annotations  # noqa: E402
from biomedqa.harness import git_sha  # noqa: E402
from biomedqa.schema import (  # noqa: E402
    CostRecord,
    QueryRecord,
    read_cost_records,
    read_query_records,
)
from biomedqa.scoring.calibration import (  # noqa: E402
    G3_AUROC_MIN,
    G3_COST_RATIO_MIN,
    gate_g3,
    join_scores_and_labels,
)
from biomedqa.scoring.cost import _get_val, overhead_ratio  # noqa: E402

DEFAULT_VERIFIER_NAME = "minicheck"


def detect_verifier_name(records: list[QueryRecord], preferred: str | None = None) -> str:
    """Detect verifier score name from records or fallback to preferred/default."""
    if preferred:
        return preferred
    names = {v.name for r in records for c in r.claims for v in c.verifier_scores}
    if "lytang/MiniCheck-Flan-T5-Large" in names:
        return "lytang/MiniCheck-Flan-T5-Large"
    if "minicheck" in names:
        return "minicheck"
    if names:
        return next(iter(names))
    return DEFAULT_VERIFIER_NAME

def file_sha256(path: Path) -> str | None:
    """Compute sha256 hex digest of a file, returning None if non-existent."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()




def sanitize_nans_for_json(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN, inf) with None for valid JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_nans_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nans_for_json(v) for v in obj]
    return obj


def score_records(
    records: list[QueryRecord],
    verifier_name: str = DEFAULT_VERIFIER_NAME,
    primary_annotator: str | None = None,
) -> tuple[list[float], list[bool], list[str], dict[str, Any]]:
    """Compute join diagnostics and perform score/label join via `join_scores_and_labels`."""
    n_records = len(records)
    n_claims = sum(len(r.claims) for r in records)
    n_citations = sum(len(c.citations) for r in records for c in r.claims)
    n_scored = 0
    n_missing_scores = 0
    n_missing_annotations = 0

    for r in records:
        for c in r.claims:
            if not c.citations:
                continue
            matching_scores = [v for v in c.verifier_scores if v.name == verifier_name]
            for cit_idx in range(len(c.citations)):
                if cit_idx < len(matching_scores):
                    n_scored += 1
                else:
                    n_missing_scores += 1

                matching_labels = [
                    h for h in c.human_labels if h.citation_index == cit_idx or (h.citation_index is None and cit_idx == 0)
                ]
                if not matching_labels:
                    n_missing_annotations += 1
    n_extra_citations = n_citations - (n_scored + n_missing_scores)
    scores: list[float] = []
    labels: list[bool] = []
    clusters: list[str] = []
    n_no_majority = 0
    no_majority_rate = 0.0

    if n_missing_annotations == 0 and n_scored > 0 and (n_scored + n_missing_scores) > 0:
        try:
            joined = join_scores_and_labels(
                records, verifier_name=verifier_name, primary_annotator=primary_annotator
            )
            n_no_majority = joined.n_no_majority
            no_majority_rate = joined.no_majority_rate
            scores = [r.score for r in joined]
            labels = [r.is_supporting for r in joined]
            clusters = [r.question_id for r in joined]
        except ValueError:
            pass

    diagnostics = {
        "n_records": n_records,
        "n_claims": n_claims,
        "n_citations": n_citations,
        "n_scored": n_scored,
        "n_missing_scores": n_missing_scores,
        "n_missing_annotations": n_missing_annotations,
        "n_no_majority": n_no_majority,
        "no_majority_rate": no_majority_rate,
        "n_extra_citations": n_extra_citations,
    }
    return scores, labels, clusters, diagnostics


def compute_verdict(
    scores: list[float],
    labels: list[bool],
    clusters: list[str],
    cost_ratio: float | None,
    diagnostics: dict[str, Any],
    missing_judge_evidence: bool = False,
) -> dict[str, Any]:
    """Pass score/label pairs into `gate_g3` and adjust reasons if evidence is missing."""
    verdict = gate_g3(
        scores,
        labels,
        clusters=clusters if scores else None,
        cost_ratio=cost_ratio,
        n_no_majority=diagnostics["n_no_majority"],
        no_majority_rate=diagnostics["no_majority_rate"],
    )

    reasons: list[str] = []
    if diagnostics["n_missing_annotations"] > 0 or len(scores) == 0:
        reasons.append("missing_human_labels")

    if missing_judge_evidence:
        reasons.append("missing_judge_cost_evidence")
    elif cost_ratio is None:
        reasons.append("cost_ratio_missing")
    elif not math.isfinite(cost_ratio):
        reasons.append("verifier_cost_unpriced")
    elif cost_ratio < G3_COST_RATIO_MIN:
        reasons.append(f"cost_ratio_below_threshold ({cost_ratio} < {G3_COST_RATIO_MIN})")
    elif not math.isnan(verdict.get("auroc", float("nan"))) and verdict["auroc"] < G3_AUROC_MIN:
        reasons.append(f"auroc_below_threshold ({verdict['auroc']:.4f} < {G3_AUROC_MIN})")

    if reasons:
        verdict["reason"] = "; ".join(reasons)

    return verdict


def print_verdict_block(verdict: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    """Print short human-readable verdict block ending in G3 PASSES: true/false."""
    print("=== G3 Gate Verdict — Cheap Verifier Evaluation (ADR-0020) ===")
    print(
        f"Records: {diagnostics['n_records']} | Claims: {diagnostics['n_claims']} | Citations: {diagnostics['n_citations']}"
    )
    print(
        f"Scored: {diagnostics['n_scored']} | Missing Scores: {diagnostics['n_missing_scores']} | "
        f"Missing Labels: {diagnostics['n_missing_annotations']}"
    )

    auroc_val = verdict.get("auroc", float("nan"))
    auroc_str = f"{auroc_val:.4f}" if not math.isnan(auroc_val) else "N/A"
    cost_val = verdict.get("cost_ratio")
    cost_str = f"{cost_val:.2f}x" if cost_val is not None and not math.isnan(cost_val) else "N/A"

    print(
        f"AUROC: {auroc_str} (min {verdict['auroc_min']}) — {'passes' if verdict['auroc_passes'] else 'FAILS'}"
    )
    print(
        f"Cost Ratio: {cost_str} (min {verdict['cost_ratio_min']}x) — {'passes' if verdict['cost_passes'] else 'FAILS'}"
    )
    print(f"Reason: {verdict['reason']}")
    passes_str = "true" if verdict["passes"] else "false"
    print(f"G3 PASSES: {passes_str}")


def build_report(
    args: argparse.Namespace,
    diagnostics: dict[str, Any],
    verdict: dict[str, Any],
    finished_at: str,
    verifier_name: str = DEFAULT_VERIFIER_NAME,
    cost_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build self-describing JSON report dict containing gate verbatim, diagnostics, and provenance."""
    ann_sources = (
        [str(p) for p in args.annotations]
        if isinstance(args.annotations, list)
        else ([str(args.annotations)] if args.annotations else None)
    )
    ann_sha = (
        [file_sha256(p) for p in args.annotations]
        if isinstance(args.annotations, list)
        else ([file_sha256(args.annotations)] if args.annotations and args.annotations.exists() else None)
    )
    keyfile_source = str(args.keyfile) if getattr(args, "keyfile", None) else None
    keyfile_sha = (
        file_sha256(args.keyfile)
        if getattr(args, "keyfile", None) and args.keyfile.exists()
        else None
    )
    cost_prov = cost_provenance or {}

    report: dict[str, Any] = {
        "script": "scripts/g3_report.py",
        "finished_at": finished_at,
        "records_source": str(args.records),
        "records_sha256": file_sha256(args.records),
        "annotations_source": ann_sources,
        "annotations_sha256": ann_sha,
        "keyfile_source": keyfile_source,
        "keyfile_sha256": keyfile_sha,
        "cost_source_type": cost_prov.get("source_type", "none"),
        "cost_records_source": cost_prov.get("records_source"),
        "cost_records_sha256": cost_prov.get("records_sha256"),
        "cost_provenance": cost_prov,
        "judge_run_specification": {
            "required_component": "judge",
            "required_backend": "anthropic:claude-opus-5",
            "required_cost_record_fields": [
                "run_id",
                "query_id",
                "component",
                "backend",
                "input_tokens",
                "output_tokens",
                "usd",
                "wall_s",
            ],
            "required_population": "All 1,257 (claim, cited span) evaluation units in gold evaluation set",
            "pricing_provenance": "Anthropic listed API rates (ADR-0004:73-79)",
            "verifier_hardware_provenance": "NVIDIA A4000 (ADR-0008, research_roadmap.md:519)",
            "doc_reference": "docs/harvest/runbooks/g3_judge_run_spec.md",
        },
        "git_commit": git_sha(_REPO),
        "verifier": verifier_name,
        "thresholds": {
            "auroc_min": G3_AUROC_MIN,
            "cost_ratio_min": G3_COST_RATIO_MIN,
        },
        "diagnostics": diagnostics,
        "gate": verdict,
    }
    # Unpack verdict keys into top level for direct access
    for k, v in verdict.items():
        report[k] = v
    return report

def main() -> int:
    ap = argparse.ArgumentParser(description="G3 Gate Report Driver (CPU)")
    ap.add_argument("--records", type=Path, required=True, help="Path to QueryRecord JSONL file")
    ap.add_argument(
        "--annotations",
        type=Path,
        nargs="+",
        default=None,
        help="Path(s) to human annotation JSONL file(s)",
    )
    ap.add_argument(
        "--keyfile",
        type=Path,
        default=None,
        help="Path to blinding keyfile JSONL mapping unit_id to record metadata",
    )
    ap.add_argument(
        "--primary-annotator",
        type=str,
        default=None,
        help="Primary annotator ID for tie-breaking majority votes (ADR-0016)",
    )
    ap.add_argument("--cost-ratio", type=float, default=None, help="Cost reduction ratio vs Opus")
    ap.add_argument("--costs", type=Path, default=None, help="Path to CostRecord JSONL file")
    ap.add_argument(
        "--verifier-gpu-hourly-rate",
        type=float,
        default=None,
        help="Operator-supplied verifier GPU hourly rate ($/hr) for hardware-normalized cost evaluation",
    )
    ap.add_argument("--verifier", type=str, default=None, help="Verifier score name filter")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON file path")
    args = ap.parse_args()

    if args.cost_ratio is not None and args.costs is not None:
        print("Error: --cost-ratio and --costs are mutually exclusive", file=sys.stderr)
        return 1

    if not args.records.exists():
        print(f"Error: --records file does not exist: {args.records}", file=sys.stderr)
        return 1

    finished_at = datetime.now(timezone.utc).isoformat()

    cost_source_type = "none"
    cost_records_source = None
    cost_records_sha256 = None
    cost_overhead_summary = None
    n_ours_cost_records = None
    n_judge_cost_records = None
    cost_ratio = None
    missing_judge_evidence = False

    if args.cost_ratio is not None:
        cost_source_type = "hand_passed"
        cost_ratio = args.cost_ratio
    elif args.costs is not None:
        cost_source_type = "derived"
        cost_records_source = str(args.costs)
        if not args.costs.exists():
            print(f"Error: --costs file does not exist: {args.costs}", file=sys.stderr)
            return 1
        cost_records_sha256 = file_sha256(args.costs)
        cost_records = list(read_cost_records(args.costs))
        judge_records = [c for c in cost_records if (_get_val(c, "component") or "") == "judge"]
        ours_records = [c for c in cost_records if (_get_val(c, "component") or "") != "judge"]
        n_judge_cost_records = len(judge_records)
        n_ours_cost_records = len(ours_records)
        if args.verifier_gpu_hourly_rate is not None and args.verifier_gpu_hourly_rate > 0:
            rate_per_sec = args.verifier_gpu_hourly_rate / 3600.0
            priced_ours = []
            for c in ours_records:
                c_wall = _get_val(c, "wall_s")
                if c_wall is not None and c_wall > 0:
                    priced_ours.append(
                        CostRecord(
                            run_id=_get_val(c, "run_id") or "unspecified",
                            query_id=_get_val(c, "query_id"),
                            component=_get_val(c, "component") or "verify",
                            backend=_get_val(c, "backend") or "unknown",
                            input_tokens=_get_val(c, "input_tokens"),
                            output_tokens=_get_val(c, "output_tokens"),
                            usd=float(c_wall * rate_per_sec),
                            wall_s=c_wall,
                        )
                    )
                else:
                    priced_ours.append(
                        CostRecord(
                            run_id=_get_val(c, "run_id") or "unspecified",
                            query_id=_get_val(c, "query_id"),
                            component=_get_val(c, "component") or "verify",
                            backend=_get_val(c, "backend") or "unknown",
                            input_tokens=_get_val(c, "input_tokens"),
                            output_tokens=_get_val(c, "output_tokens"),
                            usd=None,
                            wall_s=c_wall,
                        )
                    )
            ours_records = priced_ours

        if not judge_records:
            cost_ratio = None
            missing_judge_evidence = True
            cost_overhead_summary = {
                "usd_ratio": None,
                "note": "empty judge series: component='judge' CostRecords absent",
            }
        else:
            cost_overhead_summary = overhead_ratio(ours_records, judge_records)
            cost_ratio = cost_overhead_summary.get("usd_ratio")

    cost_provenance = {
        "source_type": cost_source_type,
        "records_source": cost_records_source,
        "records_sha256": cost_records_sha256,
        "hand_passed_cost_ratio": args.cost_ratio,
        "verifier_gpu_hourly_rate": args.verifier_gpu_hourly_rate,
        "n_ours_cost_records": n_ours_cost_records,
        "n_judge_cost_records": n_judge_cost_records,
        "cost_overhead_summary": cost_overhead_summary,
    }

    records = list(read_query_records(args.records))
    if args.annotations is not None:
        if args.keyfile is None:
            print("Error: --keyfile is required when --annotations is specified", file=sys.stderr)
            return 1
        try:
            ingest_annotations(
                label_files=args.annotations,
                keyfile=args.keyfile,
                records=records,
            )
        except Exception as err:
            print(f"Error loading annotations from {args.annotations}: {err}", file=sys.stderr)
            return 1

    verifier_name = detect_verifier_name(records, args.verifier)
    scores, labels, clusters, diagnostics = score_records(
        records, verifier_name=verifier_name, primary_annotator=args.primary_annotator
    )
    verdict = compute_verdict(
        scores,
        labels,
        clusters,
        cost_ratio,
        diagnostics,
        missing_judge_evidence=missing_judge_evidence,
    )

    print_verdict_block(verdict, diagnostics)
    report = build_report(
        args,
        diagnostics,
        verdict,
        finished_at,
        verifier_name=verifier_name,
        cost_provenance=cost_provenance,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    clean_report = sanitize_nans_for_json(report)
    args.out.write_text(
        json.dumps(clean_report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
