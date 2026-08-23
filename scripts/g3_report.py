#!/usr/bin/env python3
"""G3 Gate Report Driver (ADR-0020).

Consumes verifier-scored QueryRecords and human label annotations to report G3 gate status.

Usage:
    uv run python scripts/g3_report.py --records <path> [--annotations <path>] [--cost-ratio <float>] --out <path>

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

from biomedqa.harness import git_sha  # noqa: E402
from biomedqa.schema import (  # noqa: E402
    HumanLabel,
    QueryRecord,
    SupportLabel,
    read_query_records,
)
from biomedqa.scoring.calibration import (  # noqa: E402
    G3_AUROC_MIN,
    G3_COST_RATIO_MIN,
    gate_g3,
    join_scores_and_labels,
)

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


def load_and_apply_annotations(records: list[QueryRecord], annotations_path: Path) -> None:
    """Load human label annotations from file and attach to matching claims on records."""
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    content = annotations_path.read_text(encoding="utf-8").strip()
    if not content:
        return

    rows: list[dict[str, Any]] = []
    if content.startswith("["):
        rows = json.loads(content)
    else:
        for line in content.splitlines():
            if line.strip():
                rows.append(json.loads(line))

    claim_map = {(r.query_id, c.claim_id): c for r in records for c in r.claims}

    for r in rows:
        if "query_id" in r and "claim_id" in r:
            key = (str(r["query_id"]), str(r["claim_id"]))
            if key in claim_map:
                claim = claim_map[key]
                citation_idx = r.get("citation_index")
                annotator_id = str(r.get("annotator_id", "human"))
                lbl_val = r["support_label"]
                support_lbl = SupportLabel(lbl_val) if isinstance(lbl_val, str) else lbl_val
                claim_validity = bool(r.get("claim_validity", True))
                notes = r.get("notes")

                dup = any(
                    h.annotator_id == annotator_id and h.citation_index == citation_idx
                    for h in claim.human_labels
                )
                if not dup:
                    claim.human_labels.append(
                        HumanLabel(
                            annotator_id=annotator_id,
                            support_label=support_lbl,
                            claim_validity=claim_validity,
                            citation_index=citation_idx,
                            notes=notes,
                        )
                    )


def score_records(
    records: list[QueryRecord], verifier_name: str = DEFAULT_VERIFIER_NAME
) -> tuple[list[float], list[bool], list[str], dict[str, Any]]:
    """Compute join diagnostics and perform score/label join via `join_scores_and_labels`."""
    n_records = len(records)
    n_claims = sum(len(r.claims) for r in records)
    n_citations = sum(len(c.citations) for r in records for c in r.claims)
    n_scored = 0
    n_missing_scores = 0
    n_missing_annotations = 0
    n_extra_citations = sum(
        max(0, len(c.citations) - 1) for r in records for c in r.claims if c.citations
    )

    for r in records:
        for c in r.claims:
            if not c.citations:
                continue
            has_score = any(v.name == verifier_name for v in c.verifier_scores)
            if has_score:
                n_scored += 1
            else:
                n_missing_scores += 1

            if not c.human_labels:
                n_missing_annotations += 1
    scores: list[float] = []
    labels: list[bool] = []
    clusters: list[str] = []
    n_no_majority = 0
    no_majority_rate = 0.0

    if n_missing_annotations == 0 and n_scored > 0 and (n_scored + n_missing_scores) > 0:
        try:
            joined = join_scores_and_labels(records, verifier_name=verifier_name)
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

    if diagnostics["n_missing_annotations"] > 0 or len(scores) == 0:
        reasons: list[str] = []
        if diagnostics["n_missing_annotations"] > 0 or len(scores) == 0:
            reasons.append("missing_human_labels")
        if cost_ratio is None:
            reasons.append("cost_ratio_missing")
        elif math.isnan(cost_ratio) or cost_ratio < G3_COST_RATIO_MIN:
            reasons.append(f"cost_ratio_below_threshold ({cost_ratio} < {G3_COST_RATIO_MIN})")
        elif not math.isnan(verdict.get("auroc", float("nan"))) and verdict["auroc"] < G3_AUROC_MIN:
            reasons.append(f"auroc_below_threshold ({verdict['auroc']:.4f} < {G3_AUROC_MIN})")

        verdict["reason"] = "; ".join(reasons) if reasons else verdict["reason"]

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
) -> dict[str, Any]:
    """Build self-describing JSON report dict containing gate verbatim, diagnostics, and provenance."""
    report: dict[str, Any] = {
        "script": "scripts/g3_report.py",
        "finished_at": finished_at,
        "records_source": str(args.records),
        "records_sha256": file_sha256(args.records),
        "annotations_source": str(args.annotations) if args.annotations else None,
        "annotations_sha256": file_sha256(args.annotations)
        if args.annotations and args.annotations.exists()
        else None,
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
    ap.add_argument("--annotations", type=Path, default=None, help="Path to human annotations file")
    ap.add_argument("--cost-ratio", type=float, default=None, help="Cost reduction ratio vs Opus")
    ap.add_argument("--verifier", type=str, default=None, help="Verifier score name filter")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON file path")
    args = ap.parse_args()

    if not args.records.exists():
        print(f"Error: --records file does not exist: {args.records}", file=sys.stderr)
        return 1

    finished_at = datetime.now(timezone.utc).isoformat()

    records = list(read_query_records(args.records))
    if args.annotations is not None:
        try:
            load_and_apply_annotations(records, args.annotations)
        except Exception as err:
            print(f"Error loading annotations from {args.annotations}: {err}", file=sys.stderr)
            return 1

    verifier_name = detect_verifier_name(records, args.verifier)
    scores, labels, clusters, diagnostics = score_records(records, verifier_name=verifier_name)
    verdict = compute_verdict(scores, labels, clusters, args.cost_ratio, diagnostics)

    print_verdict_block(verdict, diagnostics)

    report = build_report(args, diagnostics, verdict, finished_at, verifier_name=verifier_name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
