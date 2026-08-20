"""Citation-F1 paired contrast between joint and post-hoc attribution (Goal 6).

Calculates corpus-level citation precision, recall, and F1 for joint and post-hoc arms,
and computes the paired joint-minus-post_hoc citation-F1 delta with a bootstrap
confidence interval resampled over queries (ADR-0011 §2).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

from biomedqa.schema import CostRecord, QueryRecord, System, read_jsonl, read_query_records
from biomedqa.scoring.calibration import bootstrap_ci
from biomedqa.scoring.citation import Phi, _concat, _spans, answered_claims, citation_f1
from biomedqa.verify import MiniCheckVerifier, phi_from_scores


def load_run(prefix_str: str | Path) -> tuple[list[QueryRecord], list[CostRecord], dict]:
    """Load records, costs, and summary for a given run prefix.

    Tolerates missing costs or summary files, and tolerates cost records
    with null input/output token fields.
    """
    prefix = Path(prefix_str)
    rec_path = prefix.with_name(f"{prefix.name}.records.jsonl")
    cost_path = prefix.with_name(f"{prefix.name}.costs.jsonl")
    sum_path = prefix.with_name(f"{prefix.name}.summary.json")

    if not rec_path.exists():
        raise FileNotFoundError(f"Records file not found: {rec_path}")

    records = list(read_query_records(rec_path))

    costs: list[CostRecord] = []
    if cost_path.exists():
        for d in read_jsonl(cost_path):
            # CostRecord dataclass accepts input_tokens=None, output_tokens=None
            costs.append(CostRecord(**d))

    summary: dict = {}
    if sum_path.exists():
        summary = json.loads(sum_path.read_text(encoding="utf-8"))

    return records, costs, summary


def collect_phi_pairs(records: Iterable[QueryRecord]) -> set[tuple[str, str]]:
    """Collect all (premise, hypothesis) pairs queried by citation_f1 for these records."""
    pairs: set[tuple[str, str]] = set()
    for record in records:
        passages = {p.passage_id: p.text for p in record.retrieved if p.text is not None}
        for claim in answered_claims(record):
            spans = _spans(claim, passages)
            if spans:
                pairs.add((_concat(spans), claim.text))
                for i, span in enumerate(spans):
                    pairs.add((span, claim.text))
                    rest = spans[:i] + spans[i + 1 :]
                    if rest:
                        pairs.add((_concat(rest), claim.text))
    return pairs


def pair_queries(
    records: Iterable[QueryRecord],
    costs: Iterable[CostRecord] | None = None,
    summary: dict | None = None,
) -> tuple[list[tuple[QueryRecord, QueryRecord]], list[dict]]:
    """Pair joint and post_hoc query records by query_id.

    Returns:
        (paired_list, dropped_list)
        where paired_list is a list of (joint_record, post_hoc_record) sorted by query_id,
        and dropped_list contains dicts with {"query_id": qid, "reason": reason}.
    """
    records_list = list(records)
    joint_map = {r.query_id: r for r in records_list if r.system == System.JOINT}
    post_hoc_map = {r.query_id: r for r in records_list if r.system == System.POST_HOC}

    all_qids = sorted(set(joint_map.keys()) | set(post_hoc_map.keys()))

    # Track rejected calls per (query_id, system)
    rejected_set: set[tuple[str, str]] = set()

    if summary and "rows" in summary:
        for row in summary["rows"]:
            qid = row.get("query_id")
            sys_val = row.get("system")
            if qid and sys_val:
                call_fail = row.get("call_failure_count", 0) > 0
                errs = row.get("errors", [])
                has_rej_err = any(
                    "call rejected" in str(e).lower()
                    or "call failure" in str(e).lower()
                    or str(e).lower().startswith("call ")
                    for e in errs
                )
                if call_fail or has_rej_err:
                    rejected_set.add((qid, sys_val))

    paired: list[tuple[QueryRecord, QueryRecord]] = []
    dropped: list[dict] = []

    for qid in all_qids:
        j_rec = joint_map.get(qid)
        p_rec = post_hoc_map.get(qid)

        if j_rec is None:
            dropped.append({"query_id": qid, "reason": "missing joint arm record"})
        elif p_rec is None:
            dropped.append({"query_id": qid, "reason": "missing post_hoc arm record"})
        elif (qid, System.JOINT.value) in rejected_set or (
            j_rec.raw_generation == "" and len(j_rec.claims) == 0
        ):
            dropped.append({"query_id": qid, "reason": "call rejected in joint arm"})
        elif (qid, System.POST_HOC.value) in rejected_set or (
            p_rec.raw_generation == "" and len(p_rec.claims) == 0
        ):
            dropped.append({"query_id": qid, "reason": "call rejected in post_hoc arm"})
        elif len(j_rec.claims) == 0:
            dropped.append({"query_id": qid, "reason": "zero claims in joint arm"})
        elif len(p_rec.claims) == 0:
            dropped.append({"query_id": qid, "reason": "zero claims in post_hoc arm"})
        else:
            paired.append((j_rec, p_rec))

    # Sort paired list deterministically by query_id
    paired.sort(key=lambda u: u[0].query_id)

    return paired, dropped


def compute_citation_contrast(
    paired: Sequence[tuple[QueryRecord, QueryRecord]],
    phi: Phi,
    *,
    seed: int = 0,
    n_boot: int = 10000,
    confidence: float = 0.95,
) -> dict:
    """Compute joint and post_hoc citation metrics and the paired bootstrap delta.

    Resamples queries (ADR-0011 §2), pairing joint and post_hoc readings for each query.
    """
    if not paired:
        raise ValueError("Cannot compute citation contrast with zero paired queries")

    # Sort deterministically by query_id so order of input records does not change resample draws
    paired_sorted = sorted(paired, key=lambda u: u[0].query_id)

    joint_recs = [u[0] for u in paired_sorted]
    post_hoc_recs = [u[1] for u in paired_sorted]

    joint_stats = citation_f1(joint_recs, phi)
    post_hoc_stats = citation_f1(post_hoc_recs, phi)

    def _delta_stat(units_sample: Sequence[tuple[QueryRecord, QueryRecord]]) -> float:
        j_sample = [u[0] for u in units_sample]
        p_sample = [u[1] for u in units_sample]
        j_f1 = citation_f1(j_sample, phi)["f1"]
        p_f1 = citation_f1(p_sample, phi)["f1"]
        return j_f1 - p_f1

    clusters = [u[0].query_id for u in paired_sorted]

    ci_res = bootstrap_ci(
        units=paired_sorted,
        statistic=_delta_stat,
        clusters=clusters,
        cluster_unit="query",
        n_boot=n_boot,
        confidence=confidence,
        seed=seed,
    )

    delta_point = joint_stats["f1"] - post_hoc_stats["f1"]

    return {
        "joint": joint_stats,
        "post_hoc": post_hoc_stats,
        "delta": {
            "point": delta_point,
            "lower": ci_res["lower"],
            "upper": ci_res["upper"],
            "width": ci_res["width"],
            "n_boot": n_boot,
            "confidence": confidence,
            "seed": seed,
            "excludes_zero": bool(ci_res["lower"] > 0 or ci_res["upper"] < 0),
        },
        "bootstrap_ci": ci_res,
        "n_paired": len(paired_sorted),
    }


def get_phi_from_cache_or_verifier(
    records: Iterable[QueryRecord],
    cache_path: str | Path = "docs/harvest/minicheck_cache.json",
    threshold: float = 0.5,
    batch_size: int = 32,
) -> Phi:
    """Load MiniCheck scores from cache file and run MiniCheckVerifier for any missing pairs."""
    records_list = list(records)
    required_pairs = collect_phi_pairs(records_list)

    cache_file = Path(cache_path)
    cache: dict[tuple[str, str], float] = {}

    if cache_file.exists():
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            cache = {tuple(k.split("|||")): float(v) for k, v in raw.items()}
        except Exception as err:
            print(f"Warning: could not load cache from {cache_file}: {err}", file=sys.stderr)

    missing = [p for p in required_pairs if p not in cache]

    print(
        f"MiniCheck pairs: {len(cache)} cached, {len(missing)} missing out of {len(required_pairs)} required pairs",
        flush=True,
    )

    if missing:
        print(f"Scoring {len(missing)} missing pairs on MiniCheck CPU verifier...", flush=True)
        try:
            import torch
            torch.set_num_threads(16)
        except ImportError:
            pass
        verifier = MiniCheckVerifier(batch_size=batch_size)
        scored_results = verifier.score_pairs(missing)
        for pair, vscore in zip(missing, scored_results, strict=True):
            cache[pair] = vscore.score
        # Write the enlarged cache back. Scoring 1113 pairs costs ~9 CPU-minutes; a run that
        # discards them makes every re-read of the same contrast pay that cost again, and a
        # re-scored pair is only bit-identical by luck of thread count and batching.
        tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps({f"{p}|||{h}": s for (p, h), s in sorted(cache.items())}),
            encoding="utf-8",
        )
        tmp.replace(cache_file)
        print(f"Cache written: {len(cache)} pairs -> {cache_file}", flush=True)

    return phi_from_scores(cache, threshold)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute citation-F1 paired contrast between joint and post-hoc arms."
    )
    parser.add_argument(
        "prefix",
        type=str,
        help="Run prefix (e.g. docs/harvest/generate_fp05_n100_guided_batched)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap CI (default: 0)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10000,
        help="Number of bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="MiniCheck phi threshold (default: 0.5)",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="docs/harvest/minicheck_cache.json",
        help="Path to MiniCheck cache file (default: docs/harvest/minicheck_cache.json)",
    )

    args = parser.parse_args()

    print(f"=== Citation-F1 Paired Contrast Analysis ===")
    print(f"Run Prefix: {args.prefix}")
    print(f"Bootstrap Seed: {args.seed}, Resamples: {args.n_boot}")
    print(f"MiniCheck Threshold: {args.threshold}")

    records, costs, summary = load_run(args.prefix)
    print(f"Loaded {len(records)} total query records and {len(costs)} cost records.")

    paired, dropped = pair_queries(records, costs, summary)

    print(f"\n--- Query Pairing Summary ---")
    print(f"Paired queries count: {len(paired)}")
    print(f"Dropped queries count: {len(dropped)}")

    if dropped:
        reason_counts = Counter(d["reason"] for d in dropped)
        for reason, count in reason_counts.items():
            print(f"  - {reason}: {count}")

    if not paired:
        print("ERROR: No queries could be paired between joint and post_hoc arms.", file=sys.stderr)
        return 1

    # Extract all records from paired list for phi lookup
    paired_records = [r for pair in paired for r in pair]
    phi = get_phi_from_cache_or_verifier(
        paired_records, cache_path=args.cache_path, threshold=args.threshold
    )

    result = compute_citation_contrast(
        paired, phi, seed=args.seed, n_boot=args.n_boot, confidence=0.95
    )

    j_stats = result["joint"]
    p_stats = result["post_hoc"]
    d_stats = result["delta"]

    print(f"\n--- Arm Performance ---")
    print("Joint Arm:")
    print(f"  Precision:         {j_stats['precision']:.4f}")
    print(f"  Recall (answered): {j_stats['recall']:.4f}")
    print(f"  Citation F1:       {j_stats['f1']:.4f}")
    print(f"  Recall (all):      {j_stats['recall_all_claims']:.4f}")
    print(f"  F1 (all):          {j_stats['f1_all_claims']:.4f}")
    print(f"  Claims (ans/abst): {j_stats['n_answered']} / {j_stats['n_abstentions']} (total: {j_stats['n_claims']})")
    print(f"  Citations (rel):   {j_stats['n_relevant_citations']} / {j_stats['n_citations']}")

    print("\nPost-Hoc Arm:")
    print(f"  Precision:         {p_stats['precision']:.4f}")
    print(f"  Recall (answered): {p_stats['recall']:.4f}")
    print(f"  Citation F1:       {p_stats['f1']:.4f}")
    print(f"  Recall (all):      {p_stats['recall_all_claims']:.4f}")
    print(f"  F1 (all):          {p_stats['f1_all_claims']:.4f}")
    print(f"  Claims (ans/abst): {p_stats['n_answered']} / {p_stats['n_abstentions']} (total: {p_stats['n_claims']})")
    print(f"  Citations (rel):   {p_stats['n_relevant_citations']} / {p_stats['n_citations']}")

    print(f"\n--- Paired Contrast (Joint - Post-Hoc) ---")
    print(f"Citation-F1 Delta:  {d_stats['point']:+.4f}")
    print(
        f"95% Bootstrap CI:   [{d_stats['lower']:+.4f}, {d_stats['upper']:+.4f}] (width: {d_stats['width']:.4f})"
    )
    print(f"Excludes Zero:      {d_stats['excludes_zero']}")
    print(f"Resampling Details: unit=query, n_clusters={result['n_paired']}, n_boot={d_stats['n_boot']}, seed={d_stats['seed']}")

    joint_summary = (summary.get("per_system") or {}).get("joint") or {}
    clean = joint_summary.get("clean_parses")
    n_joint = joint_summary.get("n")
    qnf = joint_summary.get("quote_not_found")
    if clean is not None and n_joint:
        rate = clean / n_joint
        print(f"\nJoint arm clean parses: {clean}/{n_joint} ({rate:.0%}), quote_not_found = {qnf}")
        if rate < 0.95:
            # The bar is stated on the arm, not on the contrast: an arm that cannot parse cannot
            # carry a gate figure, however the delta lands.
            print(
                "WARNING: the joint arm is under the Gate G2 >=95% parse bar on this run, so this "
                "delta is a diagnostic reading and not a gate figure."
            )

    out_path = Path(f"{args.prefix}.citation_f1.minicheck.json")
    out_path.write_text(
        json.dumps(
            {
                "run_prefix": str(args.prefix),
                "verifier": "lytang/MiniCheck-Flan-T5-Large",
                "threshold": args.threshold,
                "joint_clean_parses": clean,
                "joint_quote_not_found": qnf,
                "n_dropped_queries": len(dropped),
                **result,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
