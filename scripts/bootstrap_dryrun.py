#!/usr/bin/env python3
"""ADR-0011 §2's W4 dry-run: compute every interval **both ways** and record what clustering costs.

ADR-0011 made question-clustered resampling a standing rule for every CI in the paper and said the
cost out loud — "clustering widens every interval, including the one G2 gates on" — then required
this dry-run so the width is seen in August rather than discovered at the gate in September.

Three arms, all on already-committed records. Nothing here chooses a threshold or a method; τ is
`τ_confusable = 0.7`, fixed in W2 and recorded in `docs/harvest/confusability_probe.json`.

1. **Harness check — hit@5 over dev questions.** One observation per question, so the cluster *is*
   the unit and clustering must be a no-op to within Monte-Carlo noise. If this arm moves, the
   clustering code is doing something other than what it says.
2. **Clustered vs unclustered — fraction of reranked distractors at or above τ.** 414 passages in
   100 question clusters: a corpus-level proportion micro-averaged over sub-units of a question,
   which is the same shape as citation-F1 micro-averaged over claims.
3. **Paired, the G2 shape.** Retrieved minus random-control on that same fraction, paired by
   question. Pairing lives in the statistic; clustering lives in the resampling unit.

**This is a lower bound on what G2 will see.** The probe carries 4.14 passages per question; the
generator emits 9.2 claims per question (ADR-0011 §Context), so G2's clusters are ~2.2× larger and
correlated units inflate the unclustered *n* correspondingly harder.

Usage (CPU, no GPU, ~1 min):
    uv run python scripts/bootstrap_dryrun.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.schema import query_record_from_dict  # noqa: E402
from biomedqa.scoring.calibration import bootstrap_ci  # noqa: E402
from biomedqa.scoring.retrieval import gold_rank  # noqa: E402

#: Set post-hoc in W2 on the pre-rerank probe and unchanged since. Not re-chosen here.
TAU_CONFUSABLE = 0.7

#: Table 1 row 4 — the full cascade, the same row G1 is read on (ADR-0015 §3).
CASCADE_ROW = 4

#: The ratio of two bootstrap widths is itself noisy; report its spread over a fixed seed panel
#: rather than one seed's reading (ADR-0011 §2 Consequences).
SEEDS = (20260804, 1, 7, 42, 999983)


def hit_at_5_units(records_path: Path) -> tuple[list[bool], list[str]]:
    """One boolean per dev question: did any gold chunk land in the top 5?"""
    units: list[bool] = []
    keys: list[str] = []
    with records_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.pop("table1_row") != CASCADE_ROW:
                continue
            record = query_record_from_dict(raw)
            rank = gold_rank(record)
            units.append(rank is not None and rank <= 5)
            keys.append(record.query_id)
    return units, keys


def confusable_units(probe: dict, tau: float) -> tuple[list[bool], list[str]]:
    """One boolean per scored distractor: is it confusable enough to be a plausible mis-citation?"""
    units: list[bool] = []
    keys: list[str] = []
    for question in probe["per_question"]:
        for score in question["passage_max_scores"]:
            units.append(score >= tau)
            keys.append(question["pubid"])
    return units, keys


def paired_units(probe: dict, control: dict, tau: float) -> tuple[list[tuple[bool, bool]], list[str]]:
    """One (retrieved, random) pair per passage slot, matched within a question.

    The control drew the same number of passages per question as the probe scored, so the slots
    correspond by position inside a question. Pairing across questions would be meaningless.
    """
    by_pubid = {q["pubid"]: q for q in control["per_question"]}
    units: list[tuple[bool, bool]] = []
    keys: list[str] = []
    for question in probe["per_question"]:
        pubid = question["pubid"]
        drawn = by_pubid[pubid]["passage_max_scores"]
        retrieved = question["passage_max_scores"]
        if len(drawn) != len(retrieved):
            raise ValueError(
                f"{pubid}: control drew {len(drawn)} passages against {len(retrieved)} retrieved; "
                "the arms are not paired and a paired bootstrap over them is not defined"
            )
        for r, c in zip(retrieved, drawn, strict=True):
            units.append((r >= tau, c >= tau))
            keys.append(pubid)
    return units, keys


def _paired_delta(units) -> float:
    n = len(units)
    return sum(r for r, _ in units) / n - sum(c for _, c in units) / n


def both_ways(units, keys, statistic=None, *, n_boot: int, seeds: Sequence[int]) -> dict:
    """The same interval computed twice — resampling observations, then resampling questions.

    Repeated over several seeds, because the deliverable is a *ratio of two bootstrap widths* and
    a single seed reports it to more precision than it has. The intervals shown are the first
    seed's; the ratio carries its spread.
    """
    per_seed = []
    for seed in seeds:
        flat = bootstrap_ci(units, statistic, n_boot=n_boot, seed=seed)
        clustered = bootstrap_ci(units, statistic, clusters=keys, n_boot=n_boot, seed=seed)
        per_seed.append((flat, clustered, clustered["width"] / flat["width"]))

    ratios = [r for _, _, r in per_seed]
    flat, clustered, ratio = per_seed[0]
    median = statistics.median(ratios)
    return {
        "unclustered": flat,
        "clustered": clustered,
        "width_ratio": round(ratio, 4),
        "width_ratio_over_seeds": {
            "seeds": list(seeds),
            "values": [round(r, 4) for r in ratios],
            "median": round(median, 4),
            "min": round(min(ratios), 4),
            "max": round(max(ratios), 4),
        },
        # Kish's design effect, off the median ratio: the factor by which correlated units inflate
        # an unclustered n. effective_n is what the unclustered interval is really worth.
        "design_effect": round(median**2, 4),
        "effective_n": round(flat["n_units"] / median**2, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="ADR-0011 clustered-vs-unclustered bootstrap dry-run")
    ap.add_argument(
        "--records", type=Path, default=Path("docs/harvest/table1_rows_1_4.records.jsonl")
    )
    ap.add_argument("--probe", type=Path, default=Path("docs/harvest/confusability_probe_reranked.json"))
    ap.add_argument(
        "--control", type=Path, default=Path("docs/harvest/confusability_probe_reranked_control.json")
    )
    ap.add_argument("--out", type=Path, default=Path("docs/harvest/bootstrap_dryrun.json"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    control = json.loads(args.control.read_text(encoding="utf-8"))

    hits, hit_keys = hit_at_5_units(args.records)
    confusable, conf_keys = confusable_units(probe, TAU_CONFUSABLE)
    pairs, pair_keys = paired_units(probe, control, TAU_CONFUSABLE)

    arms = {
        "harness_hit_at_5": both_ways(hits, hit_keys, n_boot=args.n_boot, seeds=args.seeds),
        "confusable_fraction": both_ways(
            confusable, conf_keys, n_boot=args.n_boot, seeds=args.seeds
        ),
        "paired_retrieved_minus_control": both_ways(
            pairs, pair_keys, _paired_delta, n_boot=args.n_boot, seeds=args.seeds
        ),
    }

    harness = arms["harness_hit_at_5"]
    harness_gap = abs(harness["clustered"]["width"] - harness["unclustered"]["width"])
    harness_passed = harness_gap < 0.01

    for name, arm in arms.items():
        flat, clustered = arm["unclustered"], arm["clustered"]
        print(
            f"{name:34s} point {flat['point']:+.4f}  "
            f"observation [{flat['lower']:+.4f}, {flat['upper']:+.4f}] w={flat['width']:.4f}  ·  "
            f"question [{clustered['lower']:+.4f}, {clustered['upper']:+.4f}] "
            f"w={clustered['width']:.4f}  ×{arm['width_ratio']:.2f}"
        )
    print(
        f"\nharness check: widths differ by {harness_gap:.4f} — "
        f"{'PASSES' if harness_passed else 'FAILS'} (one observation per cluster must be a no-op)"
    )

    report = {
        "script": "scripts/bootstrap_dryrun.py",
        "adr": "ADR-0011 §2 — every bootstrap cluster-resamples questions, not claims",
        "question": "what does question-clustering cost in CI width, measured in August?",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "records": str(args.records),
            "probe": str(args.probe),
            "control": str(args.control),
        },
        "config": {
            "tau_confusable": TAU_CONFUSABLE,
            "table1_row": CASCADE_ROW,
            "n_boot": args.n_boot,
            "seeds": list(args.seeds),
            "confidence": 0.95,
        },
        "harness_check": {
            "arm": "harness_hit_at_5",
            "rule": "one observation per cluster; clustering must not move the interval",
            "width_gap": round(harness_gap, 6),
            "passed": harness_passed,
        },
        "arms": arms,
        "cluster_size": {
            "probe_units_per_question": round(len(confusable) / len(set(conf_keys)), 2),
            "generator_claims_per_question": 9.2,
            "note": (
                "The probe's clusters are smaller than G2's will be, so every widening below is a "
                "lower bound on what the headline gate pays (ADR-0011 §Context)."
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
