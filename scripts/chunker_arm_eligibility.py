#!/usr/bin/env python3
"""Which chunker arms are readable as retrieval, and which measure a gold-only signal?

`chunker_pool_eval.py` answered "can a chunker rescue G1?" and one arm cleared the bar: `section`,
at hit@5 0.9400 upper bound against `abstract`'s 0.8600. `g1_miss_analysis.json` registered, before
that run, that no arm would reach 0.90 and that **the full ~2 h build is owed to any arm that
does**. So the prediction is falsified as written and a debt is outstanding.

This script decides whether that debt is payable, and it does so on a rule that predates the
measurement rather than one invented to survive it.

The rule
--------
ADR-0014 §2 rejects "the one property every gold passage shares and no distractor has" as a
systematic signal sitting in the space hit@5 is measured in. `chunk.py`'s module docstring already
generalises it past titles: *"One splitter cuts gold and distractors both … ADR-0014 §2 rejects any
property that every gold passage shares and no distractor has, and how the text was cut is such a
property."* Both statements were written before this sweep ran.

What trips it
-------------
`encode_corpus.py` builds gold with `chunk_instance` — real PubMedQA section spans — and MedRAG
rows with `chunk_text(sections=None)`, because the corpus carries no section labels. Strategies
that ignore `sections` take the same path twice and cut gold exactly as they cut a distractor. The
`section` strategy does not: gold splits on real BACKGROUND/METHODS boundaries while every
distractor stays one whole abstract. `chunk_text` degrades `"section"` to `"abstract"` when
`sections is None`, so under symmetric treatment `section` **is** `abstract` — the two are not
merely close, they are the same chunking.

That makes `section` a gold-only transformation rather than a corpus chunking strategy, and its
+0.08 the size of the leak.

Why no GPU is owed
------------------
The verdict is a property of the chunkers, not of any ranking, so it is computed from
`chunk.py` alone and joined onto the committed arms. Refusing the build costs nothing to check and
the check is exact — not a bound, not a resample.

    uv run python scripts/chunker_arm_eligibility.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from biomedqa.data import Instance, load_instances, load_splits  # noqa: E402
from biomedqa.scoring.retrieval import wilson_interval  # noqa: E402
from chunker_pool_eval import gold_cut_asymmetry  # noqa: E402  — one detector, defined once
from chunker_sweep import SWEEP  # noqa: E402

#: G1 as written: point >= 0.90 **and** Wilson lower > 0.85. Never relaxed to make an arm fit.
G1_POINT = 0.90
G1_WILSON_LOWER = 0.85


def _instances(split: str) -> list[Instance]:
    pubids = set(load_splits()[split])
    return [i for i in load_instances() if i.pubid in pubids]


def verdict(arms: list[dict], instances: list[Instance]) -> list[dict]:
    """Join the cut-symmetry check onto each measured arm and say what it earns."""
    out = []
    for arm in arms:
        name = arm["chunker"]
        curve = arm["hit_at_k_upper_bound"]["hit_at_5"]
        asym = gold_cut_asymmetry(SWEEP[name], instances)
        clears = curve["point"] >= G1_POINT and curve["wilson_lower"] > G1_WILSON_LOWER
        eligible = not asym["cuts_gold_unlike_distractors"]
        if not eligible:
            earns, why = False, (
                "Cut differently from every distractor, so the bound measures ADR-0014 §2's "
                "rejected signal rather than retrieval. A build would reproduce the leak "
                "faithfully and still not be a reading of hit@5."
            )
        elif not clears:
            earns, why = False, (
                "Upper bound misses G1, and the real build reports no more than the bound, so "
                "the ~2 h is refused on evidence."
            )
        else:
            earns, why = True, "Eligible and clears G1 as an upper bound: the full build is owed."
        out.append(
            {
                "chunker": name,
                "hit_at_5_upper_bound": curve["point"],
                "wilson_lower": curve["wilson_lower"],
                "hit_at_10_upper_bound": arm["hit_at_k_upper_bound"]["hit_at_10"]["point"],
                "clears_g1_as_upper_bound": clears,
                "gold_cut_asymmetry": asym,
                "eligible": eligible,
                "earns_a_full_build": earns,
                "reason": why,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Rule on the chunker sweep's arms (CPU-only)")
    ap.add_argument(
        "--arms", type=Path, default=Path("docs/harvest/chunker_pool_eval.json")
    )
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument(
        "--out", type=Path, default=Path("docs/harvest/chunker_arm_eligibility.json")
    )
    args = ap.parse_args()

    source = json.loads(args.arms.read_text(encoding="utf-8"))
    if not (source.get("harness_check") or {}).get("passed"):
        print(
            f"{args.arms} did not pass its harness check; its arms are not readable.",
            file=sys.stderr,
        )
        return 1

    instances = _instances(args.split)
    rulings = verdict(source["arms"], instances)

    print(f"{'chunker':<22}{'hit@5 (UB)':>12}{'clears G1':>11}{'eligible':>10}{'build?':>9}")
    print("-" * 64)
    for r in rulings:
        print(
            f"{r['chunker']:<22}{r['hit_at_5_upper_bound']:>12.4f}"
            f"{('yes' if r['clears_g1_as_upper_bound'] else 'no'):>11}"
            f"{('yes' if r['eligible'] else 'NO'):>10}"
            f"{('OWED' if r['earns_a_full_build'] else 'refused'):>9}"
        )

    owed = [r["chunker"] for r in rulings if r["earns_a_full_build"]]
    leaking = [r["chunker"] for r in rulings if not r["eligible"]]
    best = max(
        (r for r in rulings if r["eligible"]), key=lambda r: r["hit_at_5_upper_bound"]
    )

    report = {
        "script": "scripts/chunker_arm_eligibility.py",
        "question": "Does any chunker arm earn its ~2 h build, and does G1 survive at k=5?",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "arms_source": str(args.arms),
        "rule": (
            "An arm is eligible only if it cuts gold the way it cuts a distractor. ADR-0014 §2 "
            "and chunk.py's module docstring both predate this sweep."
        ),
        "g1": {"k": 5, "point": G1_POINT, "wilson_lower_must_exceed": G1_WILSON_LOWER},
        "arms": rulings,
        "ineligible_arms": leaking,
        "builds_owed": owed,
        "best_eligible_arm": {
            "chunker": best["chunker"],
            "hit_at_5_upper_bound": best["hit_at_5_upper_bound"],
            "cannot_pass_g1_at_k5": best["hit_at_5_upper_bound"] < G1_POINT,
        },
        "registered_prediction_outcome": (
            "FALSIFIED as written: 'section' reached 0.94. The arm that falsified it is "
            "ineligible under a rule that predates the measurement, and the debt it created — "
            "a full build — is refused on that ground and not on its number. Every eligible arm "
            "is at or below "
            f"{best['hit_at_5_upper_bound']}, so no chunker rescues G1 at k=5."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nineligible: {', '.join(leaking) or 'none'}")
    print(f"builds owed: {', '.join(owed) or 'none'}")
    print(f"best eligible arm: {best['chunker']} at {best['hit_at_5_upper_bound']:.4f} UB")
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
