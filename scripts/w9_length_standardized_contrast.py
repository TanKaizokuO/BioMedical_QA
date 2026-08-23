#!/usr/bin/env python3
"""ADR-0009 §5 asymmetric scrutiny: is C2's citation-F1 gap an artifact of granularity?

Usage:
    uv run python scripts/w9_length_standardized_contrast.py docs/harvest/generate_fp05_n100_guided_v4

ADR-0009's Context names the exact confound this script measures: "Coarser claims are harder to
entail per claim, so if post-hoc's claims are systematically coarser, post-hoc is systematically
penalised -- and **C2's gap appears without joint grounding doing any work.**" §5's one-sided
fallback makes the W9 stratified check mandatory whenever the residual granularity gap favours C2,
which it does on every run to date.

`w9_stratified_parity_report.py` answers the *precondition* of that worry -- how far apart the two
arms' claim lengths are. It cannot answer the worry itself, because a granularity gap is only a
confound if it *transmits* to citation-F1. This script answers the worry directly, by removing the
gap arithmetically instead of trying to remove it from the model's behaviour with prompt text:

    joint's citation-recall is re-weighted to post-hoc's own claim-length distribution
    (direct standardisation over `CLAIM_LENGTH_BANDS`), then recombined with each arm's observed
    precision into the frozen `citation_f1` harmonic mean.

Post-hoc is the reference distribution, so its standardised recall is its observed recall by
construction -- the script asserts that, since it is the one identity that proves the weighting is
not silently rescaling the baseline too.

**Why standardise rather than tune the prompt to close the gap.** Tuning joint's claim length to
make the parity number pass would be steering *joint's* granularity, which ADR-0009 §4 confines to
`POST_HOC_ANSWER_TEMPLATE`, with citation-F1 already unblinded (§6, 2026-08-14) -- and §1/§3 fix
parity as a diagnostic "disclosed whatever it says" whose "tolerance does not need to be
achievable". Standardisation is a scoring-side answer to a scoring-side question, so it leaves both
prompts frozen and re-derives from stored records with no new inference. Same posture as ADR-0009's
Second amendment, which kept the 731-word claim a scoring-side guard rather than a prompt edit.

Interval convention follows ADR-0011 §2: queries are the resampled cluster, seed 0, 10000 draws.
The statistic is recomputed inside every draw -- standardisation weights included -- because
weights estimated once on the full sample and reused would understate the width.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from biomedqa.schema import QueryRecord  # noqa: E402
from biomedqa.scoring.abstention import answered_claims  # noqa: E402
from biomedqa.scoring.calibration import bootstrap_ci  # noqa: E402
from biomedqa.scoring.citation import _precision_counts, citation_recall  # noqa: E402
from biomedqa.scoring.granularity import CLAIM_LENGTH_BANDS  # noqa: E402
from citation_contrast import (  # noqa: E402
    get_phi_from_cache_or_verifier,
    load_run,
    pair_queries,
)

Pair = tuple[QueryRecord, QueryRecord]


#: Empty-text claims (`len(text.split()) == 0`) are a guided-JSON artifact of the joint arm only --
#: 3 of them on `..._v4`, up to 11 on `..._v8`, none in post-hoc on any run -- and
#: `CLAIM_LENGTH_BANDS` starts at 1, so they fall outside every band. They are folded into the
#: lowest band rather than skipped, because they are real claims in `citation_f1`'s denominator and
#: skipping them would quietly *remove* joint's own defect from the comparison: post-hoc has none,
#: so a dropped-band policy would standardise them away and flatter joint. ADR-0009's Context is
#: explicit that a residual pointing toward the hypothesis is the direction that must never go
#: unmeasured, so the fold keeps joint charged for them. `compute_claim_length_strata` skips them
#: instead; that only shifts a median by well under a word and is not worth unfreezing.
def band_of(n_words: int) -> str:
    lowest = CLAIM_LENGTH_BANDS[0][0]
    if n_words < CLAIM_LENGTH_BANDS[0][1]:
        return lowest
    for label, lo, hi in CLAIM_LENGTH_BANDS:
        if lo <= n_words <= hi:
            return label
    raise ValueError(f"no band covers {n_words} words")


def _harmonic(precision: float, recall: float) -> float:
    """`citation_f1`'s combination rule, reused verbatim so the two are comparable."""
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _tally(records: Sequence[QueryRecord], phi) -> tuple[dict[str, list[float]], int, int]:
    """Per-band (recall sum, claim count) plus corpus precision counts for one arm."""
    bands: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    good = cited = 0
    for record in records:
        passages = {p.passage_id: p.text for p in record.retrieved if p.text is not None}
        for claim in answered_claims(record):
            band = bands[band_of(len(claim.text.split()))]
            band[0] += citation_recall(claim, phi, passages=passages)
            band[1] += 1
            g, n = _precision_counts(claim, phi, passages)
            good += g
            cited += n
    return bands, good, cited


def contrast(pairs: Sequence[Pair], phi, *, standardize: bool = True) -> dict:
    """Citation-F1 delta, optionally with joint's recall standardised to post-hoc's lengths.

    `standardize=False` reproduces `citation_contrast.py`'s delta exactly, which is what makes the
    standardised number interpretable: the two differ only in the recall weighting.
    """
    j_bands, j_good, j_cited = _tally([p[0] for p in pairs], phi)
    p_bands, p_good, p_cited = _tally([p[1] for p in pairs], phi)

    j_precision = j_good / j_cited if j_cited else 0.0
    p_precision = p_good / p_cited if p_cited else 0.0

    if standardize:
        # Bands empty in either arm carry no comparison and would inject a 0/0 weight.
        shared = [b for b in p_bands if p_bands[b][1] and j_bands.get(b, [0.0, 0])[1]]
        total = sum(p_bands[b][1] for b in shared)
        if not total:
            raise ValueError("no claim-length band is populated in both arms")
        weights = {b: p_bands[b][1] / total for b in shared}
        j_recall = sum(w * (j_bands[b][0] / j_bands[b][1]) for b, w in weights.items())
        p_recall = sum(w * (p_bands[b][0] / p_bands[b][1]) for b, w in weights.items())
    else:
        j_n = sum(v[1] for v in j_bands.values())
        p_n = sum(v[1] for v in p_bands.values())
        j_recall = sum(v[0] for v in j_bands.values()) / j_n if j_n else 0.0
        p_recall = sum(v[0] for v in p_bands.values()) / p_n if p_n else 0.0

    j_f1 = _harmonic(j_precision, j_recall)
    p_f1 = _harmonic(p_precision, p_recall)
    return {
        "joint": {"precision": j_precision, "recall": j_recall, "f1": j_f1},
        "post_hoc": {"precision": p_precision, "recall": p_recall, "f1": p_f1},
        "delta": j_f1 - p_f1,
        "bands": {
            b: {
                "joint_n": j_bands.get(b, [0.0, 0])[1],
                "joint_recall": (
                    j_bands[b][0] / j_bands[b][1] if j_bands.get(b, [0.0, 0])[1] else None
                ),
                "post_hoc_n": p_bands.get(b, [0.0, 0])[1],
                "post_hoc_recall": (
                    p_bands[b][0] / p_bands[b][1] if p_bands.get(b, [0.0, 0])[1] else None
                ),
            }
            for b, _, _ in CLAIM_LENGTH_BANDS
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prefix", help="run prefix under docs/harvest/, without a suffix")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confidence", type=float, default=0.95)
    args = ap.parse_args()

    records, costs, summary = load_run(args.prefix)
    phi = get_phi_from_cache_or_verifier(records)
    pairs, dropped = pair_queries(records, costs, summary)
    pairs = sorted(pairs, key=lambda u: u[0].query_id)
    if not pairs:
        print("no paired queries", file=sys.stderr)
        return 1

    raw = contrast(pairs, phi, standardize=False)
    std = contrast(pairs, phi, standardize=True)

    # The reference arm must be untouched by its own re-weighting; if this drifts, the weights are
    # wrong and the standardised delta means nothing.
    assert abs(std["post_hoc"]["recall"] - raw["post_hoc"]["recall"]) < 1e-9, (
        "post-hoc is the reference distribution, so standardising must leave its recall fixed"
    )

    ci = bootstrap_ci(
        units=pairs,
        statistic=lambda sample: contrast(sample, phi, standardize=True)["delta"],
        clusters=[p[0].query_id for p in pairs],
        cluster_unit="query",
        n_boot=args.n_boot,
        confidence=args.confidence,
        seed=args.seed,
    )

    name = Path(args.prefix).name
    print(f"ADR-0009 §5 length-standardised citation-F1 contrast: {name}")
    print(f"  paired queries: {len(pairs)}, dropped: {len(dropped)}")
    print(f"  bands: {', '.join(b for b, _, _ in CLAIM_LENGTH_BANDS)} words\n")

    print("Per-band citation recall (matched claim length):")
    print(f"  {'band':>8} {'joint n':>8} {'joint R':>8} {'ph n':>8} {'ph R':>8} {'dR':>8}")
    for b, _, _ in CLAIM_LENGTH_BANDS:
        row = std["bands"][b]
        if not row["joint_n"] or not row["post_hoc_n"]:
            continue
        d = row["joint_recall"] - row["post_hoc_recall"]
        print(
            f"  {b:>8} {row['joint_n']:>8} {row['joint_recall']:>8.3f} "
            f"{row['post_hoc_n']:>8} {row['post_hoc_recall']:>8.3f} {d:>+8.3f}"
        )

    print("\nCitation-F1:")
    print(
        f"  unstandardised: joint {raw['joint']['f1']:.4f} (R {raw['joint']['recall']:.4f}) vs "
        f"post-hoc {raw['post_hoc']['f1']:.4f} (R {raw['post_hoc']['recall']:.4f})  "
        f"delta {raw['delta']:+.4f}"
    )
    print(
        f"  standardised:   joint {std['joint']['f1']:.4f} (R {std['joint']['recall']:.4f}) vs "
        f"post-hoc {std['post_hoc']['f1']:.4f} (R {std['post_hoc']['recall']:.4f})  "
        f"delta {std['delta']:+.4f}"
    )
    print(
        f"  {args.confidence:.0%} clustered CI on the standardised delta: "
        f"[{ci['lower']:+.4f}, {ci['upper']:+.4f}] (width {ci['width']:.4f})"
    )
    excludes = ci["lower"] > 0 or ci["upper"] < 0
    print(f"  excludes zero: {excludes}")
    print(f"  resampling: unit=query, n_clusters={len(pairs)}, n_boot={args.n_boot}, seed={args.seed}")

    direction = "against" if std["delta"] >= raw["delta"] else "toward"
    print(
        f"\nVerdict: the granularity gap transmits {direction} C2 -- matching claim length "
        f"{'widens' if std['delta'] >= raw['delta'] else 'narrows'} the contrast "
        f"({raw['delta']:+.4f} -> {std['delta']:+.4f}), so the citation-F1 gap is "
        f"{'not' if excludes else 'not established as'} an artifact of post-hoc's coarser claims."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
