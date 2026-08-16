#!/usr/bin/env python3
"""The first citation-F1 — **ADR-0009 §6's unblinding, computed once the parity loop closed.**

    uv run python scripts/first_citation_f1.py docs/harvest/parity_iter1b

Runs only because `prompts.PARITY_LOOP_CLOSED` is set (`parity_iter1b`, 2026-08-14);
`scoring.citation.citation_f1` raises otherwise, so this script cannot be the thing that breaks the
blind.

## What this number is, and what it is not

**It is the R5 early-warning read, not the G2 gate.** Two things separated it from the paper's
number when it was first computed on 2026-08-14, and one of them is now closed:

1. **φ.** The verifier is MiniCheck-Flan-T5-Large and it is implemented — `biomedqa.verify`, since
   2026-08-17, pulled ahead of its W6 slot precisely because this read depends on it and G2 is
   Sep 6. `--phi minicheck` (the default) scores the pairs through it at
   `--threshold 0.5`, MiniCheck's own binarisation. `--phi deberta-xsmall` reproduces the interim
   read — `cross-encoder/nli-deberta-v3-xsmall` at `argmax == entailment`, φ in `notebooks/03_2`
   and `06_5` — and exists so the 2026-08-14 artifact stays reproducible, not because it is a
   number anyone should quote. **R7 predicts either model degrades on biomedical text**, so treat
   the level as a lower bound and the joint-vs-post-hoc *contrast* as the signal. Nothing is
   thresholded on a stored score: swapping φ re-runs this script and nothing else.
2. **The records are a smoke run**, `--max-tokens 3584`, 100 dev questions, `not a gate run and not a
   sample` in its own summary. Vanilla is excluded by ADR-0010 — it cites nothing by construction.

Intervals resample **questions** (ADR-0011 §2, `calibration.bootstrap_ci`), including the paired
joint − post-hoc delta, computed inside the statistic so both arms see the same drawn questions.

## Why φ is evaluated in two passes

`citation_f1` calls φ one pair at a time, and its precision rule is short-circuiting: the
`concat(C \\ {x})` call happens only when `φ(x, c)` is false. A recording pass whose φ returns `False`
therefore collects a **superset** of the pairs any real φ could need. That set is scored in batches on
whatever device is available, and the second pass reads a dict. Batched inference matters: the two
arms need ~7.7k pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.harness import costs_path, records_path  # noqa: E402
from biomedqa.prompts import PARITY_LOOP_CLOSED  # noqa: E402
from biomedqa.schema import CostRecord, System, read_jsonl, read_query_records  # noqa: E402
from biomedqa.scoring.abstention import answered_claims  # noqa: E402
from biomedqa.scoring.calibration import bootstrap_ci  # noqa: E402
from biomedqa.scoring.citation import citation_f1, citation_recall  # noqa: E402
from biomedqa.scoring.granularity import truncated_queries  # noqa: E402
from biomedqa.verify import MINICHECK_DEFAULT_THRESHOLD, phi_from_scores  # noqa: E402

#: φ implementations this script can run. `minicheck` is the real verifier (`biomedqa.verify`);
#: `deberta-xsmall` is the interim one the 2026-08-14 artifact was computed with.
INTERIM_PHI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
MINICHECK_PHI_MODEL = "lytang/MiniCheck-Flan-T5-Large"

#: Systems that can carry citations at all. ADR-0010 excludes vanilla from every citation table.
SCORED = (System.JOINT, System.POST_HOC)


def _pairs_needed(records) -> list[tuple[str, str]]:
    """Every (premise, hypothesis) pair a real φ could ask for, collected by a φ that says no."""
    seen: dict[tuple[str, str], None] = {}

    def recorder(premise: str, hypothesis: str) -> bool:
        seen.setdefault((premise, hypothesis), None)
        return False

    citation_f1(records, recorder)
    return list(seen)


def _minicheck_scores(pairs, *, batch_size: int) -> dict[tuple[str, str], float]:
    """`{(premise, hypothesis): support probability}` under the real verifier.

    Continuous, and stored continuous: the threshold is applied once, by `phi_from_scores`, so the
    same scored map can be re-read at another operating point without a second forward pass.
    """
    from biomedqa.verify import MiniCheckVerifier, score_map

    return score_map(pairs, MiniCheckVerifier(batch_size=batch_size))


def _deberta_entailment(pairs, *, batch_size: int) -> dict[tuple[str, str], bool]:
    """`{(premise, hypothesis): entailed}` under the interim φ, argmax over its own label set."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(INTERIM_PHI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(INTERIM_PHI_MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    labels = {i: name.lower() for i, name in model.config.id2label.items()}
    entail_ids = [i for i, name in labels.items() if "entail" in name]
    if len(entail_ids) != 1:
        raise RuntimeError(f"cannot identify the entailment class in {labels}")
    entail_id = entail_ids[0]

    out: dict[tuple[str, str], bool] = {}
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = tokenizer(
                [p for p, _ in batch],
                [h for _, h in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            predicted = model(**encoded).logits.argmax(dim=-1).tolist()
            out.update({pair: p == entail_id for pair, p in zip(batch, predicted)})
            print(f"  φ: {min(start + batch_size, len(pairs))}/{len(pairs)}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return out


#: Operating points the contrast is re-read at once the scores exist. Free — no second forward
#: pass — and it is the question a single-threshold read cannot answer: whether the sign of
#: joint − post_hoc is a property of the systems or of where the cutoff happened to fall.
THRESHOLD_SWEEP = (0.1, 0.3, 0.5, 0.7, 0.9)


def _score_distribution(scores: dict[tuple[str, str], float]) -> dict:
    """What the continuous φ scores look like, so a level can be read without the pairs."""
    ordered = sorted(scores.values())
    n = len(ordered)
    return {
        "n_pairs": n,
        "mean": round(sum(ordered) / n, 4),
        "median": round(ordered[n // 2], 4),
        "frac_at_or_above": {
            str(t): round(sum(s >= t for s in ordered) / n, 4) for t in THRESHOLD_SWEEP
        },
    }


def _report(name: str, result: dict, ci: dict) -> None:
    print(f"\n{name}")
    print(f"  precision {result['precision']:.3f}   recall {result['recall']:.3f}   "
          f"**F1 {result['f1']:.3f}**  [{ci['lower']:.3f}, {ci['upper']:.3f}] "
          f"({ci['resampling_unit']}-clustered, n={ci['n_clusters']})")
    print(f"  recall over ALL claims {result['recall_all_claims']:.3f} -> "
          f"F1 {result['f1_all_claims']:.3f}  (ADR-0010: both denominators, always)")
    print(f"  claims {result['n_claims']} = {result['n_answered']} answered + "
          f"{result['n_abstentions']} abstentions · citations {result['n_citations']}, "
          f"{result['n_relevant_citations']} not irrelevant")


#: Word-count bands for the length read below. The gate's medians (15 / 17) sit in the third.
LENGTH_BANDS = ((0, 10), (11, 15), (16, 20), (21, 30), (31, 10_000))


def _recall_by_length(records, phi) -> list[dict]:
    """Per-claim recall by claim length — the one diagnostic that can tell a *method* effect from a
    *granularity* effect.

    ADR-0009 exists because coarser claims are harder to entail per claim, so an arm whose claims are
    longer is penalised for its shape rather than its grounding. The gate bounds that at ±15% of a
    median; this shows the whole curve, and whether the arms differ *within* a band.
    """
    rows = []
    for low, high in LENGTH_BANDS:
        row: dict = {"band": f"{low}-{high}" if high < 10_000 else f"{low}+"}
        for system in SCORED:
            recalls = [
                citation_recall(
                    claim, phi,
                    passages={p.passage_id: p.text for p in r.retrieved if p.text is not None},
                )
                for r in records
                if r.system is system
                for claim in answered_claims(r)
                if low <= len(claim.text.split()) <= high
            ]
            row[system.value] = {
                "n_claims": len(recalls),
                "recall": sum(recalls) / len(recalls) if recalls else None,
            }
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("prefix", help="artifact prefix, e.g. docs/harvest/parity_iter1b")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="ADR-0011 §2 defaults to 10000; F1 is recomputed per replicate, so this "
                         "trades interval precision for minutes")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="the run's per-call cap; enables the untruncated-basis sensitivity read")
    ap.add_argument("--phi", choices=("minicheck", "deberta-xsmall"), default="minicheck",
                    help="minicheck is the real verifier (biomedqa.verify); deberta-xsmall "
                         "reproduces the interim 2026-08-14 read")
    ap.add_argument("--threshold", type=float, default=MINICHECK_DEFAULT_THRESHOLD,
                    help="support-probability cutoff for --phi minicheck; MiniCheck's own is 0.5, "
                         "and G3 is what sweeps it")
    ap.add_argument("--out", type=Path, default=None, help="JSON artifact path")
    args = ap.parse_args()

    if PARITY_LOOP_CLOSED is None:
        raise SystemExit("ADR-0009 §6: the parity loop is open; citation-F1 is not computable yet")

    prefix = Path(args.prefix)
    # `records_path`, not `prefix.with_suffix(...)`: with_suffix truncates at the last dot, so a
    # prefix like `freqpen_0.1` would silently read `freqpen_0.records.jsonl`.
    records = list(read_query_records(records_path(prefix)))
    by_system = {s: [r for r in records if r.system is s] for s in SCORED}

    scores: dict[tuple[str, str], float] | None = None
    pairs = _pairs_needed(records)
    if args.phi == "minicheck":
        phi_model, operating_point = MINICHECK_PHI_MODEL, f"support probability >= {args.threshold}"
        print(f"first citation-F1 · {prefix.name} · φ = {phi_model} @ {operating_point}")
    else:
        phi_model, operating_point = INTERIM_PHI_MODEL, "argmax == entailment"
        print(f"first citation-F1 · {prefix.name} · φ = {phi_model} (interim, not MiniCheck)")
    print(f"parity loop closed {PARITY_LOOP_CLOSED.date} on {PARITY_LOOP_CLOSED.run}: gap "
          f"{PARITY_LOOP_CLOSED.gap:+.1%} [{PARITY_LOOP_CLOSED.interval[0]:+.1%}, "
          f"{PARITY_LOOP_CLOSED.interval[1]:+.1%}] — the blind is lifted, ADR-0009 §6")

    print(f"φ pairs to score: {len(pairs)}")
    if args.phi == "minicheck":
        scores = _minicheck_scores(pairs, batch_size=args.batch_size)
        phi = phi_from_scores(scores, args.threshold)
    else:
        entailed = _deberta_entailment(pairs, batch_size=args.batch_size)

        def phi(premise: str, hypothesis: str) -> bool:
            return entailed[(premise, hypothesis)]

    results = {s: citation_f1(by_system[s], phi) for s in SCORED}
    intervals = {}
    for system in SCORED:
        arm = by_system[system]
        intervals[system] = bootstrap_ci(
            arm,
            lambda drawn: citation_f1(drawn, phi)["f1"],
            clusters=[r.query_id for r in arm],
            n_boot=args.n_boot,
        )
        _report(system.value, results[system], intervals[system])

    #: The C2 contrast. Paired: the statistic scores both arms on the questions that were drawn.
    indexed = {s: {r.query_id: r for r in by_system[s]} for s in SCORED}
    shared = [q for q in indexed[System.JOINT] if q in indexed[System.POST_HOC]]

    def delta(drawn_ids) -> float:
        joint = citation_f1([indexed[System.JOINT][q] for q in drawn_ids], phi)["f1"]
        post_hoc = citation_f1([indexed[System.POST_HOC][q] for q in drawn_ids], phi)["f1"]
        return joint - post_hoc

    paired = bootstrap_ci(shared, delta, clusters=shared, n_boot=args.n_boot)
    print(f"\njoint − post_hoc F1: {paired['point']:+.3f} "
          f"[{paired['lower']:+.3f}, {paired['upper']:+.3f}] on {paired['n_clusters']} shared "
          f"questions")
    print("  crosses zero -> C2's direction is not established by this read"
          if paired["lower"] <= 0 <= paired["upper"] else
          "  interval excludes zero")

    sweep = None
    if scores is not None:
        # The scores are already computed, so re-reading the contrast at other cutoffs costs only
        # the scoring pass. A sign that flips inside the sweep is a fact about the threshold, and
        # the point estimate at 0.5 would then be reporting an arbitrary choice as a result.
        sweep = []
        for cutoff in THRESHOLD_SWEEP:
            swept = phi_from_scores(scores, cutoff)
            arms = {s: citation_f1(by_system[s], swept) for s in SCORED}
            sweep.append({
                "threshold": cutoff,
                **{s.value: {k: arms[s][k] for k in ("precision", "recall", "f1")} for s in SCORED},
                "delta_f1": arms[System.JOINT]["f1"] - arms[System.POST_HOC]["f1"],
            })
        print("\nthe same contrast at other operating points (no bootstrap — point estimates)")
        print(f"  {'τ':>5}{'joint F1':>11}{'post F1':>10}{'delta':>9}")
        for row in sweep:
            print(f"  {row['threshold']:>5}{row[System.JOINT.value]['f1']:>11.3f}"
                  f"{row[System.POST_HOC.value]['f1']:>10.3f}{row['delta_f1']:>+9.3f}")

    length_rows = _recall_by_length(records, phi)
    print("\nper-claim recall by claim length — a granularity effect looks like a level shift here")
    print(f"  {'band':>8}{'joint n':>10}{'joint R':>10}{'post n':>10}{'post R':>10}")
    print("  " + "-" * 48)
    for row in length_rows:
        j, p = row[System.JOINT.value], row[System.POST_HOC.value]
        print(f"  {row['band']:>8}{j['n_claims']:>10}"
              f"{(f'{j['recall']:.3f}' if j['recall'] is not None else '-'):>10}"
              f"{p['n_claims']:>10}"
              f"{(f'{p['recall']:.3f}' if p['recall'] is not None else '-'):>10}")

    censored = None
    if args.max_tokens is not None:
        costs = [CostRecord(**d) for d in read_jsonl(costs_path(prefix))]
        truncated = truncated_queries(records, costs, args.max_tokens)
        drop = truncated[System.JOINT.value] | truncated[System.POST_HOC.value]
        kept = [q for q in shared if q not in drop]
        censored = {
            "n_queries": len(kept),
            "dropped": len(drop),
            "delta_f1": delta(kept),
            **{s.value: citation_f1([indexed[s][q] for q in kept], phi) for s in SCORED},
        }
        print(f"\nsame-queries untruncated basis ({len(kept)} of {len(shared)}, dropping the "
              f"{len(drop)} where either arm hit the cap)")
        print(f"  joint F1 {censored[System.JOINT.value]['f1']:.3f} · post_hoc F1 "
              f"{censored[System.POST_HOC.value]['f1']:.3f} · delta {censored['delta_f1']:+.3f}")
        print("  this is the basis that excludes joint's runaway records (see "
              "docs/harvest/parity_iter1b.md)")

    artifact = {
        "script": "scripts/first_citation_f1.py",
        "purpose": "ADR-0009 §6 unblinding read on smoke-run records; not the G2 gate",
        "run": prefix.name,
        "phi": {
            "model": phi_model,
            "operating_point": operating_point,
            "interim": args.phi != "minicheck",
            # The continuous scores are kept so the read can be moved to another operating point
            # without a second forward pass — the sweep G3 owns, previewed on Table 2's own pairs.
            "score_distribution": _score_distribution(scores) if scores else None,
        },
        "parity_termination": {
            "date": PARITY_LOOP_CLOSED.date,
            "run": PARITY_LOOP_CLOSED.run,
            "gap": PARITY_LOOP_CLOSED.gap,
            "interval": list(PARITY_LOOP_CLOSED.interval),
        },
        "per_system": {s.value: results[s] | {"f1_ci": intervals[s]} for s in SCORED},
        "paired_delta_f1": paired,
        "threshold_sweep": sweep,
        "recall_by_claim_length": length_rows,
        "same_queries_untruncated": censored,
        "n_phi_pairs": len(pairs),
    }
    out = args.out or prefix.with_name(f"{prefix.name}.citation_f1.{args.phi}.json")
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
