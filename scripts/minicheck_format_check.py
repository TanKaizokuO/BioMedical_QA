#!/usr/bin/env python3
"""Does MiniCheck get the format it was trained on? — **the weights-side check for `verify.py`**.

    uv run python scripts/minicheck_format_check.py \\
      --records docs/harvest/parity_iter1b.records.jsonl \\
      --sample 200 --device cpu \\
      --out docs/harvest/minicheck_format_check.json

`lytang/MiniCheck-Flan-T5-Large` is a `T5ForConditionalGeneration` with no classification head, so
**every** prompt returns a number and a wrong prompt returns a wrong number *silently*. Nothing in
`tests/test_verify.py` can catch that: the tests pin the rendering and the arithmetic against fakes,
and a fake cannot say whether the real weights were trained on the string being rendered.

This script is that evidence, and it is the reason it exists rather than a one-off in a shell:

1. **Known-answer pairs.** Six pairs whose label is not in dispute — a paraphrase the document
   states, a negation it contradicts, a number it changes, an unrelated sentence. A correctly
   invoked MiniCheck separates them; an incorrectly invoked one does not have to.
2. **The retired framing, measured beside it.** `scripts/confusability_probe.py` scored pairs as
   `"premise: {p} hypothesis: {h}"` and compared the sequence loss of the strings `"1"` and `"0"` —
   plausible NLI framing, and not what this checkpoint was trained on (`minicheck/inference.py`
   renders `"predict: {doc}</s>{claim}"` and reads the first-position logits at ids 3 and 209).
   Both are run here on the same pairs so the size of the difference is a measured quantity rather
   than an argument, and so ADR-0012's `τ_confusable = 0.7` can be re-derived knowing which
   distribution it was set on.
3. **Real Table 2 pairs.** A sample of (cited span, claim) pairs out of a real run, because the
   known-answer set is six sentences of English and the biomedical degradation R7 predicts will not
   show up on it.

Runs on CPU in a few minutes for a couple of hundred pairs; `--device cuda` if the GPU is free.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from biomedqa.schema import read_query_records  # noqa: E402
from biomedqa.scoring.citation import _span_text  # noqa: E402
from biomedqa.verify import MiniCheckVerifier, minicheck_input  # noqa: E402

#: (document, claim, expected support) — the labels are the point; none of them is a judgement call.
KNOWN_ANSWER_PAIRS: tuple[tuple[str, str, bool], ...] = (
    (
        "A group of students gather in the school library to study for their upcoming final exams.",
        "The students are preparing for an examination.",
        True,
    ),
    (
        "A group of students gather in the school library to study for their upcoming final exams.",
        "The students are playing football on the field.",
        False,
    ),
    (
        "Metformin reduced all-cause mortality in patients with type 2 diabetes.",
        "Metformin lowered deaths from any cause in type 2 diabetics.",
        True,
    ),
    (
        "Metformin reduced all-cause mortality in patients with type 2 diabetes.",
        "Metformin did not affect all-cause mortality in patients with type 2 diabetes.",
        False,
    ),
    (
        "The trial randomised 412 patients across nine centres between 2011 and 2014.",
        "The trial randomised 412 patients.",
        True,
    ),
    (
        "The trial randomised 412 patients across nine centres between 2011 and 2014.",
        "The trial randomised 1,412 patients.",
        False,
    ),
)


def legacy_score(model, tokenizer, premise: str, hypothesis: str, device) -> float:
    """The retired framing, reproduced exactly as `confusability_probe.py` had it.

    Kept verbatim rather than tidied: its numbers are in a committed artifact and in ADR-0012's
    `τ_confusable`, so the thing measured here has to be the thing that produced them.
    """
    import torch

    inputs = tokenizer(
        f"premise: {premise} hypothesis: {hypothesis}",
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        positive = tokenizer("1", return_tensors="pt").input_ids.to(device)
        negative = tokenizer("0", return_tensors="pt").input_ids.to(device)
        log_positive = model(**inputs, labels=positive).loss.neg().item()
        log_negative = model(**inputs, labels=negative).loss.neg().item()
    top = max(log_positive, log_negative)
    exp_positive = math.exp(log_positive - top)
    exp_negative = math.exp(log_negative - top)
    return exp_positive / (exp_positive + exp_negative)


def real_pairs(records_path: Path, sample: int, seed: int) -> list[tuple[str, str]]:
    """(cited span, claim text) out of a real run — Table 2's own premises."""
    pairs: list[tuple[str, str]] = []
    for record in read_query_records(records_path):
        passages = {p.passage_id: p.text for p in record.retrieved if p.text is not None}
        for claim in record.claims:
            for citation in claim.citations:
                try:
                    pairs.append((_span_text(citation, passages), claim.text))
                except ValueError:
                    continue
    unique = list(dict.fromkeys(pairs))
    random.Random(seed).shuffle(unique)
    return unique[:sample]


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, ties averaged. The two framings can agree on order and disagree on level;
    that is a different defect from disagreeing on both, and τ only survives the first."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            mean_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = mean_rank
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    return statistics.correlation(rx, ry) if len(set(rx)) > 1 and len(set(ry)) > 1 else float("nan")


def describe(scores: list[float]) -> dict:
    thresholds = (0.3, 0.5, 0.7)
    return {
        "n": len(scores),
        "mean": round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "frac_at_or_above": {
            str(t): round(sum(s >= t for s in scores) / len(scores), 4) for t in thresholds
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--records", type=Path, default=None,
                    help="a run's records.jsonl; real (cited span, claim) pairs are sampled from it")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: cuda when available)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=_REPO / "docs/harvest/minicheck_format_check.json")
    args = ap.parse_args()

    verifier = MiniCheckVerifier(device=args.device, batch_size=args.batch_size, fp16=False)
    verifier.load()
    model, tokenizer = verifier._model, verifier._tokenizer
    device = verifier.device
    print(f"MiniCheck loaded on {device}")

    # -- 1. known answers --------------------------------------------------------------------
    known_pairs = [(d, c) for d, c, _ in KNOWN_ANSWER_PAIRS]
    reference = [s.score for s in verifier.score_pairs(known_pairs)]
    legacy = [legacy_score(model, tokenizer, d, c, device) for d, c in known_pairs]

    known_rows = []
    for (document, claim, supported), ref, leg in zip(KNOWN_ANSWER_PAIRS, reference, legacy):
        known_rows.append({
            "claim": claim,
            "expected_supported": supported,
            "reference_format": round(ref, 4),
            "retired_framing": round(leg, 4),
        })
        print(f"  expect {'support ' if supported else 'refute  '} "
              f"reference {ref:.4f}  retired {leg:.4f}  {claim[:60]}")

    def separation(scores: list[float]) -> float:
        """Mean supported score minus mean unsupported score — the only thing a usable φ must do."""
        positives = [s for s, (_, _, y) in zip(scores, KNOWN_ANSWER_PAIRS) if y]
        negatives = [s for s, (_, _, y) in zip(scores, KNOWN_ANSWER_PAIRS) if not y]
        return statistics.mean(positives) - statistics.mean(negatives)

    known = {
        "pairs": known_rows,
        "separation": {
            "reference_format": round(separation(reference), 4),
            "retired_framing": round(separation(legacy), 4),
        },
        "ranks_every_supported_pair_above_every_unsupported_one": {
            name: min(s for s, (_, _, y) in zip(scores, KNOWN_ANSWER_PAIRS) if y)
            > max(s for s, (_, _, y) in zip(scores, KNOWN_ANSWER_PAIRS) if not y)
            for name, scores in (("reference_format", reference), ("retired_framing", legacy))
        },
    }
    print(f"\nseparation (supported − unsupported): reference "
          f"{known['separation']['reference_format']:+.4f} · retired "
          f"{known['separation']['retired_framing']:+.4f}")

    # -- 2. real pairs -----------------------------------------------------------------------
    real = None
    if args.records is not None:
        pairs = real_pairs(args.records, args.sample, args.seed)
        print(f"\nscoring {len(pairs)} real (cited span, claim) pairs from {args.records.name}")
        real_reference = [s.score for s in verifier.score_pairs(pairs)]
        real_legacy = [legacy_score(model, tokenizer, p, h, device) for p, h in pairs]
        disagreement = [
            abs(a - b) for a, b in zip(real_reference, real_legacy)
        ]
        real = {
            "records": str(args.records),
            "n_pairs": len(pairs),
            "reference_format": describe(real_reference),
            "retired_framing": describe(real_legacy),
            "spearman": round(spearman(real_reference, real_legacy), 4),
            "mean_abs_difference": round(statistics.mean(disagreement), 4),
            "crossings_at_0.5": sum(
                (a >= 0.5) != (b >= 0.5) for a, b in zip(real_reference, real_legacy)
            ),
            "crossings_at_0.7": sum(
                (a >= 0.7) != (b >= 0.7) for a, b in zip(real_reference, real_legacy)
            ),
        }
        print(f"  reference mean {real['reference_format']['mean']:.4f} · retired mean "
              f"{real['retired_framing']['mean']:.4f} · spearman {real['spearman']:.4f}")
        print(f"  the two framings disagree about support for "
              f"{real['crossings_at_0.5']} of {len(pairs)} pairs at 0.5, "
              f"{real['crossings_at_0.7']} at 0.7")

    artifact = {
        "script": "scripts/minicheck_format_check.py",
        "purpose": "weights-side evidence that verify.py invokes MiniCheck the way it was trained",
        "written_at": datetime.now(UTC).isoformat(),
        "model": verifier.name,
        "device": device,
        "reference_format_example": minicheck_input("{document}", "{claim}", tokenizer.eos_token),
        "retired_framing_example": "premise: {premise} hypothesis: {hypothesis}",
        "known_answer": known,
        "real_pairs": real,
    }
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
