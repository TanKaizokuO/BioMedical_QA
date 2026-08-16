#!/usr/bin/env python3
"""ADR-0012 §2 — Distractor confusability probe.

For each dev question, retrieve the RRF-fused top-5 (no reranker by default), drop gold passages,
then score
the question's gold claims against the non-gold passages with MiniCheck-Flan-T5-Large.  Reports the
distribution of entailment scores; **no threshold is pre-committed** — this is a first observation.

RUNS ON THE A4000 (needs GPU for MiniCheck and the MedCPT dense index).

    uv run python scripts/confusability_probe.py \\
      --index-dir data/index \\
      --split dev \\
      --out docs/harvest/confusability_probe.json

**Random control.**  A retrieved-distractor score is uninterpretable on its own: a third of passages
clearing 0.5 could mean RRF surfaces confusable neighbours, or that MiniCheck's base rate is a third,
or that a max over ~9 gold sentences inflates anything.  ``--random-control`` re-scores the same gold
sentences against the same *number* of passages drawn uniformly from the corpus, paired per question,
and reports the contrast.  It reads the retrieved-side run rather than recomputing it, so it needs no
dense index and cannot perturb the recorded retrieval numbers.

    uv run python scripts/confusability_probe.py \\
      --index-dir data/index \\
      --random-control docs/harvest/confusability_probe.json \\
      --out docs/harvest/confusability_probe_control.json

**Reranked arm (W3 item 2).**  ``--rerank`` puts the cross-encoder between the pool and the top-k,
so the distractors scored are the ones the deployed cascade actually shows the generator. Run it
into its own ``--out``, then pair a control against *that* file — the control's ``paired_against``
is what says which arm it belongs to.

**Scores before 2026-08-17 are off-distribution and must be re-derived.**  The private
``minicheck_score`` this script carried until then framed each pair with a standard NLI template —
the passage labelled as a premise, the claim labelled as a hypothesis — and read the decision off a
comparison of the sequence losses of the strings ``"1"`` and ``"0"``, truncated at 512.  It is a
reasonable-looking framing and the checkpoint was never trained on it.  MiniCheck-Flan-T5-Large
declares ``text-classification`` on the hub but is a ``T5ForConditionalGeneration`` with no
classification head, so every prompt returns *a* number and a wrong prompt returns a wrong number
without raising.  The reference (``Liyan06/MiniCheck``, ``minicheck/inference.py``) instead renders
``"predict: " + document + <eos> + claim``, runs one decoder step from a zero decoder input id, and
softmaxes the first-position logits at exactly two vocabulary ids — so the retired path had the
wrong input string, the wrong decode, and the wrong readout.

Every score this script produced before 2026-08-17 therefore came from a mis-invoked model.  The
committed artifacts ``docs/harvest/confusability_probe*.json`` are reads of that mis-invocation and
are left exactly as they are, as the record of what was run; ADR-0012's ``τ_confusable = 0.7`` was
set on that distribution and has to be re-derived from a re-run of this script against the corrected
path.  ``scripts/minicheck_format_check.py`` measured how far apart the two framings score on the
A4000 and keeps the retired template deliberately for that comparison.

**Scoring now routes through ``biomedqa.verify.MiniCheckVerifier``**, which is the single
MiniCheck implementation in the package.  The probe batch-scores all
(non-gold passage, gold sentence) pairs for a question in one ``score_pairs`` call, then takes the
max per passage — same aggregation shape as before, measurably faster.

**One intentional behavioural difference from the original.**  Truncation moves from 512 tokens
(original hard cut on the input string) to MiniCheck's 2048 with sentence-aligned chunking at
~500 words per chunk.  A long abstract is no longer silently cut at 512; the pair's score is the
maximum over all chunks, matching the reference implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap (scripts/ lives outside src/)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomedqa.config import RetrievalConfig  # noqa: E402
from biomedqa.data import Instance, load_splits, load_instances  # noqa: E402
from biomedqa.retrieve import RetrievalIndex, retrieve  # noqa: E402
from biomedqa.verify import MiniCheckVerifier  # noqa: E402


# ---------------------------------------------------------------------------
# Sentence splitting (proxy for gold-claim decomposition — decompose.py now exists as the C7
# granularity-ablation tool, off the C2 headline path; its regex is left alone here since editing
# it would invalidate the committed probe artifacts this script already produced)
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Rough sentence boundary split.  Good enough for a first distribution read."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile(data: list[float], p: float) -> float:
    """p in [0, 100].  Linear interpolation."""
    if not data:
        return float("nan")
    sorted_d = sorted(data)
    idx = (len(sorted_d) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_d) - 1)
    return sorted_d[lo] + (sorted_d[hi] - sorted_d[lo]) * (idx - lo)


#: Entailment cutoffs the distribution is reported against.  Chosen after seeing the first
#: distribution, as ADR-0012 §2 permits — the probe gates no tuning, so a post-hoc sweep is honest.
THRESHOLD_SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def describe(scores: list[float]) -> dict:
    """Distribution summary for a flat list of entailment scores."""
    if not scores:
        return {"n": 0}
    return {
        "n": len(scores),
        "mean": round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "p90": round(percentile(scores, 90), 4),
        "max": round(max(scores), 4),
        "min": round(min(scores), 4),
        "frac_at_or_above": {
            str(t): round(sum(s >= t for s in scores) / len(scores), 4) for t in THRESHOLD_SWEEP
        },
    }


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test under p=0.5.  Ties are excluded by the caller."""
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def run_random_control(args, instances, index, verifier: MiniCheckVerifier) -> int:
    """Score gold sentences against uniformly-drawn corpus passages, paired to a prior run.

    The pairing is what makes the contrast readable: for each question the control draws exactly
    ``n_non_gold`` passages, so the max-over-sentences aggregation sees the same number of chances
    on both sides and the inflation it causes cancels.
    """
    prior = json.loads(args.random_control.read_text(encoding="utf-8"))
    prior_by_pubid = {q["pubid"]: q for q in prior["per_question"]}
    print(f"Paired against {args.random_control} ({len(prior_by_pubid)} questions)")

    n_index = min(len(index.passage_ids), len(index.passage_texts))
    if n_index == 0:
        print("Index has no passage texts — control cannot run.", file=sys.stderr)
        return 1
    print(f"Sampling uniformly from {n_index:,} passages, seed={args.seed}")

    rng = random.Random(args.seed)
    per_question: list[dict] = []
    control_scores: list[float] = []

    for inst in instances:
        prior_q = prior_by_pubid.get(inst.pubid)
        if not prior_q or not prior_q.get("passage_max_scores"):
            continue
        n_draw = int(prior_q["n_non_gold"])
        gold_ids = set(inst.gold_passage_ids)

        gold_sentences = split_sentences(inst.abstract_text)
        if not gold_sentences:
            continue

        picked: list[int] = []
        seen: set[int] = set()
        for _ in range(n_draw * 50):
            if len(picked) == n_draw:
                break
            j = rng.randrange(n_index)
            if j in seen or index.passage_ids[j] in gold_ids:
                continue
            if not (index.passage_texts[j] or "").strip():
                continue
            seen.add(j)
            picked.append(j)

        # Batch all (passage, sentence) pairs for this question through one score_pairs call.
        pairs = [
            (index.passage_texts[j], sentence)
            for j in picked
            for sentence in gold_sentences
        ]
        if not pairs:
            continue
        raw_scores = [vs.score for vs in verifier.score_pairs(pairs)]

        # Re-aggregate: max over gold sentences per passage.
        n_sent = len(gold_sentences)
        q_scores: list[float] = []
        for passage_idx in range(len(picked)):
            offset = passage_idx * n_sent
            passage_max = max(raw_scores[offset : offset + n_sent])
            q_scores.append(passage_max)

        control_scores.extend(q_scores)
        per_question.append(
            {
                "pubid": inst.pubid,
                "n_drawn": len(q_scores),
                "n_gold_sentences": len(gold_sentences),
                "drawn_passage_ids": [index.passage_ids[j] for j in picked],
                "passage_max_scores": [round(s, 4) for s in q_scores],
                "q_mean": round(statistics.mean(q_scores), 4),
                "q_max": round(max(q_scores), 4),
                "retrieved_q_mean": prior_q["q_mean"],
                "retrieved_q_max": prior_q["q_max"],
            }
        )

        if len(per_question) % 10 == 0:
            print(f"  {len(per_question)} questions controlled …")

    retrieved_scores = [
        s
        for q in prior["per_question"]
        for s in q.get("passage_max_scores", [])
        if q["pubid"] in {p["pubid"] for p in per_question}
    ]

    wins = sum(1 for q in per_question if q["retrieved_q_mean"] > q["q_mean"])
    losses = sum(1 for q in per_question if q["retrieved_q_mean"] < q["q_mean"])
    deltas = [round(q["retrieved_q_mean"] - q["q_mean"], 4) for q in per_question]

    summary = {
        "n_questions_paired": len(per_question),
        "retrieved": describe(retrieved_scores),
        "random_control": describe(control_scores),
        "paired_q_mean_delta": {
            "mean": round(statistics.mean(deltas), 4) if deltas else None,
            "median": round(statistics.median(deltas), 4) if deltas else None,
            "retrieved_higher": wins,
            "control_higher": losses,
            "ties": len(per_question) - wins - losses,
            "sign_test_p": round(sign_test_p(wins, losses), 6),
        },
    }

    print("\n=== Random control summary ===")
    print(json.dumps(summary, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "script": "scripts/confusability_probe.py --random-control",
                "adr": "ADR-0012 §2",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "index_dir": str(args.index_dir),
                    "split": args.split,
                    "seed": args.seed,
                    "paired_against": str(args.random_control),
                    "minicheck_model": verifier.model_id,
                    "n_index_passages": n_index,
                },
                "summary": summary,
                "per_question": per_question,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nResults written to {args.out}")
    return 0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="ADR-0012 §2 distractor confusability probe (A4000 GPU required)"
    )
    ap.add_argument("--index-dir", required=True, type=Path, help="Directory with the prebuilt index")
    ap.add_argument(
        "--split",
        default="dev",
        choices=["dev", "test"],
        help="Which split to probe (default: dev)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/harvest/confusability_probe.json"),
        help="Output JSON path",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to retrieve before dropping gold (default: 5)",
    )
    ap.add_argument(
        "--no-gpu-check",
        action="store_true",
        help="Skip the CUDA availability guard (for dry-run syntax checks)",
    )
    ap.add_argument(
        "--random-control",
        type=Path,
        default=None,
        help=(
            "Path to a completed probe run.  Switches to control mode: draws the same number of "
            "uniformly-random corpus passages per question and scores them against the same gold "
            "sentences.  Skips retrieval entirely (no dense index loaded)."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for the random-control draw (default: 12345)",
    )
    ap.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Run the cross-encoder over the pool before taking the top-k (Week 3 re-confirm). "
            "Off by default: ADR-0012 §2's first distribution is pre-rerank."
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="MiniCheckVerifier batch size (default: 16)",
    )
    args = ap.parse_args()

    # A dropped --random-control writes the plain retrieved-side probe to a path that claims to be
    # a control, and the two are indistinguishable downstream except by the absent `seed` key.
    # That has already happened once and reached main as d9d6a13.  Refuse the combination.
    if args.random_control is None and "control" in args.out.stem.lower():
        print(
            f"--out is {args.out} but --random-control was not passed, so this run would write the "
            "ordinary retrieved-side probe under a control filename. Pass --random-control "
            "<prior probe json>, or choose an --out that does not say 'control'.",
            file=sys.stderr,
        )
        return 1

    import torch

    if not torch.cuda.is_available() and not args.no_gpu_check:
        print("CUDA not available — this script must run on the A4000.", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Load split
    # ------------------------------------------------------------------
    splits = load_splits()
    split_pubids: set[str] = set(splits[args.split])
    print(f"Split '{args.split}': {len(split_pubids)} questions")

    all_instances = load_instances()
    instances: list[Instance] = [i for i in all_instances if i.pubid in split_pubids]
    print(f"Loaded {len(instances)} instances for split '{args.split}'")

    # ------------------------------------------------------------------
    # Load retrieval index
    # ------------------------------------------------------------------
    # Control mode needs passage text only — skip the 3.1 GB dense matrix and the BM25 model.
    control = args.random_control is not None
    config = RetrievalConfig(
        bm25=not control,
        dense=not control,
        rrf=not control,
        # ADR-0012 §2's first observation is pre-rerank, so this stays off unless asked. W3 item 2
        # re-confirms the probe with --rerank, on both the retrieved and the control arm, because
        # a reranked top-5 is a different set of distractors than the fused one.
        rerank=args.rerank and not control,
        top_k=args.top_k,
    )
    if control and args.rerank:
        # Not an error: a paired control run is invoked alongside a reranked probe and will carry
        # the same flags. The control draws uniformly and never retrieves, so there is nothing to
        # rerank; `paired_against` in the output names the probe it belongs to.
        print("--rerank is inert in control mode (the control never retrieves); ignoring it.")
    print(f"Loading index from {args.index_dir} …")
    index = RetrievalIndex.load(args.index_dir, config)

    # ------------------------------------------------------------------
    # Load MiniCheck via biomedqa.verify — the single implementation in the package
    # ------------------------------------------------------------------
    verifier = MiniCheckVerifier(device=device, batch_size=args.batch_size)
    verifier.load()
    print(f"MiniCheck loaded: {verifier.model_id}")

    if control:
        return run_random_control(args, instances, index, verifier)

    # ------------------------------------------------------------------
    # Probe loop
    # ------------------------------------------------------------------
    started_at = datetime.now(timezone.utc).isoformat()
    per_question: list[dict] = []
    all_scores: list[float] = []

    for inst in instances:
        gold_ids = set(inst.gold_passage_ids)
        passages = retrieve(inst.question, config, index)

        # Drop gold passages — probe only non-gold distractors
        non_gold = [p for p in passages if p.passage_id not in gold_ids]
        if not non_gold:
            # All retrieved passages are gold — nothing to probe for this question
            per_question.append(
                {
                    "pubid": inst.pubid,
                    "question": inst.question,
                    "note": "all_retrieved_are_gold",
                    "scores": [],
                }
            )
            continue

        # Gold claims = sentences from the gold abstract (proxy)
        gold_sentences = split_sentences(inst.abstract_text)
        if not gold_sentences:
            per_question.append(
                {
                    "pubid": inst.pubid,
                    "question": inst.question,
                    "note": "no_gold_sentences",
                    "scores": [],
                }
            )
            continue

        # Batch all (passage, sentence) pairs for this question through one score_pairs call, then
        # re-aggregate: max over gold sentences per passage.  Same aggregation shape as the original
        # one-at-a-time loop, faster because the GPU sees a whole batch at once.
        #
        # The empty-text filter is the original's `if not (passage.text or ""): continue`, kept
        # verbatim: a whitespace-only passage was scored before and is scored now, so
        # `passage_max_scores` stays the same length it was on the committed artifacts.
        scorable = [p for p in non_gold if (p.text or "")]
        pairs = [
            (p.text, sentence)
            for p in scorable
            for sentence in gold_sentences
        ]
        raw_scores = [vs.score for vs in verifier.score_pairs(pairs)]

        n_sent = len(gold_sentences)
        q_scores: list[float] = []
        pair_details: list[dict] = []
        for passage_idx, passage in enumerate(scorable):
            offset = passage_idx * n_sent
            chunk_scores = raw_scores[offset : offset + n_sent]
            passage_max = max(chunk_scores)
            q_scores.append(passage_max)
            for sentence, s in zip(gold_sentences, chunk_scores):
                pair_details.append(
                    {
                        "passage_id": passage.passage_id,
                        "sentence_prefix": sentence[:60],
                        "score": round(s, 4),
                    }
                )

        all_scores.extend(q_scores)
        per_question.append(
            {
                "pubid": inst.pubid,
                "question": inst.question,
                "n_non_gold": len(non_gold),
                "n_gold_sentences": len(gold_sentences),
                "passage_max_scores": [round(s, 4) for s in q_scores],
                "q_mean": round(statistics.mean(q_scores), 4) if q_scores else None,
                "q_max": round(max(q_scores), 4) if q_scores else None,
                # Omit pair_details from final output to keep JSON manageable;
                # they are available in the per-run debug log
            }
        )

        if len(per_question) % 10 == 0:
            print(f"  {len(per_question)}/{len(instances)} questions processed …")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    summary: dict = {}
    if all_scores:
        summary = {
            "n_questions": len(instances),
            "n_questions_with_scores": sum(1 for q in per_question if q.get("passage_max_scores")),
            "n_scores_total": len(all_scores),
            "mean": round(statistics.mean(all_scores), 4),
            "median": round(statistics.median(all_scores), 4),
            "p90": round(percentile(all_scores, 90), 4),
            "max": round(max(all_scores), 4),
            "min": round(min(all_scores), 4),
        }
    else:
        summary = {"n_questions": len(instances), "n_scores_total": 0, "note": "no_scores_computed"}

    print("\n=== Confusability probe summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "script": "scripts/confusability_probe.py",
        "adr": "ADR-0012 §2",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "index_dir": str(args.index_dir),
            "split": args.split,
            "top_k": args.top_k,
            "minicheck_model": verifier.model_id,
            "retrieval": {
                "bm25": config.bm25,
                "dense": config.dense,
                "rrf": config.rrf,
                "rerank": config.rerank,
                "rrf_k": config.rrf_k,
                "corpus_id": config.corpus_id,
                "corpus_fingerprint": config.corpus_fingerprint,
            },
        },
        "summary": summary,
        "per_question": per_question,
    }
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
