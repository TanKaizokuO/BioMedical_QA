#!/usr/bin/env python3
"""ADR-0012 §2 — Distractor confusability probe.

For each dev question, retrieve the RRF-fused top-5 (no reranker), drop gold passages, then score
the question's gold claims against the non-gold passages with MiniCheck-Flan-T5-Large.  Reports the
distribution of entailment scores; **no threshold is pre-committed** — this is a first observation.

RUNS ON THE A4000 (needs GPU for MiniCheck and the MedCPT dense index).

    python scripts/confusability_probe.py \\
      --index-dir data/index \\
      --split dev \\
      --out docs/harvest/confusability_probe.json
"""

from __future__ import annotations

import argparse
import json
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

MINICHECK_MODEL_ID = "lytang/MiniCheck-Flan-T5-Large"


# ---------------------------------------------------------------------------
# Sentence splitting (proxy for gold-claim decomposition — decompose.py is W3)
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Rough sentence boundary split.  Good enough for a first distribution read."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# MiniCheck wrapper
# ---------------------------------------------------------------------------

def load_minicheck(device: "torch.device"):  # type: ignore[name-defined]
    """Return (model, tokenizer) for MiniCheck-Flan-T5-Large."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    print(f"Loading MiniCheck from {MINICHECK_MODEL_ID} …")
    tok = AutoTokenizer.from_pretrained(MINICHECK_MODEL_ID)
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(MINICHECK_MODEL_ID, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    return model, tok


def minicheck_score(
    model,
    tok,
    premise: str,
    hypothesis: str,
    device,
) -> float:
    """Return the entailment probability in [0, 1].

    MiniCheck is framed as a Seq2Seq classification task where the positive class token is "1".
    The input format is ``premise: {passage} hypothesis: {claim}`` (standard NLI framing for
    MiniCheck-style T5 models).  We take the probability of generating "1" as the score.
    """
    import torch

    input_text = f"premise: {premise} hypothesis: {hypothesis}"
    inputs = tok(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        # Force-decode "1" and "0" to get their log-probabilities
        labels_pos = tok("1", return_tensors="pt").input_ids.to(device)
        labels_neg = tok("0", return_tensors="pt").input_ids.to(device)

        logp_pos = model(**inputs, labels=labels_pos).loss.neg()
        logp_neg = model(**inputs, labels=labels_neg).loss.neg()

    # Softmax over the two classes
    import math
    lp = [logp_pos.item(), logp_neg.item()]
    max_lp = max(lp)
    exp_pos = math.exp(lp[0] - max_lp)
    exp_neg = math.exp(lp[1] - max_lp)
    return exp_pos / (exp_pos + exp_neg)


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
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available() and not args.no_gpu_check:
        print("CUDA not available — this script must run on the A4000.", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    config = RetrievalConfig(
        bm25=True,
        dense=True,
        rrf=True,
        rerank=False,   # ADR-0012 §2: no reranker until W3
        top_k=args.top_k,
    )
    print(f"Loading index from {args.index_dir} …")
    index = RetrievalIndex.load(args.index_dir, config)

    # ------------------------------------------------------------------
    # Load MiniCheck
    # ------------------------------------------------------------------
    model, tok = load_minicheck(device)

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

        # Score every (non-gold passage, gold sentence) pair; take the max per passage
        q_scores: list[float] = []
        pair_details: list[dict] = []
        for passage in non_gold:
            passage_text_str = passage.text or ""
            if not passage_text_str:
                continue
            passage_max = 0.0
            for sentence in gold_sentences:
                s = minicheck_score(model, tok, passage_text_str, sentence, device)
                pair_details.append(
                    {
                        "passage_id": passage.passage_id,
                        "sentence_prefix": sentence[:60],
                        "score": round(s, 4),
                    }
                )
                passage_max = max(passage_max, s)
            q_scores.append(passage_max)

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
            "minicheck_model": MINICHECK_MODEL_ID,
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
