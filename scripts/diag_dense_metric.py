#!/usr/bin/env python3
"""Does L2-normalising MedCPT embeddings cost us dense recall?  **RUNS ON THE A4000.**

Table 1 row 2 came in at dev hit@5 = 0.59, *below* BM25's 0.71. MedCPT exists to beat lexical
retrieval on PubMed, so that ordering is a bug smell rather than a ceiling.

The leading suspect: both MedCPT model cards score with a **raw dot product on unnormalised CLS
vectors** —

    embeds = model(**encoded).last_hidden_state[:, 0, :]
    scores = query_embeds @ article_embeds.T

— and NCBI ships its pre-computed PubMed embeddings unnormalised. Our pipeline L2-normalises both
sides (`encode_corpus._encode_batch`, `retrieve._encode_query`), which silently turns the dot
product into cosine. MedCPT was trained contrastively *with* the dot product, so ‖v‖ carries signal
that normalising discards.

`data/index/empty/dense.npy` was normalised at encode time, so the norms are gone and the question
cannot be answered from the existing index. This script re-encodes a *sample* unnormalised, then
scores the same vectors both ways. One encode, both metrics, paired over the same questions — so
the comparison isolates the metric and nothing else.

It deliberately does **not** decide anything on its own: a sample of `--distractors` passages is
easier than the full 2.16M index, so the absolute hit@5 here is not a Table 1 number. What
transfers is the *sign and size of the gap* between cosine and dot, and the norm dispersion — if
‖v‖ is nearly constant, normalising cannot be the culprit and the search moves elsewhere.

    uv run python scripts/diag_dense_metric.py --distractors 200000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomedqa.chunk import chunk_instance  # noqa: E402
from biomedqa.config import ChunkConfig  # noqa: E402
from biomedqa.corpus import passage_text  # noqa: E402
from biomedqa.data import Instance, load_instances, load_splits  # noqa: E402
from biomedqa.scoring.retrieval import wilson_interval  # noqa: E402

ARTICLE_ENCODER = "NCBI/MedCPT-Article-Encoder"
QUERY_ENCODER = "NCBI/MedCPT-Query-Encoder"
ARTICLE_MAX_LENGTH = 512
QUERY_MAX_LENGTH = 64      # the length the query-encoder card uses
DENSE_CHUNK_ROWS = 200_000


# ---------------------------------------------------------------------------
# Encoding — unnormalised, which is the entire point of this script
# ---------------------------------------------------------------------------

def _encode(
    texts: list[str],
    tok,
    mdl,
    device: torch.device,
    *,
    max_length: int,
    pair: bool,
    batch_size: int,
    label: str,
) -> np.ndarray:
    """Raw CLS embeddings, float32, **no L2 normalisation**."""
    out = np.empty((len(texts), 768), dtype=np.float32)
    started = time.time()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        if pair:
            enc = tok([""] * len(batch), batch, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
        else:
            enc = tok(batch, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            cls = mdl(**enc).last_hidden_state[:, 0, :]
        out[start:start + len(batch)] = cls.cpu().float().numpy()

        done = start + len(batch)
        if done % 20_000 == 0 or done == len(texts):
            rate = done / max(time.time() - started, 1e-9)
            print(f"    {label}: {done:,}/{len(texts):,}  {rate:.0f}/s", flush=True)
    return out


def _scores(emb: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """Chunked matvec, mirroring retrieve._dense_scores so the two cannot drift."""
    sims = np.empty(emb.shape[0], dtype=np.float32)
    for start in range(0, emb.shape[0], DENSE_CHUNK_ROWS):
        stop = min(start + DENSE_CHUNK_ROWS, emb.shape[0])
        sims[start:stop] = emb[start:stop] @ query_vec
    return sims


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(norms, 1e-9)


def _rank_of_gold(sims: np.ndarray, gold_positions: np.ndarray) -> int:
    """1-indexed rank of the best gold row. Full ranking, not a truncated top-k.

    Computed as "how many passages outscore the best gold, plus one", which needs no sort and is
    exact even when the gold sits at rank 800,000 — the ranks that matter most for diagnosis are
    precisely the ones a top-k list would throw away.
    """
    best_gold = float(sims[gold_positions].max())
    return int((sims > best_gold).sum()) + 1


# ---------------------------------------------------------------------------
# Corpus sampling
# ---------------------------------------------------------------------------

def _load_distractors(corpus_path: Path, limit: int, chunk_cfg: ChunkConfig) -> list[str]:
    from biomedqa.chunk import chunk_text

    if limit <= 0:      # --distractors 0 exercises the plumbing without the 5.8 GB corpus
        return []

    texts: list[str] = []
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            if len(texts) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                body = passage_text(row)
            except ValueError:
                continue
            row_id = str(row.get("id", row.get("PMID", "")))
            for chunk in chunk_text(body, row_id, chunk_cfg):
                texts.append(chunk.text)
                if len(texts) >= limit:
                    break
    return texts


def _gold_passages(instances: list[Instance], chunk_cfg: ChunkConfig):
    """(texts, owner_pubids) for every gold chunk, in a stable order."""
    texts: list[str] = []
    owners: list[str] = []
    for inst in instances:
        for chunk in chunk_instance(inst, chunk_cfg):
            texts.append(chunk.text)
            owners.append(inst.pubid)
    return texts, owners


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus/corpus.jsonl"))
    ap.add_argument("--distractors", type=int, default=200_000,
                    help="Corpus passages to encode as distractors (default: 200,000)")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("docs/harvest/dense_metric_probe.json"))
    ap.add_argument("--no-gpu-check", action="store_true")
    args = ap.parse_args()

    if not args.no_gpu_check and not torch.cuda.is_available():
        print("CUDA not available — this needs the A4000. Use --no-gpu-check to override.")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fp16 matmul is not implemented on CPU; a --no-gpu-check dry run must stay in fp32.
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Using device: {device} ({dtype})")

    chunk_cfg = ChunkConfig(strategy="abstract", max_chars=2000)

    splits = load_splits()
    pubids = set(splits[args.split])
    instances = [i for i in load_instances() if i.pubid in pubids]
    print(f"  {len(instances)} {args.split} instances")

    gold_texts, gold_owners = _gold_passages(instances, chunk_cfg)
    print(f"  {len(gold_texts):,} gold passages")

    print(f"Streaming {args.distractors:,} distractor passages …")
    distractors = _load_distractors(args.corpus, args.distractors, chunk_cfg)
    print(f"  {len(distractors):,} distractors")

    passages = gold_texts + distractors
    owner_of = gold_owners + [None] * len(distractors)
    gold_rows: dict[str, list[int]] = {}
    for idx, owner in enumerate(owner_of):
        if owner is not None:
            gold_rows.setdefault(owner, []).append(idx)

    print(f"Encoding {len(passages):,} passages (unnormalised, empty-title pair) …")
    a_tok = AutoTokenizer.from_pretrained(ARTICLE_ENCODER)
    a_mdl = AutoModel.from_pretrained(ARTICLE_ENCODER, dtype=dtype).eval().to(device)
    emb_raw = _encode(passages, a_tok, a_mdl, device, max_length=ARTICLE_MAX_LENGTH,
                      pair=True, batch_size=args.batch_size, label="passages")
    del a_mdl
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Encoding queries (unnormalised) …")
    q_tok = AutoTokenizer.from_pretrained(QUERY_ENCODER)
    q_mdl = AutoModel.from_pretrained(QUERY_ENCODER, dtype=dtype).eval().to(device)
    questions = [i.question for i in instances]
    q_raw = _encode(questions, q_tok, q_mdl, device, max_length=QUERY_MAX_LENGTH,
                    pair=False, batch_size=args.batch_size, label="queries")
    del q_mdl
    if device.type == "cuda":
        torch.cuda.empty_cache()

    p_norms = np.linalg.norm(emb_raw, axis=1)
    q_norms = np.linalg.norm(q_raw, axis=1)
    print("\nPassage embedding norms:")
    print(f"  mean={p_norms.mean():.4f}  std={p_norms.std():.4f}  "
          f"CV={p_norms.std() / p_norms.mean():.4f}")
    print(f"  min={p_norms.min():.4f}  p1={np.percentile(p_norms, 1):.4f}  "
          f"p50={np.percentile(p_norms, 50):.4f}  p99={np.percentile(p_norms, 99):.4f}  "
          f"max={p_norms.max():.4f}")

    emb_cos = _l2(emb_raw)
    q_cos = _l2(q_raw)

    variants = {
        "cosine": (emb_cos, q_cos),   # what the shipped index does
        "dot": (emb_raw, q_raw),      # what both MedCPT model cards do
    }

    per_query: list[dict] = []
    summary: dict[str, dict] = {}

    for name, (passage_mat, query_mat) in variants.items():
        ranks: list[int] = []
        for qi, inst in enumerate(instances):
            sims = _scores(passage_mat, query_mat[qi])
            ranks.append(_rank_of_gold(sims, np.array(gold_rows[inst.pubid])))
        hits = sum(1 for r in ranks if r <= args.k)
        point, lo, hi = wilson_interval(hits, len(ranks))
        summary[name] = {
            "hit_at_k": point,
            "wilson_lower": lo,
            "wilson_upper": hi,
            "hits": hits,
            "n": len(ranks),
            "rank_median": float(np.median(ranks)),
        }
        for inst, rank in zip(instances, ranks):
            per_query.append({"metric": name, "query_id": inst.pubid, "gold_rank": rank})
        print(f"\n  {name:8s}  hit@{args.k}={point:.4f}  Wilson [{lo:.4f}, {hi:.4f}]  "
              f"median gold rank={np.median(ranks):.0f}")

    cos_ranks = [r["gold_rank"] for r in per_query if r["metric"] == "cosine"]
    dot_ranks = [r["gold_rank"] for r in per_query if r["metric"] == "dot"]
    better = sum(1 for c, d in zip(cos_ranks, dot_ranks) if d < c)
    worse = sum(1 for c, d in zip(cos_ranks, dot_ranks) if d > c)
    print(f"\nPaired over the same {len(cos_ranks)} questions: "
          f"dot ranks gold higher on {better}, lower on {worse}, tied on "
          f"{len(cos_ranks) - better - worse}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "k": args.k,
        "n_gold_passages": len(gold_texts),
        "n_distractor_passages": len(distractors),
        "note": (
            "Sampled distractors, NOT the full 2.16M index — absolute hit@k here is not a "
            "Table 1 number. The transferable quantities are the paired cosine-vs-dot gap and "
            "the norm dispersion."
        ),
        "passage_norms": {
            "mean": float(p_norms.mean()), "std": float(p_norms.std()),
            "cv": float(p_norms.std() / p_norms.mean()),
            "min": float(p_norms.min()), "max": float(p_norms.max()),
            "p1": float(np.percentile(p_norms, 1)),
            "p50": float(np.percentile(p_norms, 50)),
            "p99": float(np.percentile(p_norms, 99)),
        },
        "query_norms": {
            "mean": float(q_norms.mean()), "std": float(q_norms.std()),
            "cv": float(q_norms.std() / q_norms.mean()),
        },
        "summary": summary,
        "paired": {"dot_better": better, "dot_worse": worse,
                   "tied": len(cos_ranks) - better - worse},
        "gold_rank_per_query": per_query,
    }, indent=2), encoding="utf-8")
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
