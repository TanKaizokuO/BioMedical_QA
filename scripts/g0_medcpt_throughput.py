#!/usr/bin/env python3
"""G0 — measure MedCPT encode throughput on 1,000 abstracts, so the 2M encode stops being a guess.

ADR-0003 budgets ~2 h on the A4000 for the 2M-abstract corpus, and §3 of research_roadmap.md marks
that row "estimates, not measurements". G0 requires converting it before committing to the encode.
The decision this informs is real: if the measured extrapolation lands well over ~4 h, R1 says fall
back to 1M distractors — never to the 1,000 gold contexts (ADR-0003).

RUNS ON THE A4000, not the laptop (which has no CUDA). Copy it over and run there:

    scp scripts/g0_medcpt_throughput.py a4000:~/
    ssh a4000 'uv run --with torch --with transformers --with datasets python g0_medcpt_throughput.py'

Prints a table you paste into research_roadmap.md §3, replacing the estimated encode column.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = "NCBI/MedCPT-Article-Encoder"
EMBED_DIM = 768  # MedCPT is a BERT-base; fp16 storage is 2 bytes per dimension

CORPUS_SIZES = [1_000_000, 2_000_000, 23_900_000]  # 1M fallback, 2M chosen, full MedRAG (2027)


def load_abstracts(n: int) -> list[tuple[str, str]]:
    """(title, abstract) pairs — MedCPT's article encoder is trained on the pair, not on raw text.

    pqa_labeled is used only as a convenient source of real biomedical abstracts of realistic
    length. This measures the encoder, not the dataset; nothing here is a retrieval result.
    """
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    out = []
    for row in ds:
        ctx = row["context"]
        passages = list(ctx["contexts"]) if "contexts" in ctx else [str(ctx)]
        out.append((row["question"], " ".join(passages)))
        if len(out) >= n:
            break
    # pqa_labeled holds 1,000 rows; repeat if a larger sample was asked for, and say so.
    while len(out) < n:
        out.extend(out[: n - len(out)])
    return out[:n]


def fmt_hours(seconds: float) -> str:
    h = seconds / 3600
    return f"{h:.1f} h" if h >= 1 else f"{seconds / 60:.0f} min"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="abstracts to encode (default: %(default)s)")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", default="g0_medcpt_throughput.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        print("CUDA not available — this must run on the A4000, not the laptop.")
        return 1

    dev = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"gpu:    {gpu_name}  ({total_vram_gb:.1f} GB)")
    print(f"corpus: {args.n} abstracts, max_length={args.max_length}\n")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(dev).eval()

    pairs = load_abstracts(args.n)
    results = {}

    for bs in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Warm up and discard — kernel autotuning and the first allocation are not throughput
        # (docs/harvest/latency-benchmark-methodology.md, step 1).
        with torch.no_grad():
            enc = tok([p[0] for p in pairs[:bs]], [p[1] for p in pairs[:bs]],
                      truncation=True, padding=True, max_length=args.max_length,
                      return_tensors="pt").to(dev)
            model(**enc)
        torch.cuda.synchronize()

        batch_times = []
        t_start = time.perf_counter()
        with torch.no_grad():
            for i in range(0, len(pairs), bs):
                chunk = pairs[i : i + bs]
                t0 = time.perf_counter()
                enc = tok([c[0] for c in chunk], [c[1] for c in chunk],
                          truncation=True, padding=True, max_length=args.max_length,
                          return_tensors="pt").to(dev)
                out = model(**enc)
                _ = out.last_hidden_state[:, 0, :]  # CLS — MedCPT's article embedding
                torch.cuda.synchronize()
                batch_times.append(time.perf_counter() - t0)
        wall = time.perf_counter() - t_start

        per_batch = [t / bs for t in batch_times]
        rate = len(pairs) / wall
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        results[bs] = {
            "wall_s": round(wall, 2),
            "abstracts_per_s": round(rate, 1),
            "s_per_abstract_median": round(statistics.median(per_batch), 5),
            "s_per_abstract_min": round(min(per_batch), 5),
            "s_per_abstract_max": round(max(per_batch), 5),
            "peak_vram_gb": round(peak_gb, 2),
        }
        # Report the range, never only the mean — the base repo's 88s headline hid a 1.8x spread.
        print(f"batch {bs:>3}: {rate:>6.1f} abstracts/s   "
              f"{wall:>6.1f}s total   peak VRAM {peak_gb:.2f} GB   "
              f"per-abstract {min(per_batch)*1000:.1f}-{max(per_batch)*1000:.1f} ms")

    best_bs = max(results, key=lambda b: results[b]["abstracts_per_s"])
    best = results[best_bs]

    print(f"\nbest: batch {best_bs} at {best['abstracts_per_s']} abstracts/s\n")
    print("Extrapolated encode time (single A4000, fp16, this batch size):")
    print(f"  {'corpus':>12}  {'encode':>10}  {'embeddings fp16':>16}")
    extrapolation = {}
    for size in CORPUS_SIZES:
        secs = size / best["abstracts_per_s"]
        gb = size * EMBED_DIM * 2 / 1024**3
        extrapolation[size] = {"encode_s": round(secs), "embeddings_gb": round(gb, 1)}
        print(f"  {size:>12,}  {fmt_hours(secs):>10}  {gb:>13.1f} GB")

    two_m_hours = extrapolation[2_000_000]["encode_s"] / 3600
    print()
    if two_m_hours <= 4:
        print(f"2M lands at {two_m_hours:.1f} h — inside R1's ~4 h trigger. Proceed with 2M (ADR-0003).")
    else:
        print(f"2M lands at {two_m_hours:.1f} h — over R1's ~4 h trigger.")
        print("R1 response: fall back to 1M distractors "
              f"({fmt_hours(extrapolation[1_000_000]['encode_s'])}). "
              "Never fall back to the 1,000 gold contexts.")
    print("\nCaveats this measurement does not cover: tokenisation of a real 2M PubMed dump may run "
          "longer per abstract than pqa_labeled's, and the extrapolation assumes no I/O stall. "
          "Budget the encode as a checkpoint/resume job either way.")

    payload = {
        "kind": "g0_medcpt_throughput",
        "utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "gpu": gpu_name,
        "total_vram_gb": round(total_vram_gb, 1),
        "model": MODEL_ID,
        "n_abstracts": args.n,
        "max_length": args.max_length,
        "by_batch_size": results,
        "best_batch_size": best_bs,
        "extrapolation": {str(k): v for k, v in extrapolation.items()},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {args.out}   (scp it back into docs/harvest/g0/ — tracked; runs/ is gitignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
