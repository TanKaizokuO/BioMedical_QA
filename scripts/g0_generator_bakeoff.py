#!/usr/bin/env python3
"""G0 stage B — choose the 8B AWQ generator on citation-format compliance, not benchmark scores.

The paper's failure mode is a model that writes fluent biomedical prose and ignores the citation
contract. General benchmark scores do not detect that; this does. A candidate that is fast, fluent,
and emits `[1]` markers unreliably is disqualified no matter how well it scores elsewhere
(research_roadmap.md §0 row 3, §2 D1).

What this measures
  - citation-format compliance, per the contract frozen in CONTEXT.md: every claim carries >= 1
    citation, markers are well-formed, indices are in range, and the <= 3 cap is respected
  - per-call latency and tokens/sec, following docs/harvest/latency-benchmark-methodology.md:
    warm up first and exclude it, record wall-clock separately from the server's own token counts,
    and report the range, never only the mean

What this does NOT measure, and must not be quoted as
  - retrieval quality (retrieve.py does not exist; the gold abstract is handed to the model)
  - answer accuracy (that is the test set, run once, in W9)
  - attribution correctness (that needs the gold set, W6)

Usage — one run per candidate, against a vLLM server already serving that model:

    ./scripts/g0_smoke.sh a4000                       # stage A must pass first
    ssh -L 8000:localhost:8000 a4000 '~/vllm-env/bin/vllm serve MODEL --quantization awq --port 8000'
    uv run scripts/g0_generator_bakeoff.py --model MODEL
    uv run scripts/g0_generator_bakeoff.py --model OTHER_MODEL     # after restarting the server
    uv run scripts/g0_generator_bakeoff.py --compare               # ranks every run in runs/g0/

Writes runs/g0/bakeoff_<model>_<timestamp>.json. The measured per-call latency goes into
research_roadmap.md §2, as G0 requires.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "runs" / "g0"

# The splits are not frozen until Aug 7 (research_roadmap.md §3). These 10 questions are a
# deterministic sample for a format probe only; their pubids are recorded in the output so the
# choice is auditable, and nothing measured here enters the paper.
PROBE_SEED = 20260731
PROBE_N = 10

MAX_CITATIONS = 3  # the fairness cap from CONTEXT.md — identical for every system, always

PROMPT = """You are answering a biomedical question using only the numbered passages below.

Write your answer as a list of atomic claims, one per line. Each claim must:
  - state exactly one fact
  - be self-contained (resolve every pronoun and implicit subject)
  - end with citations to the passages supporting it, as [n] or [n][m]
  - cite at most {max_citations} passages

Cite only from the passages. If the passages do not answer the question, say so as a single claim
citing the closest passage.

Passages:
{passages}

Question: {question}

Claims:"""

# A claim line ends with one or more [n] markers. Captured separately so that a line with no
# citation at all is still counted as a claim — silently dropping those would flatter the model.
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def load_probe_questions(n: int = PROBE_N, seed: int = PROBE_SEED) -> list[dict]:
    """PubMedQA pqa_labeled, deterministic sample. Gold contexts are kept as a list, not joined —
    see docs/harvest/pubmedqa-loading.md for why the base repo's `" ".join(contexts)` is a mistake
    this project does not repeat."""
    import random

    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    idx = random.Random(seed).sample(range(len(ds)), n)
    out = []
    for i in sorted(idx):
        row = ds[i]
        ctx = row["context"]
        passages = list(ctx["contexts"]) if "contexts" in ctx else [str(ctx)]
        out.append(
            {
                "pubid": str(row["pubid"]),
                "question": row["question"],
                "passages": passages,
                "labels": list(ctx.get("labels", [])) if isinstance(ctx, dict) else [],
            }
        )
    return out


def build_prompt(item: dict) -> str:
    numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(item["passages"], start=1))
    return PROMPT.format(
        max_citations=MAX_CITATIONS, passages=numbered, question=item["question"]
    )


def score_compliance(answer: str, n_passages: int) -> dict:
    """Mechanical checks only. No judgement about whether the claim is true or supported — that is
    the verifier's job (Phase 3) and the annotators' (W6). This asks one question: did the model
    honour the output contract?"""
    lines = [
        LIST_PREFIX_RE.sub("", ln).strip()
        for ln in answer.strip().splitlines()
        if ln.strip()
    ]
    claims = [ln for ln in lines if len(ln) > 15]  # drop headers, stray punctuation

    n_cited = 0
    n_over_cap = 0
    n_out_of_range = 0
    n_malformed = 0
    indices_seen: set[int] = set()

    for claim in claims:
        matches = CITATION_RE.findall(claim)
        if not matches:
            # A bare [ or a prose citation ("Passage 1 says...") is a contract failure, not an
            # alternative format — the parser downstream is not going to guess.
            if "[" in claim or re.search(r"\bpassage\s+\d", claim, re.I):
                n_malformed += 1
            continue
        n_cited += 1
        idxs = [int(x) for m in matches for x in re.split(r"\s*,\s*", m)]
        indices_seen.update(idxs)
        if len(idxs) > MAX_CITATIONS:
            n_over_cap += 1
        if any(i < 1 or i > n_passages for i in idxs):
            n_out_of_range += 1

    n = len(claims) or 1
    return {
        "n_claims": len(claims),
        "n_claims_cited": n_cited,
        "cited_rate": n_cited / n,
        "over_cap_rate": n_over_cap / n,
        "out_of_range_rate": n_out_of_range / n,
        "malformed_rate": n_malformed / n,
        "distinct_passages_cited": len(indices_seen),
        # One number for ranking. Deliberately harsh: a contract is not partially satisfied.
        "compliance": (n_cited - n_over_cap - n_out_of_range - n_malformed) / n,
    }


def call(client: httpx.Client, model: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,  # seedable locally — the Claude API rejects this (ADR-0004)
            "seed": 0,
        },
    )
    wall = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    return {
        "text": data["choices"][0]["message"]["content"],
        "wall_s": wall,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": data["choices"][0].get("finish_reason"),
    }


def run(args: argparse.Namespace) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url=args.base_url, timeout=args.timeout)

    try:
        served = [m["id"] for m in client.get("/v1/models").json()["data"]]
    except Exception as exc:
        print(f"cannot reach a vLLM server at {args.base_url}: {exc}", file=sys.stderr)
        print("Is the tunnel up?  ssh -L 8000:localhost:8000 <host> '... vllm serve ...'", file=sys.stderr)
        return 1
    if args.model not in served:
        print(f"server is serving {served}, not {args.model!r}", file=sys.stderr)
        return 1

    items = load_probe_questions(args.n)
    print(f"model:   {args.model}")
    print(f"probe:   {len(items)} pqa_labeled questions (seed {PROBE_SEED}, pre-split-freeze)")

    # Warm up and discard. Cold start is a separate quantity and must not contaminate the
    # distribution (harvest methodology, step 1).
    print("warmup...", end="", flush=True)
    call(client, args.model, "Reply with the single word: ready.", 8)
    print(" done")

    records = []
    for i, item in enumerate(items, start=1):
        prompt = build_prompt(item)
        try:
            r = call(client, args.model, prompt, args.max_tokens)
        except Exception as exc:
            print(f"[{i}/{len(items)}] pubid {item['pubid']}: FAILED — {exc}")
            records.append({"pubid": item["pubid"], "error": str(exc)})
            continue

        comp = score_compliance(r["text"], len(item["passages"]))
        tps = (r["completion_tokens"] or 0) / r["wall_s"] if r["wall_s"] else 0.0
        records.append(
            {
                "pubid": item["pubid"],
                "n_passages": len(item["passages"]),
                "wall_s": round(r["wall_s"], 3),
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "tokens_per_s": round(tps, 2),
                "finish_reason": r["finish_reason"],
                "compliance": comp,
                # Least-processed value: keep the raw generation. Every metric above is derivable
                # from it, and none of them are derivable from each other (CONTEXT.md, §2).
                "answer": r["text"],
            }
        )
        print(
            f"[{i}/{len(items)}] pubid {item['pubid']}: {r['wall_s']:.2f}s | "
            f"{r['completion_tokens']} tok | {tps:.1f} tok/s | "
            f"{comp['n_claims']} claims, compliance {comp['compliance']:.2f}"
            + ("  [TRUNCATED]" if r["finish_reason"] == "length" else "")
        )

    ok = [r for r in records if "error" not in r]
    if not ok:
        print("every call failed — nothing to summarise", file=sys.stderr)
        return 1

    walls = [r["wall_s"] for r in ok]
    comps = [r["compliance"]["compliance"] for r in ok]
    summary = {
        "n_ok": len(ok),
        "n_failed": len(records) - len(ok),
        "latency_s": {
            "mean": round(statistics.mean(walls), 2),
            "median": round(statistics.median(walls), 2),
            "min": round(min(walls), 2),
            "max": round(max(walls), 2),
        },
        "tokens_per_s_median": round(statistics.median(r["tokens_per_s"] for r in ok), 2),
        "compliance_mean": round(statistics.mean(comps), 3),
        "compliance_min": round(min(comps), 3),
        "cited_rate_mean": round(
            statistics.mean(r["compliance"]["cited_rate"] for r in ok), 3
        ),
        "over_cap_any": any(r["compliance"]["over_cap_rate"] > 0 for r in ok),
        "out_of_range_any": any(r["compliance"]["out_of_range_rate"] > 0 for r in ok),
        "truncated": sum(r["finish_reason"] == "length" for r in ok),
    }

    payload = {
        "kind": "g0_generator_bakeoff",
        "model": args.model,
        "base_url": args.base_url,
        "utc": datetime.now(timezone.utc).isoformat(),
        "client_host": platform.node(),
        "probe": {"seed": PROBE_SEED, "n": args.n, "pubids": [i["pubid"] for i in items]},
        "params": {"max_tokens": args.max_tokens, "temperature": 0.0, "seed": 0,
                   "max_citations": MAX_CITATIONS},
        "prompt_template": PROMPT,
        "summary": summary,
        "records": records,
    }
    slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model).strip("-").lower()
    out = OUT_DIR / f"bakeoff_{slug}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 64)
    print(f"{args.model}")
    print(f"  latency      {summary['latency_s']['median']:.2f}s median, "
          f"range {summary['latency_s']['min']:.2f}-{summary['latency_s']['max']:.2f}s")
    print(f"  throughput   {summary['tokens_per_s_median']:.1f} tok/s median")
    print(f"  compliance   {summary['compliance_mean']:.3f} mean, "
          f"{summary['compliance_min']:.3f} worst")
    print(f"  claims cited {summary['cited_rate_mean']:.1%}")
    if summary["over_cap_any"]:
        print(f"  ! exceeded the {MAX_CITATIONS}-citation cap — the cap is a fairness control")
    if summary["out_of_range_any"]:
        print("  ! cited passage indices that do not exist")
    if summary["truncated"]:
        print(f"  ! {summary['truncated']} generations hit max_tokens — raise --max-tokens and re-run")
    print("=" * 64)
    print(f"saved: {out.relative_to(REPO_ROOT)}")
    print("\nWrite the median latency into research_roadmap.md §2 (G0 requires it).")
    return 0


def compare() -> int:
    """Rank every bake-off in runs/g0/. Compliance first — that is the deciding axis."""
    runs = sorted(OUT_DIR.glob("bakeoff_*.json"))
    if not runs:
        print(f"no bake-off runs in {OUT_DIR}", file=sys.stderr)
        return 1
    rows = []
    for p in runs:
        d = json.loads(p.read_text())
        s = d["summary"]
        rows.append((s["compliance_mean"], s["latency_s"]["median"],
                     s["tokens_per_s_median"], d["model"], d["utc"][:16]))
    rows.sort(reverse=True)
    print(f"{'compliance':>10}  {'median s':>9}  {'tok/s':>7}  model")
    for c, lat, tps, model, when in rows:
        print(f"{c:>10.3f}  {lat:>9.2f}  {tps:>7.1f}  {model}   ({when})")
    print("\nDecide on compliance. Latency only breaks ties — the A4000 already retired that risk.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="model id as served by vLLM (exactly as in /v1/models)")
    ap.add_argument("--base-url", default="http://localhost:8000",
                    help="vLLM OpenAI-compatible endpoint (default: %(default)s, via ssh -L)")
    ap.add_argument("--n", type=int, default=PROBE_N, help="probe questions (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--compare", action="store_true", help="rank existing runs and exit")
    args = ap.parse_args()

    if args.compare:
        return compare()
    if not args.model:
        ap.error("--model is required (or use --compare)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
