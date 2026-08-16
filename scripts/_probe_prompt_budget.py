#!/usr/bin/env python3
"""Reproduce the exact re-citation request that 400'd, and measure the prompt budget.

The n=100 guided run died on query 17224424 with a bare 400. The schema is not the cause (the full
292-member enum posts fine on its own). This builds the real `recite_json` prompt for that query
and posts it with the run's real sampling parameters, then tokenises every query's prompt so the
boundary is a measured number rather than a guess.

Usage (on the A4000, inside the repo): .venv/bin/python3 scripts/_probe_prompt_budget.py
"""

from __future__ import annotations

import json

import httpx

from biomedqa.prompts import System, build_citation_response_format, build_prompt
from biomedqa.schema import RetrievedPassage

BASE = "http://localhost:8000"
MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
CONTEXTS = "docs/harvest/parity_iter1b.records.jsonl"
MAX_MODEL_LEN = 8192
MAX_TOKENS = 4096
DEPTH = 10
MAX_CITATIONS = 3


def n_tokens(client: httpx.Client, prompt: str) -> int:
    r = client.post("/tokenize", json={"model": MODEL, "prompt": prompt})
    r.raise_for_status()
    return r.json()["count"]


def main() -> int:
    records = [json.loads(line) for line in open(CONTEXTS)]

    with httpx.Client(base_url=BASE, timeout=300) as client:
        # --- 1. the exact failing request -------------------------------------------------------
        record = next(r for r in records if str(r["query_id"]).startswith("17224424"))
        passages = [RetrievedPassage(**p) for p in record["retrieved"]][:DEPTH]
        rendered = "\n".join(f"CLAIM {i}: placeholder claim text" for i in range(1, 6))
        rf = build_citation_response_format(passages, 5, MAX_CITATIONS)
        prompt = build_prompt(
            System.POST_HOC, record["question"], passages, MAX_CITATIONS,
            stage="recite_json", answer=rendered, depth=DEPTH, claim_count=5,
        )
        tok = n_tokens(client, prompt)
        print(f"17224424 recite_json prompt: {tok} tokens")
        print(f"  max_model_len={MAX_MODEL_LEN}  max_tokens={MAX_TOKENS}  "
              f"sum={tok + MAX_TOKENS}  over_by={tok + MAX_TOKENS - MAX_MODEL_LEN}")

        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "frequency_penalty": 0.5,
            "stop": [],
            "seed": 0,
            "response_format": rf,
        }
        r = client.post("/v1/chat/completions", json=body)
        print(f"  real request -> {r.status_code} {r.text[:400] if r.is_error else 'OK'}")

        # --- 2. how many of the 100 queries are over the line -----------------------------------
        print("\nprompt tokens per query (recite_json, 5 claims, depth 10):")
        seen: dict[str, int] = {}
        for record in records:
            qid = str(record["query_id"])
            if qid in seen:
                continue
            passages = [RetrievedPassage(**p) for p in record["retrieved"]][:DEPTH]
            prompt = build_prompt(
                System.POST_HOC, record["question"], passages, MAX_CITATIONS,
                stage="recite_json", answer=rendered, depth=DEPTH, claim_count=5,
            )
            seen[qid] = n_tokens(client, prompt)

        counts = sorted(seen.values())
        over = {q: t for q, t in seen.items() if t + MAX_TOKENS > MAX_MODEL_LEN}
        print(f"  n={len(counts)} min={counts[0]} median={counts[len(counts) // 2]} max={counts[-1]}")
        print(f"  budget for the prompt at max_tokens={MAX_TOKENS}: {MAX_MODEL_LEN - MAX_TOKENS}")
        print(f"  queries that cannot fit: {len(over)}")
        for q, t in sorted(over.items(), key=lambda kv: -kv[1])[:15]:
            print(f"    {q}: {t} tokens (over by {t + MAX_TOKENS - MAX_MODEL_LEN})")

        # --- 3. what completion budget would fit every query ------------------------------------
        print(f"\n  largest prompt is {counts[-1]}; a completion cap of "
              f"{MAX_MODEL_LEN - counts[-1]} tokens fits every query")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
