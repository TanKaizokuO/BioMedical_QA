#!/usr/bin/env python3
"""Slice A Probe: verify vLLM guided/structured decoding on ~5 questions.

Compares unconstrained (control) vs guided decoding on 5 post-hoc questions.
Per claim, prints: emitted quote, locate_quote result, and total latency.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

os.environ["VLLM_BASE_URL"] = "http://localhost:8000/v1"

from biomedqa import decompose as D
from biomedqa.config import GenerationConfig
from biomedqa.generate import cite_claims, split_stages
from biomedqa.schema import Granularity, System, read_query_records

def main():
    n_questions = 5
    records_path = Path("docs/harvest/parity_iter1b.records.jsonl")
    source_records = [r for r in read_query_records(records_path) if r.system is System.POST_HOC][:n_questions]

    config = GenerationConfig(
        backend="vllm",
        model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        max_tokens=4096,
        frequency_penalty=0.5,
        granularity=Granularity.ATOMIC.value,
    )

    print(f"=== Running Slice A Probe on N={len(source_records)} questions ===", flush=True)

    control_quotes = []
    control_located = 0
    control_total = 0
    guided_quotes = []
    guided_located = 0
    guided_total = 0

    t0_control = time.perf_counter()
    for idx, src in enumerate(source_records, 1):
        print(f"[CONTROL {idx}/{n_questions}] {src.query_id}...", flush=True)
        answer = split_stages(src.raw_generation)[-1]
        decomp = D.decompose(answer, config, question=src.question, seed=0, run_id="probe_ctl", query_id=src.query_id)
        recite = cite_claims(
            decomp.claims, src.question, src.retrieved, config, seed=0,
            run_id="probe_ctl", query_id=src.query_id, guided_decoding=False
        )
        for c in recite.claims:
            for cit in c.citations:
                control_total += 1
                found = (cit.quoted_text is not None)
                if found:
                    control_located += 1
                control_quotes.append((c.claim_id, cit.passage_id, cit.quoted_text, found))
    t_control = time.perf_counter() - t0_control

    t0_guided = time.perf_counter()
    for idx, src in enumerate(source_records, 1):
        print(f"[GUIDED {idx}/{n_questions}] {src.query_id}...", flush=True)
        answer = split_stages(src.raw_generation)[-1]
        decomp = D.decompose(answer, config, question=src.question, seed=0, run_id="probe_gd", query_id=src.query_id)
        recite = cite_claims(
            decomp.claims, src.question, src.retrieved, config, seed=0,
            run_id="probe_gd", query_id=src.query_id, guided_decoding=True
        )
        for c in recite.claims:
            for cit in c.citations:
                guided_total += 1
                found = (cit.quoted_text is not None)
                if found:
                    guided_located += 1
                guided_quotes.append((c.claim_id, cit.passage_id, cit.quoted_text, found))
    t_guided = time.perf_counter() - t0_guided

    # Summary report
    print("\n=== PROBE RESULTS (N=5) ===", flush=True)
    print(f"Control Wall Time: {t_control:.2f}s | Citations: {control_total} | Located: {control_located}/{control_total} ({control_located/max(1, control_total)*100:.1f}%)")
    print(f"Guided Wall Time:  {t_guided:.2f}s | Citations: {guided_total} | Located: {guided_located}/{guided_total} ({guided_located/max(1, guided_total)*100:.1f}%)")

    print("\n--- Control Quotes Sample (Unconstrained) ---", flush=True)
    for cid, pid, qtext, found in control_quotes[:5]:
        status = "LOCATED" if found else "NOT_FOUND"
        print(f"  [{status}] claim={cid} passage={pid} quote={qtext!r}")

    print("\n--- Guided Quotes Sample (Constrained) ---", flush=True)
    for cid, pid, qtext, found in guided_quotes[:5]:
        status = "LOCATED" if found else "NOT_FOUND"
        print(f"  [{status}] claim={cid} passage={pid} quote={qtext!r}")

if __name__ == "__main__":
    main()
