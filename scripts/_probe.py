#!/usr/bin/env python3
"""Throwaway: decompose-only A/B over the first N queries, counting duplicate claims per prompt
variant. The cite stage is skipped — this isolates the repetition loop."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

os.environ["VLLM_BASE_URL"] = "http://localhost:8000/v1"

from biomedqa import decompose as D  # noqa: E402
from biomedqa.config import GenerationConfig  # noqa: E402
from biomedqa.generate import split_stages  # noqa: E402
from biomedqa.schema import Granularity  # noqa: E402

sys.path.insert(0, str(_REPO / "scripts"))
from decompose_smoke import load_post_hoc  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
FP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
ONE_CLAIM_EXAMPLE = len(sys.argv) > 3 and sys.argv[3] == "one"

if ONE_CLAIM_EXAMPLE:
    D.FORMAT_BLOCK = """Reply in exactly this format and add nothing else:

CLAIM 1: the one thing the sentence asserts

Rules:
- Every line starts with "CLAIM <n>:", numbered from 1. Write nothing before the first CLAIM line
  and nothing after the last one.
- Most sentences assert exactly one thing and must produce exactly one CLAIM line. Write a second
  CLAIM line only when the sentence asserts two things that could be true or false independently,
  and then only for the second thing.
- Never write the same claim twice, and never restate in different words a claim you have already
  written.
- Stop as soon as the sentence you were asked to split has been covered."""

config = GenerationConfig(
    backend="vllm", model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    max_tokens=4096, frequency_penalty=FP, granularity=Granularity.ATOMIC.value,
)

queries = clean = dups = claims_n = sentences_n = 0
for src in load_post_hoc(Path("docs/harvest/parity_iter1b.records.jsonl"), N):
    answer = split_stages(src.raw_generation)[-1]
    d = D.decompose(answer, config, question=src.question, seed=0, run_id="probe",
                    query_id=src.query_id)
    queries += 1
    clean += not d.errors
    dups += sum(1 for e in d.errors if "repeats" in e)
    claims_n += len(d.claims)
    sentences_n += len(D.sentence_units(answer))

print(f"fp={FP} one_claim_example={ONE_CLAIM_EXAMPLE} n={queries}")
print(f"  clean_decompose_rate {clean/queries:.2f}  duplicates {dups}  "
      f"claims {claims_n} over {sentences_n} sentences ({claims_n/sentences_n:.2f}/sentence)")
