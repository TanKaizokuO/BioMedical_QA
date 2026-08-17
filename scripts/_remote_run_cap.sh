#!/usr/bin/env bash
set -euo pipefail
cd ~/BioMedical_QA
uv run python scripts/decompose_smoke.py \
  --model Qwen/Qwen2.5-14B-Instruct-AWQ \
  --base-url http://localhost:8000/v1 \
  --contexts docs/harvest/parity_iter1b.records.jsonl \
  --n 100 \
  --max-tokens 4096 \
  --frequency-penalty 0.5 \
  --out-prefix docs/harvest/decompose_cap \
  --overwrite
