#!/usr/bin/env bash
# Throwaway: run the C7 smoke on the A4000 box at whatever n the caller asks for.
set -euo pipefail
cd ~/BioMedical_QA
N="${1:-10}"
PREFIX="${2:-docs/harvest/_diag}"
uv run python scripts/decompose_smoke.py \
  --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --base-url http://localhost:8000/v1 \
  --n "$N" --max-tokens 4096 --frequency-penalty 0.5 \
  --out-prefix "$PREFIX" --overwrite >"${PREFIX}.log" 2>&1
tail -8 "${PREFIX}.log"
