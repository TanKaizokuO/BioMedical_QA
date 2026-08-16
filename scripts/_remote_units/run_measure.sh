#!/usr/bin/env bash
# The C7 decompose measurement, as run by the `biomedqa-run.service` --user unit.
#
# This is `run_all.sh` with the two things a unit makes unnecessary removed: it no longer starts
# the vLLM server (`vllm-8b.service` owns that, so the server outlives the measurement and a
# follow-up run does not pay the model load again), and it no longer ends in `sleep 7200`. The
# sleep existed to hold the pts/0 session open so the distro would not tear the artifacts down
# with it; a unit under a lingering `systemd --user` has no session to hold.
set -uo pipefail

MODEL=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
BASE=http://localhost:8000/v1
# The sanity artifacts are written OUTSIDE the repository on purpose. `harness.git_sha()` suffixes
# the manifest `-dirty` whenever `git status --porcelain` is non-empty, and that includes untracked
# files — so a sanity run that dropped four files into docs/harvest/ would make the n=100 manifest
# that follows it unreproducible from any commit, which is exactly what a Gate G2 number may not be.
SANITY_PREFIX=/home/user/sanity_out/_sanity_guided
mkdir -p "$(dirname "$SANITY_PREFIX")"
# `--max-tokens 2048`, not 4096. The server runs `--max-model-len 8192`, and vLLM refuses any
# request where prompt + requested completion exceeds that. Measured 2026-08-16: the largest
# `recite_json` prompt of the 100 queries is 17224424 at 4327 tokens, so 4327 + 4096 = 8423 and the
# n=100 run died on it at query 24 of 100 with a bare 400. 4096 was never needed — across the
# sanity run the largest completion any stage produced was 1298 tokens (`decompose_cite`; the
# decompose stage peaks at 135), none truncated. 2048 leaves 58% headroom over the worst observed
# completion and a 6144-token prompt budget against a ~4400-token worst case.
COMMON=(--model "$MODEL" --base-url "$BASE"
        --contexts docs/harvest/parity_iter1b.records.jsonl
        --max-tokens 2048 --frequency-penalty 0.5 --overwrite)
stamp() { date -u +%H:%M:%S; }

# The unit is `WantedBy=default.target`, so a WSL2 distro restart starts it again — and the
# measurement below opens with `rm -f` and `--overwrite`. Without this guard, a reboot *after* a
# finished run silently destroys the result it took 3.3 h to produce. A finished run is one whose
# summary covers the full n, so that is what is checked.
if .venv/bin/python3 - <<'PY'
import json, sys
try:
    rows = json.load(open("docs/harvest/decompose_guided_v2.summary.json"))
except Exception:
    sys.exit(1)
rows = rows.get("per_row", rows)
sys.exit(0 if rows.get("sentence", {}).get("n_queries") == 100 else 1)
PY
then
  echo "[$(stamp)] decompose_guided_v2 already complete (n=100) — refusing to overwrite it"
  exit 0
fi

echo "[$(stamp)] waiting for vllm-8b.service to answer /v1/models"
for _ in $(seq 1 90); do
  curl -s --max-time 3 "$BASE/models" 2>/dev/null | grep -q '"id"' && break
  sleep 10
done
if ! curl -s --max-time 3 "$BASE/models" | grep -q '"id"'; then
  echo "[$(stamp)] SERVER NOT READY after 15 min — aborting"
  tail -20 /home/user/vllm_8b.log
  exit 1
fi
echo "[$(stamp)] server READY"

# --- gate: n=3 integration proof of the guided citation path before spending 3.3 h --------------
rm -f "$SANITY_PREFIX".*
.venv/bin/python3 -u scripts/decompose_smoke.py "${COMMON[@]}" \
  --n 3 --out-prefix "$SANITY_PREFIX" > /home/user/sanity.log 2>&1
echo "[$(stamp)] sanity exit=$?"

GATE=$(SANITY_PREFIX="$SANITY_PREFIX" .venv/bin/python3 - <<'PY'
import json, os
try:
    d = json.load(open(os.environ["SANITY_PREFIX"] + ".summary.json"))
except Exception as exc:
    print("FAIL no summary:", exc); raise SystemExit
rows = d.get("per_row", d)
ok = True
for row in ("atomic", "decontextualized_atomic"):
    e = rows.get(row, {})
    q, c = e.get("quote_located_rate"), e.get("claim_parse_rate")
    print(f"{row}: quote_located={q} claim_parse={c} clean_cite={e.get('clean_cite_rate')} "
          f"claims={e.get('total_claims')} qnf={e.get('quote_not_found_count')}")
    if q is None or c is None or q < 0.95 or c < 0.95:
        ok = False
print("PASS" if ok else "FAIL")
PY
)
echo "$GATE"
if ! printf '%s' "$GATE" | tail -1 | grep -q PASS; then
  echo "[$(stamp)] GATE FAILED — not spending the 3.3 h; vllm-8b.service stays up for debugging"
  exit 1
fi

# --- the measurement ----------------------------------------------------------------------------
# The manifest records `git_sha()` when the run starts, so the tree has to be clean *now* — the
# previous run's own `decompose_guided_v2.*` files are untracked and would dirty it.
echo "[$(stamp)] launching n=100 guided v2 (expect ~3.3 h)"
rm -f docs/harvest/decompose_guided_v2.*
if [ -n "$(git status --porcelain)" ]; then
  echo "[$(stamp)] WARNING — tree is not clean, the manifest will be marked -dirty:"
  git status --porcelain | head -20
fi
.venv/bin/python3 -u scripts/decompose_smoke.py "${COMMON[@]}" \
  --n 100 --out-prefix docs/harvest/decompose_guided_v2 > /home/user/guided_v2.log 2>&1
rc=$?
echo "[$(stamp)] guided_v2 exit=$rc"
exit "$rc"
