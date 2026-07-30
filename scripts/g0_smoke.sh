#!/usr/bin/env bash
# G0 stage A — is the A4000 box actually usable?
#
# This is the true first measurement of the project (research_roadmap.md §0, row 3): no successful
# vLLM load on that machine has ever been recorded, and the laptop has no CUDA fallback, so every
# downstream deliverable is behind this script. Run it BEFORE the bake-off, not as part of it.
#
# Runs entirely over SSH using tools that ship with any Linux box — nothing is installed here.
# A failure at this stage is a driver/access problem to solve today, not a modelling problem.
#
#   ./scripts/g0_smoke.sh <ssh-host>          # e.g. ./scripts/g0_smoke.sh a4000
#
# Writes runs/g0/smoke_<timestamp>.txt and prints a verdict. Paste the verdict into
# research_roadmap.md §2 alongside the G0 latency number.

set -uo pipefail

HOST="${1:-}"
if [[ -z "$HOST" ]]; then
    echo "usage: $0 <ssh-host>" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/runs/g0"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/smoke_$(date -u +%Y%m%dT%H%M%SZ).txt"

# Budget for the GPU-resident set at G3, when all three models are co-resident and the overhead
# measurement needs them to be: 8B AWQ ~6 GB + MiniCheck-770M ~1.5 GB + cross-encoder ~1.3 GB.
REQUIRED_FREE_MIB=10240   # ~10 GB of the card's 16 GB, leaving KV-cache headroom
REQUIRED_FREE_DISK_GB=60  # model weights + a 2M-abstract index (~12 GB peak, ADR-0003) + slack

pass=0; fail=0; warn=0
say()  { echo -e "$*" | tee -a "$OUT"; }
ok()   { pass=$((pass+1)); say "  [ OK ]   $*"; }
bad()  { fail=$((fail+1)); say "  [FAIL]   $*"; }
note() { warn=$((warn+1)); say "  [WARN]   $*"; }

# Single multiplexed connection: one auth, and it proves the link is stable rather than merely
# reachable once.
CTL="$(mktemp -u /tmp/g0-ssh-%C.XXXXXX)"
SSH=(ssh -o ControlMaster=auto -o ControlPath="$CTL" -o ControlPersist=60
         -o ConnectTimeout=10 -o BatchMode=yes "$HOST")
r() { "${SSH[@]}" "$@" 2>&1; }

say "G0 stage A — A4000 preflight"
say "host: $HOST   at: $(date -u +%Y-%m-%dT%H:%M:%SZ)   from: $(hostname)"
say ""

say "1. Reachability"
if ! r true >/dev/null; then
    bad "cannot ssh to '$HOST' non-interactively."
    say ""
    say "  Nothing else can be checked. Likely causes, in order:"
    say "    - no Host entry for '$HOST' in ~/.ssh/config"
    say "    - key not yet installed on the box (BatchMode disallows password prompts)"
    say "    - box powered off or not on this network"
    say ""
    say "VERDICT: BLOCKED — G0 cannot proceed. This is today's only critical-path problem."
    exit 1
fi
ok "ssh works non-interactively"
say "  remote: $(r 'uname -sr; . /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"' | tr '\n' ' ')"

say ""
say "2. GPU and driver"
if ! r 'command -v nvidia-smi >/dev/null'; then
    bad "nvidia-smi not found — the NVIDIA driver is not installed."
    say "         A driver install will eat the afternoon. Start it now, then re-run."
else
    gpu_csv="$(r 'nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader,nounits')"
    say "  $gpu_csv"
    name="$(echo "$gpu_csv" | cut -d, -f1 | xargs)"
    free_mib="$(echo "$gpu_csv" | cut -d, -f4 | xargs)"
    used_mib="$(echo "$gpu_csv" | cut -d, -f3 | xargs)"

    [[ "$name" == *A4000* ]] && ok "card is $name" \
                             || note "card reports '$name', expected an RTX A4000 (ADR-0004)"

    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= REQUIRED_FREE_MIB )); then
        ok "${free_mib} MiB VRAM free (need >= ${REQUIRED_FREE_MIB} for the co-resident set)"
    else
        bad "${free_mib} MiB VRAM free, need >= ${REQUIRED_FREE_MIB} MiB"
        say "         ${used_mib} MiB is in use — the box is supposed to be exclusive."
        say "         Check for a stray process:  ssh $HOST nvidia-smi"
    fi

    # Compute capability decides whether the AWQ kernels will even load. Ampere is 8.6.
    cc="$(r 'nvidia-smi --query-gpu=compute_cap --format=csv,noheader' | xargs)"
    [[ -n "$cc" ]] && say "  compute capability: $cc (Ampere = 8.6; AWQ kernels need >= 7.5)"
fi

say ""
say "3. Python toolchain"
if r 'command -v uv >/dev/null'; then
    ok "uv present: $(r 'uv --version')"
else
    note "uv not found. Install:  curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
say "  python: $(r 'command -v python3 >/dev/null && python3 -V || echo none')"

say ""
say "4. vLLM"
# vllm is deliberately NOT a dependency of this project (see pyproject.toml): it pins torch
# exactly, and adding it backtracks the whole workspace. It lives in its own environment on the
# box and is reached over HTTP.
if r 'python3 -c "import vllm, sys; print(vllm.__version__)"' | grep -qE '^[0-9]'; then
    ok "vllm importable: $(r 'python3 -c "import vllm; print(vllm.__version__)"')"
else
    note "vllm not importable in the default python3 on the box."
    say "         Expected until you install it. It gets its own env, not this project's:"
    say "           ssh $HOST 'uv venv ~/vllm-env && ~/vllm-env/bin/uv pip install vllm'"
fi
if r 'command -v nvcc >/dev/null'; then
    say "  nvcc: $(r 'nvcc --version | tail -2 | head -1' | xargs)"
else
    say "  nvcc: absent (fine — vllm wheels ship their own CUDA runtime)"
fi

say ""
say "5. Capacity"
disk_gb="$(r 'df -BG --output=avail ~ | tail -1 | tr -dc "0-9"')"
if [[ "$disk_gb" =~ ^[0-9]+$ ]] && (( disk_gb >= REQUIRED_FREE_DISK_GB )); then
    ok "${disk_gb} GB free in \$HOME (need >= ${REQUIRED_FREE_DISK_GB})"
else
    bad "${disk_gb:-?} GB free in \$HOME, need >= ${REQUIRED_FREE_DISK_GB} GB for weights + the 2M index"
fi
say "  RAM:  $(r "free -g | awk '/^Mem:/ {print \$2\" GB total, \"\$7\" GB available\"}'")"
say "  CPU:  $(r "nproc") cores"
r 'test -w "$HOME"' >/dev/null && ok "\$HOME is writable" || bad "\$HOME is not writable"

say ""
say "6. HuggingFace access (model weights must download before the bake-off)"
if r 'test -n "$HF_TOKEN" || test -f ~/.cache/huggingface/token'; then
    ok "an HF token is present on the box"
else
    note "no HF token found. Gated repos will 401 mid-download, during the bake-off, not before it."
fi

ssh -o ControlPath="$CTL" -O exit "$HOST" 2>/dev/null || true

say ""
say "----------------------------------------------------------------"
say "pass: $pass   warn: $warn   fail: $fail"
if (( fail > 0 )); then
    say "VERDICT: BLOCKED — fix the [FAIL] lines before the bake-off. Everything is behind this."
    say "saved: $OUT"
    exit 1
fi
if (( warn > 0 )); then
    say "VERDICT: USABLE, setup incomplete — the [WARN] lines are install steps, not faults."
else
    say "VERDICT: READY — proceed to stage B (scripts/g0_generator_bakeoff.py)."
fi
say ""
say "Stage B needs a vLLM server up on the box. From a second terminal:"
say "  ssh -L 8000:localhost:8000 $HOST '~/vllm-env/bin/vllm serve <MODEL_ID> \\"
say "      --quantization awq --max-model-len 8192 --port 8000'"
say "then:  uv run scripts/g0_generator_bakeoff.py --model <MODEL_ID> ..."
say ""
say "saved: $OUT"
