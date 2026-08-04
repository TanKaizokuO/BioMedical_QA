# Runbook — vLLM on the A4000 (Windows host, WSL2 guest)

**Verified end-to-end 2026-08-04** on `DESKTOP-5C6NFL8`. Reproduces a working vLLM server from a
bare Windows install. Every step below was executed; nothing here is inferred.

See [ADR-0008](../../adr/0008-a4000-is-a-windows-box-vllm-runs-in-wsl2.md) for *why* this shape.

---

## Host facts as verified

| | |
|---|---|
| GPU | RTX A4000, 16376 MiB, **WDDM mode with a display attached** |
| Driver | 582.08, reporting a CUDA **13.0** ceiling |
| Idle VRAM held by desktop | ~500–700 MiB, **drifts upward during use** (Chrome, Edge, dwm, M365Copilot) |
| Disk | 743 GB free on `C:` |
| `sshd` | `AUTO_START` — safe to reboot remotely |
| SSH session | already elevated (`whoami /groups` shows `High Mandatory Level`) |

---

## 1. Install WSL2

The `wsl.exe` in `System32` on a fresh box is the **inbox stub**, with a stale distro catalog. Its
usage banner lists only four arguments (`--install`, `--list`, `--status`, `--help`); real WSL has
many more. That is how you recognise it.

```cmd
wsl --update --web-download
wsl --install -d Ubuntu-24.04 --web-download
```

`--web-download` pulls from Microsoft's CDN instead of the Store — necessary here, and generally
necessary on managed boxes where Store access is blocked by policy.

**Do not** diagnose the `Invalid distribution name` error as a permissions problem. It fails at
argument parsing, before any privilege check.

Reboot, reconnect, then `wsl` to complete first-run setup (UNIX username and password).

## 2. Verify GPU passthrough

Inside WSL:

```bash
nvidia-smi
```

This must print the same A4000 table as the Windows side. **Do not install an NVIDIA driver inside
the distro** — the Windows driver is passed through, and installing one in the guest breaks it.

`nvidia-smi` succeeding inside WSL is itself proof of WSL2: GPU passthrough does not exist under
WSL1. (`wsl -l -v` does not work *inside* the distro — it is a Windows-side command. Check version
from the Windows side if you need to.)

## 3. Toolchain the distro does not ship

Two separate needs, discovered one after the other:

```bash
sudo apt-get update
sudo apt-get install -y build-essential          # torch.compile / Triton JIT needs a C compiler
```

```bash
# FlashInfer JIT-compiles its sampling kernel on first use and needs nvcc specifically.
# vLLM's own wheels bundle their CUDA runtime, so this is NOT "install CUDA for vLLM" —
# it is narrower than that, and only surfaces at first sampling.
# Match the toolkit to the DRIVER's reported ceiling (13.0 here), not the newest patch line.
sudo apt-get install -y cuda-toolkit-13-0        # from NVIDIA's WSL-Ubuntu apt repo
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH                 # persist both in ~/.bashrc
```

## 4. vLLM in its own environment

Isolated from this project's env by design — see the note in `pyproject.toml`. vLLM pins torch
exactly and will drag the workspace back to pydantic 1.10.x if it is ever added as a dependency.

```bash
uv venv ~/venvs/vllm-server --python 3.12
source ~/venvs/vllm-server/bin/activate
uv pip install vllm      # ~8 GB of CUDA wheels; start it and walk away
```

## 5. Launch

```bash
source ~/venvs/vllm-server/bin/activate
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve <model> \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0
```

Both non-default settings are load-bearing:

- **`VLLM_USE_V2_MODEL_RUNNER=0`** — vLLM 0.26.0's `GPUModelRunnerV2` (default since 0.25.0)
  hard-requires pinned host memory for its `StagedWriteTensor`/`UvaBuffer` path. WSL2's
  `cudaHostAlloc` support is unreliable, so this crashes with `RuntimeError: UVA is not available`
  on *every* model, including a 0.6B. Older vLLM only warned (`pin_memory=False`); the new runner
  made it fatal. **`--no-async-scheduling` does not avoid it** — the UVA buffer is built
  unconditionally. Forcing the legacy runner is the fix.
- **`--gpu-memory-utilization 0.85`** — the WDDM concession. The display baseline drifts upward
  during desktop use; a tighter cap keeps a browser tab from OOM-ing a benchmark mid-run.

## 6. Reach it from outside

WSL2's localhost mirroring works on this box, verified with `curl http://localhost:8000/v1/models`
from a Windows `cmd.exe` prompt. **No `netsh portproxy` rule and no explicit port forward were
needed.** From the laptop, tunnel to the Windows host and let Windows forward inward:

```bash
ssh -L 8000:localhost:8000 <host-alias>
```

---

## Failure signatures, for recognition

Each of these only became visible after the previous one was cleared, so expect them serially
rather than all at once.

| Signature | Cause |
|---|---|
| `Invalid distribution name` + unversioned `--list --online` catalog | inbox WSL stub, stale catalog — **not** permissions |
| `RuntimeError: UVA is not available` in `UvaBuffer.__init__` | `GPUModelRunnerV2` + WSL2 pinned memory |
| `InductorError: Failed to find C compiler` | no `build-essential` |
| `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` | FlashInfer sampling-kernel JIT wants `nvcc` |
| `wsl: command not found` | you are inside the distro; `wsl` is Windows-side |
