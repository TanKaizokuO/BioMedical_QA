# Runbook — reaching the two A4000 boxes from the agent environment

**Verified 2026-09-08.** Every command below was executed from the agent environment against the
live boxes; the outputs quoted are what came back. Steps that were *not* executed are marked
**UNVERIFIED** and say so in place.

See [ADR-0022](../../adr/0022-second-a4000-is-a-shared-linux-box-ssh-replaces-copy-paste.md) for
*why* the two boxes are used differently, and
[`wsl-vllm-a4000.md`](wsl-vllm-a4000.md) for the vLLM server on Box A.

---

## The two boxes

| | **Box A** | **Box B** |
|---|---|---|
| SSH alias | `vllm-box` | `a4000-linux` |
| Address | `user@172.16.84.110` | `user@172.16.84.58` |
| Hostname | `DESKTOP-5C6NFL8` | `user-DIT400TR-55RL` |
| OS | Windows 10 19045.6466 (WSL2 Ubuntu-24.04, **Stopped**) | Ubuntu 22.04.2, kernel 6.8.0-101 |
| Default remote shell | `cmd.exe` | `bash` |
| GPU / driver | RTX A4000 16376 MiB / 582.08 | RTX A4000 16376 MiB / 590.48.01 |
| CPU / RAM | 2× Xeon Gold 6226R (64 threads) / 128 GB | 2× Xeon Gold 6226R (64 threads) / 125 GB |
| Free disk | 743 GB on `C:` | 572 GB on `/` |
| Role | generation box of record (vLLM) | CPU/storage + idle GPU; **no project checkout yet** |

Both are on the rater LAN (`172.16.84.0/24`), which is the premise ADR-0017 relies on.

## Credentials

`.env.local` at the repo root holds `SSH_HOST_1/USER_1/PASS_1` (Box A) and `SSH_HOST_2/USER_2/PASS_2`
(Box B). It is **gitignored — never commit it, never paste a password into a doc, a commit message,
or an issue.** After the key install below, the passwords are needed only for `sudo` on Box B.

## `~/.ssh/config` (agent environment, not in the repo)

```
Host vllm-box
    HostName 172.16.84.110
    User user
    ServerAliveInterval 30
    ServerAliveCountMax 6
    LocalForward 8000 127.0.0.1:8000

Host a4000-linux
    HostName 172.16.84.58
    User user
    ServerAliveInterval 30
    ServerAliveCountMax 6
    LocalForward 8001 127.0.0.1:8000
```

**The forwards deliberately differ.** Box A's vLLM lands on local `8000`, Box B's would land on
local `8001`, so both sessions can be open at once without a bind clash. A plain command run over an
alias still binds its forward; if the local port is taken, SSH warns and the command still runs.

## Key install (done once per box; Box B was done 2026-09-08)

Box A already accepted the key. Box B needed it, and password auth is interactive, so it must run on
a PTY — a bare `ssh-copy-id` in a captured-output shell will hang at the prompt:

```bash
ssh-copy-id -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519.pub user@172.16.84.58
```

Confirmed by `Number of key(s) added: 1`, then verified non-interactively:

```bash
ssh -o BatchMode=yes a4000-linux 'hostname; nvidia-smi --query-gpu=name --format=csv,noheader'
# user-DIT400TR-55RL
# NVIDIA RTX A4000
```

`-o BatchMode=yes` is the check that matters: it fails rather than falling back to a password
prompt, so a green result proves key auth.

---

## Box A — the shell is `cmd.exe`, and it bites

`ssh vllm-box 'echo OK; hostname'` prints the literal string `OK; hostname`. `cmd.exe` does not
split on `;`. Three consequences:

- **Separate commands with `&`**, not `;`: `ssh vllm-box "hostname & ver"`.
- **PowerShell for anything structured**:
  `ssh vllm-box "powershell -NoProfile -Command \"(Get-CimInstance Win32_Processor).Name\""`.
- **PowerShell output arrives UTF-16-ish** — pipe through `tr -d '\000'` or the text looks
  interleaved with nulls.

WSL is Windows-side: `wsl -l -v` works over SSH, and reported `Ubuntu-24.04  Stopped  2` on
2026-09-08. Running anything under WSL boots the distro; the vLLM sequence is in
[`wsl-vllm-a4000.md`](wsl-vllm-a4000.md). Nothing was listening on `:8000`, `:8765` or `:11434`, and
`tasklist` showed no `python`, `vllm` or `ollama` process, so **no server was running on Box A** at
the time of writing.

## Box B — Ubuntu, shared, and bare

```bash
ssh a4000-linux 'uname -r; nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader'
# 6.8.0-101-generic
# NVIDIA RTX A4000, 16376 MiB, 293 MiB, 590.48.01
```

What is there, and what is not:

| | |
|---|---|
| `python3` | 3.10.12 (system), Anaconda 3.12.4 at `~/anaconda3` |
| `uv` | **absent** |
| `nvcc` / CUDA toolkit | **absent** (driver only) |
| `docker` | present |
| `git` | present |
| `sudo` | works, **password required** (`SSH_PASS_2`); `echo "$PASS" \| sudo -S -v` returns 0 |
| Network | PyPI and huggingface.co both return 200 |
| Port 8000 | free |
| Port 11434 | **taken — Ollama, serving `qwen2.5:7b`** |
| Home directory | another person's work (`churn`, `edu`, `hadoopdata`, coursework on the desktop) |

**Box B is somebody's workstation.** The GPU drives a display (293 MiB baseline, no compute
processes at the time of writing) and the Ollama server is not ours: ADR-0022 §6 forbids using it as
a backend and forbids stopping it. Check for other people's GPU work before starting anything long:

```bash
ssh a4000-linux 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```

Empty output means the card is free right now — not that it is reserved.

### Setting Box B up for project work — **UNVERIFIED**

Not executed as of 2026-09-08. There is **no project checkout on Box B**, no `uv`, and no HF cache
(`~/.cache/huggingface` does not exist). The intended shape, when it is first needed, is: install
`uv`, clone from `origin/main`, `uv sync`, and run CPU/verifier work per ADR-0022 §2. Treat each
step as unproven until someone runs it and updates this file. In particular, **nothing is known
about vLLM on Box B** — the WSL2-specific settings in
[`wsl-vllm-a4000.md`](wsl-vllm-a4000.md) §5 (`VLLM_USE_V2_MODEL_RUNNER=0` above all) exist for a
WSL2 defect and have no measured meaning on a native box.

---

## Discipline that survives SSH access

1. **One line per command.** Multi-line blocks with `\` continuations have silently dropped flags
   three times. SSH does not fix that; it just makes the loss faster.
2. **`uv run python`, never bare `python`** — on either box, in any checkout.
3. **Name the box in anything that carries a wall clock** (ADR-0022 §3). Two hosts, one GPU model:
   an unlabelled `wall_s` is unusable and an accidental cross-box comparison is invisible.
4. **Pull before you push, on every box.** `git pull --rebase` first; Box A pushes to `origin/main`
   directly.
5. **Do not commit an artifact from a dirty checkout**, and do not commit `.env.local`.
