# ADR-0022 — A second A4000 box, shared and Linux-native; SSH replaces copy-paste

**Status:** Accepted · **Date:** 2026-09-08 · **Refines** ADR-0008 · **Changes an operational
premise of** ADR-0017 (it does not overturn either)

## Context

ADR-0008 recorded one machine: `DESKTOP-5C6NFL8`, Windows, RTX A4000 in WDDM mode, vLLM inside
WSL2. Everything downstream — the `gpu_idle_confirmed` manifest field, Table 4's wall clock,
ADR-0017's "the A4000 is copy-paste only from the agent environment", and `CLAUDE.md`'s one-line
command rule — was written for a single box reached by a human with a clipboard.

Two things changed on 2026-09-08, both verified from the agent environment rather than reported:

1. **A second A4000 box exists.**
2. **Both boxes take key-based SSH from the agent environment.** The first box already did; the key
   for the second was installed this session.

Inventory as measured (`nvidia-smi`, `lscpu`, `/etc/os-release`, `ss -ltn`, `df -h`; full command
transcript in [`docs/harvest/runbooks/a4000-boxes-access.md`](../harvest/runbooks/a4000-boxes-access.md)):

| | **Box A** | **Box B** |
|---|---|---|
| Alias | `vllm-box` | `a4000-linux` |
| Host | `DESKTOP-5C6NFL8` · 172.16.84.110 | `user-DIT400TR-55RL` · 172.16.84.58 |
| OS | Windows 10 19045.6466; WSL2 Ubuntu-24.04 (Stopped) | Ubuntu 22.04.2, kernel 6.8.0-101, **native** |
| GPU | RTX A4000, 16376 MiB, **WDDM, display attached** | RTX A4000, 16376 MiB, **display attached** (293 MiB idle) |
| Driver | 582.08 (CUDA 13.0 ceiling) | 590.48.01 |
| CPU / RAM | 2× Xeon Gold 6226R, 64 threads / 128 GB | 2× Xeon Gold 6226R, 64 threads / 125 GB |
| Free disk | 743 GB on `C:` (ADR-0008) | 572 GB on `/` |
| Project toolchain | vLLM venv, `serve_8b.sh` (in WSL2) | **none** — no `uv`, no `nvcc`, system Python 3.10.12, Anaconda 3.12.4, Docker |
| Other occupants | none observed | `~/churn`, `~/edu`, `~/hadoopdata`, coursework on the desktop; **Ollama on 127.0.0.1:11434 serving `qwen2.5:7b`** |

The second box is **not a dedicated machine**. It is somebody's Ubuntu workstation that happens to
hold an A4000, and it drives a display like Box A does. It is the same CPU and the same card as Box
A; what differs is the substrate (native Linux vs WSL2), the driver, and who else is on it.

## Decision

**1. Box A stays the generation box of record.** Every committed generation artifact —
`generate_fp05_n100_guided_v4` above all, the Gate G2 run of record — was produced on Box A under
WSL2, vLLM 0.26.0, `VLLM_USE_V2_MODEL_RUNNER=0`, driver 582.08. Box B is a different substrate:
native Linux, driver 590.48.01, and no reason to force the legacy runner. A generation run on Box B
is **not** comparable to `v4` and may not be mixed into a contrast with it. If a gate-relevant
generation ever moves to Box B, the whole contrast moves with it and is re-run there.

**2. Box B carries substrate-independent and freshly-measured work.** Verifier scoring (MiniCheck
φ, AlignScore), index and corpus builds, scoring and bootstrap passes, and any CPU-bound batch are
free to run on Box B — their outputs are determined by inputs and weights, not by the host. This is
where the second box actually pays: 64 threads and 572 GB of free disk that nobody is waiting on.

**3. Every artifact that carries a wall clock names its box.** `wall_s`, GPU-hour costs, and the
G3 verifier-cost figures are host-dependent, and there are now two hosts with the same card and
different substrates. A timing without a host name is unusable, and **no wall-clock number may be
compared across boxes** — including the two A4000s, whose identical GPU model makes the mistake
easy to make. The G3 verifier cost per claim must therefore record which box produced it.

**4. `gpu_idle_confirmed` means less on Box B than on Box A, and must be read that way.** ADR-0008
already downgraded it to "desktop session confirmed quiescent at run time" on a WDDM box. Box B is
shared with another person's work, so on Box B the field means "quiescent when checked, not
reserved". Anything sensitive to contention states its box and keeps its status as a *secondary*
measure (ADR-0004: tokens and dollars are primary).

**5. SSH is the access path for both boxes; the copy-paste rule is retired.** An agent may read
logs, start and stop servers, and run measurement scripts over SSH on either box, without routing
each command through a human. The **one-line command discipline survives** for anything a human
still runs, and for anything sent over SSH: multi-line blocks with `\` continuations have silently
dropped flags three times, and a lost flag over SSH is the same wasted GPU run as a lost flag on a
clipboard.

**6. The Ollama server on Box B is not a project backend.** `qwen2.5:7b` on 127.0.0.1:11434 is the
retired harvest generator (ADR-0007 retired that pipeline; ADR-0004 fixed the generator as a local
8B AWQ behind vLLM). It belongs to the box's owner. Do not point `backends.py` at it, do not stop
it, and do not count its VRAM as available.

## Consequences

- **ADR-0017's second premise no longer holds.** "The A4000 is copy-paste only from the agent
  environment. Every restart and log read is a human step" was the reason a down annotation
  collector could look like three idle raters for three days. An agent can now read
  `annotation_collect.py`'s log and restart it directly. **ADR-0017's decision is unchanged** — the
  sidecar stays write-mostly, `localStorage` stays primary, and the panel still shows counts and
  times, because those defend against a rater restoring the wrong copy, not against a human being
  slow to check a log.
- **The `pyproject.toml` isolation argument is unchanged and now applies twice.** vLLM on Box B
  would be a third environment, still not a dependency, and still bound by the `uv run --no-project`
  caution in the WSL runbook — Box B has no `uv` at all today.
- **Box B's vLLM path is unverified.** Nothing about WSL2's failure signatures (`UVA is not
  available`, Python.h, FlashInfer's `nvcc`) is known to transfer, and the legacy-runner setting
  exists for a WSL2 defect that a native box should not have. None of that has been measured. Until
  it is, Box B is a CPU and storage resource plus an idle GPU, not a second vLLM server.
- **Reproducibility appendix gains a row.** The paper says "Hardware: one RTX A4000"
  (`paper/skeleton.md` §4). That stays true per measurement, but the appendix must name *which*
  box produced each measured number once anything runs on Box B.
- **A second machine invites a third failure mode: silent divergence.** Two checkouts, two caches,
  two model copies. The runbook fixes the checkout path and states that Box B pulls from
  `origin/main` like everyone else; no artifact is committed from a box whose checkout is dirty.

## What this does *not* change

ADR-0004 (local 8B AWQ generator, Opus 5 judge), ADR-0008 (the WSL2 substrate and its three
load-bearing settings), the frozen prompt digests, `CONFIG_VERSION 1.5.0`, and the Gate G2 sign-off
on `v4`. No gate is re-opened by acquiring hardware, and no artifact is re-run because a faster
path exists.

## Alternatives rejected

- **Move everything to Box B because it is native Linux.** Would void the comparability of every
  committed generation artifact against a substrate change nobody measured, on a machine the
  project does not own, to buy throughput the project has not shown it needs. G2 is signed; the
  remaining gates are G3 (verifier AUROC and cost) and G4 (human annotation), and neither is
  generation-bound.
- **Run generation on both boxes in parallel to halve wall clock.** Two substrates inside one run
  is exactly the provenance defect §3 exists to prevent, and the runs left are small.
- **Treat Box B as exclusive project hardware.** It has another person's coursework and a running
  Ollama on it. Claiming exclusivity in a manifest field would make `gpu_idle_confirmed` a lie
  rather than a weak claim.
- **Say nothing and use the box ad hoc.** The failure this ADR prevents is a G3 cost-per-claim
  number measured on one box and compared against a baseline measured on the other.

## Related

ADR-0004 (generator, judge, "exclusive" GPU) · ADR-0007 (retired base pipeline, i.e. the Ollama
model on Box B) · ADR-0008 (Box A's WSL2 substrate) · ADR-0017 (annotation sidecar; premise 2) ·
[`docs/harvest/runbooks/a4000-boxes-access.md`](../harvest/runbooks/a4000-boxes-access.md) (the
verified access sequence and the inventory transcript) ·
[`docs/harvest/runbooks/wsl-vllm-a4000.md`](../harvest/runbooks/wsl-vllm-a4000.md) (Box A only)
