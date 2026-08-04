# ADR-0008 — The A4000 is a Windows box; the vLLM server runs inside WSL2

**Status:** Accepted · **Date:** 2026-08-04 · **Refines** ADR-0004 (it does not overturn it)

## Context

ADR-0004 decided the generator is "a local 8B AWQ model on the A4000" and `pyproject.toml` records
that vLLM is a **network boundary, not an import**. Both are silent on the host OS, because it was
assumed to be Linux.

On first access (2026-08-04) the box turned out to be **Windows** — `DESKTOP-5C6NFL8`, RTX A4000 in
**WDDM mode with a display attached**, driver 582.08 (CUDA 13.0 ceiling), no WSL installed. vLLM
publishes no Win32 wheels and does not support Windows natively.

This is not a small operational detail. ADR-0004's "exclusive GPU access" premise, the W8 backend
decision, and every wall-clock number in Table 4 all depend on where and how the generator runs.

## Decision

**The vLLM server runs inside WSL2 (Ubuntu-24.04) on the Windows host, using GPU passthrough.**
Verified working end to end on 2026-08-04 with vLLM 0.26.0; the exact sequence is
[`docs/harvest/runbooks/wsl-vllm-a4000.md`](../harvest/runbooks/wsl-vllm-a4000.md).

Three settings are part of the decision, not incidental configuration:

1. **`VLLM_USE_V2_MODEL_RUNNER=0`.** vLLM 0.26.0's default `GPUModelRunnerV2` requires pinned host
   memory, which WSL2 does not reliably provide; it crashes with `UVA is not available` on every
   model, a 0.6B included. We run the legacy runner.
2. **`--gpu-memory-utilization 0.85`.** The card drives a display, so its VRAM baseline drifts
   during desktop use.
3. **WSL2 localhost mirroring** carries the HTTP boundary — verified, no `netsh portproxy` needed.

## Consequences

- **ADR-0004's `pyproject.toml` isolation now has a second reason to exist.** vLLM is not merely in
  a separate venv; it is in a separate *operating system*. Nothing about the project env changes.
- **"Exclusive GPU access" is now qualified.** ADR-0004 promised a clean wall-clock secondary:
  *"median of ≥5 runs, spread shown, GPU otherwise idle."* On a WDDM box with a live desktop, "GPU
  otherwise idle" is a claim about human behaviour, not a machine state. The `gpu_idle_confirmed`
  manifest field (required from W5) must therefore mean **"desktop session confirmed quiescent at
  run time"**, and the paper must state that the generator ran under WSL2 on a display-attached
  card. It stays a *secondary* measure — ADR-0004 already made tokens and $ primary, which absorbs
  most of this.
- **The legacy model runner is a reproducibility fact**, not a workaround to be silently dropped.
  It belongs in the reproducibility appendix alongside the vLLM version; a future re-run on the
  default runner is not the same configuration.
- **Throughput expectations should be treated as unmeasured, not merely inherited.** WSL2 adds a
  virtualization layer and the legacy runner forgoes V2's optimizations. Whether that matters for
  an 8B AWQ at this scale is exactly what G0's bake-off measures — but no Linux-native benchmark
  should be quoted as if it applied here.
- **R1's MedCPT encode estimate inherits the same caveat.** The ~4 h trigger was estimated for a
  bare-metal encode. Measure it on this box, under WSL2, before treating it as decided.

## What this does *not* change

The generator decision itself (local 8B AWQ), the judge (Claude Opus 5), the W8 deferral, and the
seed plan are all untouched. ADR-0004 stands; this ADR supplies the substrate it left unstated.

Notably, **G0's stop rule did not fire.** Issue #1 says an unusable box "invalidates ADR-0004's
compute decision; escalate rather than re-dating a third time." The box was awkward, not unusable —
one day of setup, no gate slip. G1 remains 2026-08-23.

## Related

ADR-0004 (generator and judge) · `pyproject.toml` (why vLLM is not a dependency) ·
`docs/harvest/runbooks/wsl-vllm-a4000.md` (the verified sequence and failure signatures)
