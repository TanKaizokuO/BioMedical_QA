# HANDOFF — 2026-08-04 (evening)

Snapshot for resuming work in a fresh session. Regenerate with `/handoff`; do not append to it.

---

## 1. Where the project is

**Gate G0 (due today, 2026-08-04) — infrastructure unblocked, both measurements still unmeasured.**

The blocker recorded this morning is resolved: **vLLM 0.26.0 runs on the A4000 under WSL2**, GPU
passthrough verified, server reachable from the Windows host. That took a full day and five serial
failures; the sequence is now written down (see §3).

G0's two deliverables are both still **unmeasured**:

- **Generator bake-off** — not started. The candidate models were never chosen (see §2).
- **MedCPT encode throughput** — not started. This is the number that decides R1.

Week 0's code is complete and pushed: `main` is at `dbd9ed4` and matches `origin/main`.
G1 remains **2026-08-23** and has not moved. G0's escalation rule did **not** fire — the box was
awkward, not unusable.

---

## 2. What is blocking right now

**Nothing technical. G0 is blocked on a decision that has never been made.**

`research_roadmap.md` line 162 still carries the open marker: *"Still open — decide by Aug 4: which
8B AWQ model."* ADR-0004 says only "a local 8B AWQ model" and names no IDs. Issue #1 says
`--model <ID>` per candidate. **There is nothing to look up — naming the candidates is the G0
decision itself.** A previous session lost time searching for a prior choice that does not exist.

Proposed candidates, to be confirmed by the user and then recorded in the roadmap:

- **A:** `Qwen/Qwen2.5-7B-Instruct-AWQ` — already pulled on the box. Note it is **7B, not 8B**;
  ADR-0004's real constraint is the ~6 GB VRAM budget, which this satisfies. Record as a deliberate
  choice, not drift.
- **B:** *unchosen.* A second AWQ-INT4 instruct model in the 7–8B range. Verify the repo ID
  resolves on HuggingFace before pulling — hub IDs get renamed.

**Decision rule (issue #1, do not improvise around it):** rank on **citation-format compliance**,
not benchmark scores. A model that is fast, fluent and unreliable with `[n]` markers is
disqualified. Latency only breaks ties.

**Next actions, in order:**

1. Name candidate B.
2. Serve A, run the bake-off from the laptop through the tunnel, swap to B, re-run, then `--compare`.
   `scripts/g0_generator_bakeoff.py` takes `--model` and `--base-url` (default
   `http://localhost:8000`); `--compare` ranks existing runs and exits.
3. Copy `scripts/g0_medcpt_throughput.py` into WSL and run it **on the box** — it needs the GPU
   directly and refuses without CUDA.
4. Write the measured latency into `research_roadmap.md` §2 and the throughput into §3.

**Still not started, human-only, long-lead, unaffected by any of the above:**

- **Issue #7** — annotator ask. Blocks G4 (2026-09-27), which needs ≥250 claims at α ≥ 0.6.
- **Issue #8** — MedNLI / PhysioNet application. Insurance for G3. Credentialing is slow and cannot
  be compressed later.

---

## 3. What changed since the last handoff

**The A4000 turned out to be a Windows box** (`DESKTOP-5C6NFL8`, WDDM, display attached, driver
582.08 / CUDA 13.0 ceiling) with no WSL installed. vLLM has no Win32 support, so a GPU-passthrough
path had to be built from nothing. Five failures, each visible only after the previous was cleared:

1. inbox WSL stub with a stale distro catalog → `wsl --update --web-download`
2. `UVA is not available` on every model → `VLLM_USE_V2_MODEL_RUNNER=0`
3. missing C compiler for the Triton JIT → `build-essential`
4. FlashInfer sampling kernel wants `nvcc` → `cuda-toolkit-13-0`
5. `wsl -l -v` inside the distro → not a real problem; `wsl` is Windows-side

Two new documents, both **uncommitted** as of this handoff:

- **`docs/harvest/runbooks/wsl-vllm-a4000.md`** — the verified sequence plus a failure-signature
  table for recognition. (An earlier session believed this file was already written; it was not in
  this repo. Verified absent, then written.)
- **`docs/adr/0008-a4000-is-a-windows-box-vllm-runs-in-wsl2.md`** — refines, does not overturn,
  ADR-0004.

Also uncommitted: `.claude/commands/handoff.md` (the `/handoff` command) and this file.

**The consequence in ADR-0008 that is easiest to miss:** ADR-0004 promised a clean wall-clock
secondary — *"GPU otherwise idle."* On a WDDM box with a live desktop that is a claim about human
behaviour, not a machine state. `gpu_idle_confirmed` (required in manifests from W5) now means
**"desktop session confirmed quiescent at run time"**, and the paper must say the generator ran
under WSL2 on a display-attached card.

---

## 4. What to read

In order, stopping when context is sufficient:

1. `CONTEXT.md` — project language and the frozen annotation protocol
2. `research_roadmap.md` §0 (audit), §2 (decisions), §5 (week plan)
3. `docs/adr/0003`, `0004`, `0005`, `0007`, `0008` — corpus, generator/judge, attribution unit,
   retirement of the base pipeline, and the WSL2 substrate
4. `docs/harvest/runbooks/wsl-vllm-a4000.md` — before touching the box
5. `src/biomedqa/schema.py` — the frozen data contract; read before writing anything that emits data
6. `paper/skeleton.md` — the five tables and the C1–C5 claim ledger every result must land in

Not needed: `docs/project2_biomedical_attribution_rag_implementation_plan.md` (§4, §8, §9 superseded
and banner-marked), and `notebooks/` (all toy/simulated; `07_4` simulates 3 labels where
`CONTEXT.md` freezes 4 — a correctness bug, not a scale assumption).

---

## 5. Standing constraints

- **Least-processed value.** Store `phi_score: 0.83`, never `supported: true`. Store `gold_rank` or
  the ranked list, never a precomputed hit@5. Store the 4-way `support_label`, never its binary
  collapse — the collapse is derived at scoring time.
- **Wilson, not Wald**, on every gate proportion. G1 passes iff point ≥ 0.90 **and** Wilson lower
  > 0.85. Both, not either.
- **vLLM is a network boundary, not an import** — and now a separate OS as well. It is deliberately
  absent from `pyproject.toml`, including optional groups. Do not "fix" this by adding it.
- **`RAG_Debate_Agent` is retired.** Do not re-run it, do not import from it. Cite `docs/harvest/`.
- **Index identity is a content hash**, never a document count (the ADR-0007 lesson).
- **≤3 citations per claim**, enforced as a fairness control across all three systems.
- **`validate()` reports violations and never repairs them.**

---

## 6. Traps

- **`scripts/g0_smoke.sh` is now wrong.** It SSHes into a POSIX login shell; the box answers with
  `cmd.exe`, and the real target is inside `wsl --`. It has never run successfully. Either fix it or
  work from the runbook — do not trust a `VERDICT: PASS` from it.
- **`docs/` is gitignored** via `docs/*` with `!docs/adr/` and `!docs/harvest/` negations. New docs
  outside those two trees are silently untracked. `docs/harvest/runbooks/` is fine (verified with
  `git check-ignore`). Always check `git status` before believing a doc was saved.
- **VRAM drifts** on this box — WDDM, display attached, ~500–700 MiB baseline that grows with
  desktop use. Always launch with `--gpu-memory-utilization 0.85`.
- **Do not install an NVIDIA driver inside WSL.** The Windows driver is passed through; a guest
  driver breaks it.
- **`uv pip install vllm` pulls ~8 GB.** Start it and walk away.
- **Match the CUDA toolkit to the driver's reported ceiling** (13.0), not the newest patch line.
- **Do not inspect `~/.ssh/` without being asked.** This was declined once.
- Dates have drifted in conversation before. Today is **2026-08-04**; the November deadline is hard.
