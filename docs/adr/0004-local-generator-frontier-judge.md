# ADR-0004 — Local generator, frontier judge, deferred frozen-run backend

**Status:** Accepted · **Date:** 2026-07-30 · **Decided in:** grilling session Q4, Q7, Q13

## Context

Two roles need a model: the **generator** that produces claim-attributed answers, and the
**expensive judge** that C4/C5 are measured against. `learning_roadmap.md` assumed the generator
would be a Claude model. That assignment turned out to be backwards.

- **The judge should be the frontier model.** C5's point is beating an expensive judge on cost.
  "We match Claude Opus 5 at 1/Nth the cost" is materially stronger than matching a 14B local judge.
- **The generator is where reproducibility lives.** The reproducibility appendix is built on frozen
  schemas, run manifests, and seeds. A closed API model can drift and cannot be re-run
  bit-for-bit in 2027.

Hardware (established by lookup): RTX A4000, 16 GB VRAM, **exclusive** to the project. That caps a
local generator at ~7–8B fp16 or ~14B at 4-bit; an 8B AWQ (~6 GB) leaves room for
MiniCheck-770M and the cross-encoder concurrently.

A full cost analysis was produced before this decision. All options came in between ~$4 and ~$274 —
i.e. **all affordable**, so price was not the deciding axis. Dev iteration, not the frozen test
runs, was the largest line item in every API option, and it is the one workload that cannot be
batched.

## Decision

1. **Judge: Claude Opus 5 (`claude-opus-5`), ~$23.** Not negotiable — the headline cost claim needs
   a credibly expensive baseline.
2. **Generator: a local 8B AWQ model for all development iteration.**
3. **The frozen-test-run generator backend is deferred to W8 code-freeze**, when the 8B model's
   citation-format compliance rate is *measured* rather than guessed.

Cost: ~$28 typical; ceiling ~$125 (if switched to Sonnet 5) or ~$185 (Opus 5).

## Consequences

- **`generate.py` needs a backend adapter layer** (vLLM ↔ Anthropic) beneath its existing
  method-level API. The roadmap's "baselines behind one API" abstracts *methods* (joint / post-hoc /
  vanilla), not *backends*. ~½ day, scheduled in **W2**.
- **The seed plan is only implementable locally.** Opus 5 and Sonnet 5 reject `temperature`,
  `top_p`, and `top_k` outright (HTTP 400). The "≥3 seeds, paired by question" plan has no sampling
  knob to seed on the API. Headline systems therefore run local and seeded.
- **If the W8 switch happens, seeded variance is lost on switched systems.** Mitigation: keep
  headline systems local, use an API generator for the swap check only.
- **The swap check** — hold everything fixed, swap only the generator, re-run ours-vs-post-hoc on
  ~100 questions, report both gaps (~$2–10, under an hour). Answers two otherwise-unanswerable
  reviewer objections: *"your method is just prompt engineering on a weak model"* and *"your
  generator's format compliance is the confound."* Absolute scores will move; what matters is
  whether the **gap persists**.
- **W8 gains a real decision point.** Pick the backend; do not let it drift.
- **Overhead reporting follows from the split** (Q7): the verifier runs locally and the judge runs
  over a network, a boundary exclusivity cannot close. **Tokens and $ are primary**; wall-clock is
  secondary with hardware stated; **judge overhead is reported in tokens/$ only**, since per-claim
  judge wall-clock includes round-trip latency unrelated to model cost.
- **Exclusive GPU access** means W8–W9 needs no queue slack and the wall-clock secondary is a clean
  measurement: median of ≥5 runs, spread shown, GPU otherwise idle, batch policy stated.
- **Manifest fields required from W5:** `input_tokens`, `output_tokens`, `usd_at_listed_rate`,
  `wall_clock_s`, `gpu_idle_confirmed` — for both verifier and judge. Retrofitting token accounting
  in W7 means re-running Phase 3.

## Alternatives rejected

- **Claude generator throughout** (Opus 5 ~$274 / Sonnet 5 ~$165). Kills the seed plan outright.
  Also carries adverse coupling: the scenario where iteration is most needed — C2 looks null at G2
  and method must be separated from prompt — is exactly where a per-call meter bites. 3× dev volume
  takes the Opus option to ~$532.
- **All local, including the judge** (~$4). Saves $23 and forfeits the credibility of the headline
  cost claim. The $23 judge is the best-value line in the entire analysis.
- **Commit to local now, no deferral.** Nearly identical, but forecloses the strong-generator path
  at zero benefit. The deferred version dominates it.

## Pricing reference (verified 2026-06-24)

Opus 5 `claude-opus-5` **$5 / $25** per MTok · Sonnet 5 `claude-sonnet-5` **$3 / $15**
(intro $2/$10 **expires 2026-08-31** — late-September test runs pay full rate) · Haiku 4.5 $1 / $5.
Batch API −50%, results within 24 h — fine for frozen test runs, unusable for dev iteration.
Prompt-cache reads ~0.1×; Opus 5's minimum cacheable prefix is 512 tokens, so an ~800-token joint
generation system prompt qualifies.
