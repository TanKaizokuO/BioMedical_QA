# Gate G3 Judge Run Requirement Specification

Operational requirements for Gate G3 cost clause evaluation via `scripts/g3_report.py --costs <path>`.

The judge benchmark run must emit a `costs.jsonl` containing `CostRecord` entries meeting all of the following requirements (referencing pricing and judge strategy defined in [ADR-0004](../../adr/0004-local-generator-frontier-judge.md)):

1. **Component**: Every judge record MUST set `component="judge"`.
2. **Schema & Fields**: Each `CostRecord` must populate `run_id`, `query_id` (matching the record query ID), `backend` (`"anthropic:claude-opus-5"`), `input_tokens` (int), `output_tokens` (int), `usd` (float computed at listed rate), and `wall_s` (float).
3. **Population Coverage**: Must cover all 1,257 (claim, cited span) evaluation units across all questions in the gold evaluation set (`records.jsonl`), evaluated by `JudgeVerifier` (`src/biomedqa/verify.py`).
4. **Hardware/Pricing Provenance**: USD costs MUST reflect published Anthropic rates ([ADR-0004](../../adr/0004-local-generator-frontier-judge.md)); local verifier costs reflect NVIDIA A4000 timing ([ADR-0008](../../adr/0008-a4000-is-a-windows-box-vllm-runs-in-wsl2.md), `research_roadmap.md:519`).
