# Session Codebase Review

## Overall Status

**NOT READY**

## Review Scope

- Repository state reviewed: Clean `main` branch with 11 modified working tree files, 14 untracked files/artifacts (`docs/harvest/generate_fp05_n100_*`).
- Files reviewed: 11 modified files (`CLAUDE.md`, `HANDOFF.md`, `scripts/generate_smoke.py`, `src/biomedqa/backends.py`, `src/biomedqa/config.py`, `src/biomedqa/generate.py`, `src/biomedqa/prompts.py`, `src/biomedqa/scoring/granularity.py`, `tests/test_backends.py`, `tests/test_generate.py`, `tests/test_scoring_granularity.py`), plus `src/biomedqa/decompose.py`, `scripts/parity_report.py`, `scripts/analyze_guided_experiment.py`, `ADR-0009`, and harvest artifacts.
- Tests/checks executed: Full pytest test suite (455 passed in 16.97s), static analysis of prompt window guard, digest validation, and guided JSON metric extractions.
- Gemini 3.6 Flash review: COMPLETE
- Gemini 3.1 Pro review: COMPLETE

## 1. Code Base Defect Fixes

### `scripts/parity_report.py`

- `output_tokens=null` handling: PASS
- Numeric behavior preserved: PASS
- Regression risk: PASS
- Tests: PASS

Evidence:
- `stage_output_tokens()` in `src/biomedqa/scoring/granularity.py` treats `call.output_tokens is None` as 0 (`0 if call.output_tokens is None else int(call.output_tokens)`), preventing `TypeError` on rejected calls.
- Numeric `output_tokens` behavior remains unchanged (`int(call.output_tokens)`).
- `tests/test_scoring_granularity.py::test_stage_output_tokens_handles_null_output_tokens` verifies handling of `CostRecord(output_tokens=None)` and passes.

### `src/biomedqa/backends.py`

- Prompt-window guard: PASS
- Correct inequality: PASS
- Boundary accepted: PASS
- Overflow rejected pre-dispatch: PASS
- Tests: PASS

Evidence:
- `_check_prompt_window_guard()` checks `prompt_tokens + config.max_tokens > model_max_len` before request dispatch in `_vllm_complete` and `_anthropic_complete`.
- Uses `_count_tokens(prompt)` token-counting mechanism and `_get_model_max_len(config.model)` context window mapping.
- Correct strict inequality `> model_max_len` ensures exact boundary `prompt_tokens + max_tokens == model_max_len` is accepted (`False` branch), while overflow is rejected pre-dispatch with `ValueError`.
- `tests/test_backends.py::test_prompt_window_guard_raises_when_oversized` and `test_prompt_window_guard_passes_through_unchanged` pass.

## 2. Guided-JSON Post-Hoc Citation Path

- Constrained decoding: PASS
- Clean/valid parse rate: 70/100 = 70.0%
- G2 >=95%: FAIL
- `quote_not_found`: 0
- Citation F1: INSUFFICIENT EVIDENCE
- Claim recovery: 0/433 = 0.0%
- Rejected calls: 0
- Stage-2 prompt length: Total 286,533 tokens (Mean 2865.3 tokens/query)
- Stage-2 context length: Depth 10 passages (~2400-2800 tokens per prompt)

### Attribution Invariance

- Claim length: PASS
- Citation behavior: PASS
- Citation recall: INSUFFICIENT EVIDENCE
- Citation F1 regression: INSUFFICIENT EVIDENCE

Evidence:
- Constrained decoding is enforced during stage-2 post-hoc generation via `response_format` (JSON schema in vLLM / `tool_choice` in Anthropic).
- `quote_not_found` count dropped from 179 (baseline) to 0 (guided), confirming constrained decoding eliminates quote transcription drift.
- Clean parse rate is 70/100 (70.0%), failing Gate G2 requirement (`valid claim parse rate >= 95%`).
- Claim length distribution remains invariant: Baseline median 17.0 words (mean 17.55, p25=14, p75=21, p90=25); Guided median 17.0 words (mean 17.95, p25=14, p75=21, p90=25).
- Quantitative Citation F1 and recall metrics under MiniCheck could not be scored due to missing GPU/verifier outputs in harvest artifacts, marked as `INSUFFICIENT EVIDENCE`.

## 3. Decomposer and Granularity Freeze

- Decomposer model: PASS
- Prompt templates: PASS
- Parser logic: PASS
- `decompose_template_digest()`: PASS
- Atomic-claim definition: PASS
- Freeze integrity: PASS

Digest:
- Canonical: `4129a884c7a1b4854739ffe2e3900a1db39626db7fd1bf076d6b549027a7d737`
- Current: `4129a884c7a1b4854739ffe2e3900a1db39626db7fd1bf076d6b549027a7d737`
- Match: PASS

Evidence:
- Decomposer model remains `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`.
- `DECOMPOSE_TEMPLATE`, `FORMAT_BLOCK`, `BARE_ATOMIC_RULE`, `DECONTEXTUALIZATION_RULE`, and `ATOMICITY_RULE` in `src/biomedqa/decompose.py` are completely unchanged (`git diff` on `decompose.py` is clean).
- `parse_decomposition()` and claim-boundary handling logic are untouched.
- `decompose_template_digest()` SHA-256 digest matches canonical baseline `4129a884c7a1b4854739ffe2e3900a1db39626db7fd1bf076d6b549027a7d737`.
- The definition of a decontextualized atomic claim is strictly preserved per ADR-0009 §8.

## 4. Scope and Regression Review

- Unrelated changes: NONE
- Suspicious changes: NONE
- Benchmark/evaluation integrity: PASS
- Public API stability: PASS
- Overall regression risk: LOW

## 5. Test Results

| Check | Result |
|---|---|
| Parity report | PASS |
| Prompt window | PASS |
| Guided JSON | FAIL |
| Citation metrics | INSUFFICIENT EVIDENCE |
| Attribution invariance | INSUFFICIENT EVIDENCE |
| Decomposer digest | PASS |
| Decomposer/parser integrity | PASS |
| Freeze integrity | PASS |

## Blocking Issues

1. Post-hoc guided-JSON clean parse rate (70/100 = 70.0%) fails Gate G2 requirement (`valid claim parse rate >= 95%`).
2. Missing quantitative MiniCheck Citation F1 evaluation artifacts for `generate_fp05_n100_guided` to verify citation recall and F1 regression.

## Final Verdict

**NOT READY**

- All code base defect fixes (`parity_report.py`/`granularity.py` null handling and backend prompt-window guard) PASS all checks and unit tests.
- Guided JSON eliminates `quote_not_found` (179 -> 0) and maintains claim length invariance, but fails Gate G2 with a 70.0% clean parse rate (threshold >= 95%).
- MiniCheck Citation F1 evaluation artifacts are missing for the guided post-hoc run, leaving Citation F1 regression as INSUFFICIENT EVIDENCE.
- Decomposer prompts, model, parser, atomic claim definitions, and `decompose_template_digest()` (SHA-256 `4129a884...`) remain 100% frozen and PASS.
- Full test suite passes completely (455/455 tests).
