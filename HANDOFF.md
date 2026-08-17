# HANDOFF — 2026-08-17 (end of seventeenth session)

Snapshot for resuming in a fresh session. Regenerate wholesale; **do not append** — a stale line here is worse than a missing one, because the next session will trust it.

`main` · working tree uncommitted at the time of writing.

Tests: `uv run python -m pytest tests/ -q` → **447 passed** (was 431). `pyproject.toml`'s `pythonpath` is `["src", "scripts"]`.

---

## 1. Where the project is

**W3 (Aug 17–23), Phase 1. The runaway-claim pathology was closed across score, split, and decode layers without prompt edits prior to the Sep 3 Decomposer Freeze and Sep 6 Gate G2 run.**

| Gate | Date | State |
|---|---|---|
| **G0** — 8B AWQ generator chosen, A4000 measured | Aug 4 | **PASSED 2026-08-04.** Issue #1 closed. |
| **G1** — hit@10 ≥ 0.90, Wilson lower > 0.85 (ADR-0015) | Aug 23 | **PASSED 2026-08-10.** hit@5 = 0.86, **hit@10 = 0.9400** (Wilson lower 0.8752). |
| **G2** — citation-F1 contrast + per-claim parse | **Sep 6** | **PATHOLOGY CLOSED IN CODE (Aug 17, ADR-0021). CONTRAST OPERATING POINT PENDING GATE RUN.** `quote_located_rate` 1.0000. Runaway chain detector, punctuation-bound splitter, call rejection guard, and fp=0.5 decoding landed. Re-read on `parity_iter1b` at MiniCheck φ gave **joint 0.428 vs post-hoc 0.418, delta +0.011 [−0.117, +0.137]**. |
| G3 · G4 · G5 | Sep 20 · Sep 27 · Oct 11 | Unstarted, with due weeks. |

---

## 2. What happened today

### 2.1 Four code changes landed and verified (Full suite: 447 passed)

1. **Nested chain detector:** `prompts.claim_stem`, `prompts.RUNAWAY_CHAIN_MIN = 3`, and `prompts.runaway_chains` wired into `parse_response` and `decompose.parse_decomposition`. Detects nested prefix-extension chains (claim $N+1$ extends claim $N$), charging chains $\ge 3$ to `errors` and length 2 to `recovered`.
2. **Punctuation-bound run-on splitter:** `decompose._split_run_on` + `sentence_units(answer, *, max_words=MAX_CLAIM_WORDS)`. Cuts units $> 50$ words only at explicit text boundaries (`;` or `,` + whitespace). Unpunctuated run-ons remain whole and flagged.
3. **`Generation.recovered` retained:** `generate_one` retains and populates `Generation.recovered`, ending the silent dropping forbidden by ADR-0019's Consequences.
4. **Call rejection guard:** `generate_one` records rejected inference calls (`httpx.HTTPStatusError` / `TransportError`) as `f"call {n} rejected: {exc}"` in `Generation.errors` with an uninstrumented `CostRecord` (`input_tokens=None`, `output_tokens=None`, measured `wall_s`) instead of propagating unhandled exceptions.

### 2.2 Live A4000 decoding sweep (`frequency_penalty`)

A live sweep on the RTX A4000 (`scripts/generate_smoke.py`, `Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, `--max-tokens 3584`, `--frequency-penalty {0.0, 0.3, 0.5}`, seed 0, vLLM `--max-model-len 8192`) over an enriched 12-question slice (`docs/harvest/generate_fp_sweep.md`) established:

- `frequency_penalty = 0.5` eliminates chain claims and $>50$w claims, collapses longest `joint` claim from 731w to 27w, and reduces `quote_not_found` (`joint`: 8 $\rightarrow$ 0, `post_hoc`: 92 $\rightarrow$ 15).
- Window-overflow HTTP 400 call rejections vanish at `fp >= 0.3` because concise Stage 1 completions keep Stage 2 prompts within vLLM's 8192 context.
- `GenerationConfig.frequency_penalty` default changed from `0.0` to `0.5`, updating `CONFIG_VERSION` to `1.5.0`.

### 2.3 Documentation and governance

- **ADR-0021** created (`docs/adr/0021-close-runaway-claim-pathology-and-set-decoding-frequency-penalty.md`), recording the four decisions, empirical bases, full sweep table, full disclosure paragraph, and alternatives rejected.
- **ADR-0009 Third amendment** appended to `docs/adr/0009-granularity-parity-is-a-measured-diagnostic-not-a-condition.md`, discharging the deferred 731-word claim defect across score/split/decode layers while keeping prompts strictly frozen.
- **ROADMAP.md** updated to reflect completed W5 milestones and G2 status.

---

## 3. Measured evidence & findings summary

### 3.1 Nested-chain scan on `parity_iter1b`
- Nested-chain length distribution across 300 records: `joint` `{2: 22, 3: 1, 10: 1, 11: 1}`, `post_hoc` `{2: 2, 5: 1}`, `vanilla` `{2: 1}`.
- Marginal coverage: 9 `joint` claims and 3 `post_hoc` claims in chains $\ge 3$ were $\le 50$ words and invisible to length-based guards.
- Exact error strings produced on `parity_iter1b`:
  - `joint` `21074975`: `c13: extends c3's claim text through 11 nested claims (non-terminating generation)`
  - `joint` `10490564`: `c17: extends c8's claim text through 10 nested claims (non-terminating generation)`
  - `joint` `17578985`: `c17: extends c15's claim text through 3 nested claims (non-terminating generation)`
  - `post_hoc` `9920954`: `c9: extends c5's claim text through 5 nested claims (non-terminating generation)`

### 3.2 Run-on splitter performance
- Units $>50$w before: 27 of 3592 (`joint`: 20, `post_hoc`: 3, `vanilla`: 4).
- Units $>50$w after: `joint` 719 $\rightarrow$ 771 units (0 exceed 50w; 731w claim split into 18 pieces), `post_hoc` 1246 $\rightarrow$ 1249 units (0 exceed 50w), `vanilla` retains 1 unsplit 52w unit (punctuation-free).
- Invariants: 0 lost/duplicated non-whitespace characters, 0 empty/overlapping units across all 300 records.

### 3.3 Generator frequency penalty sweep (n=12 enriched slice)

| fp | system | clean/12 | call fails | chain claims | chain pairs | >50w claims | longest claim | quote-not-found | mean claims/q | recovered notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | joint | 7 | 0 | 2 | 1 | 15 | **731w** | 8 | 9.75 | 44 |
| 0.0 | post_hoc | 2 | **1** | 1 | 0 | 2 | 80w | 92 | 9.92 | 20 |
| 0.0 | vanilla | 11 | 0 | 0 | 0 | 1 | 52w | 0 | 19.58 | 0 |
| 0.3 | joint | 9 | 0 | 0 | 0 | 0 | 25w | 9 | 5.33 | 23 |
| 0.3 | post_hoc | 1 | 0 | 0 | 0 | 0 | 33w | 30 | 6.42 | 11 |
| 0.3 | vanilla | 12 | 0 | 0 | 0 | 0 | 43w | 0 | 5.42 | 0 |
| 0.5 | joint | 9 | 0 | 0 | 0 | 0 | 27w | **0** | 5.42 | 30 |
| 0.5 | post_hoc | 4 | 0 | 0 | 0 | 0 | 33w | 15 | 3.75 | 12 |
| 0.5 | vanilla | 12 | 0 | 0 | 0 | 0 | 46w | 0 | 5.08 | 0 |

Words per claim (median / mean / max):
- `fp = 0.0`: `joint` 17 / 35.87 / 731, `post_hoc` 19 / 19.88 / 80, `vanilla` 18 / 19.23 / 52
- `fp = 0.3`: `joint` 15 / 15.41 / 25, `post_hoc` 17 / 17.25 / 33, `vanilla` 20 / 20.62 / 43
- `fp = 0.5`: `joint` 15 / 15.68 / 27, `post_hoc` 17 / 17.42 / 33, `vanilla` 20 / 21.30 / 46

Median claims per query:
- `fp = 0.0`: `joint` 4.5 / `post_hoc` 9.0 / `vanilla` 7.0
- `fp = 0.3`: `joint` 4.0 / `post_hoc` 6.5 / `vanilla` 5.0
- `fp = 0.5`: `joint` 5.5 / `post_hoc` 4.0 / `vanilla` 5.0

---

## 4. Standing state & operational rules

- **Working tree:** uncommitted at the time of writing. Full test suite passing (447 passed).
- **Remote helper command:**
  `uv run --with paramiko python scripts/_remote.py 'wsl.exe -d Ubuntu-24.04 -- bash -lc "bash /home/user/status.sh"'`
- **vLLM service:** `vllm-8b.service` active on A4000. Long jobs MUST run via `systemd-run --user --unit=<name>`.

---

## 5. Pending next steps

1. **Sep 3 Decomposer & Granularity Freeze** (ADR-0009 §8) — maintain prompt digests and parsers stable.
2. **Sep 6 Gate G2 Benchmark Run** — execute on dev set at `fp = 0.5` under `CONFIG_VERSION` 1.5.0.
3. **AlignScore (~355M, Table 3 row 2)** — port official `.ckpt` / establish isolated environment to resolve dependencies (`torch<2`, `pytorch_lightning<2`).
4. **Parity diagnostic re-read:** execute claims/query diagnostic re-read on $n > 12$ at `fp = 0.5`.
5. **Post-hoc clean-parse residual:** address quote-drift defect via guided-JSON citation path.
