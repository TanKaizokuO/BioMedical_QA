# W9 Stratified Robustness Check — ADR-0009 §5 (`v5` run)

**Verdict: FAIL.** Executed on 2026-08-20 against `generate_fp05_n100_guided_v5`.
The claim-length target instruction added to `JOINT_JSON_TEMPLATE` narrowed the pooled gap from $+30.8\%$ to $+21.4\%$ and brought two of three stratification schemes to PASS, but the pooled gate and the low-claim-volume stratum still exceed the $\pm 15\%$ tolerance threshold.

## 1. What was run

```bash
uv run python scripts/w9_stratified_parity_report.py docs/harvest/generate_fp05_n100_guided_v5 --max-tokens 3584
```

| field | value |
|---|---|
| run id | `generate_fp05_n100_guided_v5` |
| manifest `git_sha` | `045a96cd2b95ab65e34f1e3947f86229bc4946cd` |
| `config_hash` / `CONFIG_VERSION` | `b1d8a1c7d4f8` / `1.5.0` |
| served window | `--max-model-len 14336` |
| generation | `frequency_penalty 0.5`, `temperature 0.0`, `max_tokens 3584`, `guided_decoding true` for **both** arms |
| prompt modifications | $15$–$20$ word claim-length target added to `JOINT_JSON_TEMPLATE` |
| records / cost rows | 300 / 479 |
| tolerance | `PARITY_TOLERANCE = ±15%`, `min_queries = 5` |
| text artifact | `docs/harvest/generate_fp05_n100_guided_v5.w9_stratified_parity.txt` |

## 2. Pooled gate

| arm | median words/claim |
|---|---|
| joint | 14.0 |
| post_hoc | 17.0 |

gap **+21.4%** against ±15% → **FAIL**.

## 3. Scheme 1 — compound structure

| stratum | queries | joint claims | post_hoc claims | joint median | post_hoc median | gap | status |
|---|---|---|---|---|---|---|---|
| simple | 100 | 336 | 378 | 14.0 | 16.0 | +14.3% | **PASS** |
| compound | 96 | 97 | 249 | 17.0 | 19.0 | +11.8% | **PASS** |

Scheme verdict **PASS**, 2/2 strata powered. Up from FAIL in `generate_fp05_n100_guided_both` (where the simple stratum breached at +23.1%).

## 4. Scheme 2 — claim length bands

| stratum | queries | joint claims | post_hoc claims | joint median | post_hoc median | gap | status |
|---|---|---|---|---|---|---|---|
| 1-10 | 56 | 72 | 34 | 9.0 | 9.0 | +0.0% | **PASS** |
| 11-15 | 96 | 193 | 191 | 13.0 | 13.0 | +0.0% | **PASS** |
| 16-20 | 90 | 111 | 214 | 17.0 | 18.0 | +5.9% | **PASS** |
| 21-30 | 80 | 46 | 175 | 23.0 | 23.0 | +0.0% | **PASS** |
| 31+ | 18 | 7 | 13 | 37.0 | 33.0 | -10.8% | **PASS** |

Scheme verdict **PASS**, 5/5 strata powered.

## 5. Scheme 3 — query claim volume

| stratum | queries | joint claims | post_hoc claims | joint median | post_hoc median | gap | status |
|---|---|---|---|---|---|---|---|
| 1-5 claims | 78 | 309 | 475 | 14.0 | 17.0 | +21.4% | **FAIL** |
| 6-10 claims | 18 | 124 | 125 | 14.0 | 16.0 | +14.3% | **PASS** |
| 11+ claims | 0 | 0 | 0 | - | - | - | UNDERPOWERED |

Scheme verdict **FAIL**, 2/3 strata powered. The 1–5 claims stratum ($n=78$ queries) remains the single failing stratum in the benchmark.

## 6. Synthesis & Next Steps

1. **Progress**: The prompt-level target instruction reduced the pooled gap from $+30.8\%$ to $+21.4\%$ and restored `compound_structure` to PASS.
2. **Remaining Defect**: The 1–5 claims stratum still has a median claim length gap ($14.0$ vs $17.0$ words), driving the overall FAIL verdict.
3. **Required Action**: To pass Gate G2, a stricter claim-length floor or per-claim minimum length instruction is needed specifically targeting short claim outputs in the joint schema.
