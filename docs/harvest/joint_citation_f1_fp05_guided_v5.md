# Joint vs Post-Hoc Citation F1 at `fp = 0.5`, both arms guided (`v5` run)

**Delta +0.0933, 95% CI [+0.0259, +0.1613]. The interval excludes zero: contrast C2 is established on this run.**
This run achieves a joint arm clean parse rate of **95/100** (95%), satisfying the $\ge 95\%$ parse-rate bar and resolving the malformed-JSON decoder defect. However, sign-off for Gate G2 remains blocked by the W9 stratified robustness check (`docs/harvest/w9_stratified_parity_guided_v5.md`).

## 1. What was run

```bash
uv run python scripts/citation_contrast.py docs/harvest/generate_fp05_n100_guided_v5 --threshold 0.5
```

| field | value |
|---|---|
| run id | `generate_fp05_n100_guided_v5` |
| manifest `git_sha` | `045a96cd2b95ab65e34f1e3947f86229bc4946cd` |
| `config_hash` / `CONFIG_VERSION` | `b1d8a1c7d4f8` / `1.5.0` |
| served window | `--max-model-len 14336` |
| generation | `frequency_penalty 0.5`, `temperature 0.0`, `max_tokens 3584`, `guided_decoding true` for **both** joint and post-hoc |
| joint decoding fixes | bounded escape-valve retry at $T=0.3$ and $T=0.7$; claim-length target instruction ($15$–$20$ words/claim) |
| verifier | MiniCheck φ, `lytang/MiniCheck-Flan-T5-Large`, threshold 0.5, CPU |
| paired queries / dropped | 96 / 4 (dropped: zero claims in joint arm) |
| bootstrap | 10000 resamples, cluster unit = query, seed 0 (ADR-0011 §2) |
| artifact | `docs/harvest/generate_fp05_n100_guided_v5.citation_f1.minicheck.json` |

## 2. Result

| arm | precision | recall | citation F1 | claims | relevant / total citations |
|---|---|---|---|---|---|
| joint | 0.9653 | 0.4503 | **0.6142** | 433 | 473 / 490 |
| post_hoc | 0.9533 | 0.3583 | **0.5209** | 600 | 714 / 749 |

Paired delta (joint − post_hoc): **+0.0933**, 95% CI **[+0.0259, +0.1613]**, width 0.1354, excludes zero **True**.

## 3. Comparison across guided-arm iterations

| run | joint clean parses | joint F1 | post_hoc F1 | delta | 95% CI | excludes zero |
|---|---|---|---|---|---|---|
| `generate_fp05_n100_guided_both` | 89/100 (89%) | 0.6137 | 0.5055 | +0.1083 | [+0.0432, +0.1722] | Yes |
| `generate_fp05_n100_guided_v5` | 95/100 (95%) | 0.6142 | 0.5209 | +0.0933 | [+0.0259, +0.1613] | **Yes** |

The bounded escape-valve retry raised the clean parse rate from 89% to 95%, clearing the Gate G2 valid-parse requirement while maintaining a clear and statistically significant citation F1 advantage for the joint attribution arm.

## 4. Gate G2 Status

- **Clean parse rate**: 95/100 (≥95% bar **MET**)
- **Citation F1 contrast (C2)**: Delta +0.0933, 95% CI [+0.0259, +0.1613] (excludes zero **MET**)
- **W9 stratified robustness check**: Pooled gap +21.4% (against ±15% tolerance **NOT MET**)

Sign-off remains refused until all three criteria pass on the same run.
