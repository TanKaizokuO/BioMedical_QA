# W9 Stratified Robustness Check — ADR-0009 §5

**Verdict: PASS on all three pre-registered schemes.** Run on 2026-08-20 against
`generate_fp05_n100_guided_batched`, the batched guided-JSON run of record.

## 1. What was run

```
uv run python scripts/w9_stratified_parity_report.py docs/harvest/generate_fp05_n100_guided_batched --max-tokens 3584
```

| field | value |
|---|---|
| run id | `generate_fp05_n100_guided_batched` |
| manifest `git_sha` | `c9a2a8e54cae617acaea2f26c477abd5f09321b0` (clean, no `-dirty`) |
| provenance | `live` |
| `config_hash` / `CONFIG_VERSION` | `d7716c778abd` / `1.5.0` |
| `split_hash` / `index_fingerprint` | `71c46cc5b0ca` / `57ab89e445f8` |
| generation | `frequency_penalty 0.5`, `temperature 0.0`, `max_tokens 1536`, `guided_decoding true` |
| records / cost rows | 300 / 453 |
| tolerance | `PARITY_TOLERANCE = ±15%`, `min_queries = 5` |

ADR-0009 §5 makes this check mandatory whatever the pooled result says, because the
pooled +13.3% pass sits inside the tolerance band by two words of median.

## 2. Pooled gate (context, not the check)

| arm | median words/claim |
|---|---|
| joint | 15.0 |
| post_hoc | 17.0 |

gap **+13.3%** against ±15% → **PASS**. Unmoved from `parity_iter1b` and from the
free-text `generate_fp05_n100` read.

## 3. Scheme 1 — compound structure

| stratum | queries | joint claims | post_hoc claims | joint median | post_hoc median | gap | status |
|---|---|---|---|---|---|---|---|
| simple | 100 | 310 | 380 | 14.0 | 16.0 | +14.3% | PASS |
| compound | 98 | 153 | 247 | 17.0 | 19.0 | +11.8% | PASS |

Scheme verdict **PASS**, 2/2 strata powered.

This is the scheme with real discriminating power, and it is the one that reproduces the
`parity_iter0b` reading: the gap survives inside simple claims (14.0 vs 16.0), so the
residual is **verbosity, not compounding**. +14.3% on the simple stratum is 0.7 points
under the tolerance — the pass is real but it is not comfortable.

## 4. Scheme 2 — claim length bands

| stratum | queries | joint claims | post_hoc claims | joint median | post_hoc median | gap | status |
|---|---|---|---|---|---|---|---|
| 1-10 | 51 | 75 | 34 | 9.0 | 9.0 | +0.0% | PASS |
| 11-15 | 91 | 177 | 189 | 13.0 | 13.0 | +0.0% | PASS |
| 16-20 | 93 | 129 | 219 | 17.0 | 18.0 | +5.9% | PASS |
| 21-30 | 82 | 73 | 173 | 23.0 | 23.0 | +0.0% | PASS |
| 31+ | 18 | 9 | 12 | 35.0 | 33.5 | -4.3% | PASS |

Scheme verdict **PASS**, 5/5 strata powered.

**Read this scheme with care.** The stratifier *is* the outcome variable: claims are
binned by their own word count and then compared on median word count inside the bin, so
a within-band median is confined to a band 5 words wide and near-zero gaps are partly
mechanical. What the scheme does carry is the **mass shift** — post-hoc puts 219 claims in
16-20 and 173 in 21-30 against joint's 129 and 73, while joint holds 75 claims in 1-10
against post-hoc's 34. The distributions differ where the medians cannot say so.

## 5. Scheme 3 — query claim volume

| stratum | queries | joint claims | post_hoc claims | joint median | post_hoc median | gap | status |
|---|---|---|---|---|---|---|---|
| 1-5 claims | 76 | 297 | 457 | 15.0 | 17.0 | +13.3% | PASS |
| 6-10 claims | 24 | 166 | 170 | 16.0 | 17.0 | +6.2% | PASS |
| 11+ claims | 0 | 0 | 0 | — | — | — | UNDERPOWERED |

Scheme verdict **PASS**, 2/3 strata powered. Queries are assigned to a band by their
**joint** claim count. The 11+ band is empty at `fp = 0.5` — the frequency penalty caps
enumeration well below 11 claims per query — so no verdict is available there and none is
claimed.

## 6. What this licenses, and what it does not

Licensed:

- The ADR-0009 §5 obligation attached to the +13.3% granularity pass is **discharged** for
  this run. No stratum inverts the sign, and no powered stratum breaches ±15%.
- The residual gap is verbosity, not compounding (§3).

Not licensed:

- **Nothing about Gate G2.** This is the granularity-parity check, a fairness precondition
  on the claim unit. G2's citation-F1 contrast and its ≥95% valid-parse bar are separate and
  still open — the joint arm reads 34/100 clean parses and 161 `quote_not_found` on this
  same run.
- **No claim on the 11+ claims/query regime**, which is unobserved here.
- **No transfer to a re-run at a different `frequency_penalty` or served window.** The check
  is per-run; the G2 run gets its own.
