# W9 Stratified Robustness Check — ADR-0009 §5 (`v4` run, run of record)

**Verdict: FAIL on the parity tolerance, and the check is nonetheless discharged.** Re-executed on
2026-08-23 against `generate_fp05_n100_guided_v4`, which is the Gate G2 run of record.

Two statements that must be read together, because either alone is misleading:

1. The pooled granularity gap is $+30.8\%$ against a $\pm 15\%$ tolerance — the **widest** of any
   run to date. Joint median 13.0 words/claim, post-hoc 17.0.
2. The confound that gap exists to flag **does not transmit to citation-F1**. Standardising joint's
   citation-recall to post-hoc's own claim-length distribution *widens* the contrast, from
   $+0.1403$ to $+0.1495$ $[+0.0786, +0.2244]$. Joint's advantage is larger at matched claim length
   than at observed claim length.

ADR-0009 §1 and §3 are explicit that this is the intended reading: parity is "one quantity measured
and disclosed whatever it says", and "**the tolerance does not need to be achievable.** Missing it
is survivable by design — see §5." §5's one-sided fallback makes the stratified check *mandatory*
when the residual favours C2; it does not make passing it a condition. **W9 is not a Gate G2
criterion** — `research_roadmap.md`'s gate text gates citation-F1 and parse rate, and nothing else.

## 1. What was run

```bash
uv run python scripts/w9_stratified_parity_report.py docs/harvest/generate_fp05_n100_guided_v4 --max-tokens 3584
uv run python scripts/w9_length_standardized_contrast.py docs/harvest/generate_fp05_n100_guided_v4
```

| field | value |
|---|---|
| run id | `generate_fp05_n100_guided_v4` |
| manifest `git_sha` | `054ec6b6adb5f73cff0e61451850711733a74d9a` (clean, no `-dirty`) |
| `config_hash` / `CONFIG_VERSION` | `d6fa0d9b5491` / `1.5.0` |
| generation | `frequency_penalty 0.5`, `temperature 0.0`, `max_tokens 3584`, `guided_decoding true` for **both** arms |
| joint decoding fixes | filler-claim ban, compact-JSON instruction, bounded escape-valve retry at $T=0.3$ / $T=0.7$ |
| prompt modifications | **none on granularity.** `JOINT_JSON_TEMPLATE` carries no claim-length target |
| records / cost rows | 300 / 472 |

`054ec6b` is `045a96c^` — the commit *before* any claim-length target was added to
`JOINT_JSON_TEMPLATE`. This is why `v4` is the run of record and not `v5`–`v9`: see §4.

## 2. Pooled parity gate

| arm | median words/claim |
|---|---|
| joint | 13.0 |
| post_hoc | 17.0 |

gap **+30.8%** against ±15% → **FAIL**.

## 3. Stratification schemes

### 3.1 `compound_structure`

| stratum | queries | joint claims | post_hoc claims | joint median | post_hoc median | gap | status |
|---|---|---|---|---|---|---|---|
| simple | 100 | 301 | 379 | 13.0 | 16.0 | +23.1% | **FAIL** |
| compound | 95 | 105 | 247 | 17.0 | 19.0 | +11.8% | **PASS** |

Scheme verdict **FAIL**, 2/2 strata powered.

### 3.2 `claim_length`

| stratum | queries | joint claims | post_hoc claims | joint median | post_hoc median | gap | status |
|---|---|---|---|---|---|---|---|
| 1-10 | 58 | 92 | 34 | 9.0 | 9.0 | +0.0% | **PASS** |
| 11-15 | 96 | 165 | 190 | 13.0 | 13.0 | +0.0% | **PASS** |
| 16-20 | 90 | 102 | 215 | 18.0 | 18.0 | +0.0% | **PASS** |
| 21-30 | 78 | 41 | 174 | 24.0 | 23.0 | -4.2% | **PASS** |
| 31+ | 13 | 3 | 13 | 34.0 | 33.0 | -2.9% | **PASS** |

Scheme verdict **PASS**, 5/5 strata powered. The $+0.0\%$ gaps are structural — the scheme bins *by*
the gated quantity — which is exactly what makes these strata the right place to ask the confound
question, and is what §4 below does.

### 3.3 `query_claim_count`

| stratum | queries | joint claims | post_hoc claims | joint median | post_hoc median | gap | status |
|---|---|---|---|---|---|---|---|
| 1-5 claims | 89 | 336 | 552 | 13.0 | 18.0 | +38.5% | **FAIL** |
| 6-10 claims | 10 | 70 | 67 | 14.0 | 16.0 | +14.3% | **PASS** |
| 11+ claims | 0 | 0 | 0 | — | — | — | UNDERPOWERED |

Scheme verdict **FAIL**, 2/3 strata powered.

## 4. Discharging §5's asymmetric scrutiny

A granularity gap is a confound only if it **transmits** to citation-F1. The pooled gate measures
the gap's *size*; it cannot measure its *effect*. ADR-0009's Context names the mechanism to test:

> Coarser claims are harder to entail per claim, so if post-hoc's claims are systematically coarser,
> post-hoc is systematically penalised — and **C2's gap appears without joint grounding doing any
> work.**

Per-claim citation recall within matched length bands (`scripts/w9_length_standardized_contrast.py`):

| band | joint n | joint R | post_hoc n | post_hoc R | ΔR |
|---|---|---|---|---|---|
| 1-10 | 95 | 0.526 | 34 | 0.529 | -0.003 |
| 11-15 | 165 | 0.497 | 187 | 0.358 | **+0.139** |
| 16-20 | 102 | 0.480 | 214 | 0.322 | **+0.158** |
| 21-30 | 41 | 0.585 | 172 | 0.384 | **+0.202** |
| 31+ | 3 | 0.667 | 12 | 0.333 | **+0.333** |

Joint leads in four of five bands and ties in the shortest. **ΔR grows monotonically with claim
length** — the opposite of the confound's signature, which requires joint's advantage to come from
its claims being shorter.

Direct standardisation of joint's recall to post-hoc's claim-length distribution:

| quantity | joint | post_hoc | delta |
|---|---|---|---|
| citation-F1, unstandardised | 0.6651 | 0.5248 | **+0.1403** |
| citation-F1, length-standardised | 0.6743 | 0.5248 | **+0.1495** |

95% clustered CI on the standardised delta: **[+0.0786, +0.2244]** (width 0.1458, unit = query,
$n$ = 99 clusters, 10000 draws, seed 0, ADR-0011 §2). Excludes zero.

Post-hoc is the reference distribution, so its standardised recall equals its observed recall by
construction — the script asserts this, since it is the identity that proves the weighting is not
quietly rescaling the baseline. The unstandardised path reproduces `citation_contrast.py`'s
$+0.1403$ exactly, so standardisation is the *only* difference between the two rows.

**Conclusion: the granularity gap transmits *against* C2, not for it.** Post-hoc's coarser claims
were, on net, making joint's measured advantage look *smaller* than it is at matched granularity.
The gap is reported at $+30.8\%$ and is not tuned away.

### Empty-text claims are folded in, not dropped

`len(text.split()) == 0` claims are a guided-JSON artifact of the joint arm only — 3 here, none in
post-hoc on any run. They fall outside `CLAIM_LENGTH_BANDS` (which starts at 1) and are folded into
the `1-10` band rather than skipped. Skipping them would standardise joint's own defect out of the
comparison and flatter joint, and ADR-0009's Context is explicit that a residual pointing toward the
hypothesis is the direction that must never go unmeasured.

## 5. Why `v4` and not `v5`–`v9`

`v5` through `v9` each edited `JOINT_JSON_TEMPLATE`'s claim-length target to move this check
(`045a96c`, `95dd958`, `dab7a68`, `dc08914`, `b29e74c`; three of the four commit subjects name W9
parity outright). All five are reverted as of 2026-08-23, on two independent grounds.

**Protocol.** ADR-0009 §4 confines the granularity lever to `POST_HOC_ANSWER_TEMPLATE`. §6's blind
lifted 2026-08-14, so every one of those edits steered *joint's* granularity with citation-F1
visible — "the one thing §6 exists to prevent", in `PARITY_LOOP_CLOSED`'s own words. And §5's check
is a disclosure instrument: *"A pre-registered asymmetric check is not retracted because the
iteration that closed the loop passed; that retraction is the post-hoc steering §3 and §6 exist to
prevent."* Tuning it into passing is that retraction by another route.

**Measurement.** The five runs read verdicts in no stable relation to the target's wording:

| run | claim-length target | joint clean parses | W9 verdict | citation-F1 delta | CI excludes 0 |
|---|---|---|---|---|---|
| `v4` | none | 97/100 | FAIL (+30.8%) | +0.1403 | **yes** `[+0.0751, +0.2066]` |
| `v5` | 15–20 words | 95/100 | FAIL (+21.4%) | +0.0933 | **yes** `[+0.0259, +0.1613]` |
| `v6` | 16–22 words | 96/100 | PASS (+6.2%) | +0.0557 | no `[-0.0078, +0.1226]` |
| `v7` | 15–20 words | 96/100 | FAIL (+13.3%) | +0.1114 | **yes** `[+0.0507, +0.1752]` |
| `v8` | 16–20 words | 98/100 | PASS (+13.3%) | +0.0634 | no `[-0.0019, +0.1296]` |
| `v9` | 16–21 words | 91/100 | FAIL (+13.3%) | +0.0851 | **yes** `[+0.0146, +0.1554]` |

Note `v5` and `v7` carry the **same** target text and land on different W9 verdicts and different
parse rates. Parse rate swings 98 → 91 on a one-word change. The gated statistic is an integer
median of 14–20 words, where one word is $\approx 6.7\%$ against a two-word tolerance: this is
verbatim the "run out of resolution" argument `PARITY_LOOP_CLOSED` used to stop the parity loop at 1
of 10 iterations. Continuing would have been fitting run-to-run noise, and — because W9-pass and
CI-excludes-zero are anti-correlated across these six runs — eventually manufacturing a
simultaneous pass by chance.

`v6`–`v9` are void as evidence for anything. `v4` is the last run whose joint prompt carried no
granularity instruction at all.
