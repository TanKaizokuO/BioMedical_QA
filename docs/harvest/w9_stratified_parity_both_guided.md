# W9 Stratified Robustness Check — ADR-0009 §5 (both arms guided)

**Verdict: FAIL.** Run on 2026-08-20 against `generate_fp05_n100_guided_both`, the first run
where both the joint and post-hoc arms use guided-JSON decoding. This supersedes the PASS
recorded on `generate_fp05_n100_guided_batched`, which compared a guided post-hoc arm against
an **unguided** joint arm and is no longer the relevant baseline.

## 1. What was run

```
uv run python scripts/w9_stratified_parity_report.py docs/harvest/generate_fp05_n100_guided_both --max-tokens 3584
```

| field | value |
|---|---|
| run id | `generate_fp05_n100_guided_both` |
| manifest `git_sha` | `958cf179602fa0eb8c52d19f2b8b494b1f72bcbf` (clean, no `-dirty`) |
| provenance | `live` |
| `config_hash` / `CONFIG_VERSION` | `4ea12ab3eae4` / `1.5.0` |
| served window | `--max-model-len 14336` |
| generation | `frequency_penalty 0.5`, `temperature 0.0`, `max_tokens 3584`, `guided_decoding true` for **both** arms |
| records / cost rows | 300 / 453 |
| tolerance | `PARITY_TOLERANCE = ±15%`, `min_queries = 5` |

## 2. Pooled gate (context, not the check)

| arm | median words/claim |
|---|---|
| joint | 14.0 |
| post_hoc | 17.0 |

gap **+21.4%** against ±15% → **FAIL**. This crosses the tolerance for the first time —
every earlier read (`parity_iter1b`, the unguided-joint `generate_fp05_n100_guided_batched`
read at +13.3%) passed. Guiding the joint arm's JSON schema shortened its median claim
(15.0 → 14.0 words) while post-hoc held flat (17.0 both times), widening the gap.

## 3. Scheme 1 — compound structure

| stratum | queries | joint claims | post_hoc claims | joint median | post_hoc median | gap | status |
|---|---|---|---|---|---|---|---|
| simple | 100 | 289 | 378 | 13.0 | 16.0 | +23.1% | **FAIL** |
| compound | 96 | 102 | 249 | 17.0 | 19.0 | +11.8% | PASS |

Scheme verdict **FAIL**, 2/2 strata powered. The simple stratum — the one with real
discriminating power — now breaches tolerance. The residual is still verbosity inside
simple claims (as in every earlier read), but the guided joint schema pushed the simple
median down further (14.0 → 13.0) while post-hoc's simple median held at 16.0, moving the
gap from +14.3% (PASS) to +23.1% (FAIL).

## 4. Scheme 2 — claim length bands

| stratum | queries | joint claims | post_hoc claims | joint median | post_hoc median | gap | status |
|---|---|---|---|---|---|---|---|
| 1-10 | 61 | 75 | 34 | 9.0 | 9.0 | +0.0% | PASS |
| 11-15 | 95 | 155 | 190 | 13.0 | 13.0 | +0.0% | PASS |
| 16-20 | 93 | 96 | 215 | 17.0 | 18.0 | +5.9% | PASS |
| 21-30 | 77 | 55 | 176 | 23.0 | 23.0 | +0.0% | PASS |
| 31+ | 14 | 6 | 12 | 34.5 | 33.5 | −2.9% | PASS |

Scheme verdict **PASS**, 5/5 strata powered. As before, the stratifier is the outcome
variable — claims are binned by their own word count — so within-band near-zero gaps are
partly mechanical and do not contradict the Scheme 1 finding.

## 5. Scheme 3 — query claim volume

| stratum | queries | joint claims | post_hoc claims | joint median | post_hoc median | gap | status |
|---|---|---|---|---|---|---|---|
| 1-5 claims | 73 | 280 | 441 | 13.0 | 17.0 | +30.8% | **FAIL** |
| 6-10 claims | 16 | 111 | 114 | 16.0 | 19.5 | +21.9% | **FAIL** |
| 11+ claims | 0 | 0 | 0 | — | — | — | UNDERPOWERED |

Scheme verdict **FAIL**, 2/3 strata powered (both powered strata fail). Queries are
assigned to a band by their **joint** claim count. The 11+ band remains empty at
`fp = 0.5` under the wider 14336-token window.

## 6. What this licenses, and what it does not

Licensed:

- **The granularity-parity precondition for Gate G2 is not discharged on this run.** Two of
  three pre-registered schemes fail (compound structure, query claim volume), and the pooled
  gap itself breaches tolerance (+21.4%). Guiding the joint arm's decoding shortened its
  claims relative to post-hoc, widening a gap that previously passed.
- The direction of the effect is consistent with §3 of the earlier report: the residual is
  verbosity (post-hoc claims run longer), not compounding — that qualitative finding survives.
  What changed is magnitude, not sign.

Not licensed:

- **No Gate G2 sign-off.** This W9 check gates claim-unit fairness before Gate G2's
  citation-F1 contrast can be trusted, and it now fails on the run that also carries the
  citation-F1 diagnostic read (`docs/harvest/joint_citation_f1_fp05_both_guided.md`). The
  citation-F1 delta (+0.1083, CI excludes zero) was measured on claims that are not parity-
  matched in length, so part of that delta may be attributable to shorter, more precise
  joint claims rather than to attribution quality per se.
- **No claim on the 11+ claims/query regime**, unobserved at this `frequency_penalty` /
  window combination.
- **No transfer to a different `frequency_penalty` or served window.** The check is
  per-run.

## 7. Consequence

Before a Gate G2 run of record, the joint-arm schema (`JOINT_JSON_TEMPLATE` /
`build_citation_response_format(..., is_joint=True)`) needs either a claim-length floor or a
prompt-level nudge toward parity with post-hoc's median claim length, followed by a repeat of
this check. ADR-0009 §8 freezes the decomposer prompts/parsers, not the joint-arm schema
introduced this session, so this is in scope without a new ADR.
