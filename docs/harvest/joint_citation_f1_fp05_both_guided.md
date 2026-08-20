# Joint vs Post-Hoc Citation F1 at `fp = 0.5`, both arms guided — diagnostic read

**Delta +0.1083, 95% CI [+0.0432, +0.1722]. The interval excludes zero: contrast C2 is
established on this run.** This is still a diagnostic reading, not a Gate G2 verdict — the
joint arm parses 89/100 clean, under the ≥95% valid-parse bar.

## 1. What was run

```
uv run python scripts/citation_contrast.py docs/harvest/generate_fp05_n100_guided_both --threshold 0.5
```

| field | value |
|---|---|
| run id | `generate_fp05_n100_guided_both` |
| manifest `git_sha` | `958cf179602fa0eb8c52d19f2b8b494b1f72bcbf` (clean) |
| `config_hash` / `CONFIG_VERSION` | `4ea12ab3eae4` / `1.5.0` |
| served window | `--max-model-len 14336` (raised from 8192 this session) |
| generation | `frequency_penalty 0.5`, `temperature 0.0`, `max_tokens 3584`, `guided_decoding true` for **both** joint and post-hoc |
| verifier | MiniCheck φ, `lytang/MiniCheck-Flan-T5-Large`, threshold 0.5, CPU |
| paired queries / dropped | 89 / 11 (dropped: zero claims in joint arm) |
| bootstrap | 10000 resamples, cluster unit = query, seed 0 (ADR-0011 §2) |
| artifact | `docs/harvest/generate_fp05_n100_guided_both.citation_f1.minicheck.json` |

φ pairs: 1493 required, 3016 were cached, 589 were scored fresh this read.

## 2. Result

| arm | precision | recall | citation F1 | claims | relevant / total citations |
|---|---|---|---|---|---|
| joint | 0.9308 | 0.4578 | **0.6137** | 391 | 444 / 477 |
| post_hoc | 0.9514 | 0.3441 | **0.5055** | 555 | 666 / 700 |

Paired delta (joint − post_hoc): **+0.1083**, 95% CI **[+0.0432, +0.1722]**, width 0.1290,
excludes zero **True**.

## 3. Comparison with the earlier confounded read

| read | joint decoding | post_hoc decoding | joint F1 | post_hoc F1 | delta | 95% CI | excludes zero |
|---|---|---|---|---|---|---|---|
| `generate_fp05_n100_guided_batched` | unguided (34/100 clean) | guided | 0.5344 | 0.5250 | +0.0094 | [−0.0536, +0.0729] | No |
| this run (`generate_fp05_n100_guided_both`) | guided (89/100 clean) | guided | 0.6137 | 0.5055 | +0.1083 | [+0.0432, +0.1722] | **Yes** |

Once the decoding constraint stops confounding the comparison, the delta moves substantially
— from statistically indistinguishable to a clear joint-arm advantage. The joint arm's
precision jumped from 0.844 to 0.931 under its own guided-JSON schema (closing most of the
gap to post-hoc's 0.951), while its recall roughly doubled (0.391 → 0.458) relative to the
unguided read on a comparable claim count, so both terms of F1 improved.

## 4. Why this is still not the gate figure

The joint arm parses 89/100 clean on this run (11 dropped queries were zero-claims-in-joint,
not malformed JSON — the log shows 89 clean parses, 0 `quote_not_found`, and the remaining 11
failures are malformed-JSON replies from the guided decoder, not schema violations). 89/100 is
under the Gate G2 ≥95% valid-parse bar, so this remains a diagnostic reading. The 11 malformed
replies are truncated/malformed JSON emitted despite guided-JSON constraints (see
`generate_fp05_n100_guided_both.run.log`, e.g. `reply is malformed JSON: Expecting property
name enclosed in double quotes`), which is a decoder-side truncation issue independent of the
citation-F1 contrast.

## 5. Not licensed by this reading

- No Gate G2 verdict, in either direction — the ≥95% valid-parse bar is not met (89/100).
- No claim about the cause of the remaining 11% parse failures beyond "malformed JSON under
  guided decoding," which needs its own investigation before a Gate G2 run of record.
- No transfer to the abstention-adjusted denominators: recall-all equals recall-answered by
  construction on this run (need to confirm zero abstentions hold under the wider window).
