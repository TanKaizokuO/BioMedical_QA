# Joint vs Post-Hoc Citation F1 at `fp = 0.5` — diagnostic read

**Delta +0.0094, 95% CI [-0.0536, +0.0729]. The interval crosses zero: contrast C2 is not
established.** This is a diagnostic reading, not a gate figure — the joint arm on this run is
unguided and parses 34/100.

## 1. What was run

```
uv run python scripts/citation_contrast.py docs/harvest/generate_fp05_n100_guided_batched --threshold 0.5
```

| field | value |
|---|---|
| run id | `generate_fp05_n100_guided_batched` |
| manifest `git_sha` | `c9a2a8e54cae617acaea2f26c477abd5f09321b0` (clean) |
| verifier | MiniCheck φ, `lytang/MiniCheck-Flan-T5-Large`, threshold 0.5, CPU |
| paired queries / dropped | 100 / 0 |
| bootstrap | 10000 resamples, cluster unit = query, seed 0 (ADR-0011 §2) |
| artifact | `docs/harvest/generate_fp05_n100_guided_batched.citation_f1.minicheck.json` |

φ pairs: 2131 required, 1903 were cached, 1113 were scored fresh. (The two counts do not
add up because the cache carries pairs from earlier runs that this run does not need.)

## 2. Result

| arm | precision | recall | citation F1 | claims | relevant / total citations |
|---|---|---|---|---|---|
| joint | 0.8443 | 0.3909 | **0.5344** | 463 | 580 / 687 |
| post_hoc | 0.9550 | 0.3620 | **0.5250** | 627 | 743 / 778 |

Paired delta (joint − post_hoc): **+0.0094**, 95% CI **[−0.0536, +0.0729]**, width 0.1265,
excludes zero **False**.

## 3. Comparison with the only earlier paired read

| read | `fp` | joint | post_hoc | delta | 95% CI |
|---|---|---|---|---|---|
| `parity_iter1b` | 0.0 | 0.428 | 0.418 | +0.011 | [−0.117, +0.137] |
| this run | 0.5 | 0.534 | 0.525 | +0.009 | [−0.054, +0.073] |

Both arms gain ~0.11 F1 at `fp = 0.5`, and the interval halves in width (0.254 → 0.127) on
the same 100 queries. **The delta does not move.** The penalty and the guided post-hoc path
bought precision on both sides, not separation between them.

## 4. Why this is not the gate figure

The joint arm here is the **unguided** free-text arm: 34/100 clean parses, 161
`quote_not_found`. Post-hoc is guided and batched: 99/100 clean, 0 `quote_not_found`. So the
comparison is confounded by decoding constraint, and the joint arm is far under the Gate G2
≥95% valid-parse bar. Post-hoc's higher precision (0.955 vs 0.844) is exactly what a verbatim
schema buys, so part of the remaining gap is a decoding artefact rather than an arm property.

The joint guided path landed in code on 2026-08-20 (`System.JOINT` branch in
`src/biomedqa/generate.py`, `JOINT_JSON_TEMPLATE`). **The read must be repeated on a run where
both arms are guided.** Until then, C2 has no measured operating point.

## 5. Not licensed by this reading

- No Gate G2 verdict, in either direction.
- No claim that joint attribution is at parity with post-hoc — an arm that parses 34/100
  cannot support a parity claim.
- No transfer to the abstention-adjusted denominators: this run records zero abstentions in
  both arms, so recall-all equals recall-answered by construction, not by finding.
