# Joint vs Post-Hoc Citation F1 at `fp = 0.5`, both arms guided (`v4` run) — Gate G2 run of record

**Delta +0.1403, 95% CI [+0.0751, +0.2066]. The interval excludes zero, and the joint arm reaches
97/100 clean parses. Both Gate G2 criteria are met on this run.**

Gate G2, verbatim from `research_roadmap.md`:

> **Gate G2 (by Sep 6): on dev, joint attribution beats post-hoc citation on citation-F1 by a margin
> exceeding the paired-bootstrap CI, and ≥95% of emitted claims parse into the schema with
> resolvable spans.**

Two criteria, both met here. The ADR-0009 §5 W9 stratified check is a mandatory *disclosed
diagnostic* rather than a third criterion (§1: "one quantity measured and disclosed whatever it
says"; §3: "the tolerance does not need to be achievable"). It reads FAIL at $+30.8\%$ on this run
and is discharged in `docs/harvest/w9_stratified_parity_guided_v4.md`, which shows the granularity
gap transmits *against* C2: at matched claim length the contrast **widens** to $+0.1495$
$[+0.0786, +0.2244]$.

## 1. What was run

```bash
uv run python scripts/citation_contrast.py docs/harvest/generate_fp05_n100_guided_v4
uv run python scripts/w9_length_standardized_contrast.py docs/harvest/generate_fp05_n100_guided_v4
```

| field | value |
|---|---|
| run id | `generate_fp05_n100_guided_v4` |
| manifest `git_sha` | `054ec6b6adb5f73cff0e61451850711733a74d9a` (clean, no `-dirty`) |
| `config_hash` / `CONFIG_VERSION` | `d6fa0d9b5491` / `1.5.0` |
| generation | `frequency_penalty 0.5`, `temperature 0.0`, `max_tokens 3584`, `guided_decoding true` for **both** joint and post-hoc |
| joint decoding fixes | filler-claim ban, compact-JSON instruction, bounded escape-valve retry at $T=0.3$ / $T=0.7$ |
| prompt modifications | **none on granularity.** `JOINT_JSON_TEMPLATE` carries no claim-length target |
| verifier | MiniCheck φ, `lytang/MiniCheck-Flan-T5-Large`, threshold 0.5, CPU |
| records / cost rows | 300 / 472 |

## 2. Arm performance

| quantity | joint | post_hoc |
|---|---|---|
| precision | 0.9563 | 0.9545 |
| recall (answered) | 0.5099 | 0.3619 |
| **citation F1** | **0.6651** | **0.5248** |
| recall (all claims) | 0.5099 | 0.3619 |
| F1 (all claims) | 0.6651 | 0.5248 |
| claims (answered / abstained) | 406 / 0 | 619 / 0 |
| relevant citations / citations | 460 / 481 | 734 / 769 |

Precision is effectively tied (0.9563 vs 0.9545); **the entire contrast is a recall difference**
(0.5099 vs 0.3619). That is the expected shape for C2 — joint grounding changes whether a claim ends
up supported by what was actually cited, not whether the citations it emits are relevant.

## 3. Paired contrast

Paired delta (joint − post_hoc): **+0.1403**, 95% CI **[+0.0751, +0.2066]**, width 0.1314, excludes
zero **True**. Resampling: unit = query, $n$ = 99 clusters, 10000 draws, seed 0 (ADR-0011 §2).

One query dropped from pairing: zero claims in the joint arm.

Joint arm clean parses: **97/100 (97%)**, `quote_not_found = 0`. Clears the $\ge 95\%$ bar. The
remaining 3 are xgrammar whitespace death-loop failures that exhausted the bounded escape-valve
retry; see `prompts.JOINT_JSON_TEMPLATE`'s comment and `generate.py`'s `System.JOINT` guided branch.

## 4. Comparison across the guided runs

| run | claim-length target | joint clean parses | joint F1 | post_hoc F1 | delta | 95% CI | excludes zero |
|---|---|---|---|---|---|---|---|
| **`v4`** | **none** | **97/100** | **0.6651** | **0.5248** | **+0.1403** | **[+0.0751, +0.2066]** | **yes** |
| `v5` | 15–20 words | 95/100 | 0.6142 | 0.5209 | +0.0933 | [+0.0259, +0.1613] | yes |
| `v6` | 16–22 words | 96/100 | 0.5785 | 0.5228 | +0.0557 | [-0.0078, +0.1226] | no |
| `v7` | 15–20 words | 96/100 | 0.6275 | 0.5161 | +0.1114 | [+0.0507, +0.1752] | yes |
| `v8` | 16–20 words | 98/100 | 0.5833 | 0.5200 | +0.0634 | [-0.0019, +0.1296] | no |
| `v9` | 16–21 words | 91/100 | 0.5914 | 0.5063 | +0.0851 | [+0.0146, +0.1554] | yes |

`v4` carries the **largest** delta and the **widest** granularity gap simultaneously, and `v6`/`v8`
carry the smallest deltas and the narrowest gaps. Read naively that pattern looks like the
granularity confound ADR-0009 warned about. It is not: the length-standardised analysis shows the
mechanism runs the other way (matching lengths *widens* the delta on `v4`). What actually drives the
run-to-run spread is that pushing joint's claims longer trades recall away — a longer claim is
harder to entail from the same three cited spans — which lowers joint's F1 while narrowing the
parity gap. Both movements come from one knob, which is why W9-pass and CI-excludes-zero never
co-occur across `v5`–`v9`.

`v5`–`v9` are void as evidence: each tuned `JOINT_JSON_TEMPLATE`'s granularity against a
pre-registered check with citation-F1 already unblinded, contrary to ADR-0009 §4 and §6. All five
edits are reverted as of 2026-08-23; `prompts.JOINT_JSON_TEMPLATE` now matches `054ec6b`
byte-for-byte, the state `v4` ran on. Full argument in
`docs/harvest/w9_stratified_parity_guided_v4.md` §5.

## 5. Gate G2 status

| criterion | source | result |
|---|---|---|
| citation-F1 margin exceeds the paired-bootstrap CI | `research_roadmap.md` Gate G2 | **MET** — +0.1403 [+0.0751, +0.2066] |
| $\ge 95\%$ of emitted claims parse with resolvable spans | `research_roadmap.md` Gate G2 | **MET** — 97/100, `quote_not_found = 0` |
| W9 stratified granularity check *(disclosed diagnostic, ADR-0009 §1/§3/§5 — not a gate)* | ADR-0009 §5 | run, FAIL at +30.8%, disclosed, and discharged by the length-standardised contrast |

**Gate G2 signs off on `generate_fp05_n100_guided_v4`.**
