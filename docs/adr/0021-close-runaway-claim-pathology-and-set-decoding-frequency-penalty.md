# ADR-0021 — Close runaway-claim pathology and set decoding frequency penalty

**Status:** Accepted · **Date:** 2026-08-17 · **Decided in:** Runaway-claim triage and decoding parameter sweep session
**Refines** ADR-0009 (Second amendment), ADR-0019 §1 · **Constrained by** ADR-0009 §8 (Sep 3 freeze)

## Context

Closing the runaway-claim pathology (`docs/adr/0009-granularity-parity-is-a-measured-diagnostic-not-a-condition.md` *One defect found in the process* and *Second amendment*; `docs/harvest/citation_f1_minicheck.md` "Joint's 31+ band is still its worst, and it is still joint's defect, not φ's ... Fixing the splitter and the non-terminating generation still moves a reported number, and still has to happen before the G2 read") is mandatory before the **Sep 3 2026 Decomposer & Granularity Freeze** and the **Sep 6 2026 Gate G2** benchmark run.

Prior to these fixes, non-terminating repetition loops in model generations produced claims exceeding 50 words (up to 731 words on joint query `21074975`), nested prefix-extension chains where claim $N+1$ extends claim $N$, sentence units over the `MAX_CLAIM_WORDS` bound that were not split by the decomposer, and prompt-length window overflows causing HTTP 400 call rejections on post-hoc Stage 2 calls.

This ADR records the four decisions governing detector logic, splitter mechanics, generator parameter defaults, and call rejection handling.

## Decision

### 1. A nested prefix-extension chain is a non-terminating generation, charged at length >= 3

A sequence of consecutive claims where claim $N+1$ extends claim $N$'s text (evaluated via `prompts.claim_stem`, requiring strict prefix matching after stripping punctuation/whitespace and enforcing non-alphanumeric boundaries) represents a non-terminating generation loop.

The chain is charged to `errors` at length $\ge 3$ and to `recovered` at length 2, mirroring ADR-0019 §1's multiplicity taxonomy.

#### Empirical basis for the ×3 threshold

A re-scan of the unchanged `docs/harvest/parity_iter1b.records.jsonl` (300 records across 100 queries $\times$ 3 systems) established the nested-chain length distribution:

- `joint`: `{2: 22, 3: 1, 10: 1, 11: 1}`
- `post_hoc`: `{2: 2, 5: 1}`
- `vanilla`: `{2: 1}`

This exhibits a large $\times 2$ mode plus a sparse loop tail, establishing the exact population split that justified ADR-0019's $\times 3$ threshold for claim repetition.

#### Marginal coverage over MAX_CLAIM_WORDS

The detector is not redundant with `prompts.MAX_CLAIM_WORDS = 50`. Across `parity_iter1b`, `joint` contains 24 claims inside chains of length $\ge 3$, of which only 15 exceed 50 words — leaving **9 claims invisible** to the 50-word length guard. `post_hoc` contains 5 such claims, of which 2 exceed 50 words — leaving **3 claims invisible**. For example, `joint` query `17578985` contains a 3-chain (claims c15, c16, c17) at word lengths 19, 26, and 32 — wholly invisible to the 50-word length guard.

Re-parsing `raw_generation` across `parity_iter1b` flags four records with length $\ge 3$ using the exact error string emitted by `parse_response`:

- `joint` `21074975`: `c13: extends c3's claim text through 11 nested claims (non-terminating generation)`
- `joint` `10490564`: `c17: extends c8's claim text through 10 nested claims (non-terminating generation)`
- `joint` `17578985`: `c17: extends c15's claim text through 3 nested claims (non-terminating generation)`
- `post_hoc` `9920954`: `c9: extends c5's claim text through 5 nested claims (non-terminating generation)`
- `vanilla`: none.

The logic fires symmetrically across both C2 arms (`joint` and `post_hoc`).

### 2. A run-on unit is split only at boundaries the text marks

Sentence decomposition via `sentence_units(answer, *, max_words=MAX_CLAIM_WORDS)` enforces `MAX_CLAIM_WORDS = 50`. When a unit exceeds 50 words, `_split_run_on` cuts only at boundaries explicit in the text (`;` or `,` followed by whitespace). A run-on unit with no such internal punctuation is returned whole and remains flagged as exceeding the word bound rather than force-split.

#### Empirical basis and invariants

Across the 300 `parity_iter1b` records:
- **Before:** 27 of 3592 units exceeded 50 words (`joint`: 20, `post_hoc`: 3, `vanilla`: 4; longest was 731 words on `joint` `21074975`).
- **After:** `joint` units increased from 719 to 771 and **0** exceed 50 words (the 731-word claim was split into 18 pieces); `post_hoc` units increased from 1246 to 1249 and **0** exceed 50 words; `vanilla` retains exactly **1 unit at 52 words** — a clause containing no internal punctuation, correctly left unsplit and still flagged.

Across all 300 records, the following invariants were verified:
- **0** records lost or duplicated non-whitespace content.
- **0** records contained overlapping or empty units.

### 3. `frequency_penalty` default 0.0 -> 0.5, chosen by measurement

`GenerationConfig.frequency_penalty` default is set from `0.0` to `0.5`, updating `CONFIG_VERSION` to `1.5.0`.

#### Empirical basis: A4000 live sweep (2026-08-17)

A live sampling sweep was executed on the RTX A4000 using `scripts/generate_smoke.py` (`Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, `--max-tokens 3584`, `--frequency-penalty {0.0, 0.3, 0.5}`, seed 0, vLLM `--max-model-len 8192`) over an enriched 12-question slice from `docs/harvest/dev_contexts_top10.jsonl` (`10375486, 10490564, 10757151, 10759659, 10927144, 11500608, 11970923, 11977907, 12238307, 17578985, 21074975, 9920954`; 4 questions carrying known pathology).

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

#### Reasoning chain

1. **Pathology elimination:** At `fp >= 0.3`, chain claims and >50w claims fall to zero; longest `joint` claim drops from 731w to 27w.
2. **Quote extraction improvement:** `quote_not_found` does not rise but **falls** (`joint` 8 $\rightarrow$ 0, `post_hoc` 92 $\rightarrow$ 15). At `fp = 0.0`, repetition loops mangle quote formatting; suppressing repetition stabilizes output structure.
3. **Plateau selection:** `fp = 0.5` reaches 0 `quote_not_found` for `joint` while claims/query remains flat (5.33 vs 5.42), proving 0.5 sits on a stable plateau rather than the total-claims collapse observed at `fp = 1.0` in `docs/harvest/decompose_smoke_fp_sweep.md`.
4. **Pipeline unification:** `fp = 0.5` matches the decoding setting used in C7 decomposition (`scripts/decompose_smoke.py`, `scripts/_diag.sh`, `scripts/_remote_run_cap.sh`), providing uniform decoding settings across generator and decomposer stages.

#### Required disclosures

- The published parity table and ADR-0009's diagnostic were measured at `fp = 0.0`. A Gate G2 run at `fp = 0.5` uses a decoding setting not present during the closed parity loop.
- This is a **decoding parameter applied identically to all three systems, not a prompt edit**. Prompt text is unchanged; `prompts.PARITY_LOOP_CLOSED` and `decompose.decompose_template_digest()` remain untouched, preserving ADR-0009 §4's prompt freeze and §3/§6's prohibition on post-hoc prompt steering.
- The **gated** parity quantity is unaffected: ADR-0009 gates median words/claim within $\pm 15\%$. The published `parity_iter1b` baseline at `fp = 0.0` ($n=100$) reported `joint` 15 / `post_hoc` 17 (**+13.3%**). On this sweep's 12-question enriched slice, the `fp = 0.0` arm reads `joint` 17 / `post_hoc` 19 (**+11.8%**, both arms two words higher as expected for an enriched slice), while the `fp = 0.5` arm reads `joint` 15 / `post_hoc` 17 (**+13.3%**), landing exactly on the published pair. The gated quantity remains inside the $\pm 15\%$ parity window at every sweep point (+11.8% at `fp = 0.0`, +13.3% at `fp = 0.5`), showing agreement with the published figure at the adopted setting (an agreement, not population confirmation, given $n=12$). What inverts is the *reported* claims/query diagnostic (`joint` 4.5 / `post_hoc` 9.0 at `fp = 0.0` becomes `joint` 5.5 / `post_hoc` 4.0 at `fp = 0.5`). On $n=12$, a single answer shifts a median, so this is flagged for re-reading on a larger sample ($n > 12$) rather than claimed as a population finding.
- `post_hoc` clean-parse counts remain low across all settings (2, 1, 4 of 12). The residual failure is quote-drift (generated quotes failing to match source text), an independent defect unreachable by decoding parameters. The guided-JSON citation path addresses quote drift.
- $n=12$ on an enriched slice is a sanity check; no figure here represents a Gate G2 rate or benchmark statistic.

### 4. A rejected model call is recorded, not raised, and never repaired

When an inference call fails (e.g., `httpx.HTTPStatusError` or `TransportError`), `generate_one` records the rejection as `f"call {n} rejected: {exc}"` in `Generation.errors` along with an uninstrumented `CostRecord` (`input_tokens=None`, `output_tokens=None`, measured `wall_s`), rather than propagating an unhandled exception.

#### Empirical basis

At `fp = 0.0`, question `21074975` `post_hoc` Stage 2 failed with an HTTP 400 error (`call 2 rejected: vLLM returned 400 ... requested 3584 output tokens and your prompt contains at least 4609 input tokens, for a total of at least 8193`). Because `post_hoc` Stage 2 embeds Stage 1's completion, Stage 1's runaway generation inflated Stage 2's prompt past vLLM's 8192-token context window.

Under `generate_one`'s contract, an API/transport rejection belongs to the same class of execution events as an unparseable completion. Prior to this fix, an unhandled exception aborted the entire `fp = 0.0` arm while 0.3 and 0.5 completed — making run survival arm-dependent and corrupting multi-arm comparisons.

Additionally, `Generation.recovered` was previously dropped inside `generate_one`, preventing recovered parsing notes from reaching output summaries. `generate_one` now populates and retains `Generation.recovered`.

## Consequences

- Gate G2 (Sep 6, 2026) inherits `GenerationConfig.frequency_penalty = 0.5` under `CONFIG_VERSION` 1.5.0 and the 4-change detector/splitter/recovered/rejection stack.
- `claim_parse_rate` for `joint` will now be charged for nested extension chains it previously passed clean, reflecting an honest accounting of generation quality.
- Every affected diagnostic quantity is re-derivable from stored `raw_generation` records, permitting re-scoring without re-running inference.
- All changes strictly honor ADR-0009 §8's Sep 3 decomposer and granularity freeze.

## Alternatives rejected

- **Force-splitting run-on units on a word budget without marked punctuation:** Rejected because splitting at arbitrary word boundaries invents text boundaries and destroys provenance, violating ADR-0018 §1's exact-match principle (the same rationale for rejecting fuzzy quote matching).
- **Dynamically clamping or retrying `max_tokens` on window overflow (HTTP 400):** Rejected because a per-query effective cap cannot be represented in a run manifest's single `generation.max_tokens` entry. Silently altering `max_tokens` per query would render manifest metadata inaccurate.
- **Editing generation prompt templates to suppress runaway loops:** Rejected because prompt template modifications break `prompts.PARITY_LOOP_CLOSED` under ADR-0009 §4, which requires prompt text to remain frozen.
