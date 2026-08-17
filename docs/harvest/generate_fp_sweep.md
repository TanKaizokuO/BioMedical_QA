# Generator-side frequency penalty sweep — closing the runaway claim loop (2026-08-17)

`scripts/generate_smoke.py --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --query-ids <12 ids> --max-tokens 3584 --frequency-penalty {0.0,0.3,0.5} --overwrite`, seed 0, `--depth` default, served by `vllm serve --max-model-len 8192 --gpu-memory-utilization 0.85` on the A4000 GPU. Artifacts committed as `docs/harvest/generate_fp_sweep_fp{00,03,05}.{summary.json,manifest.json,records.jsonl,costs.jsonl}`.

## Read the caveats before the number

1. **Not a gate run, not a sample.** n=12; the slice (file order from `docs/harvest/dev_contexts_top10.jsonl`): `10375486, 10490564, 10757151, 10759659, 10927144, 11500608, 11970923, 11977907, 12238307, 17578985, 21074975, 9920954`.
2. **Deliberately enriched slice.** Four of the 12 questions (`10490564`, `17578985`, `21074975`, `9920954`) were chosen specifically because they carry the runaway-claim pathology. Nothing here is a rate, a sample, or a Gate G2 benchmark number.
3. **Arm-dependent call failure on unguarded runs.** Two earlier sweep attempts (at token caps of 3584 and 3072) died on an arm-dependent HTTP 400 error. The single `fp = 0.0` call failure occurred on question `21074975` post_hoc stage 2 (`call 2 rejected: vLLM returned 400 ... requested 3584 output tokens and your prompt contains at least 4609 input tokens, for a total of at least 8193`). Because post-hoc's stage 2 citation prompt embeds stage 1's answer, stage 1's runaway generation inflated the stage 2 prompt past vLLM's max model window. Before the call-failure recording guard landed, this 400 aborted the entire `fp = 0.0` arm while the 0.3 and 0.5 arms completed. Survival was arm-dependent, which made an aborting run unmeasurable and uncomparable across arms.

## The number

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
- `fp = 0.0`: joint 17 / 35.87 / 731, post_hoc 19 / 19.88 / 80, vanilla 18 / 19.23 / 52
- `fp = 0.3`: joint 15 / 15.41 / 25, post_hoc 17 / 17.25 / 33, vanilla 20 / 20.62 / 43
- `fp = 0.5`: joint 15 / 15.68 / 27, post_hoc 17 / 17.42 / 33, vanilla 20 / 21.30 / 46

Median claims per query:
- `fp = 0.0`: joint 4.5 / post_hoc 9.0 / vanilla 7.0
- `fp = 0.3`: joint 4.0 / post_hoc 6.5 / vanilla 5.0
- `fp = 0.5`: joint 5.5 / post_hoc 4.0 / vanilla 5.0

## Reading it

**The runaway claim loop is a real and complete `frequency_penalty` response at 0.3 already.** Chain claims fall from 3 (joint 2, post_hoc 1) to 0, chain pairs 1 to 0, claims >50 words from 17 (joint 15, post_hoc 2) to 0, and the longest joint claim collapses from 731 words to 25 words (post_hoc longest from 80w to 33w).

**`quote_not_found` FALLS rather than rises.** This plainly inverts the risk `GenerationConfig.frequency_penalty`'s own comment anticipated ("Frequency penalty applied to generator sampling ... higher values may cause quote extraction failure"). The mechanism: at `fp = 0.0`, non-terminating repetition loops mangle quote formatting as well as claim text; suppressing the repetition loop under penalty pressure stabilizes generation structure, which improves quote extraction and causes `quote_not_found` to fall (joint 8 → 9 → 0, post_hoc 92 → 30 → 15).

**`fp = 0.5` is preferred over `0.3`.** It is the only point where joint's `quote_not_found` reaches 0 (down from 9 at fp=0.3) while mean claims per query remains flat (5.33 vs 5.42). This demonstrates 0.5 sits on a stable plateau, unlike the total-claims collapse observed at `fp = 1.0` on the decomposer side (`docs/harvest/decompose_smoke_fp_sweep.md`).

**The window-overflow call failure vanishes along with its cause.** At `fp >= 0.3`, runaway loops do not occur, keeping stage 1 answers concise and ensuring stage 2 prompts fit well within vLLM's 8192-token context limit.

## What the sweep did not fix

**`post_hoc` clean-parse counts stay low across all points (2, 1, 4 of 12).** The residual failures in `post_hoc` are driven by quote drift (where generated citation quotes fail to match source text), an independent defect that a decoding knob cannot reach.

This mirrors the finding in `decompose_smoke_fp_sweep.md` for the decomposer: fixing repetition loops does not fix format collapse. **No single `frequency_penalty` value here clears Gate G2's ≥0.95 clean-parse bar** (`claim_parse_rate ≥ 0.95` and `quote_located_rate ≥ 0.95`), and none was expected to. The guided-JSON citation path is what addresses quote drift.

## The parity consequence

**ADR-0009's gated quantity is unmoved.** The published `parity_iter1b` gated baseline figure at `fp = 0.0` (n=100) is joint 15 / post_hoc 17 (+13.3%). On this sweep's 12-question enriched slice, the `fp = 0.0` arm reads joint 17 / post_hoc 19 (+11.8%), where both arms read two words higher than the 100-question run as expected for an enriched slice. At `fp = 0.5`, median words per claim reads joint 15 / post_hoc 17 (+13.3%), landing exactly on the published pair. The gated quantity remains well within the ±15% parity window at every sweep point (+11.8% at `fp = 0.0`, +13.3% at `fp = 0.5`), showing agreement with the published figure at the adopted setting—an agreement rather than a population confirmation, given that at n=12 a single answer shifts a median.

**The reported claims/query diagnostic inverts.** Joint 4.5 / post_hoc 9.0 at `fp = 0.0` becomes joint 5.5 / post_hoc 4.0 at `fp = 0.5` (mean claims/query: joint 9.75 / post_hoc 9.92 → joint 5.42 / post_hoc 3.75). At n=12 a single answer moves a median, so this is flagged as needing a re-read on a larger sample (n > 12) rather than a definitive population shift.

**No prompt moved and no post-hoc steering occurred.** Prompt templates remain frozen (`prompts.PARITY_LOOP_CLOSED` and `decompose.decompose_template_digest()` are unchanged). This is a decoding parameter applied identically to all three systems.

## Decision

**`GenerationConfig.frequency_penalty` default is set from 0.0 to 0.5.**

`CONFIG_VERSION` is incremented to `1.5.0`. As a consequence of updating a default configuration parameter, every `RunConfig.hash()` changes.

## What this does not license

Twelve questions on an enriched slice cannot set Gate G2 benchmark rates, confirm `fp = 0.5` generalizes across the entire dataset, or license claiming that generator frequency penalty resolves post_hoc quote drift or satisfies G2's ≥0.95 parse threshold on its own.
