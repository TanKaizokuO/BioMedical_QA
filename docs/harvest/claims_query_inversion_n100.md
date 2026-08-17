# The claims/query inversion at `fp = 0.5`, re-read on all 100 dev questions (2026-08-17)

`scripts/generate_smoke.py --base-url http://localhost:8000 --n-questions 100 --contexts docs/harvest/dev_contexts_top10.jsonl --max-claim-words 50`, seed 0, `--depth` default, `--max-tokens 3584`, `temperature = 0.0`, served by `vllm serve --max-model-len 8192 --gpu-memory-utilization 0.85` on the A4000. Artifacts: `docs/harvest/generate_fp05_n100.{summary.json,manifest.json,records.jsonl,costs.jsonl}` — 300 records, 400 cost rows, `provenance.kind = "live"`, `git_sha 9a7cb94` with no `-dirty` suffix, `config_hash 246dfc4897fe` at `CONFIG_VERSION 1.5.0`, `index_fingerprint 57ab89e445f8`, `split_hash 71c46cc5b0ca`. Started 08:47:33 UTC, finished 09:32:50 UTC; 43.7 min of summed call wall time.

**`fp = 0.5` was not passed on the command line.** `run_arguments` in the summary carries only `base_url`, `n_questions`, `contexts`, and `max_claim_words`. The penalty came from `GenerationConfig.frequency_penalty`'s new default, so this run reads the adopted configuration rather than an override of it.

**Guided JSON was NOT introduced.** The three arms ran the frozen free-text grammar: `generate_one` has no guided path, `scripts/generate_smoke.py` never touches `build_citation_response_format`, and the error strings below (`CITE line has no '||' separator`) are only producible by the free-text `cite` stage. `guided_decoding` exists solely on `generate.cite_claims`, which is the `decompose.py` re-citation path and is not called here. It remains unexecuted **on the generation-side post-hoc cite stage**, which is the arm the G2 contrast is read on.

---

## Classification: **INVERSION REPRODUCED**

`generate_fp_sweep.md` flagged the claims/query inversion at `fp = 0.5` as a 12-question reading that "needs a re-read on a larger sample (n > 12) rather than a definitive population shift". On all 100 dev questions the inversion is present, in the same direction, and significant on three paired tests:

| statistic | joint | post_hoc | delta |
|---|---|---|---|
| median claims/query | **5.0** | **4.0** | **+1.0** |
| mean claims/query | **4.56** | **3.80** | **+0.760** |
| total claims parsed | 456 | 380 | +76 |

Paired over the 100 queries (each query contributes one joint and one post_hoc count):

| test | statistic | p |
|---|---|---|
| paired *t* | t = 3.9622, df = 99 | **1.40e-4** |
| Wilcoxon signed-rank | W = 790.5 | **1.40e-4** |
| sign test (54 vs 24, binomial, 22 ties dropped) | — | **9.01e-4** |

Effect size $d_z = 0.396$. Direction counts: **54 queries joint > post_hoc, 24 post_hoc > joint, 22 tied.**

At `fp = 0.0` on the *same 100 queries* (`parity_iter1b`) the ordering was the opposite and stronger: mean delta **−5.23** (joint 7.19 vs post_hoc 12.42), 9 queries joint > post_hoc against 83 post_hoc > joint, paired *t* p = 4.08e-5, Wilcoxon p = 4.95e-10, sign test p = 3.92e-16. **The sign flip is significant at both ends**, so it is not a null result read twice.

---

## Read the caveats before the numbers

1. **This is not a gate run.** The summary's own `purpose` field says so: *"W4 live-path smoke test; not a gate run and not a sample."* Nothing here is a Gate G2 rate, and nothing here reads a citation-F1, a verifier score, or a label.
2. **It is, however, the whole dev split.** The 100 question ids are exactly `data/splits.json`'s `dev` list (`split_hash 71c46cc5b0ca`) and exactly `parity_iter1b`'s query set. The claims/query comparison against `parity_iter1b` is therefore within-query on an identical, frozen slice — not a resample. The 12-question sweep slice is a subset of it.
3. **The `fp = 0.0` reference has two confounds, named in §5.**
4. **Three of 400 calls were rejected by the server** (§4). The diagnostic is re-run without the affected queries in §2.3 and does not depend on them.

---

## 1. The run, as measured

| | joint | post_hoc | vanilla |
|---|---|---|---|
| records | 100 | 100 | 100 |
| stages seen | 1 | 2 | 1 |
| clean parses (records with **no** error string) | **34** | **23** | **99** |
| records with violations (≤3-citation cap) | 3 | 4 | 0 |
| call failures | **1** | **2** | 0 |
| median claims/query | **5.0** | **4.0** | 5.0 |
| mean claims/query | 4.56 | 3.80 | 5.59 |
| total claims | 456 | 380 | 559 |
| total citations | 677 | 563 | 0 |
| recovered notes | 323 | 168 | 0 |
| `quote_not_found` | **154** | **172** | 0 |
| claims > 50 words | 0 | 0 | **1** |
| longest claim | 40w | 37w | 51w |
| runaway chain claims (chains ≥ 3) | **0** | **0** | **0** |
| runaway chain pairs (length 2, recovered) | 0 | 1 | 0 |
| median latency | 9.28 s | 12.92 s | 2.69 s |

Words per claim (median / mean / max): joint **15.0** / 15.53 / 40, post_hoc **17.0** / 17.55 / 37, vanilla 20.0 / 20.84 / 51.

Claims-per-query distribution (queries by claim count):

| count | 0 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| joint | 1 | 5 | 24 | 18 | **29** | 12 | 6 | 3 | 1 | 1 | — | — |
| post_hoc | 2 | 1 | **46** | 29 | 12 | 7 | 2 | 1 | — | — | — | — |
| vanilla | — | 1 | 6 | 16 | **42** | 16 | 7 | 5 | 2 | 2 | 2 | 1 |

**Runaway pathology: absent.** Zero chains of length ≥ 3 across all 300 records; one length-2 pair, charged to `recovered` rather than `errors` as designed (`post_hoc` `14599616`: `c4: extends c3's claim text ('A practicing surgeon cannot detect early lymphedema reliably')`). The single over-length claim is vanilla `24315783` `c5` at **51 words** — one word over `MAX_CLAIM_WORDS = 50`. It is *not* the punctuation-free unit the `fp` sweep recorded: it contains commas, so `decompose.sentence_units`' punctuation-bound splitter would cut it. The generation path only flags length; vanilla is excluded from the parity gate by ADR-0010.

### 1.1 The 12-question sweep reproduces inside this run

The sweep's 12 ids are a subset of these 100, so the two runs can be compared on the same questions. Same server window, same cap, same `fp`; different batch composition.

| | n=100 run, those 12 | `generate_fp_sweep_fp05` (n=12) |
|---|---|---|
| joint clean / qnf / median claims / longest | 8 / **0** / 5.5 / 27w | 9 / **0** / 5.5 / 27w |
| post_hoc clean / qnf / median claims / longest | 5 / 10 / 4.0 / 33w | 4 / 15 / 4.0 / 33w |
| vanilla clean / qnf / median claims / longest | 12 / 0 / 5.0 / 46w | 12 / 0 / 5.0 / 46w |

Claim counts (median **and** mean) and longest-claim words agree exactly on all three arms; **30 of the 36 generations are byte-identical**. The six that moved, and the ±1 clean-parse and `quote_not_found` differences, are the KV-cache-layout / prefill-batching nondeterminism `parity_iter1b.md` documented — byte-identity is only a control across runs whose *batching* as well as server config matches.

**One sweep headline does not survive the re-read: joint `quote_not_found` is 0 on those 12 questions in both runs, but 154 over the full 100, touching 55 of 100 records.** The enriched slice happened to be clean of quote drift on the joint arm. §3 gives the split-wide figure.

---

## 2. The inversion, in detail

### 2.1 Across the four runs on record

| run | n | fp | server window | median claims/query (joint / post_hoc) | ordering |
|---|---|---|---|---|---|
| `parity_iter1b` | 100 (whole dev) | 0.0 | 14336 | 4.0 / **10.0** | post_hoc higher |
| `generate_fp_sweep_fp00` | 12 (enriched) | 0.0 | 8192 | 4.5 / **9.0** | post_hoc higher |
| `generate_fp_sweep_fp03` | 12 (enriched) | 0.3 | 8192 | 4.0 / **6.5** | post_hoc higher |
| `generate_fp_sweep_fp05` | 12 (enriched) | 0.5 | 8192 | **5.5** / 4.0 | joint higher |
| **`generate_fp05_n100`** | **100 (whole dev)** | **0.5** | **8192** | **5.0** / 4.0 | **joint higher** |

Mean claims/query moves the same way: post_hoc 12.42 → 3.80 on the identical 100 questions, joint 7.19 → 4.56.

### 2.2 Significance at n=12 versus n=100

| | n=12, `fp = 0.5` | n=100, `fp = 0.5` |
|---|---|---|
| mean delta (joint − post_hoc) | +1.667 | **+0.760** |
| paired *t* p | 0.0120 | **1.40e-4** |
| Wilcoxon p | 0.0234 | **1.40e-4** |
| direction (joint > / post_hoc > / tie) | 8 / 2 / 2 | **54 / 24 / 22** |

The n=100 point estimate is **less than half** the n=12 one, which is what the sweep's own caveat predicted for an enriched slice; the *evidence* is two to three orders of magnitude stronger. The inversion is smaller and better established than n=12 suggested.

### 2.3 Robustness to the rejected calls

Dropping the two queries touched by a call rejection (`12607666`, `17224424`) from both arms:

| | all 100 | the 98 unaffected |
|---|---|---|
| joint median / mean | 5.0 / 4.56 | 5.0 / 4.571 |
| post_hoc median / mean | 4.0 / 3.80 | 4.0 / 3.878 |
| mean delta | +0.760 | **+0.694** |
| paired *t* / Wilcoxon / sign p | 1.40e-4 / 1.40e-4 / 9.01e-4 | **2.21e-4 / 2.31e-4 / 1.26e-3** |
| median words/claim (joint / post_hoc) | 15.0 / 17.0 | **15.0 / 17.0** |

The inversion and the gated median survive the exclusion. The failures are not carrying either result.

---

## 3. Clean parse rates and `quote_not_found`

`clean_parses` counts records whose `errors` list is **empty** — one bad `CITE` line anywhere in a record disqualifies it. Error strings by kind (count, and the number of records each kind touches):

| kind | joint errors / records | post_hoc errors / records | vanilla |
|---|---|---|---|
| `quote_not_found` (`not found verbatim`) | **154 / 55** | **172 / 62** | 0 |
| `CITE line has no '\|\|' separator` | 41 / 17 | 13 / 9 | 0 |
| `CLAIM line carries no claim number` | 7 / 3 | 47 / 13 | 0 |
| claim is empty | 0 / 0 | 38 / 8 | 0 |
| cites a passage not in the context | 8 / 4 | 7 / 2 | 0 |
| citations exceed the cap of 3 | 3 / 3 | 4 / 4 | 0 |
| call rejected + `no DECISION line` + `no CLAIM lines` | 3 / 1 | 6 / 2 | 0 |
| claim over `max claim length` | 0 | 0 | **1 / 1** |

**`quote_not_found` is the dominant term on both cited arms** and is the reason clean-parse rates are 34/100 and 23/100 rather than something near vanilla's 99/100. Gate G2's bar (`claim_parse_rate ≥ 0.95` **and** `quote_located_rate ≥ 0.95`) is not met by either arm at this configuration, and no decoding value was expected to reach it — `decompose_smoke_fp_sweep.md` and `generate_fp_sweep.md` both recorded that fixing repetition loops does not fix format collapse. Post-hoc's second-largest term (`CLAIM line carries no claim number`, 47 strings over 13 records, plus 38 empty-claim strings over 8) is the same format-collapse family.

**These are per-claim error counts, not rates:** 154 joint `quote_not_found` strings arise against 677 joint citations, and 172 post_hoc strings against 563 citations. A `quote_located_rate` is a G2 statistic and is not computed here.

---

## 4. Call failures: 3 of 400 calls

`call_failure_check` in the summary is **`passed: false`** with `total_call_failures: 3` — the run's own guard flags them. All three are HTTP 400 rejections recorded by the call-rejection guard (ADR-0021) into `Generation.errors` with an uninstrumented `CostRecord` (`input_tokens = null`, `output_tokens = null`, measured `wall_s`), instead of aborting the arm as the pre-guard `fp = 0.0` sweep attempt did.

| query | system | call | latency | consequence |
|---|---|---|---|---|
| `17224424` | joint | **call 1** (its only call) | 0.07 s | record lost entirely: 0 claims, empty `raw_generation` |
| `17224424` | post_hoc | **call 2** (cite) | 5.25 s | stage 1 answered; 0 claims parsed |
| `12607666` | post_hoc | **call 2** (cite) | 5.90 s | stage 1 answered; 0 claims parsed |

All three carry the byte-identical server message, differing only in the call index:

```
call {1,2} rejected: vLLM returned 400 for /v1/chat/completions: {"error":{"message":"This
model's maximum context length is 8192 tokens. However, you requested 3584 output tokens and
your prompt contains at least 4609 input tokens, for a total of at least 8193 tokens. ...
(parameter=input_tokens, value=4609)","type":"BadRequestError", ...
```

**This is a static context-window overflow, not a runaway generation.** The joint failure is decisive: it was rejected in 0.07 s on its *first and only* call, so no generated text existed to inflate anything. Contrast the `fp = 0.0` sweep's single failure, which was caused by a runaway stage-1 answer inflating the stage-2 prompt — that mechanism is gone (zero chains ≥ 3, longest claim 40w).

Measured per-call prompt sizes from `costs.jsonl`, against the 4608-token prompt budget that a 3584 cap leaves in an 8192 window:

| call | n instrumented | median | max (query) | margin to 4608 |
|---|---|---|---|---|
| joint | 99 | 3020 | 4355 (`12607666`) | 253 |
| post_hoc_answer | 100 | 2803 | 4488 (`17224424`) | 120 |
| post_hoc_cite | 98 | 3223 | **4545** (`24783217`) | **63** |
| vanilla | 100 | 2671 | 4356 (`17224424`) | 252 |

**The largest call that succeeded is 63 tokens below the boundary.** Three calls crossed it (0.75% of 400). `parity_iter1b` saw none of this because it was served at `--max-model-len 14336`; the runbook window on the A4000 is 8192 (`docs/harvest/runbooks/wsl-vllm-a4000.md`), and `backends.py` still does not check `prompt_tokens + max_tokens` against the served window before posting — the robustness gap `parity_iter0.md` recorded and deferred to W5 cleanup.

*[INFERENCE]* The reported `4609` is a **bound, not a measurement**: $8192 - 3584 + 1 = 4609$ is exactly the smallest prompt length that can overflow, the message says "at least", and three distinct prompts on two different queries reporting the identical figure is what a boundary constant looks like and not what three independent token counts look like. The actual sizes of the three rejected prompts are unmeasured (§6).

---

## 5. The ADR-0009 gated quantity is unmoved: **+13.3%**

Median words per claim, computed through `scoring/granularity.py` on the committed records:

| basis | queries | joint | post_hoc | gap | gate |
|---|---|---|---|---|---|
| all 100 records | 100 / 100 | 15.0 | 17.0 | **+13.3%** | **PASS** |
| untruncated per arm | 100 / 100 | 15.0 | 17.0 | **+13.3%** | **PASS** |
| untruncated, same queries both arms | 100 | 15.0 | 17.0 | **+13.3%** | **PASS** |

**The three bases coincide because nothing truncated.** Zero of the 397 completed calls reached the 3584 cap; the largest completion any stage produced was 1396 tokens (`post_hoc_cite`, `24315783`) — 39% of the cap. `parity_iter1b` at the same cap had 8 joint, 10 post-hoc-answer, **16 post-hoc-cite**, and 7 vanilla calls pinned at 3584, which is why its three bases read +13.3% / +14.3% / +6.7% on 100 / 92-and-84 / 78 queries. Cite-stage truncation 16 → **0**.

This lands **exactly** on the published `parity_iter1b` pair (joint 15 / post_hoc 17, +13.3%) and on the `fp = 0.5` sweep's 12-question reading (+13.3%), well inside ADR-0009's ±15% window. The distribution around the median also holds its shape: joint p25/p75/p90 = 12/18/23 against post_hoc 14/21/25, and post_hoc's mean is 17.55 against joint's 15.53.

**Two honest qualifications, neither of which moves the headline:**

- **The query-level bootstrap interval widened and now straddles.** Resampling queries (4000 draws, seed 0): gap +13.3%, 95% **[+6.3%, +21.4%]** on all records — against `parity_iter1b`'s [+0.0%, +14.3%], which sat inside ±15% throughout. The point estimate passes on every basis; the interval no longer certifies the pass at every draw. The mechanism is the one ADR-0009 already documents: the statistic is an integer median of 15–17 words, so one word is ~6.7% and the tolerance is two words wide. Fewer claims per query (836 total against `parity_iter1b`'s 1961) means fewer draws per resampled query, so the same grid produces a wider interval.
- **`parity_report.py`'s coverage warning fires.** Post-hoc's median claims/query is 4.0, below the 8.0 floor the script checks, which prints *"Check that words/claim fell because claims got finer, not because the answer got shorter."* At `fp = 0.0` post-hoc held 10.0 claims/query and 1242 claims; here it holds 4.0 and 380. **The pass is no longer accompanied by "post-hoc parses more claims than joint".** That is exactly what the inversion means, and §7 says what it does and does not license.

**The parity loop is untouched.** `prompts.PARITY_LOOP_CLOSED` still records the 2026-08-14 termination at 1 of 10 iterations on `parity_iter1b`; `post_hoc_answer_template_digest() = 91bc7ddd…ac1a` and `decompose_template_digest() = 4129a884…a737` are unchanged. This run charges no iteration and edits no prompt: a decoding default applied identically to all three arms is shared run config, the precedent `parity_iter0 → parity_iter0b` and `parity_iter1 → parity_iter1b` were both taken under. `ParityGate.requires_w9_robustness_check` evaluates **False** here only because the gate passes; **ADR-0009 §5's W9 stratified robustness check remains mandatory** — it was triggered at iteration 0, the residual still favours C2 on every basis, and a passing run does not retract a pre-registered check.

**`parity_report.py` cannot be run end-to-end on this prefix.** `stage_output_tokens` does `int(call.output_tokens)` and raises `TypeError` on the three uninstrumented cost rows the call-rejection guard writes by design. The three bases above were computed by calling `parity_gate`, `arm_granularity`, and `gap_bootstrap_ci` directly with the rejected calls treated as *not truncated* (they produced no tokens); treating them as truncated instead removes at most two queries and cannot move an integer median that is identical on all 100 and on the 98 (§2.3). **Reconciling the script with its own guard is a real defect and is listed in §6.**

---

## 6. Interpretation — explicitly not fact

**Why post-hoc's claim count collapses.** `frequency_penalty` taxes tokens by how often they have already appeared, and an enumerated claim list is built out of tokens that repeat once per item: `CLAIM`, the index, `CITE`, `||`, the passage-id scaffolding. Post-hoc's cite stage emits the most repeated structure per item of the three arms, so it accrues penalty pressure fastest and stops enumerating soonest — 46 of its 100 records stop at exactly 3 claims. Joint moves the other way because at `fp = 0.0` its budget was partly consumed by non-terminating chains (mean 7.19 claims/query against a median of 4.0, and a 731-word claim on `21074975`); removing the loop leaves budget for distinct claims, and its median rises 4.0 → 5.0. **Untested.** A token-level attribution of the penalty across the two stages would settle it and was not run.

**What the inversion costs the C2 argument.** `parity_iter1b` could answer "did words/claim fall because claims got finer or because the answer got shorter?" with post-hoc's 1242 claims against joint's 719 — *"the pass is not the model saying less."* At `fp = 0.5` that sentence is no longer available: post-hoc parses 380 against joint's 456. The granularity gate still passes on the statistic ADR-0009 gates, but the coverage defence of that pass must be re-argued at G2 rather than inherited.

**What this does not license.** It does not set a Gate G2 rate; does not show `fp = 0.5` is optimal (only 0.0 and 0.5 were run at n=100, and 0.3 only on the enriched 12); does not show the penalty is responsible for the *whole* claims/query move, because the `fp = 0.0` reference differs in server window as well as penalty; does not retract W9; does not reopen the parity loop; and does not license quoting any figure here as a parse or quote-location rate.

---

## 7. Unknown / pending

1. **Two confounds on the `fp = 0.0 → 0.5` comparison.** (a) `parity_iter1b` was served at `--max-model-len 14336`, this run at 8192 — the mechanism `parity_iter1b.md` documented (KV-cache block layout and prefill batching change greedy numerics) means some of the difference is server config, not penalty. (b) `parity_iter1b`'s manifest is **backfilled**, and it lists `generation.frequency_penalty` among its `unrecovered` fields: its `fp = 0.0` is read from the `CONFIG_VERSION` default at commit `cbf5727`, not from a recorded run parameter. The *same-server* evidence for the inversion is the 12-question sweep (`fp = 0.0` and `0.5`, both at 8192), which is enriched and small.
2. **The three rejected prompts' true token counts.** The 400 message reports a bound (§4). Measuring them needs `/tokenize` on the live server, which was not reachable from the writing host. The instrumented maximum (`post_hoc_cite` 4545) is a floor on how close the run runs to the window, not the boundary itself.
3. **`parity_report.py` raises on any run containing a rejected call.** The call-rejection guard writes `output_tokens = null` by design and `stage_output_tokens` casts with `int()`. Unfixed; the gate figures in §5 were obtained by calling the library directly.
4. **No `fp` value between 0.0 and 0.5 has been read at n=100.** Whether the inversion crosses zero at 0.3 on the whole split is unmeasured; on the enriched 12 it had not yet crossed (joint 4.0 / post_hoc 6.5).
5. **Citation-F1 at `fp = 0.5` is unmeasured.** The MiniCheck re-read (`citation_f1_minicheck.md`, joint 0.428 vs post-hoc 0.418, delta +0.011 [−0.117, +0.137]) was computed on `parity_iter1b`'s `fp = 0.0` records. Whether a 69% reduction in post-hoc claim volume moves that contrast is open, and it is the question the Sep 6 G2 run answers.
6. **Guided JSON remains unexecuted on the arm that needs it.** It was not introduced here. It *is* already exercised on the other side of the pipeline — `decompose_guided_v2` (n=100, `scripts/decompose_smoke.py`, `fp = 0.5`, guided `recite_json`) reads `quote_not_found = 0` on all three granularity rows and `quote_located_rate = 1.0` on the two atomic rows (the `sentence` row records `null`) — but that is the `decompose.py` re-citation path over pre-existing answers, not the post-hoc generation arm measured here, and the result does not transfer to it. The quote-drift residual that holds both cited generation arms far below G2's 0.95 bar is the defect the generation-side guided path is meant to reach, and it is untried.

---

## What changed in the repository as a result of this run

**Nothing.** No prompt, parser, config default, or gate figure moved. This is a diagnostic re-read discharging the open item `generate_fp_sweep.md` §"The parity consequence" left behind: the inversion it flagged for a larger sample is reproduced at n=100 and classified above. Full suite at the time of writing: **447 passed**.
