# HANDOFF — 2026-08-17 (end of eighteenth session)

Snapshot for resuming in a fresh session. Regenerate wholesale; **do not append** — a stale line here is worse than a missing one, because the next session will trust it.

`main` · **working tree clean as of this commit.** The run of record for this session, `generate_fp05_n100`, was produced at **`9a7cb94`** with a clean tree (its manifest `git_sha` carries no `-dirty` suffix). Two documentation commits follow it: `fcfbd4b` (ROADMAP annotator confirmation) and this one, which lands the four `generate_fp05_n100.*` artifacts plus the diagnostic report.

Tests: `uv run python -m pytest tests/ -q` → **447 passed** in 15.7 s (unchanged from the seventeenth session; no code changed this session). `pyproject.toml`'s `pythonpath` is `["src", "scripts"]`. **Use `python -m pytest`** — bare `uv run pytest` and `uv run --group dev pytest` both fail with `Failed to spawn: pytest`.

Frozen digests, verified this session:

| pin | value |
|---|---|
| `decompose.decompose_template_digest()` | `4129a884c7a1b4854739ffe2e3900a1db39626db7fd1bf076d6b549027a7d737` |
| `prompts.post_hoc_answer_template_digest()` | `91bc7dddd62db4d6d37c26a91f05f938b22dafcca7a6e5aed4509c714f25ac1a` |
| `prompts.PARITY_LOOP_CLOSED` | 2026-08-14, `parity_iter1b`, 1 of 10 iterations, +13.3% [+0.0%, +14.3%], `residual_favours_c2=True` |
| `CONFIG_VERSION` / `GenerationConfig.frequency_penalty` | `1.5.0` / `0.5` |

---

## 1. Where the project is

**W3 (Aug 17–23), Phase 1. The runaway-claim closure adopted last session was re-read on the whole dev split this session: it holds, the ADR-0009 gated quantity is unmoved at +13.3%, and the claims/query inversion it flagged is confirmed. No prompt, parser, or default moved.**

| Gate | Date | State |
|---|---|---|
| **G0** — 8B AWQ generator chosen, A4000 measured | Aug 4 | **PASSED 2026-08-04.** Issue #1 closed. |
| **G1** — hit@10 ≥ 0.90, Wilson lower > 0.85 (ADR-0015) | Aug 23 | **PASSED 2026-08-10.** hit@5 = 0.86, **hit@10 = 0.9400** (Wilson lower 0.8752). |
| **G2** — citation-F1 contrast + per-claim parse | **Sep 6** | **PATHOLOGY CLOSED IN CODE (Aug 17, ADR-0021) AND CONFIRMED AT n=100 (Aug 17). CONTRAST OPERATING POINT STILL PENDING THE GATE RUN.** Re-read on `parity_iter1b` at MiniCheck φ gave **joint 0.428 vs post-hoc 0.418, delta +0.011 [−0.117, +0.137]** — interval crosses zero, C2 not established. **New this session:** on all 100 dev questions at the adopted `fp = 0.5`, clean-parse rates are **joint 34/100, post_hoc 23/100, vanilla 99/100**, so the ≥0.95 valid-parse half of G2 is *not* met at this configuration and quote drift is the dominant term. |
| G3 · G4 · G5 | Sep 20 · Sep 27 · Oct 11 | Unstarted, with due weeks. |

---

## 2. What happened this session

### 2.1 The n=100 `fp = 0.5` re-read was executed (the only measurement this session)

`scripts/generate_smoke.py --base-url http://localhost:8000 --n-questions 100 --contexts docs/harvest/dev_contexts_top10.jsonl --max-claim-words 50`, seed 0, `--max-tokens 3584`, `temperature 0.0`, vLLM `--max-model-len 8192` on the A4000. 300 records, 400 cost rows, 43.7 min of summed call wall time, `provenance.kind = "live"`, `config_hash 246dfc4897fe` at `CONFIG_VERSION 1.5.0`, `index_fingerprint 57ab89e445f8`, `split_hash 71c46cc5b0ca`.

Artifacts: `docs/harvest/generate_fp05_n100.{summary.json,manifest.json,records.jsonl,costs.jsonl}`. Written up in **`docs/harvest/claims_query_inversion_n100.md`**.

Two facts about *what* was run, because both are easy to misread later:

- **`fp = 0.5` came from the new default, not a flag.** The summary's `run_arguments` carries only `base_url`, `n_questions`, `contexts`, `max_claim_words`. This run reads the adopted configuration.
- **The 100 questions are the entire frozen dev split** (`split_hash 71c46cc5b0ca`) and are *exactly* `parity_iter1b`'s query set, so the `fp = 0.0` → `fp = 0.5` comparison is within-query on an identical slice. The 12-question `fp` sweep slice is a subset of it.

### 2.2 Guided JSON was NOT introduced

The three arms ran the frozen free-text grammar. `generate_one` has no guided path; `scripts/generate_smoke.py` never touches `build_citation_response_format`; the `CITE line has no '||' separator` errors in the artifacts are only producible by the free-text `cite` stage. `guided_decoding=True` exists **only** on `generate.cite_claims`, the `decompose.py` re-citation path, which this run never calls.

The guided path is measured *there* and only there: `decompose_guided_v2` (n=100, Aug 16, `fp = 0.5`) reads `quote_not_found = 0` on all three granularity rows and `quote_located_rate = 1.0` on the two atomic rows. **That result does not transfer to the post-hoc generation arm**, which is where G2's contrast is read and where quote drift is still uncontrolled.

### 2.3 Documentation

- **`docs/harvest/claims_query_inversion_n100.md`** — the formal diagnostic report: classification, per-arm table, the paired statistics, the four-run comparison, the three parity bases, the call-failure forensics, the error taxonomy, and separated Interpretation / Unknown sections.
- **`ROADMAP.md`** — W5 re-read milestone marked completed; status summary updated.
- **`HANDOFF.md`** — this file, regenerated.

### 2.4 Carried in from `9a7cb94` (committed at the start of this session, exploratory)

`9a7cb94` added `.omp/config.yml`, nine `scripts/_check_*.py` vLLM/NVML environment probes, `scripts/_search_*.py` helpers, `scripts/_remote_{dl,serve}_qwen.sh` (Qwen2.5-14B-Instruct-AWQ at `--max-model-len 14336`), `scripts/_remote_run_cap.sh`, and `scripts/_probe_guided.py` (a 5-question guided-vs-control probe). **No artifacts from any of them are committed and no gate figure depends on them.** They are exploration, not measurement — in particular `_remote_run_cap.sh` names an output prefix `docs/harvest/decompose_cap` that does not exist, so that run either was not made or was not kept.

---

## 3. Measured evidence (facts)

### 3.1 The run, per arm

| | joint | post_hoc | vanilla |
|---|---|---|---|
| records / stages | 100 / 1 | 100 / 2 | 100 / 1 |
| clean parses (no error string) | **34** | **23** | **99** |
| call failures | **1** | **2** | 0 |
| median / mean claims per query | **5.0** / 4.56 | **4.0** / 3.80 | 5.0 / 5.59 |
| total claims / citations | 456 / 677 | 380 / 563 | 559 / 0 |
| recovered notes | 323 | 168 | 0 |
| `quote_not_found` (strings / records touched) | 154 / 55 | 172 / 62 | 0 / 0 |
| median / mean / max words per claim | **15.0** / 15.53 / 40 | **17.0** / 17.55 / 37 | 20.0 / 20.84 / 51 |
| claims > 50 words | 0 | 0 | 1 |
| runaway chains ≥ 3 | **0** | **0** | **0** |
| median latency | 9.28 s | 12.92 s | 2.69 s |

Runaway pathology absent across all 300 records: zero chains ≥ 3, one length-2 pair charged to `recovered` (`post_hoc` `14599616`). The single over-length claim is vanilla `24315783` `c5` at 51 words — one over `MAX_CLAIM_WORDS = 50`, and it contains commas, so it is *not* the punctuation-free unit the earlier sweep recorded.

### 3.2 The claims/query inversion — **REPRODUCED**

Paired over the 100 queries, joint − post_hoc: mean delta **+0.760**, median delta **+1.0**, $d_z = 0.396$; **54 joint > post_hoc, 24 post_hoc > joint, 22 ties**.

| test | p |
|---|---|
| paired *t* (t = 3.9622, df = 99) | **1.40e-4** |
| Wilcoxon signed-rank (W = 790.5) | **1.40e-4** |
| sign test (54 vs 24) | **9.01e-4** |

Direction and strength across the runs on record (median claims/query, joint / post_hoc):

| run | n | fp | window | joint / post_hoc | ordering |
|---|---|---|---|---|---|
| `parity_iter1b` | 100 (dev) | 0.0 | 14336 | 4.0 / **10.0** | post_hoc higher |
| `generate_fp_sweep_fp00` | 12 (enriched) | 0.0 | 8192 | 4.5 / **9.0** | post_hoc higher |
| `generate_fp_sweep_fp03` | 12 (enriched) | 0.3 | 8192 | 4.0 / **6.5** | post_hoc higher |
| `generate_fp_sweep_fp05` | 12 (enriched) | 0.5 | 8192 | **5.5** / 4.0 | joint higher |
| **`generate_fp05_n100`** | **100 (dev)** | **0.5** | **8192** | **5.0** / 4.0 | **joint higher** |

At `fp = 0.0` on the same 100 queries the opposite ordering is *stronger* (mean delta −5.23; 9 vs 83; sign test p = 3.92e-16), so the flip is significant at both ends. The n=100 effect is **less than half** the n=12 estimate (+0.760 vs +1.667) with far stronger evidence — smaller and better established than the enriched slice suggested. Excluding the two queries touched by call failures: mean delta +0.694, p = 2.21e-4 / 2.31e-4 / 1.26e-3, medians unchanged.

### 3.3 ADR-0009 gated parity: **+13.3%, PASS on all three bases**

| basis | queries | joint | post_hoc | gap | gate |
|---|---|---|---|---|---|
| all 100 records | 100 / 100 | 15.0 | 17.0 | **+13.3%** | **PASS** |
| untruncated per arm | 100 / 100 | 15.0 | 17.0 | **+13.3%** | **PASS** |
| untruncated, same queries both arms | 100 | 15.0 | 17.0 | **+13.3%** | **PASS** |

The three bases coincide because **nothing truncated**: 0 of 397 completed calls reached the 3584 cap, largest completion 1396 tokens (39% of cap). `parity_iter1b` at the same cap had 8 joint / 10 post-hoc-answer / **16 post-hoc-cite** / 7 vanilla calls pinned at the cap. Cite-stage truncation **16 → 0**. The figure lands exactly on the published `parity_iter1b` pair (15 / 17, +13.3%) and on the 12-question `fp = 0.5` reading.

Two qualifications that do not move the headline but must travel with it:

1. **The query-level bootstrap widened and now straddles:** +13.3%, 95% **[+6.3%, +21.4%]** (4000 draws, seed 0) against `parity_iter1b`'s [+0.0%, +14.3%]. Cause is arithmetic: 836 claims here against 1961 there, on an integer median where one word is ~6.7% and the tolerance is two words wide.
2. **`parity_report.py`'s coverage warning fires:** post-hoc's median claims/query is 4.0, under the script's 8.0 floor. The pass is no longer accompanied by "post-hoc parses more claims than joint" (380 against joint's 456; at `fp = 0.0` it was 1242 against 719).

**The parity loop is untouched:** no prompt edited, digests unchanged, `PARITY_LOOP_CLOSED` intact, ledger still 1 of 10, this run charges no iteration. `requires_w9_robustness_check` evaluates `False` only because the gate passes — **ADR-0009 §5's W9 stratified check remains mandatory**; it was triggered at iteration 0 and a passing run does not retract a pre-registered check.

### 3.4 Call failures: 3 of 400 calls, static window overflow

| query | system | call | latency | consequence |
|---|---|---|---|---|
| `17224424` | joint | call 1 (its only call) | 0.07 s | record lost: 0 claims, empty `raw_generation` |
| `17224424` | post_hoc | call 2 (cite) | 5.25 s | stage 1 answered, 0 claims parsed |
| `12607666` | post_hoc | call 2 (cite) | 5.90 s | stage 1 answered, 0 claims parsed |

All three carry the byte-identical 400: *"maximum context length is 8192 tokens … you requested 3584 output tokens and your prompt contains at least 4609 input tokens, for a total of at least 8193"*. The joint failure was rejected in 0.07 s on its first and only call, so **no generated text existed to inflate the prompt** — this is a static context overflow, unlike the `fp = 0.0` sweep failure, which a runaway stage-1 answer caused.

Measured per-call prompt maxima against the 4608-token budget a 3584 cap leaves in an 8192 window: joint 4355, post_hoc_answer 4488, **post_hoc_cite 4545**, vanilla 4356. **The largest call that succeeded is 63 tokens below the boundary.** `parity_iter1b` saw none of this because it was served at 14336. `backends.py` still does not check `prompt_tokens + max_tokens` against the served window before posting (the gap `parity_iter0.md` deferred to W5 cleanup), and `call_failure_check` in the summary reports `passed: false`.

### 3.5 Clean-parse composition

Error strings by kind (count / records touched):

| kind | joint | post_hoc | vanilla |
|---|---|---|---|
| `quote_not_found` | **154 / 55** | **172 / 62** | 0 |
| `CITE line has no '\|\|' separator` | 41 / 17 | 13 / 9 | 0 |
| `CLAIM line carries no claim number` | 7 / 3 | 47 / 13 | 0 |
| empty claim | 0 | 38 / 8 | 0 |
| cites a passage not in the context | 8 / 4 | 7 / 2 | 0 |
| citations exceed the cap of 3 | 3 / 3 | 4 / 4 | 0 |
| call rejected (+ `no DECISION` / `no CLAIM lines`) | 3 / 1 | 6 / 2 | 0 |
| over `max claim length` | 0 | 0 | 1 / 1 |

`clean_parses` requires an **empty** `errors` list, so one bad `CITE` line disqualifies a record. Quote drift is the dominant term on both cited arms; post-hoc's second family (missing claim numbers, empty claims) is format collapse. **These are per-claim error counts, not rates** — no `quote_located_rate` or `claim_parse_rate` was computed on this run.

### 3.6 Reproducibility check on the shared 12 questions

The `fp` sweep's 12 ids are a subset of these 100. On them, the n=100 run reproduces the sweep exactly on median **and** mean claims per query and longest-claim words for all three arms, and **30 of the 36 generations are byte-identical**. The ±1 clean-parse and `quote_not_found` differences are the KV-cache-layout / prefill-batching nondeterminism `parity_iter1b.md` documented.

**One sweep headline did not survive:** joint `quote_not_found` is 0 on those 12 questions in both runs but **154 over the full 100**, touching 55 records. The enriched slice was clean of joint-side quote drift; the split is not.

---

## 4. Interpretation (not fact)

- **Why post-hoc's count collapses:** `frequency_penalty` taxes tokens by prior frequency, and an enumerated claim list repeats `CLAIM`, `CITE`, `||`, and passage-id scaffolding once per item. Post-hoc's cite stage carries the most repeated structure per item, so it stops enumerating soonest — 46 of its 100 records stop at exactly 3 claims. Joint rises 4.0 → 5.0 because its `fp = 0.0` budget was partly consumed by non-terminating chains. **Untested**; a token-level attribution across the two stages would settle it.
- **What the inversion costs the C2 argument:** `parity_iter1b` could answer "finer, or just shorter?" with post-hoc's 1242 claims against joint's 719. At `fp = 0.5` that sentence is unavailable. The gated statistic still passes; the *coverage* defence of the pass must be re-argued at G2 rather than inherited.
- **Not licensed by any of the above:** a Gate G2 rate; a claim that `fp = 0.5` is optimal; a claim that the penalty alone explains the claims/query move (the `fp = 0.0` reference differs in server window too); any retraction of W9; any reopening of the parity loop.

---

## 5. Unknown / pending / open defects

1. **Two confounds on the `fp = 0.0` → `fp = 0.5` comparison.** `parity_iter1b` was served at `--max-model-len 14336` and this run at 8192 (greedy numerics differ with KV-cache layout); and `parity_iter1b`'s manifest is **backfilled** with `generation.frequency_penalty` listed among its `unrecovered` fields, so its `fp = 0.0` is read from the then-current default, not a recorded parameter. The same-server `fp` evidence is the enriched 12-question sweep.
2. **`parity_report.py` raises on any run containing a rejected call.** `stage_output_tokens` does `int(call.output_tokens)` and the call-rejection guard writes `output_tokens = null` by design → `TypeError`. The three bases in §3.3 were computed by calling `parity_gate` / `arm_granularity` / `gap_bootstrap_ci` directly. **Unfixed defect; blocks the standard gate script on any future run with a rejected call.**
3. **`backends.py` posts without checking `prompt_tokens + max_tokens` against the served window** (§3.4). Recorded since `parity_iter0.md`; still open; now costing real records.
4. **Exact prompt sizes of the three rejected calls are unmeasured.** The 400 reports a bound: $8192 - 3584 + 1 = 4609$, and all three rejections report that identical figure. Measuring needs `/tokenize` on the live server.
5. **No `fp` between 0.0 and 0.5 has been read at n=100.**
6. **Citation-F1 at `fp = 0.5` is unmeasured** — the MiniCheck read used `parity_iter1b`'s `fp = 0.0` records. Whether a 69% reduction in post-hoc claim volume moves the contrast is the question the Sep 6 G2 run answers.
7. **Guided JSON is untried on the generation-side post-hoc cite stage** (§2.2), which is where the quote-drift residual lives.
8. **G2's ≥0.95 valid-parse bar is unmet at this configuration** (34/100 and 23/100 clean) and no decoding value was expected to reach it.
9. **AlignScore (~355M, Table 3 row 2)** still needs an isolated environment (`torch<2`, `pytorch_lightning<2`) and the official `.ckpt`.

---

## 6. Standing state & operational rules

- **Working tree:** clean as of this commit. Full suite **447 passed** via `uv run python -m pytest tests/ -q`.
- **Remote helper:** `uv run --with paramiko python scripts/_remote.py 'wsl.exe -d Ubuntu-24.04 -- bash -lc "bash /home/user/status.sh"'`
- **vLLM service:** `vllm-8b.service` on the A4000, `--max-model-len 8192 --gpu-memory-utilization 0.85` (`docs/harvest/runbooks/wsl-vllm-a4000.md`). It was **not reachable from the writing host** at the end of this session (`localhost:8000` refused), so anything needing `/tokenize` or generation must re-establish it.
- Long jobs MUST run via `systemd-run --user --unit=<name>`; sanity artifacts belong **outside** the repo (`harness.git_sha()` stamps `-dirty` on untracked files, which would make a Gate G2 manifest unreproducible).
- Prompts are frozen; digests in the header. Any prompt edit before **Sep 3** must be justified against ADR-0009 §8, and after it invalidates the gold set.
- **Upcoming Goals:** `Upcoming_goals.md` contains the active targets in STE. Agents MUST keep it updated as targets progress.

---

## 7. Pending next steps

1. **Fix `parity_report.py` against its own call-rejection guard** (§5.2) — it is the gate script and it currently cannot read a run with a rejected call.
2. **Add the pre-flight window check to `backends.py`** (§5.3): refuse or shrink a request whose `prompt_tokens + max_tokens` exceeds the served window instead of surfacing an unattributed 400. Three records were lost to it this session.
3. **Sep 6 Gate G2 benchmark run** — dev set at `fp = 0.5`, `CONFIG_VERSION 1.5.0`. Either raise the server window to 14336 or lower the cap so the 4545-token worst case has headroom; and settle the guided-JSON citation path first, because 34/100 and 23/100 clean parses cannot clear the ≥0.95 bar.
4. **Sep 3 Decomposer & Granularity Freeze** (ADR-0009 §8) — keep prompt digests and parsers stable; the freeze also unblocks `ANNOTATOR_GUIDE.md` §2's worked examples.
5. **AlignScore** environment and checkpoint port (§5.9).
6. **W9 stratified robustness check** remains mandatory (ADR-0009 §5) regardless of the +13.3% pass.
