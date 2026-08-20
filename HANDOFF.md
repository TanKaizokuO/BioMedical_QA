# HANDOFF — 2026-08-20 (end of twenty-first session)

Snapshot for resuming in a fresh session. Regenerate wholesale; **do not append** — a stale line
here is worse than a missing one, because the next session will trust it.

`main`. This session fixed the joint arm's malformed-JSON parse-rate defect (bounded
escape-valve retry + claim-length target instruction), re-ran the Gate G2 dev-set benchmark
(`generate_fp05_n100_guided_v5`), and re-measured both post-hoc criteria. Two of Gate G2's three
preconditions now pass on the same run — valid claim parse rate ($95/100$) and the citation-F1
contrast (excludes zero) — but the mandatory W9 stratified robustness check (ADR-0009 §5) still
**fails**, pooled gap $+21.4\%$ against $\pm15\%$. Gate G2 sign-off is refused.

Tests: `uv run python -m pytest tests/ -q` → **495 passed** in ~20 s. `pyproject.toml`'s
`pythonpath` is `["src", "scripts"]`. **Use `python -m pytest`** — bare `uv run pytest` fails with
`Failed to spawn: pytest`.

Frozen digests (unchanged this session):

| pin | value |
|---|---|
| `decompose.decompose_template_digest()` | `4129a884c7a1b4854739ffe2e3900a1db39626db7fd1bf076d6b549027a7d737` |
| `prompts.post_hoc_answer_template_digest()` | `91bc7dddd62db4d6d37c26a91f05f938b22dafcca7a6e5aed4509c714f25ac1a` |
| `CONFIG_VERSION` / `GenerationConfig.frequency_penalty` | `1.5.0` / `0.5` |

`JOINT_JSON_TEMPLATE` (`src/biomedqa/prompts.py`) is **not** frozen under ADR-0009 §8 — it is a
new stage, not an edit to the decomposer or post-hoc templates above, so tuning it for
guided-decoding defects and length-target parity stays in scope without a new ADR.

---

## 1. Where the project is

**Gate G2 is one precondition away from sign-off. That precondition is the W9 stratified
robustness check, which improved this session but still fails.**

| Gate | Date | State |
|---|---|---|
| **G0** — 8B AWQ generator chosen | Aug 4 | **PASSED 2026-08-04.** |
| **G1** — hit@10 ≥ 0.90, Wilson lower > 0.85 (ADR-0015) | Aug 23 | **PASSED 2026-08-10.** hit@10 = 0.9400, Wilson lower 0.8752. |
| **G2** — citation-F1 contrast + per-claim parse + W9 | **Sep 6** | **BLOCKED — one of three preconditions fails.** On `generate_fp05_n100_guided_v5`: joint valid-parse rate **95/100** (meets ≥95%), citation F1 joint **0.6142** vs post-hoc **0.5209**, delta **+0.0933 [+0.0259, +0.1613], excludes zero** (met). W9 stratified check **fails**, pooled gap +21.4% against ±15% (not met). |
| **W9 stratified check** (ADR-0009 §5) | before G2 sign-off | **RUN on `generate_fp05_n100_guided_v5` and FAILS.** Pooled gap +21.4% (improved from +30.8% on the intermediate prompt-only attempt). Compound-structure and claim-length schemes now PASS; query-claim-volume scheme still FAILS on the 1–5-claims stratum (+21.4%). |
| G3 · G4 · G5 | Sep 20 · Sep 27 · Oct 11 | Unstarted, with due weeks. |

---

## 2. What happened this session

### 2.1 Diagnosed and fixed the joint arm's malformed-JSON parse failures

Root cause: an xgrammar whitespace "death loop". At greedy decoding (`temperature=0.0`), the
guided JSON grammar could walk into unbounded indentation-token runs inside JSON strings,
burning the entire 3584-token completion cap on tabs before closing the object. This produced
0-claim, no-decision replies with a JSON parse error.

Two mitigations tried and rejected before landing on the fix:

- **Server-side `--structured-outputs-config '{"disable_any_whitespace": true}'`** on the A4000
  vLLM host. Crashes vLLM 0.26.0's xgrammar backend on startup (`structural_tag.py`
  `SequenceFormat.model_rebuild()` raises a pydantic-core `SchemaError`). Reverted.
- **Prompt-only mitigation** (anti-filler instruction + compact-JSON formatting rule in
  `JOINT_JSON_TEMPLATE`). Cut an 11-query smoke test's failures from 11 to 2, but a full
  100-query run (`v3`) showed the edits just moved the death-loop trajectory to different
  queries: 15/100 parse failures, worse than the 11/100 baseline.

Landed fix — a **bounded escape-valve retry** in `generate_one`'s `System.JOINT` guided branch
(`src/biomedqa/generate.py`): a zero-claim, no-decision reply with a malformed-JSON error at
`temperature=0.0` retries up to twice, at `temperature=0.3` then `0.7`. A successful retry logs
to `recovered` (clean-parse stats stay intact); exhausting both retries logs to `errors`.
`scripts/generate_smoke.py`'s and `tests/test_generate.py`'s stage-count checks were relaxed to
`joint stages_seen >= 1` (vanilla stays strict `== 1`) because a retried call adds an extra
completion stage.

### 2.2 Added a claim-length target and ran the Gate G2 candidate (`v5`)

`JOINT_JSON_TEMPLATE` also grew a target-length instruction (15–20 words/claim, noting under-10
usually misses qualifying detail), aimed at the W9 claim-length parity gap the prior session
found (§2.5 of the previous handoff). Full $n=100$ dev run `generate_fp05_n100_guided_v5`,
`git_sha 045a96c`, `config_hash b1d8a1c7d4f8`:

| arm | clean parses | `quote_not_found` | call failures | recovered notes |
|---|---|---|---|---|
| joint | 95/100 | 0 | 0 | 24 (escape-valve retries; most single-retry) |
| post_hoc | 99/100 | 0 | 0 | 0 |
| vanilla | 99/100 | 0 | 0 | 0 |

Stage-count check and call-failure check both PASS. Two queries (`25982163`, `26399179`)
exhaust both retries and still fail — the escape valve reduces but does not eliminate the
death loop; 95/100 clears the ≥95% Gate G2 bar with three points of headroom, not zero.

### 2.3 Citation F1 contrast (C2) re-read on `v5` — still excludes zero, delta narrower

```
uv run python scripts/citation_contrast.py docs/harvest/generate_fp05_n100_guided_v5
```

| arm | precision | recall | citation F1 | claims |
|---|---|---|---|---|
| joint | 0.9653 | 0.4503 | 0.6142 | 433 |
| post_hoc | 0.9533 | 0.3583 | 0.5209 | 600 |

Delta **+0.0933 [+0.0259, +0.1613]**, 96 paired queries, 4 dropped (zero claims in joint arm —
the two exhausted-retry queries plus two more with a 0-claim result), 10000 resamples clustered
on question, seed 0. Narrower than the prior guided-both read (+0.1083 [+0.0432, +0.1722]) but
still clears zero. Artifact: `docs/harvest/generate_fp05_n100_guided_v5.citation_f1.minicheck.json`.

### 2.4 W9 stratified check re-read on `v5` — improved, still FAILS

```
uv run python scripts/w9_stratified_parity_report.py docs/harvest/generate_fp05_n100_guided_v5 --max-tokens 3584
```

Pooled gap **+21.4%** against ±15% (joint median 14.0 w/c, post-hoc median 17.0 w/c) — improved
from +30.8% on the intermediate prompt-only `v3`/`v4` attempt, but still over tolerance.

| scheme | verdict | detail |
|---|---|---|
| compound_structure | **PASS** (2/2) | simple +14.3%, compound +11.8% |
| claim_length | **PASS** (5/5) | all five bands inside tolerance, largest -10.8% |
| query_claim_count | **FAIL** (1/2 powered) | 1–5 claims +21.4% FAIL, 6–10 claims +14.3% PASS, 11+ underpowered (0 queries) |

The claim-length target instruction fixed the compound-structure and claim-length schemes
outright but only partly closed the query-claim-volume gap — the 1–5-claims stratum, which
holds 78 of 100 queries, is where the remaining gap lives. Artifact:
`docs/harvest/generate_fp05_n100_guided_v5.w9_stratified_parity.txt`.

### 2.5 Gate G2 verdict: sign-off refused

Per ADR-0009 §5 and the standing Gate G2 criteria (parse rate ≥95%, citation-F1 CI excludes
zero, W9 stratified check passes — all on the same run): `generate_fp05_n100_guided_v5` clears
the first two and fails the third. Gate G2 remains **BLOCKED**. `Upcoming_goals.md` goals 4
(closed this session), 8, and 11 were updated to reflect this.

---

## 3. Open items, in priority order

1. **Close the remaining W9 gap** in the 1–5-claims stratum (+21.4%) and the pooled gate
   (+21.4%). The claim-length target instruction alone was not enough; needs either a stronger
   nudge (explicit minimum-word floor per claim in the schema/prompt) or a matching change on
   the post-hoc side, then a repeat of the W9 check. This is the only remaining Gate G2
   precondition (`Upcoming_goals.md` goal 11).
2. **Re-run Gate G2** once goal 11 passes on the same run as the parse-rate and citation-F1
   criteria, then sign off (`Upcoming_goals.md` goal 8).
3. Goals 9 (G3 verifier AUROC) and 10 (G4 gold annotation) are unstarted.

---

## 4. Standing state & operational rules

- Long jobs on the A4000 MUST run under `systemd-run --user --unit=<name>`, or under a
  detached/persisted process this session confirmed survives without `systemd-run` — the `v5`
  run in this session ran as a plain background process behind an SSH port-forward tunnel and
  survived an 18-minute unattended wait; if reusing that path, confirm the tunnel process is
  still alive with `curl -s http://127.0.0.1:8000/v1/models` before trusting a long wait.
- Sanity artifacts belong **outside** the repo, because `harness.git_sha()` stamps `-dirty` on
  untracked files and a Gate G2 manifest must be reproducible from a commit.
- **Confirm `C:\Users\user\.wslconfig` has `vmIdleTimeout=-1`** before trusting any A4000 job
  that needs to survive a multi-minute gap between polls (prior-session finding, unchanged).
- The box: repo at `/home/user/BioMedical_QA`, vLLM in `~/venvs/vllm-server`, `vllm-8b.service`
  runs `/home/user/serve_8b.sh` (`--max-model-len 14336`), measurement unit
  `biomedqa-run.service` runs `/home/user/run_measure.sh`. Copy-paste only, **one line per
  command**.
- Prompts are frozen (ADR-0009 §8) except `JOINT_JSON_TEMPLATE`, which is a new stage and stays
  in scope for guided-decoding defect fixes and length-target tuning — confirmed unchanged this
  session: `decompose_template_digest()` and `post_hoc_answer_template_digest()` both match the
  pinned values above.
- **`Upcoming_goals.md` is the live target list.** Keep it current, in ASD-STE100 STE.
- Pushing to `origin/main` needs no permission (`CLAUDE.md`); always `git pull --rebase` first.

---

## 5. Commands to resume

Confirm the server and tunnel are still reachable:

```
curl -s http://127.0.0.1:8000/v1/models
```

Re-run the contrast and stratified check on any new candidate run:

```
uv run python scripts/citation_contrast.py docs/harvest/<prefix>
```

```
uv run python scripts/w9_stratified_parity_report.py docs/harvest/<prefix> --max-tokens 3584
```

Full test suite before committing:

```
uv run python -m pytest tests/ -q
```
