# HANDOFF — 2026-08-20 (end of twentieth session)

Snapshot for resuming in a fresh session. Regenerate wholesale; **do not append** — a stale line
here is worse than a missing one, because the next session will trust it.

`main`. This session ran the joint arm under guided decoding for the first time, re-read
citation F1 with both arms guided (contrast C2 now excludes zero), and repeated the mandatory
W9 stratified check on that same run — which now **fails**. Gate G2 is closer but not signed
off: two of its own preconditions (valid parse rate, W9 parity) fail on the run that carries
the winning citation-F1 delta.

Tests: `uv run python -m pytest tests/ -q` → **493 passed** in ~18–21 s. `pyproject.toml`'s
`pythonpath` is `["src", "scripts"]`. **Use `python -m pytest`** — bare `uv run pytest` fails with
`Failed to spawn: pytest`.

Frozen digests (unchanged this session):

| pin | value |
|---|---|
| `decompose.decompose_template_digest()` | `4129a884c7a1b4854739ffe2e3900a1db39626db7fd1bf076d6b549027a7d737` |
| `prompts.post_hoc_answer_template_digest()` | `91bc7dddd62db4d6d37c26a91f05f938b22dafcca7a6e5aed4509c714f25ac1a` |
| `CONFIG_VERSION` / `GenerationConfig.frequency_penalty` | `1.5.0` / `0.5` |

---

## 1. Where the project is

**W3, Phase 1. The GPU run that Gate G2 was blocked on is done. Two new blockers replace it,
both inside the joint arm's guided-JSON schema.**

| Gate | Date | State |
|---|---|---|
| **G0** — 8B AWQ generator chosen | Aug 4 | **PASSED 2026-08-04.** |
| **G1** — hit@10 ≥ 0.90, Wilson lower > 0.85 (ADR-0015) | Aug 23 | **PASSED 2026-08-10.** hit@10 = 0.9400, Wilson lower 0.8752. |
| **G2** — citation-F1 contrast + per-claim parse | **Sep 6** | **BLOCKED — two of its own preconditions fail.** Both arms are now guided on `generate_fp05_n100_guided_both`. Citation F1: joint 0.6137 vs post-hoc 0.5055, delta **+0.1083 [+0.0432, +0.1722], excludes zero.** But joint valid-parse rate is **89/100** (< 95% bar), and the W9 stratified check **fails** on this same run. |
| **W9 stratified check** (ADR-0009 §5) | before G2 sign-off | **RUN on the Gate G2 candidate run and FAILS.** Pooled gap +21.4% against ±15%; 2 of 3 schemes fail. |
| G3 · G4 · G5 | Sep 20 · Sep 27 · Oct 11 | Unstarted, with due weeks. |

---

## 2. What happened this session

### 2.1 Served window raised to 14336, guard aligned

`serve_8b.sh` on the A4000 now runs `--max-model-len 14336` (was 8192). Confirmed live against
`/v1/models`. `_MODEL_MAX_LEN` in `src/biomedqa/backends.py` was raised to match — the
client-side pre-flight guard was still checking against 8192, which would have silently
diverged from what the server actually enforces. Two `tests/test_backends.py` cases were
resized to exercise the new 14336 bound. `scripts/_remote.py` grew a `--get` mode (WSL guest
path → local file) to pull harvest artifacts back to the writing host; `--put` was already
there for the reverse direction.

### 2.2 WSL2 VM idle-shutdown was silently killing every long job

**This was the real blocker, not GPU capacity.** Every `systemd-run --user` job launched over
SSH → `wsl.exe` died 1–3 minutes after the *SSH command that launched it* returned, because
Windows tears down the whole WSL2 lightweight VM (not just the systemd user session — the
`journalctl` boot ID changes) once no `wsl.exe` client is attached, regardless of what's
running inside. `loginctl show-user user` reporting `Linger=yes` does not prevent this; linger
only protects the *systemd user session*, not the *VM the session runs inside*. Three runs
died this way before the cause was found (killed by a `Prompt window exceeded` bug first, then
twice by this). Fixed by writing `C:\Users\user\.wslconfig`:

```
[wsl2]
vmIdleTimeout=-1
```

followed by `wsl --shutdown` to apply it. **This file is not tracked in git** — it lives on the
Windows host, not inside the WSL guest or this repo. If a fresh Windows install or a wiped user
profile loses it, long GPU jobs will silently die again after 1–3 minutes with no error in the
job's own log. Verify with `type C:\Users\user\.wslconfig` before trusting a `systemd-run` job
to survive a multi-minute gap between polls.

### 2.3 Joint arm measured under guided decoding for the first time

Run `generate_fp05_n100_guided_both`, `git_sha 958cf17`, `config_hash 4ea12ab3eae4`:

| arm | clean parses | `quote_not_found` | call failures |
|---|---|---|---|
| joint | 89/100 | 0 | 11 (malformed JSON despite guided decoding) |
| post_hoc | 99/100 | 0 | 0 |
| vanilla | 99/100 | 0 | 0 |

Up from 34/100 unguided, but under the ≥95% Gate G2 bar. The 11 failures are malformed-JSON
replies from the guided decoder (e.g. `Expecting property name enclosed in double quotes`),
not schema violations — a decoder-side truncation/formatting issue that needs its own fix
before a Gate G2 run of record. Traces in `docs/harvest/generate_fp05_n100_guided_both.run.log`.

### 2.4 Citation F1 re-read with both arms guided — C2 now excludes zero

```
uv run python scripts/citation_contrast.py docs/harvest/generate_fp05_n100_guided_both --threshold 0.5
```

| arm | precision | recall | citation F1 | claims |
|---|---|---|---|---|
| joint | 0.9308 | 0.4578 | 0.6137 | 391 |
| post_hoc | 0.9514 | 0.3441 | 0.5055 | 555 |

Delta **+0.1083 [+0.0432, +0.1722]**, 89 paired queries, 11 dropped (zero claims in joint arm),
10000 resamples clustered on question, seed 0. This is the first read where the interval
excludes zero. Against the earlier confounded read (`generate_fp05_n100_guided_batched`, joint
unguided): delta moved from +0.0094 [−0.0536, +0.0729] to +0.1083 [+0.0432, +0.1722] — removing
the decoding confound changed the finding's direction of support, not just its magnitude.
Written up in `docs/harvest/joint_citation_f1_fp05_both_guided.md`; artifact at
`docs/harvest/generate_fp05_n100_guided_both.citation_f1.minicheck.json`. Still a diagnostic
reading, not a gate figure, because of §2.3's 89% parse rate.

### 2.5 W9 stratified check repeated on the Gate G2 candidate run — FAILS

```
uv run python scripts/w9_stratified_parity_report.py docs/harvest/generate_fp05_n100_guided_both --max-tokens 3584
```

Pooled gap **+21.4%** against ±15% (was +13.3% PASS on the unguided-joint batched run).
Compound-structure scheme fails (simple stratum +23.1%), query-claim-volume scheme fails (both
powered strata breach), claim-length scheme passes (but bins by the quantity it compares, per
the standing limitation). Guiding the joint arm's JSON schema shortened its median claim (15.0
→ 14.0 words) while post-hoc held flat at 17.0, widening a gap that used to clear tolerance.
Written up in `docs/harvest/w9_stratified_parity_both_guided.md`. **This means the citation-F1
delta in §2.4 was measured on claims that are not length-parity-matched** — part of the +0.1083
delta may reflect shorter, more precise joint claims rather than attribution quality alone.

---

## 3. Open items, in priority order

1. **Fix the joint arm's 11 malformed-JSON call failures** so valid parse rate reaches ≥95%.
   The guided-JSON constraint is applied but the decoder still emits malformed replies on 11%
   of queries; needs its own diagnosis (§2.3, `docs/harvest/generate_fp05_n100_guided_both.run.log`).
2. **Restore claim-length parity in the joint arm's schema.** The W9 check fails because guided
   decoding shortened joint claims relative to post-hoc (§2.5). Needs a claim-length floor or a
   prompt-level nudge in `JOINT_JSON_TEMPLATE` / `build_citation_response_format(..., is_joint=True)`,
   then a repeat of the W9 check.
3. **Re-run Gate G2 once both of the above pass on the same run.** The citation-F1 win (§2.4)
   is real on `generate_fp05_n100_guided_both`, but Gate G2 needs a run where valid-parse-rate
   and W9 both clear their bars simultaneously with the citation-F1 contrast, not three separate
   runs each clearing one bar.
4. Goals 9 (G3 verifier AUROC) and 10 (G4 gold annotation) are unstarted.

---

## 4. Standing state & operational rules

- Long jobs on the A4000 MUST run under `systemd-run --user --unit=<name>`; sanity artifacts
  belong **outside** the repo, because `harness.git_sha()` stamps `-dirty` on untracked files
  and a Gate G2 manifest must be reproducible from a commit.
- **Confirm `C:\Users\user\.wslconfig` has `vmIdleTimeout=-1` before trusting any job that
  needs to survive a multi-minute gap between polls** (§2.2). This is a Windows-host file, not
  part of this repo or the WSL guest filesystem; nothing here restores it automatically.
- The box: repo at `/home/user/BioMedical_QA`, vLLM in `~/venvs/vllm-server`, `vllm-8b.service`
  runs `/home/user/serve_8b.sh` (now `--max-model-len 14336`), measurement unit
  `biomedqa-run.service` runs `/home/user/run_measure.sh`. Copy-paste only, **one line per
  command**.
- Remote helper: `uv run --with paramiko python scripts/_remote.py 'wsl.exe -d Ubuntu-24.04 -- bash -lc "bash /home/user/status.sh"'`.
  `--get <remote> <local>` pulls a file back (new this session); `--put <local> <remote>` pushes one.
- Prompts are frozen (ADR-0009 §8). The joint JSON template is a **new** stage, not an edit to a
  frozen one: `decompose_template_digest()` and `post_hoc_answer_template_digest()` are
  unchanged, so a claim-length fix to `JOINT_JSON_TEMPLATE` is in scope without a new ADR.
- **`Upcoming_goals.md` is the live target list.** Keep it current, in ASD-STE100 STE.
- Pushing to `origin/main` needs no permission (`CLAUDE.md`); always `git pull --rebase` first.

---

## 5. A4000 commands (copy-paste, one line each)

**On the A4000, inside WSL2 Ubuntu-24.04.** Run them in order and check each before the next.

Confirm the idle-timeout fix is still in place (§2.2) — if this prints nothing or an old value,
redo it before launching anything long:

```
type C:\Users\user\.wslconfig
```

Confirm the checkout is clean before a run:

```
cd /home/user/BioMedical_QA && git pull --rebase && git status --porcelain
```

Confirm the server is up on the 14336 window:

```
curl -s http://localhost:8000/v1/models | grep -o '"max_model_len":[0-9]*'
```

Launch a run under its own unit (adjust `--out-prefix` per attempt):

```
systemctl --user reset-failed <unit-name>.service 2>/dev/null; systemd-run --user --unit=<unit-name> --working-directory=/home/user/BioMedical_QA --setenv=HOME=/home/user bash -lc 'uv run python scripts/generate_smoke.py --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --base-url http://localhost:8000 --n 100 --contexts docs/harvest/dev_contexts_top10.jsonl --max-tokens 3584 --guided-decoding --out-prefix docs/harvest/<prefix> --overwrite > /home/user/<unit-name>.log 2>&1'
```

Watch it (safe now that the VM does not auto-shutdown):

```
systemctl --user status <unit-name>.service --no-pager
```

**On the writing host, after pulling artifacts with `scripts/_remote.py --get`** — the contrast
(cache is committed, ~10 min cold / seconds warm), then the mandatory stratified check:

```
uv run python scripts/citation_contrast.py docs/harvest/<prefix> --threshold 0.5
```

```
uv run python scripts/w9_stratified_parity_report.py docs/harvest/<prefix> --max-tokens 3584
```
