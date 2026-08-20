# HANDOFF — 2026-08-20 (end of nineteenth session)

Snapshot for resuming in a fresh session. Regenerate wholesale; **do not append** — a stale line
here is worse than a missing one, because the next session will trust it.

`main`. This session landed the joint-arm guided path, the W9 stratified check, the AlignScore
port, and the first `fp = 0.5` paired citation-F1 read, all of which had been sitting uncommitted
in the working tree.

Tests: `uv run python -m pytest tests/ -q` → **493 passed** in 18.6 s. `pyproject.toml`'s
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

**W3, Phase 1. Every G2 prerequisite that can be settled without the A4000 is settled. What
remains is one GPU run and the re-reads that hang off it.**

| Gate | Date | State |
|---|---|---|
| **G0** — 8B AWQ generator chosen | Aug 4 | **PASSED 2026-08-04.** |
| **G1** — hit@10 ≥ 0.90, Wilson lower > 0.85 (ADR-0015) | Aug 23 | **PASSED 2026-08-10.** hit@10 = 0.9400, Wilson lower 0.8752. |
| **G2** — citation-F1 contrast + per-claim parse | **Sep 6** | **BLOCKED ON ONE GPU RUN.** Post-hoc is guided, batched, 99/100 clean. Joint guided decoding is now in code but **unmeasured** — the last run has it at 34/100 clean, 161 `quote_not_found`. Paired citation-F1 at `fp = 0.5` reads **joint 0.5344 vs post-hoc 0.5250, delta +0.0094 [−0.0536, +0.0729]**, interval crosses zero. |
| **W9 stratified check** (ADR-0009 §5) | before G2 sign-off | **RUN AND PASSED** on `generate_fp05_n100_guided_batched`; must be repeated on the G2 run of record. |
| G3 · G4 · G5 | Sep 20 · Sep 27 · Oct 11 | Unstarted, with due weeks. |

---

## 2. What happened this session

### 2.1 W9 stratified robustness check — run, PASSED

`uv run python scripts/w9_stratified_parity_report.py docs/harvest/generate_fp05_n100_guided_batched --max-tokens 3584`

All three pre-registered schemes pass. Compound structure carries the finding: the gap survives
inside **simple** claims (joint 14.0 vs post-hoc 16.0 words, +14.3%), so the residual is verbosity,
not compounding — reproducing `parity_iter0b`. The claim-length scheme is reported with the
limitation that it bins claims by the same quantity it then compares. The `11+ claims/query`
stratum is empty at `fp = 0.5` and gets no verdict.

Written up in **`docs/harvest/w9_stratified_parity.md`**. ADR-0009 §5's obligation on the +13.3%
parity pass is discharged **for that run only**.

### 2.2 First paired citation-F1 read at `fp = 0.5`

`uv run python scripts/citation_contrast.py docs/harvest/generate_fp05_n100_guided_batched --threshold 0.5`

| arm | precision | recall | citation F1 | claims |
|---|---|---|---|---|
| joint | 0.8443 | 0.3909 | 0.5344 | 463 |
| post_hoc | 0.9550 | 0.3620 | 0.5250 | 627 |

Delta **+0.0094 [−0.0536, +0.0729]**, 100 paired queries, 0 dropped, 10000 resamples clustered on
question, seed 0. Against `parity_iter1b` at `fp = 0.0` (delta +0.011 [−0.117, +0.137]): both arms
gained ~0.11 F1 and the interval halved in width, **but the delta did not move**.

This is a diagnostic reading, **not** a gate figure — post-hoc is guided and joint is not, so the
decoding constraint confounds it, and joint is far under the ≥95% parse bar. Written up in
**`docs/harvest/joint_citation_f1_fp05.md`**; machine artifact at
`docs/harvest/generate_fp05_n100_guided_batched.citation_f1.minicheck.json`.

### 2.3 Joint arm guided decoding (code complete, unmeasured)

- `src/biomedqa/generate.py`: a `System.JOINT` guided branch that builds the schema, calls with
  `response_format`, unwraps a JSON-string reply, and parses with `require_decision=True`.
- `src/biomedqa/prompts.py`: `JOINT_JSON_TEMPLATE`, and `build_citation_response_format` grew an
  `is_joint=True` mode — variable claim count (1–30), plus `decision` and per-claim `text`, since
  the joint arm invents its own claims. The fixed-count post-hoc schema is untouched and has a
  regression test that says so.
- **Batching was withdrawn for this arm**, and `Upcoming_goals.md` records why: post-hoc can batch
  because stage 1 produces the claims stage 2 cites, whereas joint emits claims and citations in
  one call and the stage-count check requires it to stay at one call per query. Truncation control
  for joint is the bounded schema plus output-cap headroom.

### 2.4 `scripts/citation_contrast.py` cache defect, fixed

The script scored missing MiniCheck pairs and then **threw them away** — 1113 pairs, 9.6 CPU-minutes,
re-paid on every read. It now writes the enlarged cache back atomically; the second read of the same
contrast took 16 s instead of 578 s and reproduced every figure exactly. The hardcoded "PRE-FIX
baseline" warning was replaced by one derived from the run's own joint parse rate, and the script now
writes `<prefix>.citation_f1.minicheck.json`. Covered by
`test_newly_scored_pairs_are_written_back_to_the_cache`.

### 2.5 Stage-count check and AlignScore

- `scripts/generate_smoke.py`'s stage-count check now demands **at least two** calls per post-hoc
  query (the batched arm makes 153), and exactly one for joint and vanilla.
- AlignScore reference port is complete: `torch 1.13.1+cu117` / `pytorch_lightning 1.9.5` at
  Python 3.10.20, official `AlignScore-large.ckpt`
  (SHA256 `ff4336312b377edcbcdad5694a2d09d73dc4225422c0422d810aa7e78485e32d`), six test pairs, max
  absolute difference **0.0** at `rtol=1e-5, atol=1e-7`. `docs/harvest/alignscore_reference_port.md`,
  `scripts/alignscore_port.py`. **Caveat: the environment lives at `/tmp/alignscore_venv`** and dies
  with the host; the doc carries one-line rebuild commands.

### 2.6 `REVIEW_REPORT.MD` (pushed from the box, `9be5907`) is superseded

It reviews `generate_fp05_n100_guided`, the **pre-batching** run, and both of its blocking issues
are now closed: the 70/100 clean-parse figure was fixed by batching (`generate_fp05_n100_guided_batched`
reads 99/100 post-hoc), and the missing MiniCheck citation-F1 artifact now exists (§2.2). Its
"NOT READY" verdict and its 455-test count are both stale. Do not re-litigate it — read §1 instead.

---

## 3. Open items, in priority order

1. **Measure the joint arm under guided decoding** (Goal 4). Code is in; no run exists. Needs the
   A4000 — §5 has the commands.
2. **Raise the served window to 14336** (Goal 5). The largest stage-2 prompt is 4464 tokens against
   an 8192 window with a 3584 output cap: 144 tokens of headroom. `serve_8b.sh` on the box still
   says 8192.
3. **Confirm the pre-flight window guard against a live server** (Goal 5). Unit-tested only.
4. **Re-read citation F1 with both arms guided** (Goal 6). The present delta is confounded.
5. **Gate G2 run** (Goal 8), then **repeat W9 on it** (Goal 11).
6. Goals 9 (G3 verifier AUROC) and 10 (G4 gold annotation) are unstarted.

---

## 4. Standing state & operational rules

- Long jobs on the A4000 MUST run under `systemd-run --user --unit=<name>`; sanity artifacts belong
  **outside** the repo, because `harness.git_sha()` stamps `-dirty` on untracked files and a Gate G2
  manifest must be reproducible from a commit.
- The box: repo at `/home/user/BioMedical_QA`, vLLM in `~/venvs/vllm-server`, `vllm-8b.service`
  runs `/home/user/serve_8b.sh`, measurement unit `biomedqa-run.service` runs
  `/home/user/run_measure.sh`. Copy-paste only, **one line per command**.
- Remote helper: `uv run --with paramiko python scripts/_remote.py 'wsl.exe -d Ubuntu-24.04 -- bash -lc "bash /home/user/status.sh"'`
- Prompts are frozen (ADR-0009 §8). The joint JSON template is a **new** stage, not an edit to a
  frozen one: `decompose_template_digest()` and `post_hoc_answer_template_digest()` are unchanged.
- **`Upcoming_goals.md` is the live target list.** Keep it current, in ASD-STE100 STE.
- Pushing to `origin/main` needs no permission (`CLAUDE.md`); always `git pull --rebase` first.

---

## 5. A4000 commands (copy-paste, one line each)

**On the A4000, inside WSL2 Ubuntu-24.04.** Run them in order and check each before the next.

Raise the served window to 14336 and restart the server:

```
sed -i 's/--max-model-len 8192/--max-model-len 14336/' /home/user/serve_8b.sh && systemctl --user restart vllm-8b.service
```

Confirm the server came back with the new window:

```
sleep 90 && curl -s http://localhost:8000/v1/models | grep -o '"max_model_len":[0-9]*'
```

Confirm the checkout is clean before a run (a dirty tree makes the manifest unreproducible):

```
cd /home/user/BioMedical_QA && git pull --rebase && git status --porcelain
```

The joint-guided n=100 run (~45 min; runs under its own unit so an SSH drop cannot kill it):

```
systemd-run --user --unit=g2-guided-joint --working-directory=/home/user/BioMedical_QA --setenv=HOME=/home/user bash -lc 'uv run python scripts/generate_smoke.py --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --base-url http://localhost:8000 --n 100 --contexts docs/harvest/dev_contexts_top10.jsonl --max-tokens 3584 --guided-decoding --out-prefix docs/harvest/generate_fp05_n100_guided_both > /home/user/guided_both.log 2>&1'
```

Watch it:

```
journalctl --user -u g2-guided-joint -f
```

Read the parse rates the moment it finishes:

```
cd /home/user/BioMedical_QA && uv run python -c "import json;d=json.load(open('docs/harvest/generate_fp05_n100_guided_both.summary.json'))['per_system'];print({k:(v['clean_parses'],v['quote_not_found'],v['call_failure_count']) for k,v in d.items()})"
```

**On the writing host, after the artifacts are pulled** — the contrast (~10 min the first time,
seconds afterwards, the cache is committed), then the mandatory stratified check:

```
uv run python scripts/citation_contrast.py docs/harvest/generate_fp05_n100_guided_both --threshold 0.5
```

```
uv run python scripts/w9_stratified_parity_report.py docs/harvest/generate_fp05_n100_guided_both --max-tokens 3584
```
