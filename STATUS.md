# Project status — 2026-08-23

Current state of the experiment: what is signed off, what is frozen, and what is still open.
This file is regenerated wholesale rather than appended to, so a stale line never survives here.

`main`. **Gate G2 is signed off on `generate_fp05_n100_guided_v4`.** The Aug 23 review found that the
four previous rounds of work had been tuning against a criterion that is not in Gate G2, using a prompt
lever ADR-0009 forbids. Five joint-side granularity edits are reverted, `v5`–`v9` are void as
evidence, and the run of record moves back to `v4`, which clears both real G2 criteria: joint valid
parse rate **97/100** and citation-F1 delta **+0.1403 [+0.0751, +0.2066]**, excluding zero.

Tests: `uv run python -m pytest tests/ -q` → **495 passed** in ~19 s. `pyproject.toml`'s
`pythonpath` is `["src", "scripts"]`. **Use `python -m pytest`** — bare `uv run pytest` fails with
`Failed to spawn: pytest`.

Frozen digests (unchanged since Aug 17):

| pin | value |
|---|---|
| `decompose.decompose_template_digest()` | `4129a884c7a1b4854739ffe2e3900a1db39626db7fd1bf076d6b549027a7d737` |
| `prompts.post_hoc_answer_template_digest()` | `91bc7dddd62db4d6d37c26a91f05f938b22dafcca7a6e5aed4509c714f25ac1a` |
| `CONFIG_VERSION` / `GenerationConfig.frequency_penalty` | `1.5.0` / `0.5` |

`JOINT_JSON_TEMPLATE` **is now effectively frozen too**, and the previous handoff's claim that it
"stays in scope for guided-decoding defect fixes and length-target tuning" was wrong on its second
half. See §2.1. Its text now matches commit `054ec6b` byte-for-byte
(`sha256` prefix `1e5ac48a47befbfc`), which is the commit `v4` ran on.

---

## 1. Where the project is

**Gate G2 is PASSED on `generate_fp05_n100_guided_v4`.** Both criteria met on the same run.

| Gate | Date | State |
|---|---|---|
| **G0** — 8B AWQ generator chosen | Aug 4 | **PASSED 2026-08-04.** |
| **G1** — hit@10 ≥ 0.90, Wilson lower > 0.85 (ADR-0015) | Aug 23 | **PASSED 2026-08-10.** hit@10 = 0.9400, Wilson lower 0.8752. |
| **G2** — citation-F1 contrast + ≥95% valid claim parse | **Sep 6** | **PASSED 2026-08-23**, two weeks early, on `generate_fp05_n100_guided_v4`. Citation F1 joint **0.6651** vs post-hoc **0.5248**, delta **+0.1403 [+0.0751, +0.2066]**, excludes zero. Valid claim parse rate: record-level **97/100 (97%)**, claim-level **399/406 (98.3%)** (preregistered criterion ADR-0019), `quote_not_found = 0`. Both MET. See definitions in `docs/harvest/joint_citation_f1_fp05_guided_v4.md`. |
| **G3** — Cheap Verifier Gate | **Sep 20** | **Machinery ready, evidence pending** (`passes: false` blocked on labels, judge cost evidence, verifier timing/rate). See `docs/harvest/runbooks/g3_runbook.md`. |
| G4 · G5 | Sep 27 · Oct 11 | Unstarted, with due weeks. |

### The correction that unblocked G2

Gate G2 has **two** criteria, verbatim from `research_roadmap.md`:

> **Gate G2 (by Sep 6): on dev, joint attribution beats post-hoc citation on citation-F1 by a margin
> exceeding the paired-bootstrap CI, and ≥95% of emitted claims parse into the schema with resolvable
> spans.**

Previous handoffs asserted a **third** precondition — that the ADR-0009 §5 W9 stratified check must
*pass* — and attributed it to "ADR-0009 §5". **§5 says no such thing.** §5's one-sided fallback makes
the check mandatory to *run and disclose* when the residual favours C2. §1 lists parity as "one
quantity measured and disclosed whatever it says"; §3 states outright: "**The tolerance does not need
to be achievable.** Missing it is survivable by design." Four rounds of work were spent chasing a
non-criterion.

---

## 2. What changed on Aug 23

### 2.1 Reverted five joint-side granularity edits (ADR-0009 §4/§6 violation)

`JOINT_JSON_TEMPLATE` acquired a claim-length target in `045a96c` and had it re-tuned four times
(`95dd958`, `dab7a68`, `dc08914`, `b29e74c`). Three of the four commit subjects name the objective:
*"for W9 parity"*, *"for 16 w/c parity and W9 sign-off"*, *"for v9 parity"*. Three violations:

1. **§4 confines the granularity lever to `POST_HOC_ANSWER_TEMPLATE`.** These steer *joint's*
   granularity. The in-code freeze (`PARITY_LOOP_CLOSED.post_hoc_answer_template_sha256`, checked by
   `tests/test_prompts.py`) pins the post-hoc side only, so nothing caught it.
2. **§6's blind lifted 2026-08-14**, so all five were granularity edits made with citation-F1
   visible — what `PARITY_LOOP_CLOSED`'s own comment calls "the one thing §6 exists to prevent".
3. **§5's check was treated as a gate**, contrary to *What survives termination*: "A pre-registered
   asymmetric check is not retracted because the iteration that closed the loop passed."

All five reverted. `POST_HOC_ANSWER_TEMPLATE`, `_claim_rules()`, `PARITY_LOOP_CLOSED`, and
`decompose_template_digest()` untouched — §8's Sep 3 freeze intact. Recorded as ADR-0009's **Fourth
amendment**, which also adds a standing rule: granularity-motivated edits to *any* arm's prompt
after 2026-08-14 are prohibited.

### 2.2 The tuning was also futile — the instrument has no resolution

| run | claim-length target | joint parses | W9 verdict | citation-F1 delta | CI excludes 0 |
|---|---|---|---|---|---|
| **`v4`** | **none** | **97/100** | FAIL (+30.8%) | **+0.1403** | **yes** `[+0.0751, +0.2066]` |
| `v5` | 15–20 words | 95/100 | FAIL (+21.4%) | +0.0933 | yes `[+0.0259, +0.1613]` |
| `v6` | 16–22 words | 96/100 | PASS (+6.2%) | +0.0557 | no `[-0.0078, +0.1226]` |
| `v7` | 15–20 words | 96/100 | FAIL (+13.3%) | +0.1114 | yes `[+0.0507, +0.1752]` |
| `v8` | 16–20 words | 98/100 | PASS (+13.3%) | +0.0634 | no `[-0.0019, +0.1296]` |
| `v9` | 16–21 words | 91/100 | FAIL (+13.3%) | +0.0851 | yes `[+0.0146, +0.1554]` |

`v5` and `v7` carry the **same** target text and land on different W9 verdicts and parse rates.
Parse rate swings 98 → 91 on a one-word change. The gated statistic is an integer median of 14–20
words: one word $\approx 6.7\%$ against a two-word tolerance — verbatim the "run out of resolution"
argument `PARITY_LOOP_CLOSED` used to stop the parity loop at 1 of 10 iterations.

**W9-pass and CI-excludes-zero are anti-correlated across all six runs**, because both are driven by
joint's claim length in opposite directions: pushing joint's claims longer narrows the parity gap
while trading away the recall that produces C2's margin. Another run would eventually have
manufactured a simultaneous pass by chance.

### 2.3 Discharged §5's asymmetric scrutiny by standardisation, not tuning

A granularity gap is a confound only if it **transmits** to citation-F1; the pooled gate measures
size, not effect. New `scripts/w9_length_standardized_contrast.py` re-weights joint's citation-recall
to post-hoc's own claim-length distribution over `CLAIM_LENGTH_BANDS` (direct standardisation,
queries resampled per ADR-0011 §2). On `v4`, at the widest gap yet recorded:

| band | joint n | joint R | post_hoc n | post_hoc R | ΔR |
|---|---|---|---|---|---|
| 1-10 | 95 | 0.526 | 34 | 0.529 | -0.003 |
| 11-15 | 165 | 0.497 | 187 | 0.358 | **+0.139** |
| 16-20 | 102 | 0.480 | 214 | 0.322 | **+0.158** |
| 21-30 | 41 | 0.585 | 172 | 0.384 | **+0.202** |
| 31+ | 3 | 0.667 | 12 | 0.333 | **+0.333** |

Joint leads in four of five bands, ties in the shortest, and ΔR **grows** with claim length — the
opposite of the confound's signature. Standardised delta **+0.1495 [+0.0786, +0.2244]** against
unstandardised +0.1403. **The gap transmits *against* C2:** post-hoc's coarser claims were making
joint's advantage look *smaller* than it is at matched granularity.

Two correctness properties the script asserts or verifies:
- `standardize=False` reproduces `citation_contrast.py`'s +0.1403 exactly, so weighting is the only
  difference between the two rows.
- Post-hoc's standardised recall equals its observed recall by construction (asserted at runtime).
- Empty-text claims (`len(text.split()) == 0`) are joint-only (3 on `v4`, 11 on `v8`, none in
  post-hoc ever) and fall outside `CLAIM_LENGTH_BANDS`. They are **folded into the `1-10` band, not
  skipped** — skipping would standardise joint's own defect out and flatter joint.

### 2.4 Reports written

- `docs/harvest/joint_citation_f1_fp05_guided_v4.md` — G2 sign-off, arm performance, run comparison.
- `docs/harvest/w9_stratified_parity_guided_v4.md` — W9 FAIL disclosed + discharged, and §5 the
  run-by-run case for voiding `v5`–`v9`.
- `docs/harvest/generate_fp05_n100_guided_v4.w9_stratified_parity.txt`,
  `...v4.length_standardized_contrast.txt` — raw script output, tracked.

---

## 3. Open items, in priority order

1. **Commit and push** the Aug 23 work: the `JOINT_JSON_TEMPLATE` revert, the
   `CLAIM_LENGTH_BANDS` extraction, `scripts/w9_length_standardized_contrast.py`, ADR-0009's Fourth
   amendment, the two `v4` reports, and this file.
2. **Decide the fate of `v5`–`v9` artifacts.** They are void as evidence but currently untracked in
   `docs/harvest/` (20 files). Either delete them or keep them with a `VOID` marker; leaving them
   untracked risks one later being treated as a run of record. Recommend deleting `v6`–`v9`
   and keeping `v5` (already tracked, cited in the reports).
3. **Paper methods section** gains the length-standardised contrast alongside the pre-registered
   asymmetric rule, and reports parity as a disclosed miss with its transmission measured.
4. Goal 9 (G3 verifier AUROC, Sep 20): machinery ready, evidence pending (`passes: false` blocked on human labels [annotation opens 2026-09-07], judge cost evidence, verifier pricing). Goal 10 (G4 gold annotation, Sep 27) remains unstarted. Canonical G3 runbook pointer: `docs/harvest/runbooks/g3_runbook.md`.

**No further dev-set generation run is needed for G2.** Everything above is re-derived from stored
`v4` records with a complete MiniCheck cache — no inference, no A4000 server, no WSL2 keep-alive.

---

## 4. Standing state & operational rules

- Sanity artifacts belong **outside** the repo, because `harness.git_sha()` stamps `-dirty` on
  untracked files and a Gate G2 manifest must be reproducible from a commit. `v4`'s manifest
  `git_sha` is `054ec6b6adb5f73cff0e61451850711733a74d9a`, clean.
- The box: repo at `/home/user/BioMedical_QA`, vLLM in `~/venvs/vllm-server`, `vllm-8b.service` runs
  `/home/user/serve_8b.sh` (`--max-model-len 14336`). Copy-paste only, **one line per command**.
- **All prompts are now frozen.** ADR-0009 §8 freezes the decomposer and post-hoc templates; the
  Fourth amendment closes the `JOINT_JSON_TEMPLATE` loophole for granularity-motivated edits. A
  guided-decoding *parse* defect fix remains legitimate, but must not change claim-length guidance.
- **`TODO.md` is the live target list. Keep it current, in ASD-STE100 STE.

---

## 5. Commands to resume

Reproduce the G2 sign-off from stored records (no server needed):

```
uv run python scripts/citation_contrast.py docs/harvest/generate_fp05_n100_guided_v4
```

```
uv run python scripts/w9_stratified_parity_report.py docs/harvest/generate_fp05_n100_guided_v4 --max-tokens 3584
```

```
uv run python scripts/w9_length_standardized_contrast.py docs/harvest/generate_fp05_n100_guided_v4
```

Confirm the joint template still matches `v4`'s commit:

```
uv run python -c "import hashlib,sys; sys.path.insert(0,'src'); from biomedqa.prompts import JOINT_JSON_TEMPLATE as t; print(hashlib.sha256(t.encode()).hexdigest()[:16])"
```

Expect `1e5ac48a47befbfc`.

Full test suite before committing:

```
uv run python -m pytest tests/ -q
```
