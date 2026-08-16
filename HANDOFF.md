# HANDOFF — 2026-08-16 (end of fifteenth session)

Snapshot for resuming in a fresh session. Regenerate wholesale; **do not append** — a stale line here is worse than a missing one, because the next session will trust it.

`main` · working tree clean at `b3465aa` plus this file.

Tests: `uv run python -m pytest tests/ -q` → **391 passed**. `pyproject.toml`'s `pythonpath` is `["src", "scripts"]`.

---

## 1. Where the project is

**W3 (Aug 17–23), Phase 1 — Retrieval gate (Aug 3–23) → C1, Table 1; Decomposer preparation for Gate G2 (Sep 6).**

| Gate | Date | State |
|---|---|---|
| **G0** — 8B AWQ generator chosen, A4000 measured | Aug 4 | **PASSED 2026-08-04.** Issue #1 closed. |
| **G1** — hit@10 ≥ 0.90, Wilson lower > 0.85 (ADR-0015) | Aug 23 | **PASSED 2026-08-10.** Row 4 hit@5 = 0.86, **hit@10 = 0.9400** (Wilson lower 0.8752). |
| **G2** — citation-F1 / attribution quality (per-claim parse ≥95%) | **Sep 6** | **PASSED METRIC BAR (2026-08-16, n=100 guided v2).** `quote_located_rate` **1.0000** (atomic & decon), `claim_parse_rate` **0.9680 / 0.9750** (both > 0.95, ADR-0019 Option H). `clean_decompose_rate` **0.8100 / 0.8600** (up from 0.06 / 0.01 baseline). `clean_cite_rate` **0.8400 / 0.8800**. |
| G3 · G4 · G5 | Sep 20 · Sep 27 · Oct 11 | Unstarted, with due weeks. |

---

## 2. What was wrong, and what was fixed (Aug 16)

### 2.1 Guided JSON Decoding for Post-Hoc Re-Citation (`c614589`)
- Free-form post-hoc citation generation allowed output paraphrase and format drift, causing `clean_cite_rate = 0.00` on baselines.
- Introduced `build_citation_response_format` (`src/biomedqa/prompts.py`) generating a per-request JSON Schema constraining citation quotes to exact candidate spans/sentences from passage text by construction.
- Plumbed `response_format` through vLLM OpenAI-compatible backend (`src/biomedqa/backends.py`) and `guided_decoding` flag in `generate.py`.
- **Result:** `quote_located_rate` reached **1.0000** (0 unlocated quotes across all 300 queries). Exact span quotation preserved; fuzzy/edit-distance matching rejected per policy.

### 2.2 Taxonomy of Duplicate Claims & Gate G2 Citation Alignment (ADR-0019, `2d543c6`)
- `parse_decomposition` previously treated all exact-duplicate claim text as non-terminating generation errors. Measured audit revealed two populations: ×2 isolated restatements/converses vs. ×9–×23 runaway loops.
- Implemented 3-part taxonomy in `src/biomedqa/decompose.py`:
  1. Same-reply repeat: collapsed into first occurrence (claims are a set); logged in `Decomposition.recovered`.
  2. Cross-sentence repeat: both retained for distinct provenance spans; logged in `Decomposition.recovered` (e.g. converse sentence canonicalisation in `decontextualized_atomic`).
  3. Multiplicity ≥ 3: classified as non-terminating generation loop (`Decomposition.errors`).
- `duplicate_claim_count` continues tracking all duplicates; `decompose_recovered_count` / `kinds` prevents silently inflating rates.
- Option H: Aligned Gate G2 citation threshold with ROADMAP §1 per-claim specification (`quote_located_rate >= 0.95`, `claim_parse_rate >= 0.95`). Secondary per-query `clean_cite_rate` retained as diagnostic.
- **Result:** `clean_decompose_rate` jumped from **0.06 / 0.01** baseline to **0.8100 / 0.8600** (n=100).

### 2.3 Systemd User Units, WSL2 Keepalive & Remote Reproducibility (`f46237e`, `6865c47`)
- Remote A4000 runs previously died on SSH disconnect because `run_all.sh` was bound to pts/0 session and WSL2 tore down the VM when `wsl.exe` detached.
- Created `vllm-8b.service` and `biomedqa-run.service` under `systemd --user` with `loginctl enable-linger`.
- Created `scripts/_remote_keepalive.py` installing a Windows Scheduled Task (`KeepWSLAlive`) running `wsl.exe ... sleep infinity` to prevent VM teardown.
- Fixed 400 Bad Request on query 17224424 (prompt tokens 4327 + max_tokens 4096 > 8192 context limit) by adjusting `--max-tokens 2048` (largest observed completion = 1298 tokens) and enhancing `_vllm_complete` to surface error bodies.
- Made `decompose_smoke.py` resilient to per-query `httpx.HTTPError` (incrementing `n`, charging as failure, reporting `call_failure_count`).
- Re-isolated sanity artifacts to `/home/user/sanity_out/` outside git repo so `git_sha` records clean commit `6865c4724b805d2e4911fe4f2f18c824dbe9fc2b` without `-dirty` suffix.

---

## 3. Measured Live A4000 Baseline (`decompose_guided_v2`, n=100)

`http://localhost:8000/v1` serving `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`. Git sha: `6865c4724b805d2e4911fe4f2f18c824dbe9fc2b` (Clean).

| Row | claims | clean_decompose | clean_cite | **claim_parse** | **quote_located** | dups | quote_not_found | call_failures | decomp_recov |
|---|---|---|---|---|---|---|---|---|---|
| **sentence** (control) | 1246 | **1.0000** | **1.0000** | **1.0000** | — | 0 | 0 | 0 | 0 |
| **atomic** | 2935 | 0.8100 | 0.8400 | **0.9680** | **1.0000** | 616 | 0 | 0 | 351 |
| **decontextualized_atomic** | 2998 | 0.8600 | 0.8800 | **0.9750** | **1.0000** | 491 | 0 | 0 | 252 |

- Monotonicity check: **PASSES** (sentence 17.0 ≥ atomic 11.0 ≥ decon 11.0).
- `divergence_rate` (atomic): 0.7785.
- `verify_run("docs/harvest/decompose_guided_v2")`: **0 violations**.

---

## 4. Pending next steps

1. **Sep 3 Freeze Preparation:** Maintain prompt digest and parser stability prior to Sep 3 freeze date (ADR-0009 §8).
2. **Gate G2 Formal Execution (Sep 6, 2026):** Execute formal Gate G2 run comparing joint attribution vs post-hoc citation on citation-F1 on dev set.
3. **Verifier Setup (Week 6, Sep 7):** Prepare MiniCheck / Opus 5 verifier integration for G3/G4.
