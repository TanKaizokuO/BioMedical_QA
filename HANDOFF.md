# HANDOFF — 2026-08-17 (end of sixteenth session)

Snapshot for resuming in a fresh session. Regenerate wholesale; **do not append** — a stale line here is worse than a missing one, because the next session will trust it.

`main` · working tree clean at `ff50050` plus this file and the day's documentation.

Tests: `uv run python -m pytest tests/ -q` → **431 passed**. `pyproject.toml`'s `pythonpath` is `["src", "scripts"]`.

---

## 1. Where the project is

**W3 (Aug 17–23), Phase 1. The verifier was pulled forward out of W6 and both readings that depended on it were re-measured the same day.**

| Gate | Date | State |
|---|---|---|
| **G0** — 8B AWQ generator chosen, A4000 measured | Aug 4 | **PASSED 2026-08-04.** Issue #1 closed. |
| **G1** — hit@10 ≥ 0.90, Wilson lower > 0.85 (ADR-0015) | Aug 23 | **PASSED 2026-08-10.** hit@5 = 0.86, **hit@10 = 0.9400** (Wilson lower 0.8752). |
| **G2** — citation-F1 contrast + per-claim parse | **Sep 6** | **PARSE BAR PASSED (Aug 16). CONTRAST STILL NOT ESTABLISHED (Aug 17).** `quote_located_rate` 1.0000, `claim_parse_rate` 0.9680 / 0.9750. But the contrast, re-read with the real verifier, is **joint 0.428 vs post-hoc 0.418, delta +0.011 [−0.117, +0.137]** — the interval straddles zero. |
| G3 · G4 · G5 | Sep 20 · Sep 27 · Oct 11 | Unstarted, with due weeks. G3 is now a **prerequisite for interpreting G2's margin**, and it is fourteen days after it. |

---

## 2. What happened today

### 2.1 `verify.py` landed, pulled ahead of W6 (ADR-0020 §1, commit `f447d27`)

The schedule had `verify.py` in W6 (Sep 7–20) — **after** the Sep 6 G2 gate. G2's pass condition is a citation-F1 contrast, the only read of that contrast used `cross-encoder/nli-deberta-v3-xsmall` as a stand-in φ, and that read's own write-up named φ's premise-length sensitivity as the leading alternative explanation for its result. Deciding G2 on a placeholder φ is deciding G2 on the placeholder, so the schedule moved.

- `MiniCheckVerifier.score_pairs` is **the** single MiniCheck forward pass in the package, faithful to `Liyan06/MiniCheck`'s `minicheck/inference.py`: `predict: {document}</s>{claim}`, one decoder step from a zero `decoder_input_ids`, softmax over the **first-position** logits at vocabulary ids **3** (`"▁"`, first sub-token of `"0"`) and **209** (`"▁1"`), ~500-word sentence-aligned chunks, **max over chunks**, 2048 truncation. The two ids are asserted against the tokenizer at load (`_check_decision_tokens`).
- `JudgeVerifier` (Opus 5) sits behind the same `score_pairs`; integer 0–100 → probability, `CostRecord` stamped `component="judge"`. The whole reply must be the number — `"a 3 out of 5"` used to parse as 0.03.
- **Raw scores only.** `phi_from_scores(scores, threshold)` is the only cutoff in the module and raises for a pair it never scored rather than reporting a missing score as a grounding failure.
- `scripts/first_citation_f1.py` gained `--phi minicheck|deberta-xsmall` (default minicheck), `--threshold`, and a free threshold sweep.

### 2.2 MiniCheck was being invoked wrongly, and it was measured, not argued (ADR-0020 §2)

`scripts/confusability_probe.py` had rendered `"premise: {p} hypothesis: {h}"` and compared the sequence loss of the strings `"1"` and `"0"` at `max_length=512`. The checkpoint has no classification head, so **every prompt returns a number and a wrong prompt returns a wrong number silently**.

`scripts/minicheck_format_check.py` scored both framings against the real weights (`docs/harvest/minicheck_format_check.json`, 200 real pairs + 6 known-answer pairs): **compression, not inversion.** Spearman 0.9719, but the retired framing never leaves `[0.0754, 0.7818]` against the reference's `[0.0061, 0.9805]`, known-answer separation +0.6139 against **+0.9351**, and 11 of 200 pairs cross τ = 0.7 in one direction or the other.

### 2.3 Both dependent readings were re-run

- **Citation-F1** (`docs/harvest/citation_f1_minicheck.md`) — φ swapped, records untouched.
- **Confusability probe**, all four arms (`docs/harvest/confusability_probe_v2*.json`, write-up `confusability_probe_v2.md`) — same index, split, passages.

---

## 3. The two measured results

### 3.1 Citation-F1 on `parity_iter1b`, φ = MiniCheck @ 0.5 (4,904 pairs)

| system | precision | recall | **F1** | 95% CI | claims | citations |
|---|---|---|---|---|---|---|
| joint | 0.872 | 0.284 | **0.428** | [0.316, 0.540] | 719 | 1061 (925 not irrelevant) |
| post_hoc | 0.846 | 0.277 | **0.418** | [0.357, 0.472] | 1242 | 1807 (1528 not irrelevant) |

**joint − post_hoc = +0.011, 95% [−0.117, +0.137]** on 100 paired questions. The Aug 14 read gave **−0.081 [−0.157, +0.005]** on the *same records*. **The sign flipped when φ stopped being a placeholder.**

**It is not a result.** The interval straddles zero and is *wider* than the one it replaces (0.254 vs 0.162), and the sign flips inside the sweep: **+0.013 / −0.007 / +0.011 / −0.036 / −0.041** at τ = 0.1 / 0.3 / 0.5 / 0.7 / 0.9 — post-hoc leads at the two strictest cutoffs, by more than joint leads anywhere. Untruncated 78-question basis: 0.462 vs 0.450, +0.012.

Joint's long-claim defect survives the φ swap and is **joint's, not φ's**: the 31+-word band reads 0.088 against post-hoc's 0.289 (it was 0.000 vs 0.184). That is `parity_iter1b`'s runaway-claim pathology, and fixing the splitter still moves a reported number.

### 3.2 Confusability probe v2

| | retrieved | random control |
|---|---|---|
| mean / median / p90 | 0.3073 / 0.0982 / 0.9586 | 0.0938 / 0.0418 / 0.1436 |
| frac ≥ 0.7 | 0.2295 | 0.0351 |
| paired per-question | **retrieved higher on 85 of 100**, p < 1e-6 | — |

**The v1 anomaly was the bug.** v1 had the control scoring *higher* than retrieved passages at ≥ 0.3 (67.7% vs 62.1%) with separation only in the tail (62 of 100, p = 0.012). v2 separates everywhere. At τ = 0.7, **40 of 100** questions carry a plausible mis-citation target against **12** by chance (v1: 35 against 8); tail enrichment **6.5×** (v1: 6.9×). **τ_confusable stays 0.7 on a new argument** — the corrected distribution is bimodal (median 0.0982, p90 0.9586), so the count barely moves across 0.3–0.8. Reranking still changes nothing (mean 0.3149, frac ≥ 0.7 0.2488, paired 84 of 100). ADR-0012's direction survived; its magnitudes did not, and §3 stays untriggered.

---

## 4. Standing state

- **Decomposition (C7, Aug 16, unchanged):** `quote_located_rate` **1.0000** both rows; `claim_parse_rate` **0.9680 / 0.9750**; `clean_decompose_rate` 0.8100 / 0.8600; monotonicity passes (17.0 ≥ 11.0 ≥ 11.0); `verify_run` 0 violations. `docs/harvest/decompose_guided_v2.*`.
- **v1 probe artifacts are superseded, not deleted.** `docs/harvest/confusability_probe{,_control,_reranked,_reranked_control}.json` stay as the record of what was run. Do not quote them.
- **`docs/harvest/first_citation_f1.md` is not revised** — it is a committed measurement with its caveat stated. `citation_f1_minicheck.md` is the current read.
- **`scripts/bootstrap_dryrun.py`** reports fractions at the quarantined τ. Its answer is a CI *width*, which is a property of the resampling shape, so it does not need re-running; its fractions are not confusability numbers.
- **Remote:** `vllm-8b.service` is back up on the A4000 (it was stopped for the probe's GPU). Long jobs must run as `systemd-run --user --unit=<name> /home/user/<script>.sh` with an absolute `/home/user/.local/bin/uv` — a `nohup` launched through `wsl.exe` over SSH dies with the exec. Status: `uv run --with paramiko python scripts/_remote.py 'wsl.exe -d Ubuntu-24.04 -- bash -lc "bash /home/user/status.sh"'`.

---

## 5. Pending next steps

1. **AlignScore (Table 3's never-cut second row) is still unbuilt, and it is not a pip install.** Its package pins `torch<2`, `pytorch_lightning<2`, `protobuf<=3.20` against this project's `torch>=2.13`, and it loads a Lightning `.ckpt`, not a hub model. The routes are an isolated venv on Python ≤ 3.11, or a manual state-dict port (the nli_sp score is `softmax(tri_label_logits)[:, 0]` — **column 0**, entailment — with ~350-word chunking, max over premise chunks then mean over hypothesis sentences). A third-party HF re-upload exists (`liuyanyi/AlignScore-large-hf`) but its conversion is undocumented, so it needs a `torch.allclose` check against the official `.ckpt` before any score is publishable.
2. **Joint's runaway claims** (31+ band, recall 0.088) are now a load-bearing defect with a φ that cannot be blamed. Splitter and non-termination, before the G2 read.
3. **Sep 3 Decomposer & Granularity Freeze** (ADR-0009 §8) — keep the prompt digest and parser stable.
4. **Sep 6 Gate G2** on a real gate run, not on the `parity_iter1b` smoke run. The φ blocker is gone; the width blocker is not, and the operating point is now the open question.
5. **W6 (Sep 7):** the pilot annotation pass and the full passes are unmoved; only `verify.py` came forward.
