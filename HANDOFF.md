# HANDOFF — 2026-08-16 (end of fourteenth session)

Snapshot for resuming in a fresh session. Regenerate wholesale; **do not append** — a stale line here
is worse than a missing one, because the next session will trust it.

`main` · working tree clean at `21cac47` plus this file.

Tests: `uv run --with pytest python -m pytest tests/ -q` → **377 passed**. `pyproject.toml`'s
`pythonpath` is `["src", "scripts"]`.

---

## 1. Where the project is

**W3 (Aug 17–23), Phase 1 — Retrieval gate (Aug 3–23) → C1, Table 1; Decomposer preparation for Gate G2 (Sep 6).**

| Gate | Date | State |
|---|---|---|
| **G0** — 8B AWQ generator chosen, A4000 measured | Aug 4 | **PASSED 2026-08-04.** Issue #1 closed. |
| **G1** — hit@10 ≥ 0.90, Wilson lower > 0.85 (ADR-0015) | Aug 23 | **PASSED 2026-08-10.** Row 4 hit@5 = 0.86, **hit@10 = 0.9400** (Wilson lower 0.8752). |
| **G2** — citation-F1 / attribution quality (≥95% valid claim parses) | **Sep 6** | Structural decomposer/parser defects fixed and re-measured live (Aug 16). **`claim_parse_rate` 0.906 / 0.908**, short of 0.95. `clean_decompose_rate` 0.35 / 0.49 — sole remaining defect is claim repetition, which is model behaviour, not parser or prompt shape (five configurations tried). |
| G3 · G4 · G5 | Sep 20 · Sep 27 · Oct 11 | Unstarted, with due weeks. |

---

## 2. What was wrong, and what was actually fixed (Aug 16)

The previous session's 0.06 / 0.01 / 0.00 rates had **no attribution recorded**, so the first change
was to make the smoke test say *why* a query is unclean. Everything below follows from that.

### 2.1 The smoke test now attributes its own failures
`scripts/decompose_smoke.py` records `decompose_error_kinds`, `cite_error_kinds` and
`cite_recovered_kinds` (error strings with numbers/ids/quotes collapsed, most frequent first), prints
cite errors alongside decompose errors, and reports two rates on their proper denominators — see §2.5.
A rate of 0.0 with no attribution costs a whole GPU run to re-diagnose; that will not recur.

### 2.2 `clean_cite_rate = 0.0` was one structural bug, not tuning
The cite stage was sent the **whole** re-cut answer — 20 to 25 claims — in a single call, and asked to
reproduce every one verbatim. The 8B model stopped after 4 to 7, so
`cite stage returned N CLAIM lines for M claims sent` fired on 8 to 9 of every 10 queries. Three fixes:

1. **`generate.cite_claims` batches** at `MAX_CLAIMS_PER_CITE_CALL = 5`, each batch numbered from 1
   and matched positionally within itself. `Recitation.cost` → `Recitation.costs` (one row per call).
2. **A dedicated `"recite"` stage** (`prompts.POST_HOC_RECITE_TEMPLATE`). `POST_HOC_CITE_TEMPLATE`
   carries `_claim_rules()` — "resolve every pronoun", "split anything joined by *and*" — which for
   C7 directly contradicts "reproduce every claim exactly". A live probe caught the consequence: the
   model cited 3 of 5 claims, then emitted **twelve empty `CLAIM` lines** and a note explaining the
   rest "were not present in the original answer". The recite stage keeps the citation grammar
   unchanged and withholds only the instruction to reshape a claim that is already frozen.
3. **No `DECISION` line is requested or required** for re-citation (`parse_response(...,
   require_decision=False)`). The decision belongs to the generation being re-cited; asking again
   only bought `no DECISION line` errors on otherwise perfect replies.

### 2.3 The decomposer now makes one call per sentence, and `FROM` is gone
`CLAIM <n> FROM <k>` was the single largest error source, and its worst failure was silent. Beyond the
syntax drift already known (`CLAIM7FROM4`, `FROM S4.1`, `FROM (6)`), the index **pointed at the wrong
sentence**: the claim "CEA use is not associated with district stroke mortality" arrived stamped
`FROM 1`, whose sentence is about 14-fold variation. That corrupts `Claim.source_start/source_end`,
which the decomposition post-mortem reads as ground truth, and it is invisible in every rate.

`decompose()` now issues **one call per sentence**, with the whole answer still in the prompt as
context (the decontextualized row cannot resolve "it" from the target sentence alone). The grammar is
plain `CLAIM <n>:`; there is no index for the model to get wrong, and a claim's span is the sentence
its call was about, by construction. `MAX_SENTENCES_PER_CHUNK` and the chunk-index resolution logic
are deleted, not disabled.

This removed **entire error classes**: `sentence N does not exist`, dropped trailing sentences,
mis-attributed spans, and all `FROM` syntax drift no longer appear at n=100.

### 2.4 Widened parse acceptance is counted, never silent
`ParsedResponse.recovered` (and `Recitation.recovered`) is a third category between clean and broken:
a line whose meaning is unambiguous but whose transcription drifted. Two members, both of which
return a span the passage genuinely has:

- **Quote case / whitespace / edge-punctuation drift.** `locate_quote` matches after normalising
  them, and `quoted_text` is always copied back out of the passage, so offsets and text never
  disagree. Interior drift (`HR, 1.85` for `HR: 1.85`) and spliced spans still fail — those are
  attribution errors, not typography.
- **A passage id that dropped its chunk index** (`[pubmed23n0263_2785:]`), resolved **only** when
  exactly one context passage comes from that document. Two chunks of one document keep it an error;
  guessing would attribute evidence to the wrong chunk.

`cite_recovered_count` / `cite_recovered_kinds` report these every run, so widening acceptance can
never quietly become "the defect stopped happening". At n=100 there were **1502 / 1550** of them —
almost all id-index drift.

Also new: an empty `CLAIM n:` line is a counted defect, not a claim, so padding cannot satisfy the
positional match.

### 2.5 `clean_*_rate` is the wrong lens for G2, and both are now reported
`clean_decompose_rate` / `clean_cite_rate` are all-or-nothing per query: one drifted quote among a
query's ~60 citation lines fails the whole query. G2's actual bar is per **claim** — "≥95% valid claim
parse" (`ROADMAP.md` §1) — and citation fidelity is a separate question about the model. The summary
now carries both:

- `claim_parse_rate` — claims that came back matched from the cite stage.
- `quote_located_rate` — CITE lines whose quote was located, over lines attempted.

---

## 3. Measured live A4000 baseline (n=100, fp=0.5, max_tokens=4096)

`http://localhost:8000/v1` serving `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`.

| Row | claims | clean_decompose | clean_cite | **claim_parse** | **quote_located** | dups | quote_not_found | recovered |
|---|---|---|---|---|---|---|---|---|
| **sentence** (control) | 1246 | **1.0000** | **1.0000** | **1.0000** | — | 0 | 0 | 0 |
| **atomic** | 3006 | 0.3500 | 0.0000 | **0.9062** | **0.7398** | 442 | 1287 | 1502 |
| **decontextualized_atomic** | 3031 | 0.4900 | 0.0000 | **0.9083** | **0.7419** | 319 | 1295 | 1550 |

Against the previous session: `clean_decompose_rate` **0.06 → 0.35** (atomic) and **0.01 → 0.49**
(decontextualized). Monotonicity **PASSES** (17.0 ≥ 11.0 ≥ 11.0). `divergence_rate` (atomic) 0.7631.

**Harvest artifacts were not committed as a baseline** — the ≥0.95 bar is not cleared. The run's
outputs remain on the box at `~/BioMedical_QA/docs/harvest/decompose_smoke.*`.

### The only decompose defect left is claim repetition
At n=100 the decompose histogram contains **nothing but** `repeats ... verbatim` (641 atomic / 502
decontextualized). The model emits ~2.4 claims per sentence and restates one of them. It is not a
prompt-shape or decoding artifact — five configurations measured at n=10, decompose only:

| variant | clean_decompose | duplicates | claims/sentence |
|---|---|---|---|
| fp 0.5 (current) | 0.50 | 8 | 2.39 |
| fp 0.5, single-claim format example | 0.50 | 9 | 2.49 |
| fp 0.0 | 0.40 | 9 | 2.41 |
| fp 0.2 | 0.40 | 9 | 2.38 |
| fp 1.0 | 0.40 | 8 | 2.29 |

`frequency_penalty` is inert on this failure. The `n=15` sweep that recommended 0.5
(`docs/harvest/decompose_smoke_fp_sweep.md`) was run against the old whole-answer prompt, where
outputs were long; that premise no longer holds, and 0.5 is retained only because nothing beats it.

**Duplicates are deliberately not deduplicated** (`parse_decomposition` docstring): collapsing them
would hide the defect the flag exists to measure and would understate `total_claims`.

### Cite-side remainder, ranked (atomic, n=100)
`quote not found` 1287 · `no matching CLAIM line` 282 · count mismatch 238 · id not in context 162 ·
`CITE line has no '||'` 116 · empty CLAIM line 32. A live diff of refused quotes shows the model
**composing** them: splicing two separated spans into one, swapping `HR:` for `HR,`, and occasionally
inventing a quote outright ("Dobutamine-induced hypotension was not associated with any clinical
outcome"). These are attribution failures and must stay errors.

---

## 4. What exists on the box, and the corpus as built

### The index (built 2026-08-10 on the A4000)
- Location: `data/index/empty` (ADR-0014 §3 confirmed).
- Passages: **2,162,838** encoded in **1.99 h**.
- Gold: **1,037 gold passages present**.
- Artifacts: `dense.npy` 3.1 GB · `passage_texts.jsonl` 2.5 GB · `bm25/` (box-only, gitignored).

### The corpus (built 2026-08-06 on the A4000)
- `fingerprint`: `93321598f3f1`.
- `gold collisions`: 1,000 of 1,000.
- `duplicate rows`: 300 suppressed over 244 PMIDs.
- Split hash: `71c46cc5b0ca` (`load_splits()` dev 100 pubid strings).

### Reaching the box
The Windows host answers SSH with password auth only (`.env.local`, gitignored); the work lives in the
WSL2 guest, so every command is `wsl.exe -d Ubuntu-24.04 -- bash -lc "..."`. `scripts/_remote.py`
(paramiko, `--put` for file copy) and `scripts/_diag.sh` wrap this. A backgrounded `nohup` inside a
one-shot `wsl.exe` call **dies when the SSH channel closes** — long runs must hold the channel open.
An n=100 three-row run takes ≈3.3 h.

---

## 5. Annotator agreement & timeline

- Both annotators accepted on 2026-08-05; both free from Sep 5 onward.
- Decomposer & Granularity Freeze: **Sep 3, 2026**.
- Gate G2 Execution: **Sep 6, 2026**.
- Pilot annotation pass begins: **Sep 7, 2026** (W6).

`decompose_template_digest()` is re-pinned to
`4129a884c7a1b4854739ffe2e3900a1db39626db7fd1bf076d6b549027a7d737` (per-sentence grammar, no `FROM`).
Any further edit before Sep 3 must be a deliberate, dated re-pin.

---

## 6. Pending next steps

1. **Decide how G2 reads the C7 rows.** `claim_parse_rate` (0.906 / 0.908) is the number G2's wording
   asks for and is 4–5 points short; `clean_*_rate` cannot reach 0.95 with this model and is arguably
   measuring the wrong thing. This is a call for the maintainer, not a silent redefinition — it
   changes what the freeze certifies.
2. **Close the claim-repetition gap**, the only decompose defect left. Prompt shape and
   `frequency_penalty` are exhausted; the untried levers are a larger/undistilled decomposer model for
   the C7 rows only, or a second validation pass. Both cost more than a knob and need a decision
   before Sep 3.
3. **Decomposer & Granularity Freeze (Sep 3, 2026)** — freeze prompt fragments, rules, parser logic.
4. **Gate G2 Execution (Sep 6, 2026)** — citation-F1 against the post-hoc baseline.
