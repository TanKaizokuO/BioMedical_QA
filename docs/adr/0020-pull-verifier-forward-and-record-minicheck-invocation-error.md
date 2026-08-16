# ADR-0020 — Pull the verifier forward to W5 and record the MiniCheck invocation error

**Status:** Accepted · **Date:** 2026-08-17 · **Decided in:** W5 format-error triage and G2 readiness review
**Amends** ADR-0012 §2 (confusability-probe numbers are pending re-run) · **Depends on** ADR-0009 §6 (R5 early-warning read), ADR-0012 §2 (confusability probe)

## Context

Gate G2 (Sep 6, 2026) requires joint attribution to beat post-hoc citation on citation-F1, confirmed
by a paired-bootstrap CI excluding zero. The first unblinding read (`docs/harvest/first_citation_f1.md`,
2026-08-14) returned joint F1 0.264 vs post-hoc F1 0.345, delta −0.081 [−0.157, +0.005] — a point
estimate running against C2's direction with an interval that only just includes zero.

That document names the leading alternative explanation as premise-length sensitivity: post-hoc quotes
a median 23 words per citation against joint's 19, and a sentence-pair NLI model is more likely to
return entailment for a longer premise. φ in that read was `cross-encoder/nli-deberta-v3-xsmall` at
`argmax == entailment` — the placeholder named in `verify.py` as the thing MiniCheck replaces.
`first_citation_f1.md` §"The leading alternative explanation" states explicitly that MiniCheck is
**built for document-level premises** and that the length-sensitivity question is the first quantity
to re-read at W6.

Deciding G2 on that contrast while φ remains the placeholder is deciding G2 on the placeholder.
The schedule placed `verify.py` in Week 6 (Sep 7–20), which is **after** the Sep 6 G2 gate.
That ordering was the error.

Separately, `scripts/confusability_probe.py` (written 2026-08-10) had rendered MiniCheck inputs as
`"premise: {p} hypothesis: {h}"` and scored them by comparing sequence loss for the strings `"1"` and
`"0"` at `max_length=512`. This is not how the checkpoint was trained.
`scripts/minicheck_format_check.py` measured the size of that error against the real weights on the
A4000 (`docs/harvest/minicheck_format_check.json`, 2026-08-17, CPU, 200 real (cited span, claim)
pairs out of `parity_iter1b` plus six known-answer pairs), and the answer is **compression, not
inversion**:

| | reference invocation | retired framing |
|---|---|---|
| known-answer separation (supported − unsupported) | **+0.9351** | +0.6139 |
| known-answer ordering correct | yes | yes |
| real-pair mean / median | 0.2302 / 0.0366 | 0.2950 / 0.1843 |
| real-pair min / max | 0.0061 / 0.9805 | 0.0754 / 0.7818 |
| fraction ≥ 0.7 | 0.170 | 0.125 |

Spearman between the two on the 200 real pairs is **0.9719**, mean absolute difference **0.1423**,
and they disagree about support for **5 of 200 pairs at τ = 0.5 and 11 of 200 at τ = 0.7**. The
retired framing does not reach the ends of the interval: it never scores below 0.0754 or above
0.7818, so the correct invocation's decisive `0.98` support and `0.008` refutation both land in its
mushy middle. That is precisely the shape that matters for a threshold set at 0.7 on the *tail* of a
distribution.

## Decision

### 1. The verifier schedule moved: `verify.py` landed 2026-08-17, ahead of Gate G2

`src/biomedqa/verify.py` was committed 2026-08-17 (commit f447d27). It is THE single MiniCheck
forward pass in the package. No other file may contain a MiniCheck inference call.

The implementation follows the reference exactly (`github.com/Liyan06/MiniCheck`, `minicheck/inference.py`).
MiniCheck-Flan-T5-Large is a `T5ForConditionalGeneration` with no classification head, so inference
is not a classification forward pass — it is a single decoder step:

1. Render `"predict: " + document + tokenizer.eos_token + claim` as the encoder input.
2. Decode one step from a zero (pad) `decoder_input_ids`.
3. Softmax the first-position logits at exactly two vocabulary ids: **3** (`"▁"`, first sub-token of
   `"0"` = unsupported) and **209** (`"▁1"` = supported). Column 1 is the support probability.
4. For long documents, chunk to ~500 words (sentence-aligned via `chunk_document`); the pair score is
   the **max over chunks**.

The two token ids are asserted against the loaded tokenizer at startup (`_check_decision_tokens`),
so a tokenizer mismatch fails loudly rather than returning a wrong number silently.

`verify.py` exposes:

- `MiniCheckVerifier(model_id, batch_size, device, max_model_len, chunk_words, fp16)` with
  `.score_pairs(pairs) → list[VerifierScore]` where `VerifierScore.score` is a continuous support
  probability in [0, 1].
- `JudgeVerifier` (Opus 5) behind the same `.score_pairs` interface.
- `score_map(pairs, verifier) → dict[pair, float]` (deduplicates before scoring).
- `phi_from_scores(scores, threshold) → Phi` and `MINICHECK_DEFAULT_THRESHOLD = 0.5`.
- `minicheck_input` and `chunk_document` (the rendering and chunking primitives, exported for tests).

Raw continuous scores are the only output of `.score_pairs`; the threshold lives exclusively in
`phi_from_scores` at scoring time and is never baked into the stored scores.

**What did NOT move with the verifier.** W6's other items remain on their original schedule:
AlignScore (~355M, the never-cut second row for Table 3) and the human annotation pilot pass
(10 claims, 3 annotators, ROADMAP W6). Pulling only `verify.py` forward was sufficient to unblock
the G2 re-read; those items have no bearing on the length-sensitivity question.

### 2. The confusability-probe scores are a measurement error and must not be quoted until re-run

#### The mechanism

MiniCheck-Flan-T5-Large has no classification head. Every input, regardless of how it is formatted,
produces a logit vector — the model does not fail or warn when given an off-distribution prompt. A
wrong prompt returns a wrong number with no diagnostic signal.

The reference rendering is:

```
predict: {document}</s>{claim}
```

decoded for one step, with the support probability read from ids 3 and 209.

`scripts/confusability_probe.py` rendered:

```
premise: {p} hypothesis: {h}
```

and scored by comparing the sequence-generation loss for the strings `"1"` and `"0"` truncated at
`max_length=512`. This is off-distribution for this checkpoint on both dimensions: the prompt string
and the scoring mechanism differ from the reference.

#### Consequences for ADR-0012's reported numbers

Every number in ADR-0012 §2's result table was produced by the mis-invoked model:

- Mean 0.4245, median 0.3802, p90 0.7376.
- The threshold set post-hoc, τ_confusable = 0.7.
- The tail rates at τ = 0.7: 14.5% (retrieved distractors) vs 2.1% (uniform-random control),
  ratio ~6.9×.
- "35 of 100 dev questions carry at least one distractor that plausibly entails a gold claim."

These numbers must not be quoted in the paper until the probe is re-run with the correct
invocation. What the format check licenses saying, and no more: the *ordering* the probe reads
largely survives (Spearman 0.9719 on real pairs, and the retired framing ranks all six
known-answer pairs correctly), so the probe's qualitative claim — that the confusable mass is in
the tail, not the mean — is the part most likely to reproduce. Everything the probe *reports* is a
quantity at or above a cutoff, and that is the part the compression moves: the retired framing's
range is `[0.0754, 0.7818]`, so a threshold at 0.7 sits near its ceiling and inside the reference
distribution's ordinary upper range, and 11 of 200 real pairs cross it in one direction or the
other.

τ_confusable = 0.7 was set after seeing the distribution (as ADR-0012 §2 licensed), so it is
doubly provisional: it was calibrated on wrong scores, and it will be re-set after the re-run. The
sign-test p = 0.012 on the per-question contrast is a paired statistic on both arms' scores and
is pending for the same reason.

**The probe gates nothing** (ADR-0012 §2) — no corpus, retriever, or model is tuned against it —
so this error does not invalidate the index, G1, or any gate reading. It is a measurement error in
a diagnostic.

#### Defences committed

Three controls are now in place so this class of error cannot silently recur:

1. **Token-id assertion at load.** `_check_decision_tokens` in `verify.py` decodes ids 3 and 209
   from the loaded tokenizer and asserts they map to `"▁"` and `"▁1"` respectively. A wrong
   tokenizer raises immediately.

2. **Rendering pinned by test.** `tests/test_verify.py` asserts that `minicheck_input(document,
   claim)` produces the exact string `"predict: {document}</s>{claim}"` (using the tokenizer's
   `eos_token`). The test runs on CPU with no GPU required.

3. **Known-answer scoring against real weights.** `scripts/minicheck_format_check.py` scores
   curated pairs against the actual checkpoint weights and commits the result as
   `docs/harvest/minicheck_format_check.json`. A fixture cannot verify that the weights were trained
   on the string being rendered; only scoring a known-answer pair against the real checkpoint can.
   The size of the divergence between the correct and mis-invoked formats is recorded there, and
   the known-answer separation is the number to watch: **+0.9351 against +0.6139**. A φ that
   cannot put a stated paraphrase a full point above its own negation is not a φ.

## Consequences

- **The G2 re-read will use MiniCheck φ, not the NLI placeholder.** The length-sensitivity question
  named in `first_citation_f1.md` will be answered by the re-read rather than carried as an
  unresolved confounder into the gate.
- **ADR-0012's confusability-probe numbers are quarantined.** They must not appear in any paper
  draft until the probe is re-run at the correct format. The setup motivation (that the uniform
  pool contains plausible mis-citation targets) is still defensible in prose, and the format check
  says the *ordering* it rests on is stable; the specific rates, the ratio, and τ are not.
- **The confusability probe re-run is a W5 blocker for the setup section.** It is a CPU-side
  scoring pass over already-retrieved passages; no index rebuild is required.
- **`biomedqa.verify` is the only location a MiniCheck forward pass may live.** Any future script
  that needs entailment scores calls `MiniCheckVerifier.score_pairs` or `score_map`. A second
  inference site is a policy violation regardless of how it renders the input.
- **`docs/harvest/first_citation_f1.md` is not revised.** It is a committed measurement artifact
  recording φ = NLI-deberta with that caveat stated. The re-read at W5/W6 is a new artifact.

## Alternatives rejected

- **Keep the verifier on the W6 schedule and interpret G2 with the NLI placeholder.** Rejected
  because the leading alternative explanation for the G2 deficit (premise-length sensitivity of
  sentence-pair NLI) is exactly the property MiniCheck was pulled forward to resolve. A G2 pass or
  fail under the placeholder is uninterpretable on the axis that matters.
- **Treat the invocation error as a code smell and patch it quietly.** Rejected because the numbers
  it produced were recorded as results in ADR-0012, quoted in the confusability-probe harvest
  artifact, and may have been referenced in draft prose. A quiet patch leaves stale numbers in
  flight. The error is named here so every downstream consumer knows which numbers are pending.
- **Re-run the probe immediately and update ADR-0012 in place.** Rejected because (a) ADR-0012 is a
  committed decision record and editing its numbers in place would destroy the audit trail, and (b)
  `minicheck_format_check.py` is still running as this is written. The re-run result goes into a new
  harvest artifact, not a retroactive edit.
- **Pre-commit a threshold for the re-run.** Rejected for the same reason ADR-0012 §2 rejected it:
  the first distribution from the correct invocation is the first information anyone has; τ is set
  after seeing it, and that is honest because the probe gates nothing.
