# W4 live generation smoke — run 4 (`30186e7`)

`scripts/generate_smoke.py`, 3 dev questions × 3 systems, live vLLM on the A4000 serving
`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, depth 10, `temperature=0.0`, `max_tokens=1536`.
Artifacts: `generate_smoke.{summary.json,records.jsonl,costs.jsonl}`.

**Not a gate run and not a sample.** Three questions in file order say nothing about G0's rate. What
this run is for: proving the live path — prompt render, HTTP call, parse, citation location, schema
write — carries a citation-bearing generation end to end.

## Result

| system    | n | clean parses | cap violations | stages | mean claims | median latency |
|-----------|---|--------------|----------------|--------|-------------|----------------|
| joint     | 3 | 0            | 0              | 1      | 3.33        | 8.57 s         |
| post_hoc  | 3 | 1            | 0              | 2      | 5.0         | 12.95 s        |
| vanilla   | 3 | 3            | 0              | 1      | 5.0         | 3.05 s         |

Stage-count check passed. **Zero cap violations across all nine records** — the `c1` attribution
corruption that iterations 2–3 chased is gone, and the positional CITE grammar is what removed it.
`post_hoc` on `10757151` is the existence proof: 10 citations, 0 errors, spans in two different
passages. The 8B model can copy verbatim at this context depth; it does not do so reliably.

## The two residual failure classes

**1. Quote mutation (7 of 8 failing lines).** The model emits a semantically correct, near-verbatim
sentence that is not an exact substring of the passage. Checked against the committed dev contexts:
the source sentences exist verbatim (e.g. `There was no association between utilisation rates for
CEA a…` is present in `10375486:0`), so the divergence is in the generated string, past the 60-char
truncation in the parser's error message. Not retrieval, not `locate_quote`, not corpus formatting —
a whitespace scan over all 1,000 dev passages found 481 `[a-z][A-Z0-9]` hits and every sampled one is
legitimate scientific text (`mL`, `2peak`, `p5`).

**2. Chunk-suffix drop (1 record, 6 lines).** `joint` on `10757151` cites `pubmed23n0553_19170`,
which the parser rejects as absent from the context. The passage is present — at **rank 1**, as
`pubmed23n0553_19170:0`, the only chunk of that document in the block. The model dropped `:0`. This
is not a hallucinated identifier and not a grounding failure; it is the exact defect iteration 3
addressed, which took in `post_hoc` and did not take in `joint`.

## Decisions

**No fifth prompt cycle.** `PROMPT_ITERATIONS` stands at `joint: 4, post_hoc: 4`
(`effort_is_matched()` true). Another edit costs a cycle on *both* systems to keep the ledger
matched, and the number is reported in the paper. A generic "copy verbatim, keep the `:N` suffix"
restatement re-spends attention already spent in iterations 2 and 3 on an instruction the model
follows sometimes and ignores otherwise. Both residual classes are the generator's format
compliance, which is what G0 measures. Buying a better G0 score with prompt cycles is the thing the
equal-effort ledger exists to make visible.

**The parser stays strict, on both classes.** Two normalizations were available and both were
refused:

- *Fuzzy quote matching.* Refused previously and again here. `locate_quote` writes `char_start`/
  `char_end` that every downstream verifier reads as ground truth; a fuzzy hit fabricates a span for
  text the passage does not contain.
- *Resolving a bare document id to its unique chunk in the context.* This one fabricates no span —
  the quote would still have to match verbatim inside the resolved passage — and it would take
  `joint`/`10757151` from six errors to zero. Refused anyway: it rescues our system in the one
  comparison C2 rests on, and a normalization that only ever fires for the arm it favours is
  indistinguishable from tuning. If a reviewer asks why joint's format compliance is lower, the
  answer is a number, not a parser accommodation.

## What this does not license

The run cannot support "Llama 3.1 8B cannot do G0 citation copying" — one clean citation-bearing
generation in three refutes the categorical form. It also cannot support any rate. G0's rate comes
from the gate run.
