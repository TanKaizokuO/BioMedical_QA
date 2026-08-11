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

## What the misses actually are

`scripts/quote_misses.py` re-reads the CITE lines out of `raw_generation`, diffs each rejected
quote against the passage it names, and buckets the mutation. Over the nine records, 19 misses:

| bucket        | n | joint | post_hoc | what it is |
|---------------|---|-------|----------|------------|
| `unknown_id`  | 6 | 6     | 0        | `:N` chunk suffix dropped from the passage id |
| `spliced`     | 5 | 1     | 4        | two real spans joined across deleted material |
| `reworded`    | 3 | 1     | 2        | word order, `RV` → `right ventricular`, `Fifteen` → `15` |
| `overrun`     | 2 | 0     | 2        | correct prefix, ended early with an added full stop |
| `case`        | 2 | 0     | 2        | first character only |
| `fabricated`  | 1 | 0     | 1        | quote absent from the passage entirely |

**Not a corpus problem.** The source sentences exist verbatim; a whitespace scan over all 1,000 dev
passages found 481 `[a-z][A-Z0-9]` hits and every sampled one is legitimate scientific text (`mL`,
`2peak`, `p5`). The divergence is in the generated string.

**`spliced` and `fabricated` are the finding.** Six of 19 misses assert something the passage does
not, in the syntax of a direct quotation. On `10375486:0` the passage reads *"…for CEA and district
stroke mortality (r=-0.06, 95% CI -0.41 to 0.30) or admission rates for stroke (r=0.17…)"*; the
model repeatedly emits *"…for CEA and admission rates for stroke (r=0.17, 95% CI -0.2 to 0.49)"* —
a grammatical sentence, a plausible statistic, and a sentence the source never contains. The
`fabricated` line quotes SPRINT-shaped adverse-event numbers at a passage about hypotension
correlates. A fuzzy matcher scores both as near hits.

**`unknown_id` is not a hallucination.** All six quotes are *exact* spans of
`pubmed23n0553_19170:0`, which sits at **rank 1** in `10757151`'s context and is the only chunk of
that document there. The model dropped `:0`. That record is otherwise clean: with suffix resolution
it would parse, and `joint` would read 1/3 rather than 0/3.

## Decisions

**No fifth prompt cycle.** `PROMPT_ITERATIONS` stands at `joint: 4, post_hoc: 4`
(`effort_is_matched()` true). Another edit costs a cycle on *both* systems to keep the ledger
matched, and the number is reported in the paper. A generic "copy verbatim, keep the `:N` suffix"
restatement re-spends attention already spent in iterations 2 and 3 on an instruction the model
follows sometimes and ignores otherwise. Every bucket above is generator format compliance, which
is what G0 measures. Buying a better G0 score with prompt cycles is the thing the equal-effort
ledger exists to make visible.

**The parser stays strict, on every class.** Two normalizations were available and both were
refused:

- *Fuzzy quote matching.* Refused previously and again here. `locate_quote` writes `char_start`/
  `char_end` that every downstream verifier reads as ground truth. The `spliced` and `fabricated`
  buckets are exactly the inputs a fuzzy matcher would accept, and they are the six lines where the
  model asserted something the passage does not say.
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
