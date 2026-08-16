# The distractor confusability probe, re-run through the corrected verifier — 2026-08-17

ADR-0020 §2 quarantined every number the 2026-08-10 probe produced, because the MiniCheck call that
produced them was not MiniCheck's own invocation. `verify.py` landed with the reference invocation
(commit `f447d27`), so ADR-0012 §2's probe is re-read here. **This is the same probe, not a new one** —
the W2 first observation and the W3 post-rerank re-confirmation, both re-scored.

```
uv run python scripts/confusability_probe.py --index-dir data/index/empty --split dev \
  --out docs/harvest/confusability_probe_v2.json

uv run python scripts/confusability_probe.py --index-dir data/index/empty --split dev \
  --random-control docs/harvest/confusability_probe_v2.json \
  --out docs/harvest/confusability_probe_v2_control.json

uv run python scripts/confusability_probe.py --index-dir data/index/empty --split dev --rerank \
  --out docs/harvest/confusability_probe_v2_reranked.json

uv run python scripts/confusability_probe.py --index-dir data/index/empty --split dev \
  --random-control docs/harvest/confusability_probe_v2_reranked.json \
  --out docs/harvest/confusability_probe_v2_reranked_control.json
```

Four artifacts, beside the four v1 files they supersede: `confusability_probe_v2.json`,
`..._v2_control.json`, `..._v2_reranked.json`, `..._v2_reranked_control.json`. The controls carry
`paired_against`, which is what says which arm each belongs to; the committed files record the run
box's own output paths there (`/home/user/sanity_out/...`) rather than the `docs/harvest` paths above.
`--rerank` is inert in control mode and the script says so when passed — the control never retrieves,
so its pairing is set by `--random-control` alone.

## Only φ's invocation moved

Same script (`scripts/confusability_probe.py`), same index (`data/index/empty`, corpus
`pubmed-2m-v1`, fingerprint `93321598f3f1`), same `dev` split, same `--top-k 5` RRF fusion at
`rrf_k 60`, same control seed 12345, same gold sentences, and the same passage counts: **427 non-gold
passages over 100 dev questions pre-rerank, 414 reranked**, identical to v1. **The retrieval side was
not re-run differently and no index was rebuilt** — this was a CPU-side re-scoring pass over the
passages the same retrieval already surfaced.

What changed is the call into MiniCheck, and only that:

| | v1 (2026-08-10) | v2 (2026-08-17) |
|---|---|---|
| prompt | `"premise: {p} hypothesis: {h}"` | `"predict: {document}</s>{claim}"` |
| decision | sequence-loss comparison of the strings `"1"` / `"0"` | one decoder step, softmax over first-position logits at ids 3 and 209 |
| premise handling | `max_length 512` truncation | 500-word sentence-aligned chunks, max over chunks, 2048 truncation |

## Read the caveats before the numbers

1. **v1 was not merely noisier — it was compressed, and that is a directional distortion.**
   `minicheck_format_check.json` (2026-08-17) put the two invocations against each other on 200 real
   (cited span, claim) pairs plus 6 known-answer pairs: **Spearman 0.9719**, mean |difference|
   0.1423, so the retired framing ranks nearly correctly. What it does not do is span the interval —
   its range is **[0.0754, 0.7818]** against the reference's **[0.0061, 0.9805]**, and known-answer
   separation (mean supported − mean unsupported) is **+0.6139 retired against +0.9351 reference**.
   ADR-0020 §2's diagnosis is **compression, not inversion**. Everything below follows from that:
   rank-order conclusions had a chance of surviving, and every *level*, *rate* and *ratio* did not.
2. **The probe still gates nothing.** ADR-0012 §2 licensed setting τ_confusable post-hoc precisely
   because no corpus, retriever, reranker or model is tuned against this distribution. That is why a
   bug here forced a re-read rather than a re-decision: nothing downstream had been fitted to the
   wrong numbers. It also remains why τ may be chosen after seeing the distribution below.
3. **The score is a max over a question's gold sentences, over chunks of a passage.** A passage
   scores high if *any* gold sentence looks supported by *any* chunk of it. That is the right
   quantity for "could a system plausibly mis-cite this passage", and it is an upper bound on any
   per-sentence reading of the same data. Unchanged from v1 in intent; the chunking is new.
4. **100 dev questions.** Question-level counts below are read off 100 questions, and the two control
   draws disagree with each other by 4 questions on the same statistic (§τ) — that is the scale of
   sampling noise on any count in this document.

## The numbers

Retrieved distractors against the paired uniform-random control, both arms. `frac ≥ τ` is the
fraction of non-gold passages at or above τ; ratios are computed from those fractions.

### Pre-rerank (RRF top-5, 427 non-gold passages)

| | mean | median | p90 | max | min |
|---|---|---|---|---|---|
| retrieved | **0.3073** | 0.0982 | 0.9586 | 0.9903 | 0.0111 |
| random control | **0.0938** | 0.0418 | 0.1436 | 0.9727 | 0.0083 |

| τ | retrieved | control | ratio |
|---|---|---|---|
| 0.3 | 0.3208 | 0.0539 | **5.9×** |
| 0.4 | 0.2787 | 0.0539 | **5.2×** |
| 0.5 | 0.2623 | 0.0445 | **5.9×** |
| 0.6 | 0.2436 | 0.0422 | **5.8×** |
| 0.7 | 0.2295 | 0.0351 | **6.5×** |
| 0.8 | 0.2084 | 0.0304 | **6.9×** |

Paired per-question contrast: **mean delta +0.2151, median +0.1042, retrieved higher in 85 of 100
questions, control higher in 15, 0 ties, exact two-sided sign test p < 1e-6.** Both controls' `summary.paired_q_mean_delta.sign_test_p`
field *rounds away* a value on the order of 1e-11 at the artifacts' 4-decimal precision; the p-value
is not zero and must never be quoted as zero. The pooled means differ by 0.2135 while the paired statistic reads
+0.2151, because the paired delta weights questions equally and the pooled mean weights passages.

### Reranked (cross-encoder between pool and top-k, 414 non-gold passages)

| | mean | median | p90 | max | min |
|---|---|---|---|---|---|
| retrieved | **0.3149** | 0.1061 | 0.9552 | 0.9879 | 0.0095 |
| random control | **0.0937** | 0.0395 | 0.1594 | 0.9724 | 0.0083 |

| τ | retrieved | control | ratio |
|---|---|---|---|
| 0.3 | 0.3116 | 0.0652 | **4.8×** |
| 0.4 | 0.2850 | 0.0531 | **5.4×** |
| 0.5 | 0.2754 | 0.0507 | **5.4×** |
| 0.6 | 0.2609 | 0.0386 | **6.8×** |
| 0.7 | 0.2488 | 0.0314 | **7.9×** |
| 0.8 | 0.2222 | 0.0290 | **7.7×** |

Paired per-question contrast: **mean delta +0.2217, median +0.1614, retrieved higher in 84 of 100,
control higher in 16, 0 ties, sign test p < 1e-6.**

**The arms separate at every cutoff and in the mean, in both retrieval configurations** — 3.3× on the
mean pre-rerank (0.3073 / 0.0938) and 3.4× reranked (0.3149 / 0.0937), and between 4.8× and 7.9× on
every threshold in the sweep. ADR-0012 §2's question is answered affirmatively: the uniform pool
contains passages a system could plausibly mis-cite, at a rate several times what drawing from the
2,162,838-passage corpus at random produces.

## What changed against v1

### The anomaly is gone, and it was a symptom of the mis-invocation

v1's own recorded defect was that **at ≥0.3 the random control scored *higher* than the retrieved
passages — 67.7% against 62.1%** — so the probe's headline comparison ran backwards, and separation
existed only in the tail (14.5% vs 2.1% at ≥0.7). The v1 write-up recorded that and could not explain
it. Under v2 there is nothing to explain:

| | v1 | v2 |
|---|---|---|
| frac ≥ 0.3, retrieved vs control | **62.1% vs 67.7%** (control ahead) | **32.1% vs 5.4%** (retrieved ahead, 5.9×) |
| retrieved mean vs control mean | 0.4245 vs — (means did not separate usefully) | **0.3073 vs 0.0938** |
| retrieved higher, of 100 questions | **62** (p = 0.012) | **85** (p < 1e-6) |
| paired mean delta | +0.0524 (reranked arm) | +0.2151 |

**The anomaly was the compression.** The retired framing squeezed every score into [0.0754, 0.7818]
and the v1 probe's own observed range shows it: **min 0.1345, max 0.8312, mean 0.4245, median
0.3802** — a unimodal lump in the middle of the interval, with no mass anywhere near either end. A
cutoff at 0.3 then sits *below* almost the entire distribution, so `frac ≥ 0.3` measures how much
mass a run happened to place above a floor rather than how many passages look supportive, and the
retrieved and control arms are indistinguishable on it — or ordered by accident. Compression flattens
exactly the distinction the control exists to expose.

**The correct invocation is bimodal instead.** Pre-rerank **median 0.0982 with p90 0.9586**, min
0.0111 and max 0.9903: most retrieved distractors are decisively *un*supportive and roughly a fifth
are decisively supportive, with little in between. The mean is a mixing weight, not a central
tendency — 0.3073 against `frac ≥ 0.5` of 0.2623 is what a two-point distribution looks like when
summarised by its mean. The control has the same shape with far less high mass (median 0.0418, p90
0.1436, frac ≥ 0.8 of 0.0304).

### The direction survives; the magnitudes do not

The one v1 conclusion stated in ratio form does survive. Tail enrichment at 0.7 was **~6.9×** in v1
and is **0.2295 / 0.0351 = 6.5×** in v2.

**That stability is a coincidence and should be reported as one.** Both terms of the ratio moved, and
moved together: retrieved **14.5% → 23.0%** (a factor of 1.6) and control **2.1% → 3.5%** (a factor
of 1.7). The ratio is nearly unchanged because the numerator and denominator were compressed by
similar amounts, not because v1 measured the enrichment correctly. Nothing licenses reading the v1
ratio as having been right; what happened is that a monotone distortion (Spearman 0.9719) preserved
an ordering and a quotient of two tail fractions while corrupting both fractions.

**ADR-0012 §3 stays untriggered.** The distractor-pool redesign it holds in reserve was not triggered
by v1, and v2 argues against it more strongly than v1 did: the pool is harder than chance at every
cutoff, not only in the tail.

## Reranking still does not change confusability

This is what the W3 re-confirmation was for, and it reaches the same conclusion v1 reached — now on
correct numbers.

| quantity | pre-rerank | reranked |
|---|---|---|
| mean | 0.3073 | 0.3149 |
| median | 0.0982 | 0.1061 |
| p90 | 0.9586 | 0.9552 |
| frac ≥ 0.7 | 0.2295 | 0.2488 |
| frac ≥ 0.8 | 0.2084 | 0.2222 |
| paired: retrieved higher of 100 | 85 | 84 |
| non-gold passages | 427 | 414 |

**The cross-encoder changes which distractors reach the generator without changing how confusable
they are.** Every summary statistic moves by less than 0.02, the paired count moves by one question,
and both arms clear their controls by the same margin. v1 said this from mean 0.4244 vs 0.4245 and
`frac ≥ 0.7` 0.1425 vs 0.1452; v2 says it from the numbers above.

**One caveat that v1 could not see.** At passage level the reranked and pre-rerank tails are the same
size — **98 of 427 passages at or above 0.7 pre-rerank, 103 of 414 reranked** — but those passages are
spread over **more questions** after reranking (§τ below: 49 questions against 40). The distribution
is unchanged; the *concentration* is not. This is a redistribution of the same mass, and on 100
questions with a control draw that itself varies by 4 questions, it is not an effect worth naming.
It is recorded because it is the only place these two arms disagree.

## τ_confusable

ADR-0012 §2 permits setting this after seeing the distribution, because the probe gates nothing.
**Keep τ_confusable = 0.7 — but for a different reason than v1 had.**

v1 set 0.7 post-hoc because it was the only place on a compressed unimodal curve where the arms
separated at all. That justification is retired with the numbers that produced it. The v2
justification comes from the shape: **the distribution is bimodal, so any cutoff in the empty middle
gives nearly the same answer.**

| τ | retrieved frac, v2 | share of the τ = 0.3 count retained | retrieved frac, v1 | v1 share retained |
|---|---|---|---|---|
| 0.3 | 0.3208 | 100% | 0.621 | 100% |
| 0.4 | 0.2787 | 87% | — | — |
| 0.5 | 0.2623 | 82% | — | — |
| 0.6 | 0.2436 | 76% | — | — |
| 0.7 | 0.2295 | **72%** | 0.145 | **23%** |
| 0.8 | 0.2084 | 65% | — | — |

**Sweeping τ across the whole 0.3–0.8 range costs 35% of the count; the same sweep from 0.3 to 0.7
under v1 cost 77%.** Under v2 the answer is insensitive to τ over a 0.5-wide interval, which is the
operational meaning of bimodality and the reason the choice is close to free. Under v1 the answer
tracked τ steeply, which is what made a post-hoc choice load-bearing and uncomfortable. 0.7 is
additionally well clear of the chance distribution — the control's p90 is 0.1436, and only 3.5% of
random passages reach 0.7 at all — so it is a conservative point inside the empty region rather than
an edge of it. Continuity with the retired record is a convenience, not the argument.

**At τ = 0.7, 40 of 100 dev questions carry at least one non-gold passage a system could plausibly
mis-cite, against 12 of 100 by chance.** These are counts of questions whose `per_question[].q_max`
(the max distractor score for that question) reaches 0.7, read off
`confusability_probe_v2.json` and `confusability_probe_v2_control.json` respectively. The v1
statement this replaces was 35 of 100 against 8 of 100. On the reranked arm the same field gives
**49 of 100 against 8 of 100** — the 12-vs-8 gap between the two control draws bounds how finely any
of these counts should be read, since both draws use seed 12345 but draw per-question counts matched
to different arms (427 vs 414 passages).

## What this changes

- **The setup section can quote v2 numbers.** ADR-0020 §2's quarantine was explicitly pending this
  re-run, and the W5 checklist item that blocked the setup section from quoting "a rate, a ratio, or
  a threshold" is discharged: the rate is 40/100 questions at τ = 0.7 against 12/100 by chance, the
  ratio is 6.5× tail enrichment, and the threshold is 0.7 justified from the shape of the corrected
  distribution rather than from its position on the retired one.
- **The v1 numbers stay quarantined and stay on disk.** `confusability_probe{,_control,_reranked,_reranked_control}.json`
  are superseded, not deleted, and neither they nor ADR-0012, ADR-0020 or the ROADMAP entries that
  record them are edited. They are the record of what was actually run on 2026-08-10, and a
  quarantined number that has been overwritten cannot be audited.
- **The paper reports the re-run and the reason for it.** A probe that was re-run after a verifier bug
  is a probe whose history belongs in the record: the v1 anomaly (control ahead of retrieved at
  ≥ 0.3) was a real, published-in-repo symptom, the diagnosis was compression rather than inversion
  (Spearman 0.9719, known-answer separation +0.6139 against +0.9351), and the corrected read both
  removes the anomaly and leaves the conclusion's direction intact. Reporting that is stronger than
  reporting the corrected numbers alone, because it shows the anomaly was caught by the control the
  probe was designed with rather than by hindsight.
- **ADR-0012 §3 is not opened.** No distractor-pool redesign is owed. The uniform pool is hard enough
  for citation precision to mean something, which is the question ADR-0012 §2 asked.
- **Nothing about retrieval is reopened.** No index was rebuilt, no retrieval number moved, and the
  G1 relaxation (ADR-0015) is untouched by this read. The probe consumed already-retrieved passages.
- **The reranker's mandate is unaffected.** It changes which distractors the generator sees and not
  how confusable they are, so its justification remains retrieval quality, not distractor difficulty
  — the same conclusion W3 drew, now on numbers that support it.

## The three sentences for the setup section

> Retrieval draws distractors from a 2.16M-passage PubMed corpus, and the pool is adversarial by
> measurement rather than assumption: over 100 dev questions, 427 non-gold retrieved passages score a
> mean MiniCheck support probability of 0.3073 against 0.0938 for the same number of passages drawn
> uniformly at random, and the retrieved arm scores higher on 85 of the 100 questions (paired
> two-sided sign test, p < 1e-6). Setting τ_confusable = 0.7 — a threshold this probe gates nothing
> with, and one the distribution's bimodality makes nearly free, since sweeping it from 0.3 to 0.8
> changes the count by 35% — **40 of 100 questions carry at least one non-gold passage a system could
> plausibly mis-cite, against 12 of 100 by chance**, a 6.5× tail enrichment that cross-encoder
> reranking leaves intact (0.2488 against 0.2295 of passages at or above 0.7). These figures come
> from a re-run of the probe on 2026-08-17 after the original 2026-08-10 read was found to have used
> a non-reference MiniCheck invocation that compressed scores into [0.0754, 0.7818]; the direction of
> the original conclusion survived the correction and its magnitudes did not.
