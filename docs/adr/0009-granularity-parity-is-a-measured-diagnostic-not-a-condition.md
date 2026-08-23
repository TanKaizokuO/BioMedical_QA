# ADR-0009 — Granularity parity is a measured diagnostic, not a fifth equal-effort condition

**Status:** Accepted · **Date:** 2026-08-04 · **Decided in:** grilling session (G0 follow-up)
**Refines** ADR-0002's equal-effort protocol · **Constrained by** ADR-0005 (the attribution unit)
**Amended 2026-08-13** — §4's premise that post-hoc granularity is set by a separate `decompose.py`
decomposer was wrong; the live baseline emits claims directly, so the loop's lever is
`POST_HOC_ANSWER_TEMPLATE`. See *Amendment* under §4. The decision it supported — tune post-hoc only,
blind, hard-10 — is unchanged. Made on the user's explicit instruction, in preference to a new ADR.
**Terminated 2026-08-14** — the loop closed at **1 of 10 iterations** on `parity_iter1b`, with the
gate passing on every basis. §6's blind is lifted; the W9 stratified check survives it. See
*Termination* at the end.
**Amended 2026-08-15** — the *Termination* section's premise that the `_claim_rules()` fix for the
731-word joint claim "lands before the first citation-F1 is read" expired: F1 was read the day
before, 2026-08-14. See *Second amendment* at the end for the resulting decision — the guard stays
on the scoring side, `_claim_rules()` is not touched pre-freeze.
**Amended 2026-08-17** — the 731-word claim defect is discharged on the parser, splitter, and
decoding sides, with every prompt still frozen. See *Third amendment* at the end.
**Amended 2026-08-23** — five joint-side granularity edits (`045a96c`, `95dd958`, `dab7a68`,
`dc08914`, `b29e74c`) are **reverted**: they tuned `JOINT_JSON_TEMPLATE`'s claim length to make §5's
W9 check pass, which §4 confines to post-hoc and §6 forbids post-unblinding, and which §1/§3/§5
never made a Gate G2 precondition. §5's scrutiny is discharged by length standardisation instead —
the granularity gap transmits *against* C2. Run of record is `generate_fp05_n100_guided_v4`. See
*Fourth amendment* at the end.

## Context

C2's headline number is citation-F1, computed over **claims**. Joint generation emits claims
natively; the post-hoc baseline produces prose that `decompose.py` then splits. **Two different
mechanisms produce the unit the metric is denominated in.**

Coarser claims are harder to entail per claim, so if post-hoc's claims are systematically coarser,
post-hoc is systematically penalised — and **C2's gap appears without joint grounding doing any
work.** The bias points *toward* the hypothesis, which is the direction that must never go
unmeasured.

**What the evidence does and does not show.** G0 measured **9.2 claims/query (Llama) vs 3.8
(Qwen)** — same prompt, same passages. That is a divergence between two *models*, and D1 then fixed
the generator, so joint and post-hoc both run
`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`. **Mechanism-driven divergence has never been
measured.** It may be large; it may be 5%. This ADR builds the instrument that will tell us, and
deliberately does not assume the answer.

## Decision

### 1. Parity is a diagnostic, listed separately from the four enforced conditions

ADR-0002's protocol holds four conditions that are true **by construction** — you configure them and
they hold, and `schema.validate()` even checks the third:

| Condition | How it holds |
|---|---|
| same retriever / passages / *k* | set |
| same generator backend | set |
| same 3-citation cap + prompt token budget (±10%) | set, and checked in code |
| matched prompt-iteration budget | counted |

Granularity parity is **not** like these. It is an empirical outcome of prompt tuning: you aim at it
and then discover whether you hit it. Listing it as a fifth condition invites the reader to assume
all five hold, and converts a near-miss into a **disclosed failure of our own fairness protocol** —
*"by the authors' own criterion the comparison is unfair"* — which is a worse position than not
having named it, and strictly worse than naming it honestly.

**It is therefore reported as a separately-labelled measured fairness diagnostic:** four conditions
enforced, one quantity measured and disclosed whatever it says.

### 2. The gated quantity is median words/claim; claims/query is reported

Median words/claim drives per-claim entailment difficulty, which is the mechanism of the bias.
`claims/query` tracks answer length and is reported, not gated.

> **Open, deferred to W4 (see "Known weaknesses").** With total answer length already constrained to
> ±10% by the third enforced condition, words/claim and claims/query are near-mechanically linked.
> The choice is re-examined against the first real measurement.

### 3. Tolerance ±15%, dev only, pre-committed now and unmeasured

**±15% is fixed before any measurement exists, and it is not revised afterwards.** A tolerance chosen
after seeing the divergence is not a pre-commitment — it would be set to a number already known to be
reachable, possibly already reached, making the loop a no-op. Full blinding (§6) protects against
steering on F1; nothing but pre-commitment protects against steering on the tolerance itself.

**The tolerance does not need to be achievable.** Missing it is survivable by design — see §5.

### 4. Only the post-hoc decomposer is tuned

> **Amended 2026-08-13 — the lever is `POST_HOC_ANSWER_TEMPLATE`, not `decompose.py`.** See the
> *Amendment* block below. "The post-hoc decomposer prompt" in this section names the post-hoc
> answer template in the shipped code; the decision is unchanged.

The parity loop may edit **the post-hoc decomposer prompt only**. The joint prompt is out of bounds
for the loop's duration.

Joint's granularity is *native* — the model emits claims directly — so the only knob that moves it is
the joint prompt. A loop with both arms in scope drifts into tuning the method itself and booking it
to a line charged to nobody, which is precisely what §7's ledger treatment claims to prevent. One
direction also makes §6's blinding meaningful: a blind loop free to touch both arms is blind about
*outcomes* but unconstrained about *actions*.

With one direction fixed, the effort demonstrably went into the **baseline**, so charging it to
neither system makes the reported baseline effort an *undercount* — the safe direction to be wrong
in, and the one that strengthens the answer to objection 7.

#### Amendment, 2026-08-13 — the tunable prompt is `POST_HOC_ANSWER_TEMPLATE`, not `decompose.py`

**What was wrong: the premise about where post-hoc's granularity is set.** This section (and
`research_roadmap.md` §8 rule 8) assumed the post-hoc baseline produces prose that a separate
`decompose.py` splits into claims — so "the post-hoc decomposer prompt" read as the decomposer's
prompt. The shipped baseline does not work that way. `POST_HOC_ANSWER_TEMPLATE` emits `CLAIM` lines
**directly** (`prompts.py`; the cite stage only attaches quotes), and `CONTEXT.md`'s attribution-unit
row confirms it — "the method generates claims directly; there is no separate sentence layer." There
is no prose→decompose step on the C2 headline path. `decompose.py` is the **C7 / Table-2
granularity-ablation** tool (sentence / atomic / decontextualized rows, due W5), off the C2 path.

**What this does not change: the decision.** Tune the post-hoc arm only; the joint prompt stays out
of bounds; hard-10 or Aug 30; blind throughout. The reasoning in this section holds verbatim once
"the post-hoc decomposer prompt" is read as **`POST_HOC_ANSWER_TEMPLATE`**.

**What it does change: which knob the loop may turn.** The lever is `POST_HOC_ANSWER_TEMPLATE`'s
framing. It is **not** `_claim_rules()` — that grammar is shared by all three systems for fairness
(a baseline whose claims are shaped differently is compared on the wrong axis), so editing it would
move *joint* too and break "joint out of bounds." And it is **not** `decompose.py`: building a
decomposer to run the loop would rebuild the abandoned prose→decompose architecture and count as
method development, which §4 and §7's ledger exist to prevent.

### 5. Exactly 10 iterations, or Aug 30, whichever comes first

**A hard 10.** Not "~10" — a bound written with a tilde grants exactly the permission it exists to
deny, and pairing a soft counter with "never tune until it passes" makes the prohibition decorative.

**A hard calendar drop-dead of Aug 30 (end of W4).** Hard-10 bounds the *work*; nothing otherwise
bounds the *calendar*, and ten iterations of prompt tuning can absorb an unbounded number of days.
The loop stops on Aug 30 whether or not parity is achieved.

*The original "frozen before the first W4 run (Aug 24)" was unimplementable.* `research_roadmap.md`
§5 builds joint generation **and** both baselines in W4 — the loop compares systems that W4 is
creating. More fundamentally, under §6 the parity freeze and the first citation-F1 computation are
**the same event**, so there was never a separate freeze to schedule; what was needed was a
termination deadline that leaves G2 runway.

**One-sided fallback on the residual gap** (observable while blind — it is the *granularity* gap in
words/claim, not the F1 gap):

- residual gap **favouring C2** (post-hoc coarser) → the **stratified robustness check becomes
  mandatory**, scheduled in **W9**
- residual gap **running against C2** → note it and proceed

The asymmetry is deliberate: demand more scrutiny when the residual bias points toward the
hypothesis, less when it points away. **It is pre-registered in the paper's methods section**, not
only here — asymmetric scrutiny disclosed in advance reads as rigour; the same rule disclosed
afterwards reads as post hoc.

### 6. The loop is fully blind

**Citation-F1 is not computed on any split, in any form, until the loop terminates.** No burn slice,
no mid-loop checkpoint, no correlated proxy.

**The accepted cost, stated plainly:** first citation-F1 lands ≈ **Aug 31**; G2 is **Sep 6**. If C2 is
null, R5 and Phase 2's contingency must fire inside a six-day window. **R5's trigger is therefore
pre-armed** rather than improvised. A single pre-committed unblinding on a disjoint burn slice was
considered and rejected — it would have bought roughly a week of warning at the cost of 20 dev
questions and a carve at the Aug 7 split freeze.

### 7. Parity tuning is a third disclosed ledger line, charged to neither system

A fairness-control cost, not method development — sound because §4 confines the tuning to the
baseline.

### 8. Decomposer/granularity freeze Sep 3; guidelines in two passes

**Sep 3** is a named, dated artifact three days before G2, protecting the gold set that launches
Sep 7 — ADR-0005 establishes that changing granularity after W6 orphans it.

Annotation guidelines are written in two passes:

- **from Aug 31** — unit-independent rules: no-outside-knowledge, the SUPPORTED/PARTIAL boundary,
  hedging, numerics, jointly-necessary citations
- **Sep 3–6** — worked examples only, built from frozen decomposer output

## Consequences

- **ADR-0002's protocol is restated as "four enforced conditions plus one measured diagnostic"** in
  `research_roadmap.md` §4 Phase 2 and in the paper's setup section.
- **W9 gains the stratified robustness check** — on the evidence so far (joint emits finer claims
  natively) the triggering branch is the *likely* one, not the unlikely one.
- **The paper's methods section gains the pre-registered asymmetric rule.**
- **The six-day first-F1-to-G2 window is a schedule fact**, not a contingency. R5's response is
  decided before Aug 31.
- **The compound-claim safety net fires late.** §2 delegates compound claims to `claim_validity`,
  which is annotated from W6 — *after* the Sep 3 freeze. Parity on words/claim cannot distinguish
  one long atomic claim from one compound claim of equal length, and nothing catches that before the
  freeze.

## Known weaknesses

Recorded rather than resolved, because a future reader will find them anyway:

1. **The motivating measurement is of the wrong contrast** (models, not mechanisms) — see Context.
2. **words/claim may be near-equivalent to claims/query** under the ±10% length condition, making §2's
   claimed independence weaker than stated. Re-examined in W4.
3. **±15% has no empirical basis.** It is a pre-committed yardstick, chosen for the property of being
   fixed in advance rather than for being calibrated.

## Alternatives rejected

- **A fifth enforced condition** (as originally drafted). Creates the "failed your own criterion"
  attack surface for a quantity that cannot be enforced by construction.
- **Tolerance derived from the first measurement.** Not a pre-commitment; steerable in exactly the
  way §3 exists to prevent.
- **Both arms tunable, joint-side iterations charged to joint's ledger line.** More faithful
  accounting, but it re-couples the loop to the method and weakens §6.
- **One pre-committed unblinding on a 20-question burn slice.** Buys ≈ a week of early warning on the
  highest-risk claim in the project; rejected in favour of an unbroken blind, accepting the six-day
  window.
- **Doing nothing.** The bias is real and points toward the hypothesis. Unnamed is the one outcome
  worse than named.

## Termination, 2026-08-14 — closed at 1 of 10 iterations on `parity_iter1b`

**Recorded in code as `prompts.PARITY_LOOP_CLOSED`, not only here**, because under §6 the freeze and
the first citation-F1 computation are the same event, so something has to be able to *check* that the
loop is closed before F1 runs. `scoring.citation.citation_f1` raises while it is `None`, and
`tests/test_prompts.py` pins `POST_HOC_ANSWER_TEMPLATE`'s SHA-256 to the terminating run.

### The verdict

`parity_iter1b` — 100 dev questions × 3 systems, `--max-tokens 3584`, server `--max-model-len 14336`,
otherwise identical to `parity_iter1`. **No prompt changed**, so it charges no iteration: a cap raised
for all three arms at once is shared run config (the precedent `parity_iter0` → `parity_iter0b` was
taken under).

| basis | joint | post_hoc | gap | gate |
|---|---|---|---|---|
| all 100 records (gated) | 15 | 17 | **+13.3%** | PASS |
| untruncated per arm (92 / 84) | 14 | 16 | +14.3% | PASS |
| untruncated, same 78 queries both arms | 15 | 16 | +6.7% | PASS |

The baseline of record (`parity_iter0b`) fails all three: +25.0% / +42.9% / +37.9%. Full argument,
including the coverage and compound-marker checks that rule out "the model answered less":
`docs/harvest/parity_iter1b.md`.

### Why it stopped one iteration in rather than spending the other nine

**Not because the budget or the calendar ran out — because the instrument did.** `parity_iter1` and
`parity_iter1b` ran the **same post-hoc prompt** and read **+0.0%** and **+13.3%** on the same basis.
The gated statistic is an integer median of 14–20 words, so its resolution is **one word (~6.7%)** and
§3's ±15% is **two words wide**.

Verdicts therefore carry a query-level bootstrap (`scoring.granularity.gap_bootstrap_ci`):

| run | 95% interval, all records |
|---|---|
| `parity_iter0b` | [+18.8%, +40.0%] — outside ±15% throughout |
| `parity_iter1b` | **[+0.0%, +14.3%] — inside ±15% throughout** |

Non-overlapping, so iteration 1's edit closed a real gap. The residual is one grid step, and further
iterations would be fitting run-to-run noise — which §3's pre-commitment cannot protect against,
because it constrains the tolerance, not the number of times a noisy statistic is redrawn.

**§3's tolerance was not revised, and is not.** What changed is that a verdict is quoted with its
interval.

### What survives termination

- **§5's W9 stratified robustness check stays mandatory.** The residual favours C2 on every basis
  (post-hoc's claims are still the coarser ones). A pre-registered asymmetric check is not retracted
  because the iteration that closed the loop passed; that retraction is the post-hoc steering §3 and
  §6 exist to prevent.
- **§7's ledger line reads 1 cycle, charged to neither system.** `iteration_counts()` is unchanged at
  joint 4 / post_hoc 4.
- **The paper reports the diagnostic as measured**, per §1: four enforced conditions, one measured
  quantity, disclosed as +13.3% [+0.0%, +14.3%] with the two other bases and the resolution caveat.
- **`parity_budget_remains()` is still `True` and that is not permission.** Nine unspent iterations
  are not a licence to tune post-hoc now that F1 is knowable; the template freeze is what enforces it.

### Known weakness #2, closed by measurement

§2 deferred "words/claim may be near-equivalent to claims/query under the ±10% length condition" to
W4. It is not: across `parity_iter0b` → `parity_iter1`, post-hoc's median words/claim fell **20 → 16**
while its median claims/query rose **8 → 10** — opposite directions, one prompt edit, same run pair.
The two are separable in practice, so §2's choice to gate the first and report the second stands.

### One defect found in the process, deferred as out of bounds

Joint query `21074975` emits a single **731-word** "claim" from an `and …, and …` repetition loop
whose length scales with the output cap (164 words at 2560). 3.1% of joint's claims exceed 40 words
against post-hoc's 0.5%, and `_claim_rules()` splits on "and" yet did not split this. §4 puts
`_claim_rules()` out of bounds and the fix is W5/W6 work, but it lands **before** the first citation-F1
is read: that claim would be scored as one unit.

## Second amendment, 2026-08-15 — the 731-word claim stays a scoring-side guard, not a prompt edit

**What expired.** *Known weakness* under *Termination* read "[the fix] lands **before** the first
citation-F1 is read: that claim would be scored as one unit." First F1 was computed the same day
this ADR terminated (`docs/harvest/first_citation_f1.md`, 2026-08-14). The premise that justified
deferring the fix to W5/W6 rather than deciding its treatment now no longer holds.

**The decision, made now rather than deferred again.** `_claim_rules()` is **not** edited before the
Sep 3 decomposer freeze. Editing it moves all three prompts at once (§4's Amendment: the grammar is
shared for fairness), which would mean the parity gate, the published `parity_iter1b` table, and the
first F1 read all describe prompts that no longer exist — not a fix, a full re-run with the gate
recomputed. Nothing forces that re-run: the defect is already caught and reportable through the
mechanism §4's Consequences names for exactly this — `ScoringConfig.max_claim_words`
(`prompts.MAX_CLAIM_WORDS = 50`), the parse-side, re-scorable guard `parse_response` and
`decompose.parse_decomposition` both already apply.

**The number, at the guard's actual threshold.** The *Termination* section quoted the 40-word band
from the exploratory read that found the defect. At the guard's own threshold, 50 words, recomputed
from `docs/harvest/parity_iter1b.records.jsonl`:

| system | claims | words/claim >50 | rate |
|---|---|---|---|
| joint | 719 | 20 | **2.78%** |
| post_hoc | 1242 | 3 | **0.24%** |
| vanilla | 1622 | 4 | 0.25% |

This is not a new measurement — it is `prompts.MAX_CLAIM_WORDS`'s own justifying comment (2.78% /
0.24% / 0.25%, chosen at 50 over 30's 4.73% / 3.06% / 9.43% specifically so the guard would not tax
the three arms at three different rates and move C2's gap by instrument) — recorded here as the
disclosed defect rate rather than left to be found only in a code comment.

**If the prompt is ever fixed**, it is a dated, budgeted re-run with the parity gate recomputed
against the new prompts, per §4 — not a quiet edit to a frozen template.

## Third amendment, 2026-08-17 — runaway claim pathology closed on score/split/decode sides, prompts remain frozen

**Discharge of the deferred 731-word claim defect.** The Second amendment decided the 731-word claim "stays a scoring-side guard, not a prompt edit" and left `_claim_rules()` frozen. This defect is now fully discharged across three non-prompt layers (ADR-0021, `docs/harvest/generate_fp_sweep.md`):

1. **Parser guard strengthened (nested extension chains):** `prompts.RUNAWAY_CHAIN_MIN = 3` and `prompts.runaway_chains` detect nested prefix-extension chains ($N+1$ extends $N$), charging chains $\ge 3$ to `errors` in both `parse_response` and `decompose.parse_decomposition`.
2. **Decomposer splitter hole closed (`sentence_units` punctuation-bound splitting):** `decompose._split_run_on` splits sentence units exceeding `MAX_CLAIM_WORDS = 50` at explicit punctuation boundaries (`;` or `,` + whitespace). Units lacking marked boundaries remain whole and flagged. On `parity_iter1b`, the 731-word claim is cleanly split into 18 pieces, reducing over-length claims to 0 for `joint` and `post_hoc` while preserving content invariants (0 lost/duplicated non-whitespace characters across all 300 records).
3. **Generator-side root cause addressed (sampling frequency penalty):** `GenerationConfig.frequency_penalty` default is raised from `0.0` to `0.5` (`CONFIG_VERSION` 1.5.0). Live A4000 sampling sweeps confirm `joint` over-length claims drop to 0, longest claim falls 731w $\rightarrow$ 27w, `quote_not_found` falls $8 \rightarrow 0$, and window-overflow HTTP 400 call rejections vanish.

**Strict adherence to prompt freeze.** Still no prompt edit was performed, and `_claim_rules()` remains frozen. Prompt text, `prompts.PARITY_LOOP_CLOSED`, and `decompose.decompose_template_digest()` are unchanged.

## Fourth amendment, 2026-08-23 — five joint-side granularity edits reverted; W9 is discharged by standardisation, not by tuning

**What went wrong.** Between 2026-08-20 and 2026-08-22, `JOINT_JSON_TEMPLATE` acquired a claim-length
target and then had it re-tuned four times (`045a96c` → `95dd958` → `dab7a68` → `dc08914` →
`b29e74c`), producing runs `generate_fp05_n100_guided_v5` through `v9`. Three of the four re-tunings
name the objective in their commit subject: *"for W9 parity"*, *"for 16 w/c parity and W9
sign-off"*, *"for v9 parity"*.

This is three separate violations of this ADR, and none of them was caught by a test:

1. **§4 confines the granularity lever to `POST_HOC_ANSWER_TEMPLATE`.** These edits steer *joint's*
   granularity. The freeze that exists in code — `PARITY_LOOP_CLOSED.post_hoc_answer_template_sha256`,
   checked by `tests/test_prompts.py` — pins the post-hoc side only, so the joint side was
   unprotected precisely because §4 never contemplated tuning it.
2. **§6's blind lifted 2026-08-14.** Every one of these edits was therefore a granularity edit made
   with citation-F1 visible. `PARITY_LOOP_CLOSED`'s own comment names this as "the one thing §6
   exists to prevent" — and says so about a *post-hoc* edit, which is the milder case.
3. **§5's check was treated as a gate.** *What survives termination* states: "A pre-registered
   asymmetric check is not retracted because the iteration that closed the loop passed; that
   retraction is the post-hoc steering §3 and §6 exist to prevent." Tuning the check into passing is
   that retraction by another route. The compounding error was a belief — recorded in `HANDOFF.md`
   and `Upcoming_goals.md`, and attributed to "ADR-0009 §5" — that W9 passing was a Gate G2
   precondition. **§5 says nothing of the kind, and neither does Gate G2.** §1 lists parity as "one
   quantity measured and disclosed whatever it says"; §3 states "the tolerance does not need to be
   achievable. Missing it is survivable by design"; §5's fallback makes the stratified check
   *mandatory to run*, not mandatory to pass. `research_roadmap.md`'s gate text gates citation-F1
   and parse rate, and nothing else.

**Why it was also futile.** Across the six runs, W9 verdict and parse rate move with no stable
relation to the target's wording — `v5` and `v7` carry the *same* target text and land on different
W9 verdicts; parse rate swings 98/100 → 91/100 on a one-word change. This is the resolution argument
*Why it stopped one iteration in* already made: an integer median of 14–20 words, one word
$\approx 6.7\%$, tolerance two words wide. Worse, W9-pass and CI-excludes-zero proved
**anti-correlated** across all six runs, because both are driven by joint's claim length in opposite
directions — pushing joint's claims longer narrows the parity gap while trading away the recall that
produces C2's margin. Continuing would eventually have manufactured a simultaneous pass by chance,
on a gate whose asymmetric-scrutiny rule is pre-registered in the paper's methods section.

**Decision.** All five edits are reverted; `JOINT_JSON_TEMPLATE` now matches `054ec6b`
byte-for-byte, verified by digest. `054ec6b` is the commit `generate_fp05_n100_guided_v4` ran on
(its manifest `git_sha`, clean), so the run of record has an exact template match and its joint
prompt carries **no granularity instruction at all**. `v5`–`v9` are void as evidence. No prompt
other than `JOINT_JSON_TEMPLATE` was touched; `POST_HOC_ANSWER_TEMPLATE`, `_claim_rules()`,
`PARITY_LOOP_CLOSED`, and `decompose_template_digest()` are unchanged, so §8's Sep 3 freeze is
intact.

**§5's asymmetric scrutiny is discharged on the measurement side instead.** The scrutiny this ADR
demands was never "make the gap small" — the Context says the worry is that *"C2's gap appears
without joint grounding doing any work."* A granularity gap is a confound only if it **transmits**
to citation-F1, which the pooled gate cannot measure. `scripts/w9_length_standardized_contrast.py`
measures it directly, by re-weighting joint's citation-recall to post-hoc's own claim-length
distribution over `CLAIM_LENGTH_BANDS` (direct standardisation; queries resampled per ADR-0011 §2).
On `v4`, at the *widest* granularity gap yet recorded (+30.8%):

| quantity | joint | post_hoc | delta |
|---|---|---|---|
| citation-F1, unstandardised | 0.6651 | 0.5248 | +0.1403 `[+0.0751, +0.2066]` |
| citation-F1, length-standardised | 0.6743 | 0.5248 | **+0.1495** `[+0.0786, +0.2244]` |

Joint leads in four of five length bands, ties in the shortest, and $\Delta$recall **grows** with
claim length (+0.139 / +0.158 / +0.202 / +0.333 in the 11–15 / 16–20 / 21–30 / 31+ bands) — the
opposite of the confound's signature. **The granularity gap transmits against C2, not for it:**
post-hoc's coarser claims were making joint's advantage look smaller than it is at matched
granularity. This is the same posture as the Second amendment — a scoring-side answer to a
scoring-side question, leaving both prompts frozen and re-deriving from stored records with no new
inference.

The gap is therefore reported at +30.8%, disclosed, and **not** tuned away — which is what §1 said
would happen from the start.

**Consequence for the paper.** The methods section gains the standardised contrast alongside the
pre-registered asymmetric rule, and the parity diagnostic is reported as a miss with its transmission
measured. A disclosed miss whose mechanism is shown not to favour the hypothesis is a stronger
position than a passed tolerance reached by tuning the arm under test — and it is the position §1
chose deliberately over the "fifth enforced condition" framing.

**Standing rule added.** Granularity-motivated edits to *any* arm's prompt after 2026-08-14 are
prohibited, not merely discouraged: §4's confinement plus §6's unblinding leaves no legitimate
granularity lever on either side. `docs/harvest/w9_stratified_parity_guided_v4.md` §5 carries the
run-by-run evidence.