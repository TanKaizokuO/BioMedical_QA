"""Granularity parity — **ADR-0009's measured fairness diagnostic**, not a paper table.

Every other module here feeds a numbered table. This one feeds the blind parity loop: joint vs
post-hoc **median words/claim**, against a tolerance of ±15% that ADR-0009 §3 pre-committed before
any measurement existed and that is *never revised afterwards*. `PARITY_TOLERANCE` is that number;
it is a constant here rather than a parameter because a tolerance you can pass in is a tolerance
that gets tuned, which is precisely what §3 exists to prevent.

**Why this exists as a module.** The gate was computed ad hoc three times running (iteration 0, its
re-measure, and the write-up in `docs/harvest/parity_iter0.md`), and by the third time two of the
four supporting numbers in `PARITY_ITERATIONS[0]`'s rationale could no longer be reproduced from the
artifacts — see *The two numbers that do not reproduce* below. An ad hoc gate is a gate whose
verdict cannot be audited, on the one quantity ADR-0009 promises the paper will disclose whatever it
says.

**Blind by construction (§6).** Nothing here touches a citation, a verifier score, or a human
label. Words/claim is counted from `Claim.text` alone, so no function in this module can be turned
into a citation-F1 proxy by passing it different arguments.

**The trap this module exists to contain.** `costs.jsonl` carries **no system or stage field** —
four rows per query, positionally `joint / post_hoc answer / post_hoc cite / vanilla`. A post-hoc
record's `completion_tokens` is the **sum of both its stages**, so it cannot be compared against the
per-call `max_tokens` and per-stage truncation is invisible in `records.jsonl`. Truncation is not a
detail: the cite stage is what post-hoc claims are parsed from (`generate.py`), so a truncated cite
stage silently drops claims off the end of a record, and at `max_tokens=1536` that hit 38 of 100
post-hoc records. `stage_output_tokens` therefore **verifies the positional assumption against the
records' own totals and raises if it does not hold**, rather than trusting call order.

## The two numbers that do not reproduce

`PARITY_ITERATIONS[0]`'s rationale cites four diagnostic quantities. Re-derived from
`parity_iter0b` (2026-08-14):

| rationale claim | recomputed here, `COMPOUND_MARKERS` on all 100 records |
|---|---|
| multi-comma claims, post-hoc 13.6% vs joint 5.6% | **13.6% / 5.6% — exact** |
| subordinate clauses, post-hoc 5.6% vs joint 0.9% | 4.8% / 0.3% — same shape, ~15x either way |
| coordination by "and" "already identical" (28.4%, 26.9%) | 35.0% / 33.5% — level, as claimed |
| no-compound-marker median 17 vs 14, "+21%" | 18 vs 14, **+28.6%** |

**Only the multi-comma row survives exactly**, and no change of basis (all-records /
untruncated-only × `iter0` / `iter0b`) or of marker definition recovers the others. The four figures
came from a marker set that was never written down.

**The finding they were cited for is unaffected, and is robust to every definition tried.**
And-coordination is level across the arms — `_claim_rules()` splits on it for all three systems, so
that is the expected reading. Post-hoc's excess sits in subordinate clauses and multi-comma claims,
and post-hoc's claims are longer *even with no marker at all*. So the gap is verbosity rather than
compounding, and iteration 1's length target and qualifier-splitting rule are aimed at the right
thing — on the module's markers by a wider margin (+28.6%) than the ledger recorded (+21%).

The ledger's prose is left as the historical record it is; correcting a rationale string after the
fact would make it a claim about a computation rather than a record of one. **Comparisons for
iteration 1 and after come from this module, not from that string.**

## The gate's resolution, measured

`parity_iter1` and `parity_iter1b` ran **the same post-hoc prompt** — the second only raised the
output cap for all three arms — and read **+0.0%** and **+13.3%** on the same basis. The medians
involved are 14–20 words, so the statistic's resolution is **one word (~6.7%)** and the tolerance is
two words wide. A point estimate therefore cannot separate a prompt effect from that grid, which is
why `gap_bootstrap_ci` exists and why `GapInterval.passes` requires the *whole* interval to be inside
the tolerance. The tolerance is not revised for this (§3); what changes is that a verdict is quoted
with its interval.
"""

from __future__ import annotations

import random
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..schema import Claim, CostRecord, QueryRecord, System

#: ADR-0009 §3. Pre-committed 2026-08-04, unmeasured at the time, and **not revisable** — a
#: tolerance chosen after seeing the divergence would be set to a number already known to be
#: reachable, making the loop a no-op. Deliberately not a function parameter.
PARITY_TOLERANCE = 0.15

#: The positional contract of `costs.jsonl`, which carries no stage field of its own.
CALL_ORDER = ("joint", "post_hoc_answer", "post_hoc_cite", "vanilla")

#: Which stages can drop claims off which system's record. Post-hoc's claims are parsed from the
#: **cite** stage, but a truncated answer stage shortens the answer the cite stage is given, so both
#: disqualify the record from the untruncated basis.
STAGES_OF: dict[str, tuple[str, ...]] = {
    System.JOINT.value: ("joint",),
    System.POST_HOC.value: ("post_hoc_answer", "post_hoc_cite"),
    System.VANILLA.value: ("vanilla",),
}

#: Compound-claim markers, in code because the ad hoc versions of two of these are unreproducible
#: (see the module docstring). Each is a way one CLAIM line can carry more than one assertion, and
#: words/claim cannot tell a long atomic claim from a compound one of equal length — ADR-0009's
#: Consequences flags that `claim_validity` catches that only in W6, *after* the Sep 3 freeze, so
#: the simple-claim share is the closest pre-freeze proxy available.
COMPOUND_MARKERS: dict[str, re.Pattern[str]] = {
    #: Clause coordination. Shared across all three arms by `_claim_rules()`, which splits on it —
    #: so this rate being equal across arms is the expected reading, not a surprise.
    "and": re.compile(r"\band\b", re.IGNORECASE),
    #: A trailing qualifier promoted into the same claim: ", which ...", ", while ...".
    "subordinate": re.compile(r",\s*(?:which|while|whereas)\b", re.IGNORECASE),
    #: Two or more commas — parenthetical or list-shaped, and the marker where post-hoc's excess is
    #: largest (13.6% vs 5.6% on `parity_iter0b`).
    "multi_comma": re.compile(r",[^,]*,"),
}


def words_in_claim(text: str) -> int:
    """The gated unit: whitespace-delimited words of one CLAIM line.

    `len(text.split())` and nothing cleverer, deliberately. Tokenizing would make the gate depend on
    a tokenizer version, and the quantity is a proxy for per-claim entailment difficulty, not a
    billing figure.
    """
    return len(text.split())


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics, on a pre-sorted-or-not sequence."""
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


@dataclass(frozen=True, slots=True)
class ArmGranularity:
    """One system's claim-shape profile on one basis. `median_words_per_claim` is the gated field;
    everything else is reported alongside it because a median alone hides the tail that moves it."""

    system: str
    n_records: int
    n_claims: int
    median_words_per_claim: float
    mean_words_per_claim: float
    p25_words_per_claim: float
    p75_words_per_claim: float
    p90_words_per_claim: float
    median_claims_per_query: float


@dataclass(frozen=True, slots=True)
class ParityGate:
    """The ADR-0009 §2/§3 verdict on one basis.

    `gap` is signed and relative to joint, so **positive means post-hoc is coarser** — the direction
    that penalises post-hoc per claim and therefore produces C2's gap without joint grounding doing
    any work. That is the branch §5 makes the W9 stratified robustness check *mandatory* for, which
    is why `favours_c2` is a field and not left to the reader's arithmetic.
    """

    basis: str
    joint: ArmGranularity
    post_hoc: ArmGranularity
    gap: float
    tolerance: float = PARITY_TOLERANCE

    @property
    def passes(self) -> bool:
        return abs(self.gap) <= self.tolerance

    @property
    def favours_c2(self) -> bool:
        """Is the residual gap in the direction that flatters the hypothesis? ADR-0009 §5: if so the
        W9 stratified robustness check becomes mandatory, whether or not the gate passed."""
        return self.gap > 0

    @property
    def requires_w9_robustness_check(self) -> bool:
        return not self.passes and self.favours_c2


def arm_granularity(
    records: Iterable[QueryRecord], system: System | str, *, exclude: Iterable[str] = ()
) -> ArmGranularity:
    """Profile one arm. `exclude` is a set of `query_id`s to drop — the untruncated basis.

    Records with no claims still count toward `n_records` and contribute a zero to claims/query: a
    query the system produced nothing parseable for is a real outcome, and dropping it would quietly
    improve the arm that fails most often.
    """
    name = system.value if isinstance(system, System) else system
    skip = set(exclude)
    mine = [r for r in records if (r.system.value if isinstance(r.system, System) else r.system) == name
            and r.query_id not in skip]
    if not mine:
        raise ValueError(f"no records for system {name!r}")
    per_query = [len(r.claims) for r in mine]
    words = [words_in_claim(c.text) for r in mine for c in r.claims]
    if not words:
        raise ValueError(f"system {name!r} produced no claims; words/claim is undefined")
    return ArmGranularity(
        system=name,
        n_records=len(mine),
        n_claims=len(words),
        median_words_per_claim=statistics.median(words),
        mean_words_per_claim=statistics.mean(words),
        p25_words_per_claim=_percentile(words, 25),
        p75_words_per_claim=_percentile(words, 75),
        p90_words_per_claim=_percentile(words, 90),
        median_claims_per_query=statistics.median(per_query),
    )


def parity_gate(
    records: Iterable[QueryRecord], *, basis: str = "all", exclude: Iterable[str] = ()
) -> ParityGate:
    """The gate. `records` must hold both arms; vanilla is ignored (ADR-0010 excludes it).

    Report **both** bases, always. On `parity_iter0b` all-records reads +25.0% and untruncated-only
    reads +42.9% — an 18-point disagreement — so a single number is not an answer to the gate.
    """
    records = list(records)
    joint = arm_granularity(records, System.JOINT, exclude=exclude)
    post_hoc = arm_granularity(records, System.POST_HOC, exclude=exclude)
    gap = (post_hoc.median_words_per_claim - joint.median_words_per_claim) / joint.median_words_per_claim
    return ParityGate(basis=basis, joint=joint, post_hoc=post_hoc, gap=gap)


@dataclass(frozen=True, slots=True)
class GapInterval:
    """A resampling interval on the gate's gap, over queries. **Not a significance test** — there is
    no null hypothesis here — it is the answer to "how much of this gap is the sample I happened to
    draw?", which the gate's point estimate cannot answer and which iteration 1 made unavoidable.

    `passes` is deliberately the *conservative* reading: the whole interval must lie inside the
    tolerance. A gate whose point estimate passes while its interval straddles ±15% has not been
    shown to pass; ADR-0009 §3 fixed the tolerance, not the number of digits it is compared at.
    """

    basis: str
    lo: float
    median: float
    hi: float
    draws: int
    seed: int
    tolerance: float = PARITY_TOLERANCE

    @property
    def passes(self) -> bool:
        return max(abs(self.lo), abs(self.hi)) <= self.tolerance


def gap_bootstrap_ci(
    records: Iterable[QueryRecord],
    *,
    basis: str = "all",
    exclude: Iterable[str] = (),
    draws: int = 4000,
    seed: int = 0,
) -> GapInterval:
    """Cluster bootstrap of the gap, resampling **queries** rather than claims.

    Why queries: claims from one query are not independent draws — they come from one generation, of
    one length, on one abstract. Resampling claims would understate the interval by exactly the
    amount that matters.

    Why this exists at all. `parity_iter1` and `parity_iter1b` ran **the same post-hoc prompt** and
    read +0.0% and +13.3% on the same basis, because the gated statistic is an integer median sitting
    on the 15/16 boundary: one word is 6.7% of it, nearly half the tolerance. A point estimate
    therefore cannot distinguish a prompt effect from that quantization, and reporting one alone
    would let either run be quoted as "the" result. The interval covers both readings, which is the
    honest description of what was measured.

    Deterministic in `seed`; blind by construction, like everything else here — it resamples word
    counts.
    """
    records = list(records)
    skip = set(exclude)
    per_query: dict[str, dict[str, list[int]]] = {}
    for r in records:
        name = r.system.value if isinstance(r.system, System) else r.system
        if name not in (System.JOINT.value, System.POST_HOC.value) or r.query_id in skip:
            continue
        per_query.setdefault(r.query_id, {})[name] = [words_in_claim(c.text) for c in r.claims]

    qids = [q for q, arms in per_query.items() if len(arms) == 2]
    if not qids:
        raise ValueError("no query carries both arms; the gap is not resampleable")

    rng = random.Random(seed)
    gaps: list[float] = []
    for _ in range(draws):
        drawn = [qids[rng.randrange(len(qids))] for _ in qids]
        joint = [w for q in drawn for w in per_query[q][System.JOINT.value]]
        post_hoc = [w for q in drawn for w in per_query[q][System.POST_HOC.value]]
        if not joint or not post_hoc:
            continue
        j = statistics.median(joint)
        gaps.append((statistics.median(post_hoc) - j) / j)
    gaps.sort()
    return GapInterval(
        basis=basis,
        lo=gaps[int(0.025 * len(gaps))],
        median=statistics.median(gaps),
        hi=gaps[min(int(0.975 * len(gaps)), len(gaps) - 1)],
        draws=len(gaps),
        seed=seed,
    )


def stage_output_tokens(
    records: Iterable[QueryRecord], costs: Iterable[CostRecord]
) -> dict[str, dict[str, int]]:
    """Per-stage output tokens per query, recovered from `costs.jsonl`'s **call order**.

    `costs.jsonl` has no system or stage field: it is positional per query (`joint`,
    `post_hoc_answer`, one or more `post_hoc_cite` calls, and `vanilla`). That assumption is
    checked, not trusted — each system's stage tokens must sum to that record's
    `completion_tokens`, and a mismatch raises. If generation ever emits fewer than 4 calls per
    query, this fails loudly. If cost records carry `output_tokens: None` (e.g., from call-rejection
    guards), they are treated as 0 output tokens rather than raising TypeError.
    """
    per_query: dict[str, list[CostRecord]] = {}
    for c in costs:
        if c.query_id is not None:
            per_query.setdefault(c.query_id, []).append(c)

    by_key = {
        (r.query_id, r.system.value if isinstance(r.system, System) else r.system): r
        for r in records
    }

    out: dict[str, dict[str, int]] = {}
    for query_id, calls in per_query.items():
        if len(calls) < len(CALL_ORDER):
            raise ValueError(
                f"{query_id}: {len(calls)} cost rows, expected 4 or more "
                f"({', '.join(CALL_ORDER)}); call order cannot be assumed"
            )
        joint_out = 0 if calls[0].output_tokens is None else int(calls[0].output_tokens)
        ph_ans_out = 0 if calls[1].output_tokens is None else int(calls[1].output_tokens)
        ph_cite_out = sum(0 if c.output_tokens is None else int(c.output_tokens) for c in calls[2:-1])
        vanilla_out = 0 if calls[-1].output_tokens is None else int(calls[-1].output_tokens)
        stages = {
            "joint": joint_out,
            "post_hoc_answer": ph_ans_out,
            "post_hoc_cite": ph_cite_out,
            "vanilla": vanilla_out,
        }
        for system, owned in STAGES_OF.items():
            record = by_key.get((query_id, system))
            if record is None or record.completion_tokens is None:
                continue
            total = sum(stages[s] for s in owned)
            if total != record.completion_tokens:
                raise ValueError(
                    f"{query_id}/{system}: stages {owned} sum to {total} but the record reports "
                    f"{record.completion_tokens} completion tokens — costs.jsonl is not in "
                    f"{CALL_ORDER} order"
                )
        out[query_id] = stages
    return out


def truncated_queries(
    records: Iterable[QueryRecord], costs: Iterable[CostRecord], max_tokens: int
) -> dict[str, set[str]]:
    """Per system, the `query_id`s whose output hit the per-call cap — the untruncated basis's
    complement.

    A record is disqualified if **any** call that feeds it hit the cap. Note the cap is per *call*:
    comparing a post-hoc record's summed `completion_tokens` against `max_tokens` finds truncation
    that is not there and misses truncation that is.
    """
    per_query: dict[str, list[CostRecord]] = {}
    for c in costs:
        if c.query_id is not None:
            per_query.setdefault(c.query_id, []).append(c)

    out: dict[str, set[str]] = {system.value if isinstance(system, System) else system: set() for system in STAGES_OF}
    for query_id, calls in per_query.items():
        if len(calls) < len(CALL_ORDER):
            continue
        if any((c.output_tokens or 0) >= max_tokens for c in [calls[0]]):
            out[System.JOINT.value].add(query_id)
        if any((c.output_tokens or 0) >= max_tokens for c in calls[1:-1]):
            out[System.POST_HOC.value].add(query_id)
        if any((c.output_tokens or 0) >= max_tokens for c in [calls[-1]]):
            out[System.VANILLA.value].add(query_id)
    return out


@dataclass(frozen=True, slots=True)
class StratumResult:
    """Granularity parity outcome for a single stratum.

    `underpowered` is True when `n_queries < min_queries` or an arm produced no claims within the
    stratum. In that case, `passes` and `gap` are `None` and `reason` provides an explanation,
    preventing under-powered strata from being silently averaged or reporting misleading medians.
    """

    stratum: str
    n_queries: int
    n_joint_claims: int
    n_post_hoc_claims: int
    joint_median_words: float | None
    post_hoc_median_words: float | None
    gap: float | None
    passes: bool | None
    underpowered: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class StratifiedParityGate:
    """Outcome of a stratified robustness check under a specific scheme (ADR-0009 §5).

    - Scheme pass rule (conservative choice): ALL powered strata must pass (`abs(gap) <= tolerance`).
      Alternative reading: require a majority of powered strata to pass, or weight by claim count.
      The conservative reading enforces that parity holds across every well-powered subgroup.
    - Power threshold (conservative choice): `min_queries = 5`.
      Alternative reading: `min_queries = 1` (evaluating even on single queries) or `min_queries = 20`.
      Choice of 5 ensures stable median estimates while preventing underpowered strata from failing.
    """

    scheme: str
    strata: tuple[StratumResult, ...]
    tolerance: float = PARITY_TOLERANCE
    min_queries: int = 5

    @property
    def powered_strata(self) -> tuple[StratumResult, ...]:
        return tuple(s for s in self.strata if not s.underpowered)

    @property
    def underpowered_strata(self) -> tuple[StratumResult, ...]:
        return tuple(s for s in self.strata if s.underpowered)

    @property
    def passes(self) -> bool:
        powered = self.powered_strata
        if not powered:
            return False
        return all(s.passes is True for s in powered)


def compute_compound_strata(
    records: Iterable[QueryRecord], *, min_queries: int = 5
) -> StratifiedParityGate:
    """Stratify by compound structure: simple claims (0 compound markers) vs compound claims (>=1 marker).

    - Specified parameter: ADR-0009 §2 & `COMPOUND_MARKERS` ("and", "subordinate", "multi_comma").
    - Conservative choice: Group all marked claims into a single 'compound' stratum alongside 'simple'.
      Alternative reading: Stratify into 4 individual marker classes ('simple', 'and', 'subordinate',
      'multi_comma'). Grouping into simple/compound maintains sample size while isolating verbosity.
    """
    records = list(records)
    by_query: dict[str, dict[str, list[Claim]]] = {}
    for r in records:
        sys_name = r.system.value if isinstance(r.system, System) else r.system
        if sys_name in (System.JOINT.value, System.POST_HOC.value):
            by_query.setdefault(r.query_id, {})[sys_name] = r.claims

    strata_data = {
        "simple": {"joint": [], "post_hoc": [], "qids": set()},
        "compound": {"joint": [], "post_hoc": [], "qids": set()},
    }

    for qid, arms in by_query.items():
        if System.JOINT.value in arms and System.POST_HOC.value in arms:
            for sys_name in (System.JOINT.value, System.POST_HOC.value):
                for c in arms[sys_name]:
                    st_key = "simple" if not markers_in(c.text) else "compound"
                    w = words_in_claim(c.text)
                    strata_data[st_key][sys_name].append(w)
                    strata_data[st_key]["qids"].add(qid)

    results: list[StratumResult] = []
    for st_key in ("simple", "compound"):
        d = strata_data[st_key]
        n_q = len(d["qids"])
        j_words = d["joint"]
        ph_words = d["post_hoc"]
        if n_q < min_queries or not j_words or not ph_words:
            reason = f"too few queries ({n_q} < {min_queries})" if n_q < min_queries else "no claims in arm"
            results.append(
                StratumResult(
                    stratum=st_key,
                    n_queries=n_q,
                    n_joint_claims=len(j_words),
                    n_post_hoc_claims=len(ph_words),
                    joint_median_words=None,
                    post_hoc_median_words=None,
                    gap=None,
                    passes=None,
                    underpowered=True,
                    reason=reason,
                )
            )
        else:
            j_med = float(statistics.median(j_words))
            ph_med = float(statistics.median(ph_words))
            gap = (ph_med - j_med) / j_med
            results.append(
                StratumResult(
                    stratum=st_key,
                    n_queries=n_q,
                    n_joint_claims=len(j_words),
                    n_post_hoc_claims=len(ph_words),
                    joint_median_words=j_med,
                    post_hoc_median_words=ph_med,
                    gap=gap,
                    passes=abs(gap) <= PARITY_TOLERANCE,
                    underpowered=False,
                )
            )
    return StratifiedParityGate(scheme="compound_structure", strata=tuple(results), min_queries=min_queries)


def compute_claim_length_strata(
    records: Iterable[QueryRecord], *, min_queries: int = 5
) -> StratifiedParityGate:
    """Stratify by claim length bands: 1-10, 11-15, 16-20, 21-30, 31+ words.

    - Specified parameter: Bands from `docs/harvest/first_citation_f1.md` / `citation_f1_minicheck.md`.
    - Conservative choice: Fixed pre-registered length bands.
      Alternative reading: Dynamic quantiles/tertiles fit to run data. Fixed bands prevent data-dependent
      binning.
    """
    bands = [
        ("1-10", 1, 10),
        ("11-15", 11, 15),
        ("16-20", 16, 20),
        ("21-30", 21, 30),
        ("31+", 31, 999999),
    ]
    records = list(records)
    by_query: dict[str, dict[str, list[Claim]]] = {}
    for r in records:
        sys_name = r.system.value if isinstance(r.system, System) else r.system
        if sys_name in (System.JOINT.value, System.POST_HOC.value):
            by_query.setdefault(r.query_id, {})[sys_name] = r.claims

    strata_data = {
        label: {"joint": [], "post_hoc": [], "qids": set()} for label, _, _ in bands
    }

    for qid, arms in by_query.items():
        if System.JOINT.value in arms and System.POST_HOC.value in arms:
            for sys_name in (System.JOINT.value, System.POST_HOC.value):
                for c in arms[sys_name]:
                    w = words_in_claim(c.text)
                    for label, lo, hi in bands:
                        if lo <= w <= hi:
                            strata_data[label][sys_name].append(w)
                            strata_data[label]["qids"].add(qid)
                            break

    results: list[StratumResult] = []
    for label, _, _ in bands:
        d = strata_data[label]
        n_q = len(d["qids"])
        j_words = d["joint"]
        ph_words = d["post_hoc"]
        if n_q < min_queries or not j_words or not ph_words:
            reason = f"too few queries ({n_q} < {min_queries})" if n_q < min_queries else "no claims in arm"
            results.append(
                StratumResult(
                    stratum=label,
                    n_queries=n_q,
                    n_joint_claims=len(j_words),
                    n_post_hoc_claims=len(ph_words),
                    joint_median_words=None,
                    post_hoc_median_words=None,
                    gap=None,
                    passes=None,
                    underpowered=True,
                    reason=reason,
                )
            )
        else:
            j_med = float(statistics.median(j_words))
            ph_med = float(statistics.median(ph_words))
            gap = (ph_med - j_med) / j_med
            results.append(
                StratumResult(
                    stratum=label,
                    n_queries=n_q,
                    n_joint_claims=len(j_words),
                    n_post_hoc_claims=len(ph_words),
                    joint_median_words=j_med,
                    post_hoc_median_words=ph_med,
                    gap=gap,
                    passes=abs(gap) <= PARITY_TOLERANCE,
                    underpowered=False,
                )
            )
    return StratifiedParityGate(scheme="claim_length", strata=tuple(results), min_queries=min_queries)


def compute_query_claim_count_strata(
    records: Iterable[QueryRecord], *, min_queries: int = 5
) -> StratifiedParityGate:
    """Stratify by query claim-volume bands: 1-5 claims, 6-10 claims, 11+ claims.

    - Specified parameter: ADR-0009 §2 (claims/query).
    - Conservative choice: Classify queries based on `joint` arm claim count.
      Alternative reading: Classify queries by post_hoc claim count or average across arms. Using `joint`
      keeps query assignment anchored to the reference system.
    """
    bands = [
        ("1-5 claims", 1, 5),
        ("6-10 claims", 6, 10),
        ("11+ claims", 11, 999999),
    ]
    records = list(records)
    by_query: dict[str, dict[str, QueryRecord]] = {}
    for r in records:
        sys_name = r.system.value if isinstance(r.system, System) else r.system
        if sys_name in (System.JOINT.value, System.POST_HOC.value):
            by_query.setdefault(r.query_id, {})[sys_name] = r

    strata_data = {
        label: {"joint_records": [], "post_hoc_records": []} for label, _, _ in bands
    }

    for qid, arms in by_query.items():
        if System.JOINT.value in arms and System.POST_HOC.value in arms:
            n_c = len(arms[System.JOINT.value].claims)
            for label, lo, hi in bands:
                if lo <= n_c <= hi:
                    strata_data[label]["joint_records"].append(arms[System.JOINT.value])
                    strata_data[label]["post_hoc_records"].append(arms[System.POST_HOC.value])
                    break

    results: list[StratumResult] = []
    for label, _, _ in bands:
        d = strata_data[label]
        n_q = len(d["joint_records"])
        j_claims = [words_in_claim(c.text) for r in d["joint_records"] for c in r.claims]
        ph_claims = [words_in_claim(c.text) for r in d["post_hoc_records"] for c in r.claims]

        if n_q < min_queries or not j_claims or not ph_claims:
            reason = f"too few queries ({n_q} < {min_queries})" if n_q < min_queries else "no claims in arm"
            results.append(
                StratumResult(
                    stratum=label,
                    n_queries=n_q,
                    n_joint_claims=len(j_claims),
                    n_post_hoc_claims=len(ph_claims),
                    joint_median_words=None,
                    post_hoc_median_words=None,
                    gap=None,
                    passes=None,
                    underpowered=True,
                    reason=reason,
                )
            )
        else:
            j_med = float(statistics.median(j_claims))
            ph_med = float(statistics.median(ph_claims))
            gap = (ph_med - j_med) / j_med
            results.append(
                StratumResult(
                    stratum=label,
                    n_queries=n_q,
                    n_joint_claims=len(j_claims),
                    n_post_hoc_claims=len(ph_claims),
                    joint_median_words=j_med,
                    post_hoc_median_words=ph_med,
                    gap=gap,
                    passes=abs(gap) <= PARITY_TOLERANCE,
                    underpowered=False,
                )
            )
    return StratifiedParityGate(scheme="query_claim_count", strata=tuple(results), min_queries=min_queries)


def stratified_parity_check(
    records: Iterable[QueryRecord], *, min_queries: int = 5
) -> dict[str, StratifiedParityGate]:
    """Run all three pre-registered stratification schemes (ADR-0009 §5).

    Returns a mapping from scheme name -> StratifiedParityGate.
    """
    recs = list(records)
    return {
        "compound_structure": compute_compound_strata(recs, min_queries=min_queries),
        "claim_length": compute_claim_length_strata(recs, min_queries=min_queries),
        "query_claim_count": compute_query_claim_count_strata(recs, min_queries=min_queries),
    }

def markers_in(text: str) -> frozenset[str]:
    """Which `COMPOUND_MARKERS` a claim carries. Empty means the claim is *simple* by this test."""
    return frozenset(name for name, pattern in COMPOUND_MARKERS.items() if pattern.search(text))


def compound_profile(claims: Iterable[Claim | str]) -> dict:
    """Marker rates, the simple-claim share, and median words/claim **restricted to simple claims**.

    The last one is what separates the two explanations of a granularity gap: if post-hoc's claims
    are longer even with no compound marker present, the excess is verbosity and a length target can
    move it. If the gap lives only in marked claims, it is compounding, and the lever is a splitting
    rule. On `parity_iter0b` it was verbosity — simple-claim medians 17 vs 14, still +21%.
    """
    texts = [c if isinstance(c, str) else c.text for c in claims]
    if not texts:
        raise ValueError("compound profile of no claims")
    marked = [markers_in(t) for t in texts]
    simple = [t for t, m in zip(texts, marked) if not m]
    return {
        "n_claims": len(texts),
        "marker_rate": {
            name: sum(1 for m in marked if name in m) / len(texts) for name in COMPOUND_MARKERS
        },
        "simple_claim_share": len(simple) / len(texts),
        "n_simple_claims": len(simple),
        "median_words_per_simple_claim": (
            statistics.median(words_in_claim(t) for t in simple) if simple else None
        ),
    }
