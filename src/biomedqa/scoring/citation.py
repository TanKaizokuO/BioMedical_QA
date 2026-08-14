"""ALCE citation precision / recall / F1 — **Table 2**, and the G2 gate.

Promoted from `notebooks/03_2_citation_precision_recall.ipynb`, which is sound at any scale (pure
functions over labels) — but its φ is `cross-encoder/nli-deberta-v3-xsmall`, not MiniCheck. φ stays
a parameter here for exactly that reason: this module never loads a model, so Table 2 is
recomputable from `runs/*/records.jsonl` under a different entailment primitive without a re-run.

Semantics are reused verbatim from ALCE and are frozen in `CONTEXT.md`:

    recall(c)   = 1 iff C ≠ ∅ ∧ φ(concat(C), c) = 1
    precision   = fraction of citations that are not *irrelevant*, where x is irrelevant iff
                  φ(x, c) = 0 ∧ φ(concat(C \\ {x}), c) = 1
    F1          = harmonic mean of corpus-level precision and recall

**F1 is the reported number**, because recall alone is gamed by citing everything — which is also
what the ≤3 cap defends against, and why the cap must be identical across all three systems.
Jointly necessary citations are legitimate: the remove-it-and-see rule already handles a claim whose
dose comes from one span and whose outcome comes from another.

Two places where the frozen definition differs from the notebook, both resolved in `CONTEXT.md`'s
favour because it is authoritative where documents conflict:

1. **Precision is defined on every cited claim**, not only on ones whose union entails. The
   notebook returns `None` when recall is 0; the frozen rule makes a lone non-entailing citation
   *not irrelevant*, since removing it leaves nothing that could suffice. Read on its own that
   flatters precision — it is not read on its own. F1 pairs it with a corpus recall of 0 for that
   claim, which is the term that moves.
2. **The recall denominator excludes abstentions** (ADR-0010). A system that says "the passages do
   not report mortality" has done the right thing; counting it as an uncited claim would reward
   confabulating a citation. Precision needs no equivalent — an abstention carries no citations, so
   it contributes nothing to a denominator counted in citations. Both denominators are reported
   **always**, per ADR-0010: `recall` excludes abstentions, `recall_all_claims` does not.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from ..prompts import parity_loop_is_open
from ..schema import Citation, Claim, QueryRecord
from .abstention import abstention_claims, answered_claims

#: The entailment primitive: `phi(premise, hypothesis) -> bool`. The *verifier* is what is built on
#: it (MiniCheck + threshold + scoring) — the distinction is fixed in `CONTEXT.md`.
Phi = Callable[[str, str], bool]


def _span_text(citation: Citation, passages: Mapping[str, str] | None) -> str:
    """The text the annotator was shown, and the text φ is given as its premise.

    The passage is authoritative when it is still addressable; `quoted_text` is the fallback that
    survives a re-chunk, which is the whole reason `schema.py` stores it despite it being
    derivable. A citation with neither raises: scoring a span whose text is unknown would be
    inventing a premise.
    """
    if passages is not None:
        passage = passages.get(citation.passage_id)
        if passage is not None:
            return passage[citation.char_start : citation.char_end]
    if citation.quoted_text is not None:
        return citation.quoted_text
    raise ValueError(
        f"citation into {citation.passage_id!r} has no quoted_text and its passage is not in the "
        "record; there is no span text to give φ as a premise"
    )


def _spans(claim: Claim, passages: Mapping[str, str] | None) -> list[str]:
    return [_span_text(c, passages) for c in claim.citations]


def _concat(spans: Iterable[str]) -> str:
    """`concat(C)` from `CONTEXT.md` — the cited spans, in citation order, joined by a space.

    Order is the claim's own citation order rather than anything sorted: it is what the model
    emitted and what the annotator read, and φ over a permuted premise is not the same call.
    """
    return " ".join(spans)


def citation_recall(claim: Claim, phi: Phi, *, passages: Mapping[str, str] | None = None) -> float:
    """`1.0` iff the claim cites something and the **union** of its spans entails it.

    Union, not any-of: a claim whose dose comes from one span and whose outcome from another is
    covered by neither alone and by both together.
    """
    spans = _spans(claim, passages)
    if not spans:
        return 0.0
    return 1.0 if phi(_concat(spans), claim.text) else 0.0


def _precision_counts(
    claim: Claim, phi: Phi, passages: Mapping[str, str] | None
) -> tuple[int, int]:
    """`(citations that are not irrelevant, citations)` — the corpus-level numerator and
    denominator for one claim, kept as counts because corpus precision is micro-averaged."""
    spans = _spans(claim, passages)
    if not spans:
        return 0, 0
    good = 0
    for i, span in enumerate(spans):
        rest = spans[:i] + spans[i + 1 :]
        # Irrelevant = fails alone *and* the rest already suffice. Both halves matter: the first
        # alone would condemn every jointly-necessary citation, the second alone every claim whose
        # citations are redundant but each individually correct.
        irrelevant = (not phi(span, claim.text)) and bool(rest) and bool(phi(_concat(rest), claim.text))
        good += not irrelevant
    return good, len(spans)


def citation_precision(claim: Claim, phi: Phi, *, passages: Mapping[str, str] | None = None) -> float:
    """Per-claim fraction of citations that are not irrelevant; `nan` for an uncited claim.

    Provided for inspecting a single claim. **`citation_f1` does not average this** — corpus
    precision is the ratio of the summed counts, and the mean of per-claim ratios would weight a
    one-citation claim the same as a three-citation one.
    """
    good, total = _precision_counts(claim, phi, passages)
    return good / total if total else float("nan")


def _harmonic(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def citation_f1(records: Iterable[QueryRecord], phi: Phi) -> dict:
    """Corpus-level P/R/F1. Corpus-level, not the mean of per-claim F1s — they differ.

    Both recall denominators are reported, always (ADR-0010): `recall` over claims that ought to
    carry a citation, `recall_all_claims` over every claim. The first is the headline number and
    the second is what stops the abstention rule from being a lever on it — a rule that quietly
    grew to exclude hard claims would show up as a gap between the two.

    No confidence interval here. Every interval in this paper resamples **questions**
    (`calibration.bootstrap_ci`, ADR-0011 §2), and a CI for a harmonic mean of two corpus-level
    ratios cannot be assembled from per-claim numbers — the caller passes this whole computation
    as the bootstrap statistic.

    **Refuses to run while the ADR-0009 parity loop is open (§6).** "No citation-F1 on any split, in
    any form, until the loop terminates" is the rule the whole blind rests on, and a rule enforced
    only by remembering it is a rule that gets broken by a future session debugging something else.
    The loop closed on `parity_iter1b` (2026-08-14, `prompts.PARITY_LOOP_CLOSED`), so this is a
    guard, not a blocker — and reopening the loop turns it back into one, which is correct: the
    unblinding cannot be undone by re-opening a ledger.
    """
    if parity_loop_is_open():
        raise RuntimeError(
            "ADR-0009 §6: citation-F1 must not be computed on any split while the granularity-parity "
            "loop is open. prompts.PARITY_LOOP_CLOSED is None."
        )
    good = cited = 0
    recalled = 0.0
    answered = abstained = 0
    for record in records:
        passages = {p.passage_id: p.text for p in record.retrieved if p.text is not None}
        abstained += len(abstention_claims(record))
        # Precision iterates the same claims: an abstention carries no citations by construction,
        # so it cannot reach either precision count.
        for claim in answered_claims(record):
            answered += 1
            recalled += citation_recall(claim, phi, passages=passages)
            g, n = _precision_counts(claim, phi, passages)
            good += g
            cited += n

    n_claims = answered + abstained
    precision = good / cited if cited else 0.0
    recall = recalled / answered if answered else 0.0
    recall_all = recalled / n_claims if n_claims else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": _harmonic(precision, recall),
        "recall_all_claims": recall_all,
        "f1_all_claims": _harmonic(precision, recall_all),
        "n_claims": n_claims,
        "n_answered": answered,
        "n_abstentions": abstained,
        "n_citations": cited,
        "n_relevant_citations": good,
    }
