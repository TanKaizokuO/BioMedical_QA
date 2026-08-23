"""Populate Claim.verifier_scores on QueryRecords from cache or verifier.

ADR-0020 specifies that `biomedqa.verify` holds the sole MiniCheck inference site.
This module looks up precomputed scores from `docs/harvest/minicheck_cache.json` or calls
a `Verifier` instance to score missing pairs.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from biomedqa.schema import Claim, QueryRecord, VerifierScore
from biomedqa.scoring.citation import _span_text

if TYPE_CHECKING:
    from biomedqa.verify import Verifier

DEFAULT_VERIFIER_NAME = "lytang/MiniCheck-Flan-T5-Large"


def load_minicheck_cache(cache_path: str | Path) -> dict[tuple[str, str], float]:
    """Load MiniCheck cache from JSON file as a dict mapping (premise, hypothesis) -> score."""
    path = Path(cache_path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {tuple(k.split("|||")): float(v) for k, v in raw.items()}


def populate_verifier_scores(
    records: Iterable[QueryRecord],
    cache: Mapping[tuple[str, str], float] | None = None,
    *,
    verifier_name: str = DEFAULT_VERIFIER_NAME,
    verifier: Verifier | None = None,
    allow_verifier: bool = False,
) -> tuple[list[QueryRecord], dict[str, Any]]:
    """Populate Claim.verifier_scores for a sequence of QueryRecords.

    Returns:
        (populated_records, coverage_dict)

    One VerifierScore is appended per (claim, citation) pair in citation order, matching
    HumanLabel.citation_index alignment.

    In cache-only mode (allow_verifier=False, default), missing pairs are recorded in
    coverage_dict["missing_pairs"] and n_missing, and are NOT imputed or silently dropped.
    """
    if cache is None:
        cache = {}

    out_records: list[QueryRecord] = []
    n_records = 0
    n_claims = 0
    n_citations = 0
    n_scored = 0
    n_missing = 0
    missing_pairs: list[dict[str, Any]] = []

    for record in records:
        n_records += 1
        passages = {p.passage_id: p.text for p in record.retrieved if p.text is not None}
        new_claims: list[Claim] = []

        for claim in record.claims:
            n_claims += 1
            other_scores = [v for v in claim.verifier_scores if v.name != verifier_name]
            scores_for_verifier: list[VerifierScore] = []

            for cit_idx, citation in enumerate(claim.citations):
                n_citations += 1
                try:
                    premise = _span_text(citation, passages)
                except ValueError:
                    n_missing += 1
                    missing_pairs.append(
                        {
                            "query_id": record.query_id,
                            "claim_id": claim.claim_id,
                            "citation_index": cit_idx,
                            "reason": "span_text_resolution_failed",
                        }
                    )
                    continue

                hypothesis = claim.text
                pair = (premise, hypothesis)

                if pair in cache:
                    score = cache[pair]
                    scores_for_verifier.append(
                        VerifierScore(name=verifier_name, score=score, latency_s=None)
                    )
                    n_scored += 1
                elif allow_verifier and verifier is not None:
                    res = verifier.score_pairs([pair])
                    vscore = res[0]
                    scores_for_verifier.append(
                        VerifierScore(
                            name=verifier_name,
                            score=vscore.score,
                            latency_s=vscore.latency_s,
                        )
                    )
                    n_scored += 1
                else:
                    n_missing += 1
                    missing_pairs.append(
                        {
                            "query_id": record.query_id,
                            "claim_id": claim.claim_id,
                            "citation_index": cit_idx,
                        }
                    )

            updated_verifier_scores = other_scores + scores_for_verifier
            new_claim = dataclasses.replace(claim, verifier_scores=updated_verifier_scores)
            new_claims.append(new_claim)

        new_record = dataclasses.replace(record, claims=new_claims)
        out_records.append(new_record)

    coverage_rate = float(n_scored / n_citations) if n_citations > 0 else 1.0
    n_extra_citations = sum(
        max(0, len(c.citations) - 1) for r in out_records for c in r.claims if c.citations
    )
    coverage = {
        "n_records": n_records,
        "n_claims": n_claims,
        "n_citations": n_citations,
        "n_scored": n_scored,
        "n_missing": n_missing,
        "n_extra_citations": n_extra_citations,
        "coverage_rate": coverage_rate,
        "missing_pairs": missing_pairs,
    }

    return out_records, coverage
