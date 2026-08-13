"""Krippendorff's α and the G4 gate — `scoring/agreement.py`.

Every expected number here is computed by hand from
`α = 1 − (n − 1) · Σ_{c≠k} o_ck δ²_ck / Σ_{c≠k} n_c n_k δ²_ck`, so the tests pin the arithmetic
rather than re-assert whatever the code happens to return. The worked cases live in the comments
next to each assertion.
"""

from __future__ import annotations

import math

import pytest

from biomedqa.schema import HumanLabel, SupportLabel
from biomedqa.scoring.agreement import (
    G4_ALPHA_MIN,
    G4_MIN_CLAIMS,
    gate_g4,
    krippendorff_alpha_binary,
    krippendorff_alpha_ordinal,
    label_distribution,
)

S = SupportLabel


def _unit(*labels: SupportLabel) -> list[HumanLabel]:
    """One annotation unit's raw per-annotator labels. Only `support_label` drives α;
    `claim_validity` is required by the schema and is irrelevant here."""
    return [HumanLabel(annotator_id=f"a{i}", support_label=lab, claim_validity=True) for i, lab in enumerate(labels)]


# --- binary collapse (the G4 number) --------------------------------------------------------------

def test_binary_perfect_agreement_is_one():
    # Unit A all SUPPORTED(1), unit B all NOT_SUPPORTED(0), 3 raters. o=[[3,0],[0,3]], observed=0.
    labels = [_unit(S.SUPPORTED, S.SUPPORTED, S.SUPPORTED), _unit(S.NOT_SUPPORTED, S.NOT_SUPPORTED, S.NOT_SUPPORTED)]
    assert krippendorff_alpha_binary(labels) == pytest.approx(1.0)


def test_binary_one_label_everywhere_is_informationless_one():
    # Every rating SUPPORTED -> expected disagreement 0 -> the module returns 1.0 by convention.
    labels = [_unit(S.SUPPORTED, S.SUPPORTED), _unit(S.SUPPORTED, S.SUPPORTED)]
    assert krippendorff_alpha_binary(labels) == pytest.approx(1.0)


def test_binary_systematic_disagreement_goes_negative():
    # A=[1,0], B=[0,1]: o=[[0,2],[2,0]], marginals=[2,2], n=4.
    # observed=4, expected=8 -> α = 1 - 3·4/8 = -0.5.
    labels = [_unit(S.SUPPORTED, S.NOT_SUPPORTED), _unit(S.NOT_SUPPORTED, S.SUPPORTED)]
    assert krippendorff_alpha_binary(labels) == pytest.approx(-0.5)


def test_binary_partial_collapses_with_supported():
    # is_supporting: SUPPORTED, PARTIAL -> 1; NOT, CONTRA -> 0. So [SUPPORTED, PARTIAL] agree (both 1).
    labels = [_unit(S.SUPPORTED, S.PARTIAL), _unit(S.NOT_SUPPORTED, S.CONTRADICTED)]
    assert krippendorff_alpha_binary(labels) == pytest.approx(1.0)


def test_no_pairable_unit_is_nan():
    # One label per unit -> nothing to pair. Refusing (nan) is right; 0.0 would read as "no agreement".
    labels = [_unit(S.SUPPORTED), _unit(S.NOT_SUPPORTED)]
    assert math.isnan(krippendorff_alpha_binary(labels))


# --- 4-way ordinal --------------------------------------------------------------------------------

def test_ordinal_perfect_agreement_is_one():
    labels = [_unit(S.SUPPORTED, S.SUPPORTED), _unit(S.CONTRADICTED, S.CONTRADICTED)]
    assert krippendorff_alpha_ordinal(labels) == pytest.approx(1.0)


def test_ordinal_penalises_distance_on_the_scale():
    # Both datasets have identical marginals: one each of CONTRA/NOT/PARTIAL/SUPPORTED. marginals=[1,1,1,1].
    # δ²(0,1)=δ²(1,2)=δ²(2,3)=1, δ²(0,2)=δ²(1,3)=4, δ²(0,3)=9; expected = 2·(1+1+1+4+4+9) = 40.
    #
    # X: disagreements are one step each (SUPPORTED/PARTIAL, NOT/CONTRA). observed = 2·1 + 2·1 = 4.
    #    α_X = 1 - 3·4/40 = 0.7
    x = [_unit(S.SUPPORTED, S.PARTIAL), _unit(S.NOT_SUPPORTED, S.CONTRADICTED)]
    # Y: same marginals, but a three-step disagreement (SUPPORTED/CONTRA). observed = 2·9 + 2·1 = 20.
    #    α_Y = 1 - 3·20/40 = -0.5
    y = [_unit(S.SUPPORTED, S.CONTRADICTED), _unit(S.PARTIAL, S.NOT_SUPPORTED)]
    assert krippendorff_alpha_ordinal(x) == pytest.approx(0.7)
    assert krippendorff_alpha_ordinal(y) == pytest.approx(-0.5)
    # The whole point of the ordinal metric: the far-apart disagreement scores worse.
    assert krippendorff_alpha_ordinal(x) > krippendorff_alpha_ordinal(y)


# --- label distribution ---------------------------------------------------------------------------

def test_label_distribution_counts_every_rating_in_scale_order():
    labels = [_unit(S.SUPPORTED, S.SUPPORTED, S.PARTIAL), _unit(S.NOT_SUPPORTED, S.CONTRADICTED)]
    dist = label_distribution(labels)
    assert dist == {"CONTRADICTED": 1, "NOT_SUPPORTED": 1, "PARTIAL": 1, "SUPPORTED": 2}
    assert list(dist) == ["CONTRADICTED", "NOT_SUPPORTED", "PARTIAL", "SUPPORTED"]


# --- the gate -------------------------------------------------------------------------------------

def _agreeing(n_units: int) -> list[list[HumanLabel]]:
    # Alternate SUPPORTED/NOT_SUPPORTED units, each with 3 agreeing raters -> α = 1.0, non-degenerate.
    return [_unit(*( [S.SUPPORTED] * 3 if i % 2 else [S.NOT_SUPPORTED] * 3)) for i in range(n_units)]


def test_gate_needs_both_alpha_and_claim_volume():
    labels = _agreeing(8)  # α = 1.0, comfortably over the bar
    assert gate_g4(labels, n_claims=G4_MIN_CLAIMS)["passes"] is True
    # Same agreement, one claim short of the volume floor -> fails on volume alone.
    assert gate_g4(labels, n_claims=G4_MIN_CLAIMS - 1)["passes"] is False


def test_gate_fails_on_low_alpha():
    labels = [_unit(S.SUPPORTED, S.NOT_SUPPORTED), _unit(S.NOT_SUPPORTED, S.SUPPORTED)]  # α = -0.5
    result = gate_g4(labels, n_claims=G4_MIN_CLAIMS)
    assert result["alpha_binary"] == pytest.approx(-0.5)
    assert result["passes"] is False


def test_gate_does_not_pass_on_nan_alpha():
    # A single-rater population is unmeasurable, not passing, even with the claims.
    labels = [_unit(S.SUPPORTED), _unit(S.NOT_SUPPORTED)]
    result = gate_g4(labels, n_claims=G4_MIN_CLAIMS)
    assert math.isnan(result["alpha_binary"])
    assert result["passes"] is False


def test_gate_reports_the_interval_only_when_clusters_are_given():
    labels = _agreeing(6)
    clusters = ["q1", "q1", "q2", "q2", "q3", "q3"]
    with_ci = gate_g4(labels, n_claims=G4_MIN_CLAIMS, clusters=clusters, n_boot=200)
    assert with_ci["alpha_ci"] is not None
    # Omitting clusters omits the interval rather than silently computing the unclustered (wrong) one.
    assert gate_g4(labels, n_claims=G4_MIN_CLAIMS)["alpha_ci"] is None


def test_gate_thresholds_are_the_documented_ones():
    assert (G4_ALPHA_MIN, G4_MIN_CLAIMS) == (0.6, 250)


def test_gate_counts_units_and_ratings_separately_from_claims():
    labels = _agreeing(4)  # 4 units, 3 ratings each
    result = gate_g4(labels, n_claims=250)
    assert result["n_units"] == 4
    assert result["n_ratings"] == 12
    assert result["n_claims"] == 250  # passed in, never derived from the span-level unit count
