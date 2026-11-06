"""Tests for the selective-prediction (risk-coverage) analysis."""

from tether.selective import (
    risk_coverage_curve,
    aurc,
    selective_risk_at,
    summary,
    RiskCoveragePoint,
)

import pytest


# Convention: label 1 = faithful, 0 = hallucinated; higher faith score = more
# faithful; flag hallucinated when score <= threshold. All cases below fix the
# threshold at 0.5 so the predictions are explicit and deterministic.

LABELS = [1, 1, 1, 0, 0, 0]  # three faithful, three hallucinated


def test_curve_has_one_point_per_example_and_ascending_coverage():
    scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
    pts = risk_coverage_curve(LABELS, scores, threshold=0.5)
    assert len(pts) == len(LABELS)
    covs = [p.coverage for p in pts]
    assert covs == sorted(covs)
    assert pts[-1].coverage == 1.0
    assert pts[-1].n == len(LABELS)


def test_perfect_detector_has_zero_risk_everywhere():
    # every prediction correct at threshold 0.5
    scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
    pts = risk_coverage_curve(LABELS, scores, threshold=0.5)
    assert all(p.risk == 0.0 for p in pts)
    assert aurc(LABELS, scores, threshold=0.5) == 0.0
    assert selective_risk_at(LABELS, scores, 1.0, threshold=0.5) == 0.0


def test_full_coverage_risk_equals_overall_error_rate():
    # one faithful answer scored just under the threshold -> a single wrong call
    scores = [0.9, 0.8, 0.45, 0.1, 0.2, 0.3]
    pts = risk_coverage_curve(LABELS, scores, threshold=0.5)
    assert pts[-1].risk == pytest.approx(1 / 6)


def test_abstaining_on_least_confident_removes_the_error():
    # the single wrong call (0.45) is also the least confident (closest to 0.5),
    # so answering anything less than full coverage should drop risk to zero.
    scores = [0.9, 0.8, 0.45, 0.1, 0.2, 0.3]
    assert selective_risk_at(LABELS, scores, 1.0, threshold=0.5) == pytest.approx(1 / 6)
    assert selective_risk_at(LABELS, scores, 5 / 6, threshold=0.5) == 0.0
    assert selective_risk_at(LABELS, scores, 0.5, threshold=0.5) == 0.0


def test_informative_confidence_beats_uninformative():
    # informative: the one mistake is on the LEAST confident example (0.45)
    informative = [0.9, 0.8, 0.45, 0.1, 0.2, 0.3]
    # uninformative: the one mistake is on the MOST confident example (0.05),
    # a faithful answer scored far below the threshold
    uninformative = [0.05, 0.8, 0.7, 0.1, 0.2, 0.3]

    # both have the same overall (full-coverage) error rate
    assert selective_risk_at(LABELS, informative, 1.0, threshold=0.5) == pytest.approx(1 / 6)
    assert selective_risk_at(LABELS, uninformative, 1.0, threshold=0.5) == pytest.approx(1 / 6)

    # but abstaining helps the informative detector and not the uninformative one
    assert aurc(LABELS, informative, threshold=0.5) < aurc(LABELS, uninformative, threshold=0.5)
    assert selective_risk_at(LABELS, informative, 0.5, threshold=0.5) == 0.0
    assert selective_risk_at(LABELS, uninformative, 0.5, threshold=0.5) > 0.0


def test_summary_shape_and_keys():
    scores = [0.9, 0.8, 0.45, 0.1, 0.2, 0.3]
    out = summary(LABELS, scores, threshold=0.5)
    assert set(out) == {"aurc", "full_coverage_risk", "selective_risk"}
    assert set(out["selective_risk"]) == {"1.00", "0.90", "0.75", "0.50"}
    assert out["full_coverage_risk"] == pytest.approx(1 / 6, abs=1e-6)


def test_validation_and_edges():
    scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
    with pytest.raises(ValueError):
        risk_coverage_curve([1, 0], [0.5], threshold=0.5)  # length mismatch
    with pytest.raises(ValueError):
        selective_risk_at(LABELS, scores, 0.0, threshold=0.5)  # coverage out of range
    with pytest.raises(ValueError):
        selective_risk_at(LABELS, scores, 1.5, threshold=0.5)
    assert risk_coverage_curve([], [], threshold=0.5) == []
    assert aurc([], [], threshold=0.5) == 0.0


def test_default_threshold_is_f1_optimal():
    # with no threshold given it should fall back to the F1-optimal split and
    # still return a well-formed curve
    scores = [0.9, 0.8, 0.45, 0.1, 0.2, 0.3]
    pts = risk_coverage_curve(LABELS, scores)
    assert len(pts) == len(LABELS)
    assert isinstance(pts[0], RiskCoveragePoint)


def test_detector_selective_keys_by_name_and_package_wiring():
    from tether import detector_selective  # also checks the package export

    class _Ex:
        def __init__(self, label):
            self.label = label

    class _Stub:
        def __init__(self, name, scores):
            self.name = name
            self._scores = scores

        def score_all(self, examples):
            return self._scores

    examples = [_Ex(l) for l in LABELS]
    detectors = [_Stub("informative", [0.9, 0.8, 0.45, 0.1, 0.2, 0.3])]
    out = detector_selective(detectors, examples)
    assert "informative" in out
    assert set(out["informative"]) == {"aurc", "full_coverage_risk", "selective_risk"}
