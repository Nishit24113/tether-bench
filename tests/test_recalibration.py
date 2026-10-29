"""
Tests for post-hoc recalibration: isotonic (PAV) and Platt maps, plus the
cross-validated raw-vs-calibrated evaluation.
"""

import pytest

from tether.recalibration import (
    _pav, isotonic_fit, platt_fit, apply_map,
    cross_val_recalibration, detector_recalibration,
)


LABELS = [1] * 25 + [0] * 25


# ---------------------------------------------------------------------------
# isotonic / PAV
# ---------------------------------------------------------------------------

def test_pav_is_non_decreasing():
    out = _pav([1.0, 0.0, 0.0, 1.0, 1.0], [1, 1, 1, 1, 1])
    assert out == sorted(out)                    # monotone non-decreasing


def test_pav_pools_a_descending_run_to_its_mean():
    # a strictly decreasing input must collapse to one block at the global mean
    out = _pav([1.0, 0.0], [1, 1])
    assert out == [0.5, 0.5]


def test_isotonic_is_monotone_and_in_unit_interval():
    scores = [i / 20 for i in range(20)]
    labels = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    f = isotonic_fit(scores, labels)
    mapped = [f(x) for x in scores]
    assert mapped == sorted(mapped)
    assert all(0.0 <= m <= 1.0 for m in mapped)


def test_isotonic_constant_input_returns_base_rate():
    # a constant score maps to the base rate regardless of label order
    f1 = isotonic_fit([0.5] * 40, [1] * 20 + [0] * 20)
    f2 = isotonic_fit([0.5] * 40, [0] * 20 + [1] * 20)
    assert f1(0.5) == pytest.approx(0.5)
    assert f2(0.5) == pytest.approx(0.5)


def test_isotonic_clamps_outside_range():
    f = isotonic_fit([0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1])
    assert f(-5.0) == f(0.2)
    assert f(5.0) == f(0.8)


# ---------------------------------------------------------------------------
# Platt
# ---------------------------------------------------------------------------

def test_platt_is_monotone_increasing_for_separable_data():
    scores = [0.1] * 10 + [0.9] * 10
    labels = [0] * 10 + [1] * 10
    f = platt_fit(scores, labels)
    assert f(0.9) > f(0.1)
    assert 0.0 <= f(0.1) <= 1.0 and 0.0 <= f(0.9) <= 1.0


def test_apply_map_clamps():
    out = apply_map(lambda s: 2 * s - 0.5, [0.0, 0.5, 1.0])
    assert out == [0.0, 0.5, 1.0]


# ---------------------------------------------------------------------------
# cross-validated evaluation
# ---------------------------------------------------------------------------

def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        cross_val_recalibration([1, 0], [0.5])


def test_unsupported_method_raises():
    with pytest.raises(ValueError):
        cross_val_recalibration(LABELS, [0.5] * 50, method="spline")


def test_recalibration_fields_and_bounds():
    scores = [0.6] * 25 + [0.4] * 25          # overconfident-ish signal
    r = cross_val_recalibration(LABELS, scores, "isotonic")
    assert set(r) >= {"method", "raw_ece", "raw_brier", "cal_ece", "cal_brier"}
    assert 0.0 <= r["cal_ece"] <= 1.0 and 0.0 <= r["cal_brier"] <= 1.0


def test_calibration_helps_an_overconfident_detector():
    # scores pushed to the extremes but only moderately predictive: raw ECE is
    # high, and a fitted map should not make it worse (usually better).
    scores = [0.95] * 18 + [0.05] * 7 + [0.95] * 7 + [0.05] * 18
    r = cross_val_recalibration(LABELS, scores, "isotonic")
    assert r["cal_ece"] <= r["raw_ece"] + 1e-9


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------

def test_recalibration_reduces_free_detector_ece_on_sample():
    from tether import load_sample, default_detectors
    examples = load_sample()
    rc = detector_recalibration(default_detectors(examples), examples, method="isotonic")
    # the badly-miscalibrated lexical detectors improve substantially
    assert rc["lexical_overlap"]["cal_ece"] < rc["lexical_overlap"]["raw_ece"]
    assert rc["novel_token"]["cal_ece"] < rc["novel_token"]["raw_ece"]
    # the constant floor is already calibrated and stays so
    assert rc["majority_constant"]["cal_ece"] == pytest.approx(0.0, abs=1e-9)


def test_report_includes_recalibration_section():
    from tether import load_sample, default_detectors, generate_report
    examples = load_sample()
    report = generate_report(default_detectors(examples), examples)
    assert "## Can the cheap scores be made into probabilities?" in report
    assert "Isotonic ECE" in report
