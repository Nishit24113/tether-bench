"""
Tests for calibration analysis: reliability bins, Brier, ECE/MCE, and the
bootstrap CI over those metrics.
"""

import pytest

from tether.calibration import (
    brier_score, reliability_bins, expected_calibration_error,
    maximum_calibration_error, calibration, calibration_ci, detector_calibration,
)


LABELS = [1] * 10 + [0] * 10                       # 10 faithful, 10 hallucinated
PERFECT = [1.0] * 10 + [0.0] * 10                  # faithful->1, hallucinated->0
CONSTANT_BASE = [0.5] * 20                          # emits the base rate


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        brier_score([1, 0], [0.5])
    with pytest.raises(ValueError):
        reliability_bins([1, 0], [0.5])


def test_empty_raises():
    with pytest.raises(ValueError):
        brier_score([], [])


def test_bad_bin_count_raises():
    with pytest.raises(ValueError):
        reliability_bins(LABELS, [0.5] * 20, n_bins=0)


def test_unsupported_ci_metric_raises():
    with pytest.raises(ValueError):
        calibration_ci(LABELS, CONSTANT_BASE, metric="auroc")


# ---------------------------------------------------------------------------
# Brier
# ---------------------------------------------------------------------------

def test_brier_perfect_is_zero():
    assert brier_score(LABELS, PERFECT) == 0.0


def test_brier_constant_half_is_quarter():
    # every term is (0.5 - 0 or 1)^2 = 0.25
    assert brier_score(LABELS, CONSTANT_BASE) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# reliability bins
# ---------------------------------------------------------------------------

def test_bins_cover_all_examples_and_score_of_one_lands():
    bins = reliability_bins(LABELS, PERFECT, n_bins=10)
    assert sum(b["count"] for b in bins) == len(LABELS)        # no example dropped
    assert bins[-1]["count"] == 10                             # all the 1.0 scores
    assert bins[0]["count"] == 10                              # all the 0.0 scores


def test_empty_bins_have_none_rates():
    bins = reliability_bins(LABELS, PERFECT, n_bins=10)
    middle = bins[5]                                           # [0.5, 0.6): nothing there
    assert middle["count"] == 0
    assert middle["gap"] is None and middle["mean_score"] is None


# ---------------------------------------------------------------------------
# ECE / MCE
# ---------------------------------------------------------------------------

def test_perfect_scores_are_perfectly_calibrated():
    assert expected_calibration_error(LABELS, PERFECT) == 0.0
    assert maximum_calibration_error(LABELS, PERFECT) == 0.0


def test_base_rate_constant_is_calibrated_but_useless():
    # a constant 0.5 on a balanced set sits in one bin whose faithful rate is 0.5,
    # so ECE is 0 even though the detector has no discriminative power.
    assert expected_calibration_error(LABELS, CONSTANT_BASE) == 0.0
    assert brier_score(LABELS, CONSTANT_BASE) == pytest.approx(0.25)


def test_overconfident_wrong_scores_have_max_ece():
    # scores say "faithful" for the hallucinated half and vice versa
    flipped = [0.0] * 10 + [1.0] * 10
    assert expected_calibration_error(LABELS, flipped) == pytest.approx(1.0)
    assert maximum_calibration_error(LABELS, flipped) == pytest.approx(1.0)


def test_calibration_bundle_fields():
    c = calibration(LABELS, PERFECT)
    assert c["ece"] == 0.0 and c["mce"] == 0.0 and c["brier"] == 0.0
    assert c["n_bins"] == 10 and len(c["bins"]) == 10


# ---------------------------------------------------------------------------
# bootstrap CI
# ---------------------------------------------------------------------------

def test_ci_is_deterministic_and_ordered():
    a = calibration_ci(LABELS, CONSTANT_BASE, "brier", n_boot=200, seed=1)
    b = calibration_ci(LABELS, CONSTANT_BASE, "brier", n_boot=200, seed=1)
    assert a == b
    assert a["lo"] <= a["point"] <= a["hi"]


def test_ci_on_perfect_is_degenerate():
    ci = calibration_ci(LABELS, PERFECT, "ece", n_boot=200)
    assert ci["lo"] == 0.0 and ci["point"] == 0.0 and ci["hi"] == 0.0


# ---------------------------------------------------------------------------
# integration with the real sample
# ---------------------------------------------------------------------------

def test_detector_calibration_on_sample():
    from tether import load_sample, default_detectors
    examples = load_sample()
    cal = detector_calibration(default_detectors(examples), examples)
    # the constant detector emits the base rate, so it is trivially calibrated
    assert cal["majority_constant"]["ece"] == 0.0
    # the useful lexical detectors rank well but are not calibrated probabilities
    assert cal["novel_token"]["ece"] > 0.1


def test_report_includes_calibration_section():
    from tether import load_sample, default_detectors, generate_report
    examples = load_sample()
    report = generate_report(default_detectors(examples), examples)
    assert "## Are the scores trustworthy probabilities?" in report
    assert "ECE" in report and "Brier" in report
