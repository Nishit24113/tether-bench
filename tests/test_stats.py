"""
Tests for bootstrap confidence intervals and the paired significance test.
"""

import random

import pytest

from tether.stats import (
    _percentile, _stratified_resample, _class_indices,
    bootstrap_ci, paired_bootstrap, detector_uncertainty,
)


LABELS = [1] * 10 + [0] * 10                       # 10 faithful, 10 hallucinated
PERFECT = [0.9] * 10 + [0.1] * 10                  # separates perfectly
CONSTANT = [0.5] * 20                              # no signal


def _noisy(seed=7):
    # class means 0.6 vs 0.4 with sd 0.2: overlapping, so AUROC sits between 0.5 and 1
    rng = random.Random(seed)
    return [min(1.0, max(0.0, (0.6 if l else 0.4) + rng.gauss(0, 0.2))) for l in LABELS]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_percentile_interpolates():
    vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _percentile(vals, 0.0) == 0.0
    assert _percentile(vals, 1.0) == 4.0
    assert _percentile(vals, 0.5) == 2.0
    assert _percentile(vals, 0.125) == pytest.approx(0.5)


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        _percentile([], 0.5)


def test_stratified_resample_keeps_class_counts():
    hallu, faithful = _class_indices(LABELS)
    idx = _stratified_resample(hallu, faithful, random.Random(0))
    assert len(idx) == len(LABELS)
    assert sum(1 for i in idx if LABELS[i] == 0) == 10
    assert sum(1 for i in idx if LABELS[i] == 1) == 10


def test_single_class_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([1, 1, 1], [0.1, 0.2, 0.3])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([1, 0], [0.5])
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [0.5, 0.4], [0.3])


def test_unsupported_metric_raises():
    with pytest.raises(ValueError):
        bootstrap_ci(LABELS, PERFECT, metric="precision")


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_perfect_detector_ci_is_one():
    ci = bootstrap_ci(LABELS, PERFECT, n_boot=300)
    assert ci["point"] == 1.0 and ci["lo"] == 1.0 and ci["hi"] == 1.0


def test_constant_detector_ci_is_half():
    ci = bootstrap_ci(LABELS, CONSTANT, n_boot=300)
    assert ci["point"] == 0.5 and ci["lo"] == 0.5 and ci["hi"] == 0.5


def test_ci_is_deterministic_for_a_seed():
    s = _noisy()
    assert bootstrap_ci(LABELS, s, n_boot=500, seed=3) == bootstrap_ci(LABELS, s, n_boot=500, seed=3)


def test_ci_brackets_point_and_is_ordered():
    s = _noisy()
    ci = bootstrap_ci(LABELS, s, n_boot=1000)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert 0.0 <= ci["lo"] < ci["hi"] <= 1.0


def test_more_data_narrows_the_interval():
    small = bootstrap_ci(LABELS, _noisy(), n_boot=1000)
    big_labels = LABELS * 5
    big_scores = []
    for k in range(5):
        big_scores += _noisy(seed=100 + k)
    big = bootstrap_ci(big_labels, big_scores, n_boot=1000)
    assert (big["hi"] - big["lo"]) < (small["hi"] - small["lo"])


def test_f1_ci_runs_at_fixed_threshold():
    ci = bootstrap_ci(LABELS, _noisy(), metric="f1", n_boot=300)
    assert 0.0 <= ci["lo"] <= ci["hi"] <= 1.0
    assert ci["metric"] == "f1"


# ---------------------------------------------------------------------------
# paired_bootstrap
# ---------------------------------------------------------------------------

def test_identical_detectors_are_not_different():
    s = _noisy()
    r = paired_bootstrap(LABELS, s, s, n_boot=500)
    assert r["diff"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0
    assert r["p_value"] == 1.0 and r["significant"] is False


def test_perfect_beats_constant_significantly():
    r = paired_bootstrap(LABELS, PERFECT, CONSTANT, n_boot=500)
    assert r["diff"] == 0.5
    assert r["lo"] > 0 and r["p_value"] < 0.05 and r["significant"] is True


def test_paired_is_antisymmetric():
    s1, s2 = _noisy(1), _noisy(2)
    ab = paired_bootstrap(LABELS, s1, s2, n_boot=500, seed=4)
    ba = paired_bootstrap(LABELS, s2, s1, n_boot=500, seed=4)
    # values are rounded to 4 dp, so allow one rounding step (plus float slack)
    assert ab["diff"] == pytest.approx(-ba["diff"], abs=2e-4)
    assert ab["lo"] == pytest.approx(-ba["hi"], abs=2e-4)
    assert ab["hi"] == pytest.approx(-ba["lo"], abs=2e-4)
    assert ab["p_value"] == ba["p_value"]


def test_paired_counts_examples_that_differ():
    s = _noisy()
    assert paired_bootstrap(LABELS, s, s, n_boot=100)["n_differ"] == 0
    r = paired_bootstrap(LABELS, PERFECT, CONSTANT, n_boot=100)
    assert r["n_differ"] == 20 and r["n"] == 20


def test_zero_p_value_is_reported_as_a_bound():
    from tether.report import _fmt_p
    assert _fmt_p(0.0, 2000) == "< 0.001"
    assert _fmt_p(0.0071, 2000) == "0.007"


# ---------------------------------------------------------------------------
# integration with the real sample
# ---------------------------------------------------------------------------

def test_detector_uncertainty_on_sample():
    from tether import load_sample, default_detectors
    examples = load_sample()
    unc = detector_uncertainty(default_detectors(examples), examples, n_boot=300)
    assert unc["best"] in unc["cis"]
    assert unc["best"] not in unc["vs_best"]
    assert unc["cis"]["majority_constant"]["point"] == 0.5
    # the best detector is at least as good as the constant floor
    assert unc["cis"][unc["best"]]["point"] >= 0.5
    assert "majority_constant" in unc["vs_best"]


def test_report_includes_ci_and_significance():
    from tether import load_sample, default_detectors, generate_report
    examples = load_sample()
    report = generate_report(default_detectors(examples), examples)
    assert "AUROC 95% CI" in report
    assert "## Are the gaps real?" in report
