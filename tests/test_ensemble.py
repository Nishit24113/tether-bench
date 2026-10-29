"""
Tests for stacking the free detectors: the linear solve, the IRLS logistic fit,
and the out-of-fold stacked evaluation.
"""

import pytest

from tether.ensemble import _solve, logistic_fit, _predict, stacked_scores, cross_val_stack


# ---------------------------------------------------------------------------
# linear solve
# ---------------------------------------------------------------------------

def test_solve_matches_known_solution():
    # 2x + y = 5 ; x + 3y = 10  ->  x = 1, y = 3
    x = _solve([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
    assert x[0] == pytest.approx(1.0, abs=1e-9)
    assert x[1] == pytest.approx(3.0, abs=1e-9)


def test_solve_handles_pivoting():
    # leading zero pivot forces a row swap
    x = _solve([[0.0, 2.0], [1.0, 1.0]], [4.0, 3.0])
    assert x[0] == pytest.approx(1.0, abs=1e-9)
    assert x[1] == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# logistic fit
# ---------------------------------------------------------------------------

def test_logistic_fit_separates_a_clean_signal():
    # one feature that cleanly separates: low -> 0, high -> 1
    features = [[0.1]] * 10 + [[0.9]] * 10
    targets = [0.0] * 10 + [1.0] * 10
    w = logistic_fit(features, targets)
    assert _predict(w, [0.9]) > 0.5 > _predict(w, [0.1])


def test_logistic_fit_empty_raises():
    with pytest.raises(ValueError):
        logistic_fit([], [])


# ---------------------------------------------------------------------------
# stacked evaluation
# ---------------------------------------------------------------------------

def test_stacked_scores_shape_and_range():
    from tether import load_sample, default_detectors
    examples = load_sample()
    pooled = stacked_scores(default_detectors(examples), examples)
    assert len(pooled) == len(examples)
    assert all(0.0 <= s <= 1.0 for s in pooled)


def test_stacked_scores_deterministic():
    from tether import load_sample, default_detectors
    examples = load_sample()
    dets = default_detectors(examples)
    assert stacked_scores(dets, examples, seed=1) == stacked_scores(dets, examples, seed=1)


def test_cross_val_stack_reports_no_free_tier_win():
    from tether import load_sample, default_detectors
    examples = load_sample()
    r = cross_val_stack(default_detectors(examples), examples)
    assert r["best_single"] == "novel_token"
    assert 0.0 <= r["stack_auroc"] <= 1.0
    lo, hi = r["stack_ci"]
    assert lo <= r["stack_auroc"] <= hi
    # the redundant free signals do not combine into a significant gain
    assert r["significant"] is False


def test_report_includes_ensemble_section():
    from tether import load_sample, default_detectors, generate_report
    examples = load_sample()
    report = generate_report(default_detectors(examples), examples)
    assert "## Do the free signals combine to beat the best single detector?" in report
