"""
Tests for cross-validated operating points: stratified folds and the oracle vs
cross-validated F1 comparison.
"""

import pytest

from tether.validation import (
    stratified_folds, cross_val_operating_point, detector_validation,
)


LABELS = [1] * 25 + [0] * 25
PERFECT = [0.9] * 25 + [0.1] * 25          # cleanly separable at any reasonable split
CONSTANT = [0.5] * 50                       # flags everything regardless of threshold


# ---------------------------------------------------------------------------
# stratified folds
# ---------------------------------------------------------------------------

def test_folds_partition_every_index_once():
    folds = stratified_folds(LABELS, k=5, seed=0)
    flat = sorted(i for f in folds for i in f)
    assert flat == list(range(len(LABELS)))          # a partition, no dup or drop
    assert len(folds) == 5


def test_folds_are_class_balanced():
    folds = stratified_folds(LABELS, k=5, seed=0)
    for f in folds:
        faithful = sum(1 for i in f if LABELS[i] == 1)
        hallucinated = sum(1 for i in f if LABELS[i] == 0)
        assert faithful == 5 and hallucinated == 5   # 25 per class over 5 folds


def test_folds_are_seeded():
    assert stratified_folds(LABELS, 5, seed=1) == stratified_folds(LABELS, 5, seed=1)


def test_bad_k_raises():
    with pytest.raises(ValueError):
        stratified_folds(LABELS, k=1)


# ---------------------------------------------------------------------------
# cross-validated operating point
# ---------------------------------------------------------------------------

def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        cross_val_operating_point([1, 0], [0.5])


def test_too_few_examples_raises():
    with pytest.raises(ValueError):
        cross_val_operating_point([1, 0, 1], [0.9, 0.1, 0.8], k=5)


def test_separable_scores_have_no_optimism():
    v = cross_val_operating_point(LABELS, PERFECT, k=5, seed=0)
    assert v["oracle_f1"] == 1.0 and v["cv_f1"] == 1.0
    assert v["optimism_gap"] == 0.0


def test_constant_detector_has_no_optimism():
    # a constant score flags every answer at any threshold, so oracle and CV agree
    v = cross_val_operating_point(LABELS, CONSTANT, k=5, seed=0)
    assert v["optimism_gap"] == 0.0
    assert v["cv_recall"] == 1.0


def test_cv_f1_not_above_oracle_and_fields_present():
    v = cross_val_operating_point(LABELS, PERFECT, k=5, seed=0)
    # the oracle tunes on all the data, so it can never be beaten by held-out CV
    assert v["cv_f1"] <= v["oracle_f1"] + 1e-9
    assert v["optimism_gap"] == pytest.approx(v["oracle_f1"] - v["cv_f1"], abs=1e-9)
    assert len(v["fold_thresholds"]) == 5
    assert 0.0 <= v["cv_precision"] <= 1.0 and 0.0 <= v["cv_recall"] <= 1.0


def test_is_deterministic_for_a_seed():
    assert (cross_val_operating_point(LABELS, PERFECT, 5, 0)
            == cross_val_operating_point(LABELS, PERFECT, 5, 0))


# ---------------------------------------------------------------------------
# integration with the real sample
# ---------------------------------------------------------------------------

def test_detector_validation_on_sample():
    from tether import load_sample, default_detectors
    examples = load_sample()
    val = detector_validation(default_detectors(examples), examples, k=5, seed=0)
    # the signal-carrying detectors are optimistic at the oracle threshold
    assert val["novel_token"]["optimism_gap"] > 0
    assert val["novel_token"]["cv_f1"] < val["novel_token"]["oracle_f1"]
    # the constant floor is not
    assert val["majority_constant"]["optimism_gap"] == 0.0


def test_report_includes_validation_section():
    from tether import load_sample, default_detectors, generate_report
    examples = load_sample()
    report = generate_report(default_detectors(examples), examples)
    assert "## How optimistic is the F1?" in report
    assert "Oracle F1" in report and "Optimism" in report
