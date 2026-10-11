"""
Tests for the metrics module — confusion matrix, AUROC (incl. ties and
single-class edge cases), threshold selection, and the evaluate bundle.
"""

import pytest

from tether.metrics import (
    confusion_matrix, auroc, best_threshold, evaluate, ConfusionMatrix,
)


# ---------------------------------------------------------------------------
# Confusion matrix  (positive class = hallucinated, label 0)
# ---------------------------------------------------------------------------

def test_confusion_perfect_separation():
    # faithful (label 1) score high, hallucinated (label 0) score low
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    cm = confusion_matrix(labels, scores, threshold=0.5)
    assert (cm.tp, cm.fp, cm.tn, cm.fn) == (2, 0, 2, 0)
    assert cm.precision == 1.0
    assert cm.recall == 1.0
    assert cm.f1 == 1.0
    assert cm.accuracy == 1.0


def test_confusion_all_wrong():
    labels = [1, 1, 0, 0]
    scores = [0.1, 0.2, 0.9, 0.8]  # inverted
    cm = confusion_matrix(labels, scores, threshold=0.5)
    assert (cm.tp, cm.fp, cm.tn, cm.fn) == (0, 2, 0, 2)
    assert cm.f1 == 0.0


def test_confusion_threshold_is_inclusive_lower():
    # score exactly at threshold counts as hallucinated (score <= threshold)
    labels = [0]
    scores = [0.5]
    cm = confusion_matrix(labels, scores, threshold=0.5)
    assert cm.tp == 1


def test_confusion_length_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_matrix([1, 0], [0.5], threshold=0.5)


def test_confusion_empty_metrics_zero():
    cm = ConfusionMatrix(0, 0, 0, 0)
    assert cm.precision == 0.0
    assert cm.recall == 0.0
    assert cm.f1 == 0.0
    assert cm.accuracy == 0.0


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def test_auroc_perfect():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert auroc(labels, scores) == 1.0


def test_auroc_inverted_is_zero():
    labels = [1, 1, 0, 0]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert auroc(labels, scores) == 0.0


def test_auroc_random_half():
    # all tied scores -> no separation -> 0.5
    labels = [1, 0, 1, 0]
    scores = [0.5, 0.5, 0.5, 0.5]
    assert auroc(labels, scores) == 0.5


def test_auroc_single_class_returns_half():
    assert auroc([1, 1, 1], [0.1, 0.2, 0.3]) == 0.5
    assert auroc([0, 0, 0], [0.1, 0.2, 0.3]) == 0.5


def test_auroc_handles_ties_between_classes():
    # one faithful and one hallucinated share a score -> half credit on that pair
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.5, 0.5, 0.1]
    # pairs (faithful, hallucinated): (0.9 vs 0.5)=correct, (0.9 vs 0.1)=correct,
    # (0.5 vs 0.5)=tie(0.5), (0.5 vs 0.1)=correct -> (1+1+0.5+1)/4 = 0.875
    assert auroc(labels, scores) == pytest.approx(0.875)


def test_auroc_length_mismatch_raises():
    with pytest.raises(ValueError):
        auroc([1, 0], [0.5])


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def test_best_threshold_finds_perfect_split():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    t, f1 = best_threshold(labels, scores)
    assert f1 == 1.0
    assert 0.2 <= t <= 0.8


def test_best_threshold_empty():
    assert best_threshold([], []) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# evaluate bundle
# ---------------------------------------------------------------------------

def test_evaluate_auto_threshold():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    result = evaluate(labels, scores)
    assert result["auroc"] == 1.0
    assert result["f1"] == 1.0
    assert "threshold" in result
    assert result["confusion"]["tp"] == 2


def test_evaluate_fixed_threshold():
    labels = [1, 0]
    scores = [0.9, 0.1]
    result = evaluate(labels, scores, threshold=0.5)
    assert result["threshold"] == 0.5
