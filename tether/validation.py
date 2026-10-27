"""
Cross-validated operating points for Tether.

The leaderboard reports F1 at each detector's F1-optimal threshold. That
threshold is chosen on the same data it is then scored on, so the F1 is an
*oracle* operating point: optimistic, because a real deployment has to fix its
threshold before seeing the answers it will judge. This module measures how much
optimism that hides.

k-fold cross-validation, stratified by label and seeded: for each fold the
threshold is picked on the other folds and applied to the held-out fold, so
every example is predicted exactly once by a threshold it did not help choose.
Pooling those held-out predictions gives an honest F1; the gap to the oracle F1
is the optimism in the leaderboard number.

Pure stdlib and deterministic.
"""

import random
from typing import Dict, List, Any, Sequence

from .metrics import best_threshold, confusion_matrix, ConfusionMatrix


def stratified_folds(labels: Sequence[int], k: int = 5, seed: int = 0) -> List[List[int]]:
    """
    Partition indices into k folds, spreading each class evenly across folds.

    Indices within a class are shuffled (seeded) and then dealt round-robin, so
    every fold holds roughly the same class balance as the whole set.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    rng = random.Random(seed)
    folds: List[List[int]] = [[] for _ in range(k)]
    by_class: Dict[int, List[int]] = {}
    for i, l in enumerate(labels):
        by_class.setdefault(l, []).append(i)
    for _, idxs in sorted(by_class.items()):
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        for pos, i in enumerate(shuffled):
            folds[pos % k].append(i)
    return folds


def cross_val_operating_point(
    labels: Sequence[int],
    scores: Sequence[float],
    k: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Cross-validated F1 vs the oracle F1 for one detector's scores.

    Returns the oracle F1 (threshold tuned on all the data, as the leaderboard
    reports it), the pooled cross-validated F1 (each example predicted by a
    threshold trained without it), the mean/std of the per-fold F1, and the
    optimism gap between oracle and cross-validated F1.
    """
    labels = list(labels)
    scores = list(scores)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must be the same length")
    if len(labels) < k:
        raise ValueError("need at least k examples")

    folds = stratified_folds(labels, k, seed)
    pooled_pred_hallucinated: List[bool] = [False] * len(labels)
    per_fold_f1: List[float] = []
    fold_thresholds: List[float] = []

    for test_idx in folds:
        test_set = set(test_idx)
        train_idx = [i for i in range(len(labels)) if i not in test_set]
        thr, _ = best_threshold([labels[i] for i in train_idx], [scores[i] for i in train_idx])
        fold_thresholds.append(thr)
        if test_idx:
            cm = confusion_matrix([labels[i] for i in test_idx],
                                  [scores[i] for i in test_idx], thr)
            per_fold_f1.append(cm.f1)
        for i in test_idx:
            pooled_pred_hallucinated[i] = scores[i] <= thr

    # Pooled held-out confusion matrix (positive class = hallucinated = label 0).
    tp = fp = tn = fn = 0
    for pred_hallu, label in zip(pooled_pred_hallucinated, labels):
        actual_hallu = label == 0
        if pred_hallu and actual_hallu:
            tp += 1
        elif pred_hallu and not actual_hallu:
            fp += 1
        elif not pred_hallu and not actual_hallu:
            tn += 1
        else:
            fn += 1
    pooled = ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)

    oracle_thr, oracle_f1 = best_threshold(labels, scores)
    mean_f1 = sum(per_fold_f1) / len(per_fold_f1)
    std_f1 = (sum((f - mean_f1) ** 2 for f in per_fold_f1) / len(per_fold_f1)) ** 0.5
    return {
        "k": k,
        "seed": seed,
        "oracle_f1": round(oracle_f1, 4),
        "oracle_threshold": round(oracle_thr, 4),
        "cv_f1": round(pooled.f1, 4),
        "cv_f1_mean": round(mean_f1, 4),
        "cv_f1_std": round(std_f1, 4),
        "cv_precision": round(pooled.precision, 4),
        "cv_recall": round(pooled.recall, 4),
        "optimism_gap": round(oracle_f1 - pooled.f1, 4),
        "fold_thresholds": [round(t, 4) for t in fold_thresholds],
    }


def detector_validation(detectors, examples, k: int = 5, seed: int = 0) -> Dict[str, Any]:
    """Cross-validated operating point for every detector, keyed by name."""
    labels = [e.label for e in examples]
    return {
        d.name: cross_val_operating_point(labels, d.score_all(examples), k=k, seed=seed)
        for d in detectors
    }
