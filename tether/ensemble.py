"""
Stacking the free detectors.

The free detectors look at overlapping surface signals (token containment, novel
tokens, TF-IDF cosine), so a natural question is whether combining them beats the
best single one, and how close that gets to the semantic tiers. This fits a
logistic combiner over the detectors' scores.

The combiner is trained, so reporting its AUROC on the same data would leak. We
evaluate it out-of-fold: fit on training folds, predict the held-out fold, pool
the held-out predictions, and score those. Pure stdlib (a small IRLS logistic
regression with a Gaussian-elimination solve), seeded and deterministic.
"""

import math
from typing import List, Dict, Any, Sequence

from .metrics import auroc
from .validation import stratified_folds
from .stats import bootstrap_ci, paired_bootstrap


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _solve(matrix: List[List[float]], rhs: List[float]) -> List[float]:
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-15:
            continue  # singular column; leave this variable at its current step
        a[col], a[pivot] = a[pivot], a[col]
        piv = a[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col] / piv
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    return [a[i][n] / a[i][i] if abs(a[i][i]) > 1e-15 else 0.0 for i in range(n)]


def logistic_fit(features: Sequence[Sequence[float]], targets: Sequence[float],
                 iters: int = 50, ridge: float = 1e-6) -> List[float]:
    """
    Logistic regression by IRLS. `features` rows exclude the bias; the returned
    weight vector has the bias as its last entry. A small ridge keeps the
    Newton solve stable when the features nearly separate the classes.
    """
    if not features:
        raise ValueError("need at least one row")
    d = len(features[0])
    w = [0.0] * (d + 1)
    for _ in range(iters):
        grad = [0.0] * (d + 1)
        hess = [[0.0] * (d + 1) for _ in range(d + 1)]
        for row, y in zip(features, targets):
            feats = list(row) + [1.0]
            p = _sigmoid(sum(wj * fj for wj, fj in zip(w, feats)))
            err = y - p
            weight = p * (1.0 - p)
            for i in range(d + 1):
                grad[i] += err * feats[i]
                for j in range(d + 1):
                    hess[i][j] += weight * feats[i] * feats[j]
        for i in range(d + 1):
            hess[i][i] += ridge
        step = _solve(hess, grad)
        for i in range(d + 1):
            w[i] += step[i]
        if max(abs(s) for s in step) < 1e-9:
            break
    return w


def _predict(weights: Sequence[float], row: Sequence[float]) -> float:
    feats = list(row) + [1.0]
    return _sigmoid(sum(wj * fj for wj, fj in zip(weights, feats)))


def stacked_scores(detectors, examples, k: int = 5, seed: int = 0) -> List[float]:
    """
    Out-of-fold stacked faithfulness scores: a logistic combiner of the
    detectors' scores, each example predicted by a combiner trained without it.
    """
    labels = [e.label for e in examples]
    feats = [[d.score(e) for d in detectors] for e in examples]
    folds = stratified_folds(labels, k, seed)
    pooled = [0.0] * len(examples)
    for test_idx in folds:
        test_set = set(test_idx)
        train_idx = [i for i in range(len(examples)) if i not in test_set]
        w = logistic_fit([feats[i] for i in train_idx],
                         [1.0 if labels[i] == 1 else 0.0 for i in train_idx])
        for i in test_idx:
            pooled[i] = _predict(w, feats[i])
    return pooled


def cross_val_stack(detectors, examples, k: int = 5, seed: int = 0, n_boot: int = 2000) -> Dict[str, Any]:
    """
    Compare the out-of-fold stacked combiner with the best single input detector.

    Returns the stack's AUROC and CI, the best single detector and its AUROC, the
    difference, and a paired bootstrap of stack vs best-single on the same
    resamples (is the combination a real improvement?).
    """
    labels = [e.label for e in examples]
    pooled = stacked_scores(detectors, examples, k=k, seed=seed)
    singles = {d.name: d.score_all(examples) for d in detectors}
    best_name = max(singles, key=lambda n: auroc(labels, singles[n]))

    stack_ci = bootstrap_ci(labels, pooled, n_boot=n_boot, seed=seed)
    vs_best = paired_bootstrap(labels, pooled, singles[best_name], n_boot=n_boot, seed=seed)
    return {
        "k": k,
        "seed": seed,
        "stack_auroc": stack_ci["point"],
        "stack_ci": [stack_ci["lo"], stack_ci["hi"]],
        "best_single": best_name,
        "best_single_auroc": round(auroc(labels, singles[best_name]), 4),
        "diff": vs_best["diff"],
        "diff_ci": [vs_best["lo"], vs_best["hi"]],
        "p_value": vs_best["p_value"],
        "significant": vs_best["significant"],
    }
