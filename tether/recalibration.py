"""
Recalibrating detector scores into probabilities.

The calibration analysis (tether.calibration) showed the free detectors rank
reasonably but their raw scores are not probabilities: a `novel_token` score of
0.65 does not mean 65% faithful. This module asks the follow-up question: how
much of that gap is fixable by a post-hoc calibration map?

Two standard maps, both learned from (score -> faithful) pairs:

- isotonic: a non-decreasing step map fit by the Pool Adjacent Violators
  Algorithm (PAV). Non-parametric; only assumes higher score => not-less
  faithful.
- platt: a logistic map P = sigmoid(a * score + b), fit by a few Newton (IRLS)
  steps. Parametric; a smooth sigmoid.

The honest test is cross-validated: the map is fit on training folds and applied
to a held-out fold, so the reported post-calibration ECE/Brier is not measured on
the data the map was tuned on. Pure stdlib, seeded, deterministic.
"""

import math
from typing import Callable, Dict, List, Any, Sequence

from .calibration import expected_calibration_error, brier_score
from .validation import stratified_folds


# ---------------------------------------------------------------------------
# isotonic (PAV)
# ---------------------------------------------------------------------------

def _pav(means: Sequence[float], weights: Sequence[float]) -> List[float]:
    """
    Weighted Pool Adjacent Violators. Returns a non-decreasing fitted value for
    each input group, pooling adjacent groups that violate monotonicity.
    """
    blocks: List[List[float]] = []  # each: [weighted_sum, weight, group_count]
    for m, w in zip(means, weights):
        blocks.append([float(m) * w, float(w), 1])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            s2, w2, c2 = blocks.pop()
            s1, w1, c1 = blocks.pop()
            blocks.append([s1 + s2, w1 + w2, c1 + c2])
    out: List[float] = []
    for s, w, c in blocks:
        out.extend([s / w] * c)
    return out


def isotonic_fit(scores: Sequence[float], labels: Sequence[int]) -> Callable[[float], float]:
    """
    Fit an isotonic (monotone non-decreasing) map from score to P(faithful).

    Points with equal scores are pooled into one group first, so the fit is
    independent of their order. Returns a predictor that linearly interpolates
    between the fitted knots and clamps outside the observed score range.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    if not scores:
        raise ValueError("need at least one point")

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    knots_x: List[float] = []
    group_sum: List[float] = []
    group_cnt: List[float] = []
    for i in order:
        x = float(scores[i])
        y = 1.0 if labels[i] == 1 else 0.0
        if knots_x and knots_x[-1] == x:
            group_sum[-1] += y
            group_cnt[-1] += 1
        else:
            knots_x.append(x)
            group_sum.append(y)
            group_cnt.append(1)
    means = [s / c for s, c in zip(group_sum, group_cnt)]
    knots_y = _pav(means, group_cnt)

    def predict(s: float) -> float:
        if s <= knots_x[0]:
            return knots_y[0]
        if s >= knots_x[-1]:
            return knots_y[-1]
        lo, hi = 0, len(knots_x) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if knots_x[mid] <= s:
                lo = mid
            else:
                hi = mid
        x0, x1 = knots_x[lo], knots_x[hi]
        y0, y1 = knots_y[lo], knots_y[hi]
        if x1 == x0:
            return y1
        return y0 + (y1 - y0) * (s - x0) / (x1 - x0)

    return predict


# ---------------------------------------------------------------------------
# Platt (logistic via IRLS / Newton)
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def platt_fit(scores: Sequence[float], labels: Sequence[int], iters: int = 50) -> Callable[[float], float]:
    """
    Fit a logistic map P = sigmoid(a * score + b) by Newton/IRLS steps.

    Returns a predictor. A small ridge keeps the 2x2 solve well-conditioned when
    the scores barely separate the classes.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    if not scores:
        raise ValueError("need at least one point")

    xs = [float(s) for s in scores]
    ys = [1.0 if l == 1 else 0.0 for l in labels]
    a, b = 0.0, 0.0
    ridge = 1e-6
    for _ in range(iters):
        g0 = g1 = 0.0            # gradient of log-likelihood wrt (a, b)
        h00 = h01 = h11 = 0.0    # Hessian (negative), built as X^T W X
        for x, y in zip(xs, ys):
            p = _sigmoid(a * x + b)
            err = y - p
            g0 += err * x
            g1 += err
            w = p * (1.0 - p)
            h00 += w * x * x
            h01 += w * x
            h11 += w
        h00 += ridge
        h11 += ridge
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        # (a, b) += H^{-1} g
        da = (h11 * g0 - h01 * g1) / det
        db = (-h01 * g0 + h00 * g1) / det
        a += da
        b += db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break

    return lambda s: _sigmoid(a * float(s) + b)


# ---------------------------------------------------------------------------
# cross-validated evaluation
# ---------------------------------------------------------------------------

_METHODS = {"isotonic": isotonic_fit, "platt": platt_fit}


def apply_map(predictor: Callable[[float], float], scores: Sequence[float]) -> List[float]:
    """Map a list of raw scores through a fitted calibrator, clamped to [0, 1]."""
    return [max(0.0, min(1.0, predictor(s))) for s in scores]


def cross_val_recalibration(
    labels: Sequence[int],
    scores: Sequence[float],
    method: str = "isotonic",
    k: int = 5,
    seed: int = 0,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Raw vs post-calibration ECE/Brier, with the map fit out-of-fold.

    For each fold the calibrator is fit on the other folds and applied to the
    held-out fold; the held-out calibrated probabilities are pooled and scored,
    so the calibrated numbers are honest (never measured on the fit data). Raw
    numbers need no fitting and are measured on the same points.
    """
    labels = list(labels)
    scores = list(scores)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must be the same length")
    if method not in _METHODS:
        raise ValueError(f"unsupported method: {method!r} (use 'isotonic' or 'platt')")
    fit = _METHODS[method]

    folds = stratified_folds(labels, k, seed)
    pooled = [0.0] * len(labels)
    for test_idx in folds:
        test_set = set(test_idx)
        train_idx = [i for i in range(len(labels)) if i not in test_set]
        predictor = fit([scores[i] for i in train_idx], [labels[i] for i in train_idx])
        for i in test_idx:
            pooled[i] = max(0.0, min(1.0, predictor(scores[i])))

    return {
        "method": method,
        "k": k,
        "seed": seed,
        "raw_ece": round(expected_calibration_error(labels, scores, n_bins), 4),
        "raw_brier": round(brier_score(labels, scores), 4),
        "cal_ece": round(expected_calibration_error(labels, pooled, n_bins), 4),
        "cal_brier": round(brier_score(labels, pooled), 4),
    }


def detector_recalibration(detectors, examples, method: str = "isotonic", k: int = 5, seed: int = 0) -> Dict[str, Any]:
    """Cross-validated raw-vs-calibrated ECE/Brier for every detector, keyed by name."""
    labels = [e.label for e in examples]
    return {
        d.name: cross_val_recalibration(labels, d.score_all(examples), method=method, k=k, seed=seed)
        for d in detectors
    }
