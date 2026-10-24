"""
Calibration analysis for Tether.

AUROC measures whether a detector *ranks* faithful answers above hallucinated
ones. It says nothing about whether the score is a trustworthy probability: a
detector can rank perfectly yet output 0.55 for answers that turn out faithful
95% of the time. Calibration measures that second property — when a detector
emits a faithfulness score of p, is the answer actually faithful about p of the
time? The detector interface explicitly does not promise this ("calibration is
not required"), so this module quantifies how far off the raw scores are.

We read the faithfulness score as a predicted P(faithful); faithful is
label == 1. We report:

- reliability_bins: observed faithful rate vs mean predicted score, per bin
- brier_score: mean squared error of the probabilities (lower is better)
- expected_calibration_error (ECE): bin-count-weighted mean gap between
  predicted and observed
- maximum_calibration_error (MCE): the worst single-bin gap

Pure stdlib and deterministic. The bootstrap CI reuses tether.stats, so the
calibration numbers carry the same stratified, seeded error bars as the
leaderboard.
"""

import random
from typing import Dict, List, Any, Sequence

from .stats import _class_indices, _stratified_resample, _percentile


def _check(labels: Sequence[int], scores: Sequence[float]) -> None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must be the same length")
    if not labels:
        raise ValueError("need at least one example")


def brier_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Mean squared error of the faithfulness probabilities (0 = perfect)."""
    _check(labels, scores)
    return sum((s - (1 if l == 1 else 0)) ** 2 for l, s in zip(labels, scores)) / len(labels)


def reliability_bins(labels: Sequence[int], scores: Sequence[float], n_bins: int = 10) -> List[Dict[str, Any]]:
    """
    Partition [0, 1] into `n_bins` equal-width score bins. For each bin report
    how many scores fell in it, their mean, the observed fraction faithful, and
    the absolute gap between the two. Empty bins carry None for the rates.

    The last bin is closed on the right so a score of exactly 1.0 has a home.
    """
    _check(labels, scores)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        members = [(l, s) for l, s in zip(labels, scores)
                   if (lo <= s < hi) or (b == n_bins - 1 and s >= hi)]
        count = len(members)
        if count:
            mean_score = sum(s for _, s in members) / count
            frac_faithful = sum(1 for l, _ in members if l == 1) / count
            gap = abs(mean_score - frac_faithful)
        else:
            mean_score = frac_faithful = gap = None
        bins.append({"lo": lo, "hi": hi, "count": count, "mean_score": mean_score,
                     "frac_faithful": frac_faithful, "gap": gap})
    return bins


def expected_calibration_error(labels: Sequence[int], scores: Sequence[float], n_bins: int = 10) -> float:
    """Bin-count-weighted mean of the per-bin predicted-vs-observed gaps."""
    bins = reliability_bins(labels, scores, n_bins)
    n = len(labels)
    return sum(b["count"] / n * b["gap"] for b in bins if b["count"])


def maximum_calibration_error(labels: Sequence[int], scores: Sequence[float], n_bins: int = 10) -> float:
    """Largest single-bin gap between mean predicted score and observed rate."""
    bins = reliability_bins(labels, scores, n_bins)
    return max((b["gap"] for b in bins if b["count"]), default=0.0)


def calibration(labels: Sequence[int], scores: Sequence[float], n_bins: int = 10) -> Dict[str, Any]:
    """Full calibration bundle: ECE, MCE, Brier, and the reliability bins."""
    bins = reliability_bins(labels, scores, n_bins)
    n = len(labels)
    return {
        "ece": round(sum(b["count"] / n * b["gap"] for b in bins if b["count"]), 4),
        "mce": round(max((b["gap"] for b in bins if b["count"]), default=0.0), 4),
        "brier": round(brier_score(labels, scores), 4),
        "n_bins": n_bins,
        "bins": bins,
    }


_METRICS = {
    "ece": expected_calibration_error,
    "mce": maximum_calibration_error,
    "brier": lambda labels, scores, n_bins=10: brier_score(labels, scores),
}


def calibration_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    metric: str = "ece",
    n_bins: int = 10,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Stratified bootstrap confidence interval for a calibration metric
    ('ece', 'mce', or 'brier'), so a calibration gap comes with error bars too.
    """
    _check(labels, scores)
    if metric not in _METRICS:
        raise ValueError(f"unsupported metric: {metric!r} (use 'ece', 'mce', or 'brier')")
    fn = _METRICS[metric]
    hallucinated, faithful = _class_indices(labels)
    rng = random.Random(seed)

    point = fn(labels, scores, n_bins)
    boots = sorted(
        fn([labels[i] for i in idx], [scores[i] for i in idx], n_bins)
        for idx in (_stratified_resample(hallucinated, faithful, rng) for _ in range(n_boot))
    )
    return {
        "metric": metric,
        "point": round(point, 4),
        "lo": round(_percentile(boots, alpha / 2), 4),
        "hi": round(_percentile(boots, 1 - alpha / 2), 4),
        "n_boot": n_boot,
        "alpha": alpha,
        "seed": seed,
        "n_bins": n_bins,
    }


def detector_calibration(detectors, examples, n_bins: int = 10) -> Dict[str, Any]:
    """Calibration bundle for every detector, keyed by name."""
    labels = [e.label for e in examples]
    return {d.name: calibration(labels, d.score_all(examples), n_bins) for d in detectors}
