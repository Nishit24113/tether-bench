"""
Statistical uncertainty for Tether results.

With a small benchmark (tens of examples), a leaderboard gap of a few AUROC
points can easily be noise. This module puts error bars on the numbers:

- bootstrap_ci: a percentile bootstrap confidence interval for a detector's
  AUROC (or F1 at a fixed threshold). Resampling is STRATIFIED by class, so
  every resample keeps the same number of faithful and hallucinated examples
  and AUROC is always defined.
- paired_bootstrap: a paired test between two detectors. Both are scored on the
  SAME resampled examples, so the interval is on their difference and the
  shared difficulty of each resample cancels out.

Pure stdlib and seeded, so every interval is reproducible.
"""

import math
import random
from typing import Dict, List, Any, Sequence

from .metrics import auroc, confusion_matrix, best_threshold


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an ascending list, q in [0, 1]."""
    if not sorted_vals:
        raise ValueError("percentile of an empty list")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def _class_indices(labels: Sequence[int]):
    hallucinated = [i for i, l in enumerate(labels) if l == 0]
    faithful = [i for i, l in enumerate(labels) if l == 1]
    if not hallucinated or not faithful:
        raise ValueError("bootstrap needs at least one faithful and one hallucinated example")
    return hallucinated, faithful


def _stratified_resample(hallucinated: List[int], faithful: List[int], rng: random.Random) -> List[int]:
    """Sample with replacement within each class, keeping class counts fixed."""
    return ([hallucinated[rng.randrange(len(hallucinated))] for _ in hallucinated]
            + [faithful[rng.randrange(len(faithful))] for _ in faithful])


def _metric_fn(metric: str, labels: Sequence[int], scores: Sequence[float]):
    """Return f(indices) -> metric value on that subset."""
    if metric == "auroc":
        return lambda idx: auroc([labels[i] for i in idx], [scores[i] for i in idx])
    if metric == "f1":
        # Fix the operating point at the full-sample F1-optimal threshold, so the
        # interval reflects sampling noise at that threshold rather than re-tuning
        # the threshold on every resample (which would be optimistic).
        threshold, _ = best_threshold(list(labels), list(scores))
        return lambda idx: confusion_matrix([labels[i] for i in idx],
                                            [scores[i] for i in idx], threshold).f1
    raise ValueError(f"unsupported metric: {metric!r} (use 'auroc' or 'f1')")


def bootstrap_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    metric: str = "auroc",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Percentile bootstrap confidence interval for a detector's metric.

    Returns the point estimate on the full data and the (1 - alpha) interval.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must be the same length")
    hallucinated, faithful = _class_indices(labels)
    fn = _metric_fn(metric, labels, scores)
    rng = random.Random(seed)

    point = fn(list(range(len(labels))))
    boots = sorted(fn(_stratified_resample(hallucinated, faithful, rng)) for _ in range(n_boot))
    return {
        "metric": metric,
        "point": round(point, 4),
        "lo": round(_percentile(boots, alpha / 2), 4),
        "hi": round(_percentile(boots, 1 - alpha / 2), 4),
        "n_boot": n_boot,
        "alpha": alpha,
        "seed": seed,
    }


def paired_bootstrap(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    metric: str = "auroc",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Paired bootstrap test of metric(A) - metric(B) on the same resamples.

    p_value is two-sided: twice the smaller tail mass of the bootstrap
    differences on either side of zero, capped at 1. Identical detectors give
    a difference of 0 and p = 1. Resamples where A and B tie exactly count
    toward both tails, which keeps the test conservative when the detectors
    often agree.

    n_differ counts the examples the two detectors score differently. A paired
    comparison can only draw evidence from those, so a small n_differ explains
    a wide interval on the difference.
    """
    if not (len(labels) == len(scores_a) == len(scores_b)):
        raise ValueError("labels and both score lists must be the same length")
    hallucinated, faithful = _class_indices(labels)
    fa = _metric_fn(metric, labels, scores_a)
    fb = _metric_fn(metric, labels, scores_b)
    rng = random.Random(seed)

    full = list(range(len(labels)))
    point = fa(full) - fb(full)
    diffs = []
    for _ in range(n_boot):
        idx = _stratified_resample(hallucinated, faithful, rng)
        diffs.append(fa(idx) - fb(idx))
    diffs.sort()

    le0 = sum(1 for d in diffs if d <= 0) / n_boot
    ge0 = sum(1 for d in diffs if d >= 0) / n_boot
    p_value = min(1.0, 2 * min(le0, ge0))
    n_differ = sum(1 for a, b in zip(scores_a, scores_b) if abs(a - b) > 1e-12)
    return {
        "metric": metric,
        "diff": round(point, 4),
        "lo": round(_percentile(diffs, alpha / 2), 4),
        "hi": round(_percentile(diffs, 1 - alpha / 2), 4),
        "p_value": round(p_value, 4),
        "significant": p_value < alpha,
        "n_differ": n_differ,
        "n": len(labels),
        "n_boot": n_boot,
        "seed": seed,
    }


def detector_uncertainty(detectors, examples, n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """
    AUROC confidence intervals for every detector, plus a paired test of the
    best detector against each of the others.
    """
    labels = [e.label for e in examples]
    scores = {d.name: d.score_all(examples) for d in detectors}
    cis = {name: bootstrap_ci(labels, s, "auroc", n_boot=n_boot, seed=seed) for name, s in scores.items()}
    best = max(cis, key=lambda n: cis[n]["point"])
    vs_best = {
        name: paired_bootstrap(labels, scores[best], s, "auroc", n_boot=n_boot, seed=seed)
        for name, s in scores.items() if name != best
    }
    return {"best": best, "cis": cis, "vs_best": vs_best, "n_boot": n_boot, "seed": seed}
