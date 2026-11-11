"""
Selective prediction (risk-coverage) for Tether.

A deployed faithfulness detector does not have to answer on every example. If it
can rank its own confidence, it can ANSWER the cases it is sure about and ABSTAIN
on the rest, deferring those to a human or to a more expensive check. The
risk-coverage curve asks: as we cover fewer examples (abstain more), how fast does
the error rate on the ones we *do* answer fall?

Confidence here is the distance of a detector's faithfulness score from its
decision threshold: an answer scored far from the boundary is a confident call,
one scored right at it is a coin flip. We rank examples by that distance, sweep
coverage from full down to one example, and at each level report the selective
risk (error rate on the covered examples).

Two summaries fall out of the curve:
  - AURC, the area under the risk-coverage curve (here the mean selective risk
    across every coverage level). Lower is better. A detector whose score is
    informative about its own correctness concentrates its mistakes among the
    least-confident calls, so its risk stays low until coverage is high and AURC
    is small; an uninformative detector errs regardless of confidence, so its
    risk is roughly flat at the overall error rate and AURC approaches it.
  - selective risk at a target coverage, e.g. "if we answer the most confident
    75% and abstain on the rest, what error rate do we pay?"

All stdlib, operating on already-cached scores. No model calls, so it adds no
cost and stays trivially reproducible, like the rest of the package.

Convention (matches tether.metrics): labels are 1 = faithful, 0 = hallucinated;
faith_scores are in [0, 1] with higher = more faithful; a detector flags an answer
as hallucinated when its score <= threshold.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .metrics import best_threshold


@dataclass
class RiskCoveragePoint:
    coverage: float  # fraction of examples answered (not abstained)
    risk: float      # error rate among the answered examples
    n: int           # number of answered examples


def _wrong_and_confidence(
    labels: List[int],
    faith_scores: List[float],
    threshold: float,
):
    """Per example: whether the thresholded prediction is wrong, and how far the
    score sits from the threshold (the prediction's confidence)."""
    wrong: List[bool] = []
    confidence: List[float] = []
    for label, score in zip(labels, faith_scores):
        predicted_hallucinated = score <= threshold
        actually_hallucinated = label == 0
        wrong.append(predicted_hallucinated != actually_hallucinated)
        confidence.append(abs(score - threshold))
    return wrong, confidence


def risk_coverage_curve(
    labels: List[int],
    faith_scores: List[float],
    threshold: Optional[float] = None,
) -> List[RiskCoveragePoint]:
    """
    Risk-coverage curve for one detector.

    Examples are ranked by confidence (distance of the faithfulness score from the
    threshold), most confident first; ties break by original index so the curve is
    deterministic. At each prefix length we report the error rate on that prefix.

    If threshold is None, the F1-optimal threshold is used (the leaderboard's
    operating point), so the curve reflects the same decision rule the headline
    numbers do.

    Returns one point per coverage level, ordered from smallest coverage (a single
    most-confident example) to full coverage.
    """
    if len(labels) != len(faith_scores):
        raise ValueError("labels and scores must be the same length")
    n = len(labels)
    if n == 0:
        return []
    if threshold is None:
        threshold, _ = best_threshold(labels, faith_scores)

    wrong, confidence = _wrong_and_confidence(labels, faith_scores, threshold)
    order = sorted(range(n), key=lambda i: (-confidence[i], i))  # most confident first

    points: List[RiskCoveragePoint] = []
    errors = 0
    for k, idx in enumerate(order, start=1):
        if wrong[idx]:
            errors += 1
        points.append(RiskCoveragePoint(coverage=k / n, risk=errors / k, n=k))
    return points


def aurc(
    labels: List[int],
    faith_scores: List[float],
    threshold: Optional[float] = None,
) -> float:
    """
    Area under the risk-coverage curve, computed as the mean selective risk over
    all coverage levels (the standard empirical AURC). Lower is better.
    """
    points = risk_coverage_curve(labels, faith_scores, threshold)
    if not points:
        return 0.0
    return sum(p.risk for p in points) / len(points)


def selective_risk_at(
    labels: List[int],
    faith_scores: List[float],
    coverage: float,
    threshold: Optional[float] = None,
) -> float:
    """
    Selective risk (error rate) when answering the most-confident `coverage`
    fraction of examples and abstaining on the rest. coverage is in (0, 1].
    """
    if not (0.0 < coverage <= 1.0):
        raise ValueError("coverage must be in (0, 1]")
    points = risk_coverage_curve(labels, faith_scores, threshold)
    if not points:
        return 0.0
    n = points[-1].n
    k = max(1, min(n, round(coverage * n)))
    return points[k - 1].risk


def coverage_at_risk(
    labels: List[int],
    faith_scores: List[float],
    max_risk: float,
    threshold: Optional[float] = None,
) -> float:
    """
    Maximum coverage achievable while keeping selective risk at or below ``max_risk``.

    This is the operational dual of ``selective_risk_at``: instead of fixing how much
    we answer and reading off the error we pay, we fix the error we can tolerate and
    ask how much we can answer. "If I will auto-accept a faithfulness verdict only when
    the error on accepted cases stays under max_risk, what fraction of cases can I
    auto-accept and send the rest to a human?"

    Because the empirical risk-coverage curve is not monotonic, we take the largest
    coverage level whose selective risk is within budget, not merely the first one that
    breaches it. Returns 0.0 if even the single most-confident prediction exceeds the
    budget. max_risk is in [0, 1].
    """
    if not (0.0 <= max_risk <= 1.0):
        raise ValueError("max_risk must be in [0, 1]")
    points = risk_coverage_curve(labels, faith_scores, threshold)
    best = 0.0
    for p in points:
        if p.risk <= max_risk and p.coverage > best:
            best = p.coverage
    return best


def summary(
    labels: List[int],
    faith_scores: List[float],
    threshold: Optional[float] = None,
    coverages: Optional[List[float]] = None,
    risk_budgets: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compact risk-coverage summary: AURC, the risk at full coverage (the detector's
    plain error rate), the selective risk at a few target coverages, and the dual,
    the coverage reachable under a few target risk budgets.
    """
    if coverages is None:
        coverages = [1.0, 0.9, 0.75, 0.5]
    if risk_budgets is None:
        risk_budgets = [0.01, 0.05, 0.10]
    points = risk_coverage_curve(labels, faith_scores, threshold)
    full_risk = points[-1].risk if points else 0.0
    return {
        "aurc": round(aurc(labels, faith_scores, threshold), 6),
        "full_coverage_risk": round(full_risk, 6),
        "selective_risk": {
            f"{c:.2f}": round(
                selective_risk_at(labels, faith_scores, c, threshold), 6
            )
            for c in coverages
        },
        "coverage_at_risk": {
            f"{r:.2f}": round(
                coverage_at_risk(labels, faith_scores, r, threshold), 6
            )
            for r in risk_budgets
        },
    }


def detector_selective(
    detectors,
    examples,
    coverages: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Risk-coverage summary for every detector, keyed by name.

    Each detector is evaluated at its own F1-optimal threshold (the leaderboard's
    operating point), so the selective-risk numbers are comparable to the headline
    F1/AUROC table.
    """
    labels = [e.label for e in examples]
    return {
        d.name: summary(labels, d.score_all(examples), coverages=coverages)
        for d in detectors
    }
