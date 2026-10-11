"""
Evaluation metrics for Tether — all stdlib, no numpy/sklearn, so results are
trivially reproducible on any Python install.

Convention: the *positive* class is "hallucinated" (faithful == 0). Faithfulness
detection is really hallucination *detection* — the thing we want to catch is the
unfaithful answer — so precision/recall are reported for that class.

A detector emits a faithfulness score in [0, 1] (higher = more faithful). To turn
scores into hallucination predictions we flag an answer as hallucinated when its
faithfulness score falls at or below a threshold.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple


@dataclass
class ConfusionMatrix:
    tp: int   # predicted hallucinated, actually hallucinated
    fp: int   # predicted hallucinated, actually faithful
    tn: int   # predicted faithful, actually faithful
    fn: int   # predicted faithful, actually hallucinated

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else 0.0


def confusion_matrix(
    labels: List[int],
    faith_scores: List[float],
    threshold: float,
) -> ConfusionMatrix:
    """
    Build a confusion matrix for hallucination detection.

    Args:
        labels: gold faithfulness labels (1 = faithful, 0 = hallucinated)
        faith_scores: detector faithfulness scores (higher = more faithful)
        threshold: flag as hallucinated when score <= threshold

    """
    if len(labels) != len(faith_scores):
        raise ValueError("labels and scores must be the same length")

    tp = fp = tn = fn = 0
    for label, score in zip(labels, faith_scores):
        predicted_hallucinated = score <= threshold
        actually_hallucinated = label == 0

        if predicted_hallucinated and actually_hallucinated:
            tp += 1
        elif predicted_hallucinated and not actually_hallucinated:
            fp += 1
        elif not predicted_hallucinated and not actually_hallucinated:
            tn += 1
        else:
            fn += 1

    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


def auroc(labels: List[int], faith_scores: List[float]) -> float:
    """
    Area under the ROC curve for separating hallucinated from faithful answers,
    computed via the Mann-Whitney U statistic (rank-based, handles ties).

    The positive class is hallucinated (label == 0), so we rank by the
    *hallucination* score = -faith_score (higher hallucination score should rank
    hallucinated examples first).

    Returns 0.5 when one class is absent (no discriminative signal possible).
    """
    if len(labels) != len(faith_scores):
        raise ValueError("labels and scores must be the same length")

    pos_scores = [-s for s, l in zip(faith_scores, labels) if l == 0]  # hallucinated
    neg_scores = [-s for s, l in zip(faith_scores, labels) if l == 1]  # faithful

    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Rank all scores (average ranks for ties)
    combined = [(score, "pos") for score in pos_scores] + [(score, "neg") for score in neg_scores]
    combined.sort(key=lambda x: x[0])

    ranks = _average_ranks([c[0] for c in combined])

    rank_sum_pos = sum(rank for rank, (_, cls) in zip(ranks, combined) if cls == "pos")

    u_pos = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u_pos / (n_pos * n_neg)


def _average_ranks(sorted_values: List[float]) -> List[float]:
    """Assign 1-based ranks to an ascending-sorted list, averaging ties."""
    ranks = [0.0] * len(sorted_values)
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        # positions i..j (0-based) share the average of ranks (i+1)..(j+1)
        avg = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    return ranks


def best_threshold(
    labels: List[int],
    faith_scores: List[float],
) -> Tuple[float, float]:
    """
    Pick the threshold that maximizes F1 for hallucination detection.

    Candidate thresholds are the midpoints between consecutive unique scores,
    plus the extremes, so every possible split is considered.

    Returns:
        (threshold, f1_at_that_threshold)
    """
    uniq = sorted(set(faith_scores))
    if not uniq:
        return 0.0, 0.0

    candidates = [uniq[0] - 1e-9]
    for a, b in zip(uniq, uniq[1:]):
        candidates.append((a + b) / 2)
    candidates.append(uniq[-1] + 1e-9)

    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        f1 = confusion_matrix(labels, faith_scores, t).f1
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate(
    labels: List[int],
    faith_scores: List[float],
    threshold: float = None,
) -> Dict[str, Any]:
    """
    Full metric bundle for a detector's scores.

    If threshold is None, the F1-optimal threshold is chosen automatically — this
    reflects a detector's best achievable operating point (an oracle threshold;
    real deployments would tune on a held-out split).
    """
    if threshold is None:
        threshold, _ = best_threshold(labels, faith_scores)

    cm = confusion_matrix(labels, faith_scores, threshold)
    return {
        "threshold": round(threshold, 6),
        "precision": round(cm.precision, 4),
        "recall": round(cm.recall, 4),
        "f1": round(cm.f1, 4),
        "accuracy": round(cm.accuracy, 4),
        "auroc": round(auroc(labels, faith_scores), 4),
        "confusion": asdict(cm),
    }
