"""
Agreement analysis for Tether.

Tether's thesis is that a cheap detector is useful insofar as it *agrees with the
LLM judge*. These utilities quantify that agreement on the binary
hallucination-detection decision: raw agreement rate and Cohen's kappa (which
corrects for agreement expected by chance). All stdlib, deterministic.
"""

from typing import List, Dict, Any


def agreement_rate(a: List[int], b: List[int]) -> float:
    """Fraction of positions where two labelers agree."""
    if len(a) != len(b):
        raise ValueError("label lists must be the same length")
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohen_kappa(a: List[int], b: List[int]) -> float:
    """
    Cohen's kappa for two binary labelers (chance-corrected agreement).

    kappa = (po - pe) / (1 - pe), where po is observed agreement and pe is the
    agreement expected if both labelers assigned classes independently at their
    observed base rates. Returns 1.0 when both are perfectly, identically
    constant (pe == 1), and 0.0 when there is no better-than-chance agreement.
    """
    if len(a) != len(b):
        raise ValueError("label lists must be the same length")
    n = len(a)
    if n == 0:
        return 0.0

    po = agreement_rate(a, b)

    pe = 0.0
    for cls in (0, 1):
        p_a = sum(1 for x in a if x == cls) / n
        p_b = sum(1 for y in b if y == cls) / n
        pe += p_a * p_b

    if pe == 1.0:
        # Both labelers are the same single constant class.
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _hallucination_predictions(scores: List[float], threshold: float) -> List[int]:
    """1 = predicted hallucinated (faithfulness score at or below threshold)."""
    return [1 if s <= threshold else 0 for s in scores]


def judge_agreement(
    cheap_scores: List[float],
    judge_scores: List[float],
    cheap_threshold: float,
    judge_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compare a cheap detector's hallucination calls against the judge's.

    Both score lists are on the faithfulness scale (higher = more faithful).
    Each is turned into a binary "hallucinated" prediction at its threshold, then
    compared.

    Returns agreement rate, Cohen's kappa, and the raw confusion counts of
    cheap-vs-judge decisions.
    """
    if len(cheap_scores) != len(judge_scores):
        raise ValueError("score lists must be the same length")

    cheap = _hallucination_predictions(cheap_scores, cheap_threshold)
    judge = _hallucination_predictions(judge_scores, judge_threshold)

    both_hallucinated = sum(1 for c, j in zip(cheap, judge) if c == 1 and j == 1)
    both_faithful = sum(1 for c, j in zip(cheap, judge) if c == 0 and j == 0)
    cheap_only = sum(1 for c, j in zip(cheap, judge) if c == 1 and j == 0)
    judge_only = sum(1 for c, j in zip(cheap, judge) if c == 0 and j == 1)

    return {
        "agreement_rate": round(agreement_rate(cheap, judge), 4),
        "cohen_kappa": round(cohen_kappa(cheap, judge), 4),
        "both_flag_hallucinated": both_hallucinated,
        "both_flag_faithful": both_faithful,
        "cheap_only_flags": cheap_only,   # cheap says hallucinated, judge says faithful
        "judge_only_flags": judge_only,   # judge says hallucinated, cheap misses
        "n": len(cheap_scores),
    }
