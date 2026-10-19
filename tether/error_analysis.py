"""
Error analysis for Tether.

The leaderboard says how well a detector does; this says *where* it goes wrong.
For a detector at its operating threshold, it surfaces the specific examples it
misclassifies: the faithful answers it wrongly flags (false positives) and the
hallucinations it lets through (false negatives), each tagged with its annotated
type. On the hard dataset this is where the story lives: which paraphrases fool a
lexical detector, and which vocabulary-reusing hallucinations slip past it.
"""

from typing import List, Dict, Any, Optional

from .data import Example
from .baselines.base import Detector
from .metrics import best_threshold


def _snippet(text: str, max_len: int = 80) -> str:
    text = text.strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def misclassified(
    detector: Detector,
    examples: List[Example],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Find the examples a detector gets wrong at a threshold.

    An answer is predicted hallucinated when its faithfulness score is at or
    below the threshold (F1-optimal if none is given). Returns:
      - false_positives: faithful answers flagged as hallucinated, hardest first
        (lowest score), each with its support_type.
      - false_negatives: hallucinations scored as faithful, hardest first
        (highest score), each with its error_type.
    """
    scores = detector.score_all(examples)
    if threshold is None:
        threshold, _ = best_threshold([e.label for e in examples], scores)

    false_positives: List[Dict[str, Any]] = []
    false_negatives: List[Dict[str, Any]] = []

    for ex, score in zip(examples, scores):
        predicted_hallucinated = score <= threshold
        if ex.faithful and predicted_hallucinated:
            false_positives.append({
                "id": ex.id,
                "score": round(score, 4),
                "support_type": ex.support_type,
                "question": ex.question,
                "answer": _snippet(ex.answer),
            })
        elif not ex.faithful and not predicted_hallucinated:
            false_negatives.append({
                "id": ex.id,
                "score": round(score, 4),
                "error_type": ex.error_type,
                "question": ex.question,
                "answer": _snippet(ex.answer),
            })

    false_positives.sort(key=lambda r: r["score"])          # most confidently wrong first
    false_negatives.sort(key=lambda r: r["score"], reverse=True)

    return {
        "detector": detector.name,
        "threshold": round(threshold, 6),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "n_false_positive": len(false_positives),
        "n_false_negative": len(false_negatives),
    }


def format_errors(result: Dict[str, Any], limit: int = 10) -> str:
    """Render a detector's misclassifications as readable text."""
    lines = [
        f"Error analysis: {result['detector']} "
        f"(threshold {result['threshold']})",
        "",
        f"False negatives (hallucinations missed): {result['n_false_negative']}",
    ]
    for r in result["false_negatives"][:limit]:
        lines.append(f"  [{r['error_type']}] {r['id']} (score {r['score']}): {r['answer']}")

    lines.append("")
    lines.append(f"False positives (faithful answers wrongly flagged): {result['n_false_positive']}")
    for r in result["false_positives"][:limit]:
        lines.append(f"  [{r['support_type']}] {r['id']} (score {r['score']}): {r['answer']}")

    return "\n".join(lines)
