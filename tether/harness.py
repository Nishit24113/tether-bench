"""
Benchmark harness for Tether.

Runs a set of detectors over a dataset, computes the metric bundle for each, and
assembles a leaderboard ranked by AUROC (threshold-free, so it reflects a
detector's intrinsic ability to separate hallucinated from faithful answers
regardless of operating point).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

from .data import Example, class_balance
from .baselines.base import Detector
from .metrics import evaluate


@dataclass
class DetectorResult:
    name: str
    cost: str
    metrics: Dict[str, Any]
    scores: List[float] = field(default_factory=list)

    def to_row(self) -> Dict[str, Any]:
        return {
            "detector": self.name,
            "cost": self.cost,
            "auroc": self.metrics["auroc"],
            "f1": self.metrics["f1"],
            "precision": self.metrics["precision"],
            "recall": self.metrics["recall"],
            "accuracy": self.metrics["accuracy"],
        }


def run_detector(detector: Detector, examples: List[Example]) -> DetectorResult:
    """Score one detector over the dataset and compute its metrics."""
    scores = detector.score_all(examples)
    labels = [ex.label for ex in examples]
    metrics = evaluate(labels, scores)
    return DetectorResult(
        name=detector.name,
        cost=detector.cost,
        metrics=metrics,
        scores=scores,
    )


def run_benchmark(
    detectors: List[Detector],
    examples: List[Example],
) -> Dict[str, Any]:
    """
    Run every detector and return a leaderboard ranked by AUROC (descending),
    with the F1-optimal threshold's precision/recall/F1 as secondary columns.
    """
    results = [run_detector(d, examples) for d in detectors]
    results.sort(key=lambda r: r.metrics["auroc"], reverse=True)

    return {
        "dataset": class_balance(examples),
        "leaderboard": [r.to_row() for r in results],
        "results": {r.name: r.metrics for r in results},
    }


def format_leaderboard(benchmark: Dict[str, Any]) -> str:
    """Render the leaderboard as a fixed-width text table."""
    bal = benchmark["dataset"]
    header = (
        f"Tether leaderboard  "
        f"(n={bal['total']}, faithful={bal['faithful']}, "
        f"hallucinated={bal['hallucinated']})"
    )

    rows = benchmark["leaderboard"]
    col_names = ["detector", "cost", "auroc", "f1", "precision", "recall", "accuracy"]
    widths = {c: len(c) for c in col_names}
    for row in rows:
        for c in col_names:
            widths[c] = max(widths[c], len(str(row[c])))

    def fmt_row(values: Dict[str, Any]) -> str:
        return "  ".join(str(values[c]).ljust(widths[c]) for c in col_names)

    lines = [header, "", fmt_row({c: c for c in col_names}),
             "  ".join("-" * widths[c] for c in col_names)]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)
