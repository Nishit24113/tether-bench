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
from .metrics import evaluate, best_threshold


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


def category_breakdown(detector: Detector, examples: List[Example]) -> Dict[str, Any]:
    """
    Break a detector's performance down by annotated category.

    Uses the detector's F1-optimal threshold (its best achievable operating
    point), then reports:
      - per error_type: detection *recall* among hallucinated examples
        (what fraction of each error kind the detector actually catches)
      - per support_type: *false-positive rate* among faithful examples
        (how often it wrongly flags a grounded answer of that kind)

    This surfaces where cheap detectors break down — e.g. numeric errors, where
    a single wrong digit barely changes lexical overlap.
    """
    scores = detector.score_all(examples)
    labels = [ex.label for ex in examples]
    threshold, _ = best_threshold(labels, scores)

    by_error: Dict[str, Dict[str, Any]] = {}
    by_support: Dict[str, Dict[str, Any]] = {}

    for ex, score in zip(examples, scores):
        flagged_hallucinated = score <= threshold
        if ex.faithful:
            bucket = by_support.setdefault(
                ex.support_type, {"total": 0, "false_positives": 0}
            )
            bucket["total"] += 1
            if flagged_hallucinated:
                bucket["false_positives"] += 1
        else:
            bucket = by_error.setdefault(ex.error_type, {"total": 0, "caught": 0})
            bucket["total"] += 1
            if flagged_hallucinated:
                bucket["caught"] += 1

    for b in by_error.values():
        b["recall"] = round(b["caught"] / b["total"], 4) if b["total"] else 0.0
    for b in by_support.values():
        b["fpr"] = round(b["false_positives"] / b["total"], 4) if b["total"] else 0.0

    return {
        "detector": detector.name,
        "threshold": round(threshold, 6),
        "by_error_type": by_error,
        "by_support_type": by_support,
    }


def run_category_breakdown(
    detectors: List[Detector], examples: List[Example]
) -> Dict[str, Any]:
    """Category breakdown for several detectors, keyed by detector name."""
    return {d.name: category_breakdown(d, examples) for d in detectors}


def format_category_breakdown(breakdowns: Dict[str, Any]) -> str:
    """
    Render a per-error-type recall table (rows = error types, cols = detectors).
    Lower recall = that error type slips past that detector more often.
    """
    detectors = list(breakdowns.keys())
    error_types = sorted(
        {et for b in breakdowns.values() for et in b["by_error_type"]}
    )

    first_col = max([len("error_type")] + [len(e) for e in error_types])
    col_w = {d: max(len(d), 6) for d in detectors}

    def row(name: str, cells: List[str]) -> str:
        return name.ljust(first_col) + "  " + "  ".join(
            c.ljust(col_w[d]) for c, d in zip(cells, detectors)
        )

    lines = ["Detection recall by error type (hallucinated examples)", ""]
    lines.append(row("error_type", detectors))
    lines.append("-" * first_col + "  " + "  ".join("-" * col_w[d] for d in detectors))
    for et in error_types:
        cells = []
        for d in detectors:
            bucket = breakdowns[d]["by_error_type"].get(et)
            cells.append(f"{bucket['recall']:.2f}" if bucket else "-")
        lines.append(row(et, cells))
    return "\n".join(lines)


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
