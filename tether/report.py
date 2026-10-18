"""
Reproducible Markdown report generation for Tether.

Turns a benchmark run into Markdown tables (leaderboard, per-error-type
breakdown, cost/accuracy frontier) so the numbers in the paper and docs are
generated from a single run rather than copied by hand. Pure stdlib, so the
report is identical on any machine.
"""

from typing import List, Dict, Any

from .data import Example, class_balance
from .baselines.base import Detector
from .harness import run_benchmark, run_category_breakdown
from .frontier import build_frontier


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a GitHub-flavored Markdown table."""
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def leaderboard_markdown(benchmark: Dict[str, Any]) -> str:
    """Leaderboard as a Markdown table, ranked by AUROC (as the harness sorts it)."""
    headers = ["Detector", "Cost", "AUROC", "F1", "Precision", "Recall", "Accuracy"]
    rows = [
        [
            row["detector"],
            row["cost"],
            f"{row['auroc']:.3f}",
            f"{row['f1']:.3f}",
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['accuracy']:.3f}",
        ]
        for row in benchmark["leaderboard"]
    ]
    return _md_table(headers, rows)


def breakdown_markdown(breakdowns: Dict[str, Any]) -> str:
    """
    Per-error-type detection recall as a Markdown table (rows = error types,
    columns = detectors).
    """
    detectors = list(breakdowns.keys())
    error_types = sorted(
        {et for b in breakdowns.values() for et in b["by_error_type"]}
    )
    headers = ["Error type", *detectors]
    rows = []
    for et in error_types:
        cells = [et]
        for d in detectors:
            bucket = breakdowns[d]["by_error_type"].get(et)
            cells.append(f"{bucket['recall']:.2f}" if bucket else "-")
        rows.append(cells)
    return _md_table(headers, rows)


def frontier_markdown(points: List[Dict[str, Any]]) -> str:
    """Cost/accuracy frontier as a Markdown table; Pareto-optimal rows marked."""
    headers = ["Detector", "Cost tier", "Cost", "Quality", "Pareto"]
    rows = [
        [
            p["detector"],
            p["cost_tier"],
            f"{p['cost']:.2f}",
            f"{p['quality']:.3f}",
            "yes" if p["on_frontier"] else "",
        ]
        for p in points
    ]
    return _md_table(headers, rows)


def generate_report(
    detectors: List[Detector],
    examples: List[Example],
    metric: str = "auroc",
) -> str:
    """
    Run the benchmark and render a full Markdown results report.

    Deterministic for a given detector set and dataset, so it can be regenerated
    into RESULTS.md and cited by the paper.
    """
    benchmark = run_benchmark(detectors, examples)
    breakdowns = run_category_breakdown(detectors, examples)
    frontier = build_frontier(benchmark["leaderboard"], metric=metric)
    bal = class_balance(examples)

    parts = [
        "# Tether — Results",
        "",
        f"Dataset: {bal['total']} examples "
        f"({bal['faithful']} faithful, {bal['hallucinated']} hallucinated).",
        "",
        "## Leaderboard",
        "",
        "Ranked by AUROC (threshold-free). Precision/recall/F1 are reported at "
        "each detector's F1-optimal threshold.",
        "",
        leaderboard_markdown(benchmark),
        "",
        "## Detection recall by error type",
        "",
        "Recall of hallucination detection, per annotated error type, at each "
        "detector's F1-optimal threshold.",
        "",
        breakdown_markdown(breakdowns),
        "",
        "## Cost/accuracy frontier",
        "",
        f"Quality axis = {metric}. Cost is a relative per-tier score "
        "(free 0.0, model 0.3, api 1.0).",
        "",
        frontier_markdown(frontier),
        "",
    ]
    return "\n".join(parts)
