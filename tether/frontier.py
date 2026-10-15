"""
Cost/accuracy frontier for Tether.

A detector is only interesting if nothing else is both cheaper and at least as
accurate. This module places each detector in (cost, quality) space, computes
the Pareto frontier, and renders it — the benchmark's headline artifact for the
"how much accuracy does a dollar of judge buy?" question.

Cost is expressed as a relative score per tier (free compute vs a local model vs
a paid API call), not an absolute dollar figure — absolute cost depends on
volume, provider, and token counts, so the frontier ranks tiers rather than
pricing them. Override COST_SCORE to plug in measured costs.
"""

from typing import List, Dict, Any

# Representative relative cost per tier (0 = free, 1 = paid API call).
# These are ordinal placeholders — override with measured $/1k-examples if known.
COST_SCORE = {
    "free": 0.0,
    "model": 0.3,   # a local model (NLI, embeddings): compute + memory, no per-call fee
    "api": 1.0,     # a paid LLM-judge call
}


def cost_of(cost_tier: str) -> float:
    """Relative cost score for a tier; unknown tiers are treated as most expensive."""
    return COST_SCORE.get(cost_tier, 1.0)


def pareto_frontier(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Mark each point as on/off the Pareto frontier (maximize quality, minimize cost).

    A point is dominated if some other point is both <= cost and >= quality, with
    at least one strict (and ties broken so identical-cost points keep only the
    best quality, identical-quality points keep only the cheapest). Returns the
    same dicts with an added `on_frontier` bool.
    """
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            no_worse = q["cost"] <= p["cost"] and q["quality"] >= p["quality"]
            strictly_better = q["cost"] < p["cost"] or q["quality"] > p["quality"]
            if no_worse and strictly_better:
                dominated = True
                break
            # Exact tie on both: keep the one that sorts first by name, drop the other
            if q["cost"] == p["cost"] and q["quality"] == p["quality"] and q["detector"] < p["detector"]:
                dominated = True
                break
        p["on_frontier"] = not dominated
    return points


def build_frontier(
    leaderboard_rows: List[Dict[str, Any]],
    metric: str = "auroc",
) -> List[Dict[str, Any]]:
    """
    Build frontier points from leaderboard rows.

    Each row must have `detector`, `cost` (tier string), and the chosen `metric`.
    Returns points sorted by cost then descending quality, each with
    `on_frontier` set.
    """
    points = [
        {
            "detector": row["detector"],
            "cost_tier": row["cost"],
            "cost": cost_of(row["cost"]),
            "quality": float(row[metric]),
            "metric": metric,
        }
        for row in leaderboard_rows
    ]
    pareto_frontier(points)
    points.sort(key=lambda p: (p["cost"], -p["quality"]))
    return points


def format_frontier(points: List[Dict[str, Any]]) -> str:
    """
    Render the frontier as a cost-sorted table; frontier points marked with '*'.
    """
    if not points:
        return "No detectors to plot."

    metric = points[0]["metric"]
    name_w = max([len("detector")] + [len(p["detector"]) for p in points])
    q_w = max(len(metric), 5)

    lines = [
        f"Cost/accuracy frontier (quality = {metric}; * = Pareto-optimal)",
        "",
        f"{'detector'.ljust(name_w)}  {'cost_tier'.ljust(9)}  cost  {metric.ljust(q_w)}  frontier",
        f"{'-' * name_w}  {'-' * 9}  ----  {'-' * q_w}  --------",
    ]
    for p in points:
        star = "*" if p["on_frontier"] else ""
        quality_str = f"{p['quality']:.3f}".ljust(q_w)
        lines.append(
            f"{p['detector'].ljust(name_w)}  "
            f"{p['cost_tier'].ljust(9)}  "
            f"{p['cost']:.2f}  "
            f"{quality_str}  "
            f"{star}"
        )
    return "\n".join(lines)


def frontier_detectors(points: List[Dict[str, Any]]) -> List[str]:
    """Names of the Pareto-optimal detectors, cheapest first."""
    return [p["detector"] for p in points if p["on_frontier"]]
