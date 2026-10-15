"""
Tests for the cost/accuracy frontier — cost mapping, Pareto logic, frontier
construction from leaderboard rows, and rendering.
"""

import pytest

from tether.frontier import (
    cost_of, pareto_frontier, build_frontier, format_frontier, frontier_detectors,
)


def _pt(name, cost, quality):
    return {"detector": name, "cost": cost, "quality": quality}


# ---------------------------------------------------------------------------
# cost_of
# ---------------------------------------------------------------------------

def test_cost_of_known_tiers():
    assert cost_of("free") == 0.0
    assert cost_of("model") == 0.3
    assert cost_of("api") == 1.0


def test_cost_of_unknown_tier_is_expensive():
    assert cost_of("mystery") == 1.0


# ---------------------------------------------------------------------------
# pareto_frontier
# ---------------------------------------------------------------------------

def test_pareto_cheaper_and_better_dominates():
    pts = [_pt("a", 0.0, 0.9), _pt("b", 1.0, 0.8)]
    pareto_frontier(pts)
    assert {p["detector"]: p["on_frontier"] for p in pts} == {"a": True, "b": False}


def test_pareto_more_expensive_but_better_is_on_frontier():
    # higher cost but higher quality -> both are non-dominated
    pts = [_pt("cheap", 0.0, 0.7), _pt("pricey", 1.0, 0.95)]
    pareto_frontier(pts)
    assert all(p["on_frontier"] for p in pts)


def test_pareto_dominated_by_equal_cost_higher_quality():
    pts = [_pt("a", 0.0, 0.9), _pt("b", 0.0, 0.7)]
    pareto_frontier(pts)
    assert {p["detector"]: p["on_frontier"] for p in pts} == {"a": True, "b": False}


def test_pareto_exact_tie_keeps_one():
    pts = [_pt("a", 0.0, 0.8), _pt("b", 0.0, 0.8)]
    pareto_frontier(pts)
    on = [p["detector"] for p in pts if p["on_frontier"]]
    assert on == ["a"]  # tie broken by name, only one kept


def test_pareto_three_point_chain():
    # a cheap/low, b mid/mid, c pricey/high -> all non-dominated (a staircase)
    pts = [_pt("a", 0.0, 0.6), _pt("b", 0.3, 0.8), _pt("c", 1.0, 0.95)]
    pareto_frontier(pts)
    assert all(p["on_frontier"] for p in pts)
    # add a dominated point: mid cost, low quality
    pts.append(_pt("d", 0.3, 0.5))
    pareto_frontier(pts)
    assert next(p for p in pts if p["detector"] == "d")["on_frontier"] is False


# ---------------------------------------------------------------------------
# build_frontier
# ---------------------------------------------------------------------------

def _rows():
    return [
        {"detector": "majority", "cost": "free", "auroc": 0.5, "f1": 0.667},
        {"detector": "lexical", "cost": "free", "auroc": 0.89, "f1": 0.83},
        {"detector": "llm_judge", "cost": "api", "auroc": 0.98, "f1": 0.95},
    ]


def test_build_frontier_sorts_by_cost_then_quality():
    pts = build_frontier(_rows(), metric="auroc")
    # free tier (cost 0.0) first, highest quality first within tier
    assert pts[0]["detector"] == "lexical"
    assert pts[1]["detector"] == "majority"
    assert pts[2]["detector"] == "llm_judge"


def test_build_frontier_marks_pareto():
    pts = build_frontier(_rows(), metric="auroc")
    by_name = {p["detector"]: p["on_frontier"] for p in pts}
    assert by_name["lexical"] is True       # best free detector
    assert by_name["majority"] is False     # dominated by lexical (same cost, worse)
    assert by_name["llm_judge"] is True     # pricier but best quality


def test_build_frontier_metric_selection():
    pts = build_frontier(_rows(), metric="f1")
    assert all(p["metric"] == "f1" for p in pts)
    lexical = next(p for p in pts if p["detector"] == "lexical")
    assert lexical["quality"] == 0.83


def test_frontier_detectors_cheapest_first():
    pts = build_frontier(_rows(), metric="auroc")
    assert frontier_detectors(pts) == ["lexical", "llm_judge"]


# ---------------------------------------------------------------------------
# formatting + integration
# ---------------------------------------------------------------------------

def test_format_frontier_renders():
    pts = build_frontier(_rows(), metric="auroc")
    text = format_frontier(pts)
    assert "Cost/accuracy frontier" in text
    assert "llm_judge" in text
    assert "*" in text  # at least one frontier marker


def test_format_frontier_empty():
    assert "No detectors" in format_frontier([])


def test_integration_with_real_leaderboard():
    from tether import load_sample, default_detectors, run_benchmark
    examples = load_sample()
    benchmark = run_benchmark(default_detectors(examples), examples)
    pts = build_frontier(benchmark["leaderboard"], metric="auroc")
    # all default detectors are free; the best-AUROC one must be on the frontier,
    # and majority (0.5) must be dominated.
    assert any(p["on_frontier"] for p in pts)
    majority = next(p for p in pts if p["detector"] == "majority_constant")
    assert majority["on_frontier"] is False
