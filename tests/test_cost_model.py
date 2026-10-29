"""
Tests for the cost model — token-based API cost, compute-time model cost,
normalization to frontier scores, and feeding measured costs into the frontier.
"""

import pytest

from tether.cost_model import (
    estimate_api_cost, estimate_model_cost, free_cost, normalized_cost_scores,
    CHARS_PER_TOKEN, JUDGE_SYSTEM_TOKENS,
)
from tether.data import Example


def _ex(id, chars):
    # question+context+answer together contribute `chars` characters
    return Example(id, "q" * (chars // 3), "c" * (chars // 3), "a" * (chars - 2 * (chars // 3)), faithful=True)


# ---------------------------------------------------------------------------
# API cost
# ---------------------------------------------------------------------------

def test_api_cost_scales_with_examples_and_price():
    examples = [_ex("a", 400), _ex("b", 400)]
    cheap = estimate_api_cost(examples, input_price_per_1m=1.0, output_price_per_1m=5.0)
    pricey = estimate_api_cost(examples, input_price_per_1m=10.0, output_price_per_1m=50.0)
    assert pricey.total_usd == pytest.approx(cheap.total_usd * 10)
    assert cheap.n == 2
    assert cheap.per_example_usd == pytest.approx(cheap.total_usd / 2)


def test_api_cost_counts_input_and_output_tokens():
    # one example, 400 prompt chars -> 100 input tokens + system tokens
    ex = [_ex("a", 400)]
    est = estimate_api_cost(ex, input_price_per_1m=1_000_000, output_price_per_1m=0.0, avg_output_tokens=20)
    # input tokens = JUDGE_SYSTEM_TOKENS + 400/CHARS_PER_TOKEN
    expected_input_tokens = JUDGE_SYSTEM_TOKENS + 400 // CHARS_PER_TOKEN
    # price is $1M per 1M tokens => cost == token count in dollars
    assert est.total_usd == pytest.approx(expected_input_tokens, rel=0.05)


def test_api_cost_output_pricing():
    ex = [_ex("a", 4)]  # negligible input
    est = estimate_api_cost(ex, input_price_per_1m=0.0, output_price_per_1m=1_000_000, avg_output_tokens=20)
    assert est.total_usd == pytest.approx(20)


# ---------------------------------------------------------------------------
# Model (compute) cost
# ---------------------------------------------------------------------------

def test_model_cost_from_latency_and_rate():
    # 100 examples * 1s each = 100s; at $3600/hr -> $1/s -> $100
    est = estimate_model_cost(100, seconds_per_example=1.0, hourly_rate_usd=3600.0)
    assert est.total_usd == pytest.approx(100.0)
    assert est.per_example_usd == pytest.approx(1.0)
    assert est.tier == "model"


def test_free_cost_is_zero():
    est = free_cost(50)
    assert est.total_usd == 0.0
    assert est.per_example_usd == 0.0


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_normalized_scores_scale_to_max():
    scores = normalized_cost_scores({"free": 0.0, "model": 0.2, "api": 2.0})
    assert scores["api"] == 1.0
    assert scores["free"] == 0.0
    assert scores["model"] == pytest.approx(0.1)


def test_normalized_scores_all_zero():
    scores = normalized_cost_scores({"free": 0.0, "api": 0.0})
    assert scores == {"free": 0.0, "api": 0.0}


def test_normalized_scores_empty():
    assert normalized_cost_scores({}) == {}


# ---------------------------------------------------------------------------
# Integration with the frontier
# ---------------------------------------------------------------------------

def test_measured_costs_feed_frontier():
    from tether import build_frontier
    rows = [
        {"detector": "lexical", "cost": "free", "auroc": 0.76},
        {"detector": "nli", "cost": "model", "auroc": 0.88},
        {"detector": "llm_judge", "cost": "api", "auroc": 0.95},
    ]
    measured = normalized_cost_scores({"free": 0.0, "model": 0.05, "api": 1.0})
    pts = build_frontier(rows, metric="auroc", cost_scores=measured)
    by = {p["detector"]: p for p in pts}
    # model tier now costs only 0.05 (cheap), reflecting measured compute cost
    assert by["nli"]["cost"] == pytest.approx(0.05)
    assert by["llm_judge"]["cost"] == 1.0
    # all three are non-dominated (a cost/quality staircase)
    assert all(p["on_frontier"] for p in pts)


def test_api_cost_on_real_sample():
    from tether import load_sample
    examples = load_sample()
    est = estimate_api_cost(examples, input_price_per_1m=4.0, output_price_per_1m=20.0)
    assert est.n == len(examples)
    assert est.total_usd > 0
