"""
Cost model for Tether.

The cost/accuracy frontier ranks detectors by a relative per-tier score by
default. This module estimates the *actual* cost of running each tier so the
frontier can be priced in real units:

- api (LLM judge): token-based. Estimate input tokens from the prompt text
  (system + question + context + answer) plus a small output, and multiply by
  the model's per-token price.
- model (NLI): compute-time based. Seconds per example times an hourly
  accelerator rate.
- free: zero.

Estimates are deterministic and dependency-free. They are approximations (token
counts are char-based) meant for relative comparison on the frontier; plug in
measured numbers when you have them.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from .data import Example

# Rough chars-per-token for English text. Good enough for a relative estimate.
CHARS_PER_TOKEN = 4

# A compact judge system prompt is ~120 tokens; counted once per call.
JUDGE_SYSTEM_TOKENS = 120


@dataclass
class CostEstimate:
    tier: str
    n: int
    total_usd: float
    per_example_usd: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "tier": self.tier,
            "n": self.n,
            "total_usd": round(self.total_usd, 6),
            "per_example_usd": round(self.per_example_usd, 8),
        }


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_api_cost(
    examples: List[Example],
    input_price_per_1m: float,
    output_price_per_1m: float,
    avg_output_tokens: int = 20,
) -> CostEstimate:
    """
    Estimate the dollar cost of scoring every example with an LLM judge.

    input_price_per_1m / output_price_per_1m are the model's USD price per
    1,000,000 tokens. Input tokens are estimated from the judge prompt
    (system + question + context + answer); output is a short JSON verdict.
    """
    total_input_tokens = 0
    for ex in examples:
        prompt_chars = len(ex.question) + len(ex.context) + len(ex.answer)
        total_input_tokens += JUDGE_SYSTEM_TOKENS + _estimate_tokens(
            "x" * prompt_chars
        )
    total_output_tokens = avg_output_tokens * len(examples)

    total_usd = (
        total_input_tokens / 1_000_000 * input_price_per_1m
        + total_output_tokens / 1_000_000 * output_price_per_1m
    )
    n = len(examples)
    return CostEstimate("api", n, total_usd, total_usd / n if n else 0.0)


def estimate_model_cost(
    n_examples: int,
    seconds_per_example: float,
    hourly_rate_usd: float,
) -> CostEstimate:
    """
    Estimate the compute cost of scoring with a local model (e.g. NLI on a GPU).

    hourly_rate_usd is the accelerator's cost per hour; seconds_per_example is
    measured inference latency.
    """
    total_seconds = n_examples * seconds_per_example
    total_usd = total_seconds / 3600 * hourly_rate_usd
    return CostEstimate(
        "model", n_examples, total_usd, total_usd / n_examples if n_examples else 0.0
    )


def free_cost(n_examples: int) -> CostEstimate:
    """Free tier: zero dollars."""
    return CostEstimate("free", n_examples, 0.0, 0.0)


def normalized_cost_scores(tier_to_usd: Dict[str, float]) -> Dict[str, float]:
    """
    Turn absolute per-tier dollar costs into relative [0, 1] scores for the
    frontier: the most expensive tier maps to 1.0, free (zero) to 0.0, and the
    rest scale linearly in between.
    """
    if not tier_to_usd:
        return {}
    max_usd = max(tier_to_usd.values())
    if max_usd <= 0:
        return {tier: 0.0 for tier in tier_to_usd}
    return {tier: round(usd / max_usd, 4) for tier, usd in tier_to_usd.items()}
