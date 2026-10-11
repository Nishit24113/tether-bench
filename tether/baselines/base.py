"""
Detector interface for Tether.

A Detector takes an Example and returns a *faithfulness score* in [0, 1], where
higher means "more supported by the context". The harness turns scores into
hallucination predictions via a threshold, so detectors only need to produce a
well-ordered score — calibration is not required.
"""

import re
from abc import ABC, abstractmethod
from typing import List

from ..data import Example

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenization — deterministic, dependency-free."""
    return _TOKEN_RE.findall(text.lower())


class Detector(ABC):
    """Base class for faithfulness detectors."""

    #: Human-readable name shown on the leaderboard.
    name: str = "detector"

    #: Rough relative cost tier, for the cost/accuracy analysis.
    #: "free" = no model/API, "model" = local model, "api" = paid LLM call.
    cost: str = "free"

    @abstractmethod
    def score(self, example: Example) -> float:
        """Return a faithfulness score in [0, 1] (higher = more faithful)."""
        raise NotImplementedError

    def score_all(self, examples: List[Example]) -> List[float]:
        """Score a list of examples, preserving order."""
        return [self.score(ex) for ex in examples]
