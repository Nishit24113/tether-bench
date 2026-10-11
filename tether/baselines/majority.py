"""
Majority / constant baseline.

The floor every real detector must beat. It ignores the input entirely and
emits a constant faithfulness score, so its AUROC is 0.5 by construction — if a
detector can't clear this, it has learned nothing about grounding.
"""

from .base import Detector
from ..data import Example


class MajorityDetector(Detector):
    """Emits a constant faithfulness score for every example."""

    name = "majority_constant"
    cost = "free"

    def __init__(self, constant: float = 0.5):
        self.constant = constant

    def score(self, example: Example) -> float:
        return self.constant
