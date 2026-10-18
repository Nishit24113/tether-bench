"""
Score caching for Tether.

The free detectors are instant, but the model (NLI) and api (LLM judge) tiers
are slow and/or cost money. This module lets you run an expensive detector once
(for example as a job on an HPC cluster), persist its per-example scores to
JSON, and then reuse those scores for the offline analysis (leaderboard,
breakdown, frontier, agreement) anywhere, with no model and no API calls.

The cache is keyed by example id, so a run can be resumed: only ids missing from
the cache are recomputed.
"""

import json
import os
from typing import List, Dict, Any, Optional

from .data import Example
from .baselines.base import Detector


def write_cache(
    path: str,
    detector_name: str,
    score_map: Dict[str, float],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a detector's per-example-id scores to JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"detector": detector_name, "meta": meta or {}, "scores": score_map},
            fh,
            indent=2,
            sort_keys=True,
        )


def read_cache(path: str) -> Dict[str, Any]:
    """Load a cached score file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_and_cache(
    detector: Detector,
    examples: List[Example],
    cache_path: str,
    recompute: bool = False,
) -> List[float]:
    """
    Score examples, reusing cached scores when present.

    Loads any existing cache for this detector, computes only the example ids
    that are missing (so an interrupted run resumes), writes the merged cache
    back, and returns scores aligned to `examples`. Set `recompute=True` to
    ignore the cache and score everything fresh.
    """
    score_map: Dict[str, float] = {}
    if not recompute and os.path.exists(cache_path):
        data = read_cache(cache_path)
        if data.get("detector") == detector.name:
            score_map = dict(data.get("scores", {}))

    missing = [ex for ex in examples if ex.id not in score_map]
    if missing:
        for ex, score in zip(missing, detector.score_all(missing)):
            score_map[ex.id] = score
        write_cache(
            cache_path,
            detector.name,
            score_map,
            meta={"cost": detector.cost, "cached_ids": len(score_map)},
        )

    return [score_map[ex.id] for ex in examples]


class PrecomputedDetector(Detector):
    """
    A detector that replays cached scores by example id.

    Use this to fold an expensive detector's cached scores (e.g. an LLM judge run
    on a cluster) back into the offline harness, leaderboard, and frontier
    without re-running the model.
    """

    def __init__(self, name: str, score_map: Dict[str, float], cost: str = "api"):
        self.name = name
        self._scores = score_map
        self.cost = cost

    def score(self, example: Example) -> float:
        if example.id not in self._scores:
            raise KeyError(
                f"No cached score for example '{example.id}' in detector '{self.name}'"
            )
        return self._scores[example.id]


def precomputed_from_cache(path: str, cost: Optional[str] = None) -> PrecomputedDetector:
    """Build a PrecomputedDetector from a cache file written by write_cache."""
    data = read_cache(path)
    return PrecomputedDetector(
        name=data["detector"],
        score_map=data["scores"],
        cost=cost or data.get("meta", {}).get("cost", "api"),
    )
