"""
Tests for score caching — write/read roundtrip, compute-once-then-reuse,
resume of partial caches, and replaying cached scores through a detector.
"""

import os
import pytest

from tether.cache import (
    write_cache, read_cache, run_and_cache, PrecomputedDetector, precomputed_from_cache,
)
from tether.data import Example
from tether.baselines.base import Detector


def _examples(n):
    return [Example(f"e{i}", "q", "c", "a", faithful=(i % 2 == 0)) for i in range(n)]


class CountingDetector(Detector):
    """Counts how many examples it actually scored, to prove caching skips work."""
    name = "counting"
    cost = "api"

    def __init__(self):
        self.scored = 0

    def score(self, example):
        self.scored += 1
        return 0.5


# ---------------------------------------------------------------------------
# write / read
# ---------------------------------------------------------------------------

def test_write_read_roundtrip(tmp_path):
    path = os.path.join(tmp_path, "c.json")
    write_cache(path, "d", {"e0": 0.9, "e1": 0.1}, meta={"cost": "api"})
    data = read_cache(path)
    assert data["detector"] == "d"
    assert data["scores"] == {"e0": 0.9, "e1": 0.1}
    assert data["meta"]["cost"] == "api"


# ---------------------------------------------------------------------------
# run_and_cache
# ---------------------------------------------------------------------------

def test_run_and_cache_computes_then_reuses(tmp_path):
    path = os.path.join(tmp_path, "c.json")
    examples = _examples(4)

    d1 = CountingDetector()
    scores1 = run_and_cache(d1, examples, path)
    assert d1.scored == 4            # first run computes all
    assert scores1 == [0.5] * 4

    d2 = CountingDetector()
    scores2 = run_and_cache(d2, examples, path)
    assert d2.scored == 0            # second run reuses cache, computes nothing
    assert scores2 == [0.5] * 4


def test_run_and_cache_resumes_partial(tmp_path):
    path = os.path.join(tmp_path, "c.json")
    # pre-seed a cache missing one id, under the same detector name
    write_cache(path, "counting", {"e0": 0.5, "e1": 0.5}, meta={})
    examples = _examples(3)  # e0, e1, e2

    d = CountingDetector()
    scores = run_and_cache(d, examples, path)
    assert d.scored == 1             # only e2 was missing
    assert scores == [0.5, 0.5, 0.5]
    assert set(read_cache(path)["scores"]) == {"e0", "e1", "e2"}


def test_run_and_cache_recompute_ignores_cache(tmp_path):
    path = os.path.join(tmp_path, "c.json")
    write_cache(path, "counting", {"e0": 0.5, "e1": 0.5}, meta={})
    examples = _examples(2)

    d = CountingDetector()
    run_and_cache(d, examples, path, recompute=True)
    assert d.scored == 2             # recompute scores everything


def test_run_and_cache_ignores_mismatched_detector(tmp_path):
    path = os.path.join(tmp_path, "c.json")
    write_cache(path, "some_other_detector", {"e0": 0.9}, meta={})
    examples = _examples(2)

    d = CountingDetector()
    run_and_cache(d, examples, path)
    assert d.scored == 2             # cache was for a different detector, recompute


# ---------------------------------------------------------------------------
# PrecomputedDetector
# ---------------------------------------------------------------------------

def test_precomputed_detector_replays_scores():
    det = PrecomputedDetector("judge", {"e0": 0.8, "e1": 0.2}, cost="api")
    examples = _examples(2)
    assert det.score_all(examples) == [0.8, 0.2]
    assert det.cost == "api"


def test_precomputed_detector_missing_id_raises():
    det = PrecomputedDetector("judge", {"e0": 0.8})
    with pytest.raises(KeyError):
        det.score(Example("e9", "q", "c", "a", faithful=True))


def test_precomputed_from_cache_roundtrip(tmp_path):
    path = os.path.join(tmp_path, "judge.json")
    write_cache(path, "llm_judge", {"e0": 0.9, "e1": 0.1}, meta={"cost": "api"})
    det = precomputed_from_cache(path)
    assert det.name == "llm_judge"
    assert det.cost == "api"
    assert det.score(Example("e1", "q", "c", "a", faithful=False)) == 0.1


def test_precomputed_feeds_harness(tmp_path):
    # cached judge scores should flow through the benchmark + frontier offline
    from tether import load_sample, run_benchmark, build_frontier
    examples = load_sample()
    # fake "judge" that is a perfect oracle, cached to disk
    score_map = {e.id: (0.95 if e.faithful else 0.05) for e in examples}
    path = os.path.join(tmp_path, "judge.json")
    write_cache(path, "llm_judge", score_map, meta={"cost": "api"})

    judge = precomputed_from_cache(path)
    bench = run_benchmark([judge], examples)
    assert bench["results"]["llm_judge"]["auroc"] == 1.0
    pts = build_frontier(bench["leaderboard"], metric="auroc")
    assert pts[0]["cost_tier"] == "api"
