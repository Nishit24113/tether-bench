"""
Checks on the committed NLI cache (out/nli.json) produced by a Sol run.

These tests fold the cached mid-tier scores back into the harness exactly as the
report does, so the committed model-tier numbers stay reproducible. They skip if
the cache is absent (e.g. a checkout that has not run the HPC job).
"""

import os

import pytest

from tether import (
    load_sample, default_detectors, run_benchmark, precomputed_from_cache,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLI_CACHE = os.path.join(HERE, "out", "nli.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(NLI_CACHE), reason="out/nli.json not present (run scripts/score_detector.py nli)"
)


def _covers_all(cache_path, examples):
    from tether.cache import read_cache
    ids = set(read_cache(cache_path)["scores"])
    return all(e.id in ids for e in examples)


def test_cached_ids_score_and_are_in_range():
    examples = load_sample()
    nli = precomputed_from_cache(NLI_CACHE, cost="model")
    assert nli.name == "nli_entailment" and nli.cost == "model"
    covered = [e for e in examples if _covers_all(NLI_CACHE, [e])]
    assert covered, "cache covers no current examples"
    scores = [nli.score(e) for e in covered]      # cached ids resolve without KeyError
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_nli_tops_the_leaderboard():
    examples = load_sample()
    if not _covers_all(NLI_CACHE, examples):
        pytest.skip("NLI cache does not cover all examples yet (re-run score_detector on the new ones)")
    detectors = default_detectors(examples) + [precomputed_from_cache(NLI_CACHE, cost="model")]
    bench = run_benchmark(detectors, examples)
    top = bench["leaderboard"][0]
    assert top["detector"] == "nli_entailment"
    assert top["auroc"] >= top["f1"] >= 0.0
    # the mid tier beats the best free detector on this sample
    free_best = max(r["auroc"] for r in bench["leaderboard"] if r["cost"] == "free")
    assert top["auroc"] > free_best
