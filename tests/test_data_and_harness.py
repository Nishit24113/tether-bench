"""
Tests for dataset loading, the bundled sample, and the end-to-end harness
(leaderboard construction, ranking, and formatting).
"""

import os
import pytest

from tether import (
    load_sample, class_balance, default_detectors, run_benchmark,
    format_leaderboard, run_detector,
)
from tether.data import Example, load_jsonl, save_jsonl
from tether.baselines import LexicalOverlapDetector


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_sample_loads():
    examples = load_sample()
    assert len(examples) >= 20
    assert all(isinstance(e, Example) for e in examples)


def test_sample_is_balanced():
    bal = class_balance(load_sample())
    # roughly balanced so metrics are meaningful
    assert bal["faithful"] > 0 and bal["hallucinated"] > 0
    assert abs(bal["faithful"] - bal["hallucinated"]) <= bal["total"] * 0.25


def test_sample_ids_unique():
    examples = load_sample()
    ids = [e.id for e in examples]
    assert len(ids) == len(set(ids))


def test_roundtrip_jsonl(tmp_path):
    examples = load_sample()[:3]
    path = os.path.join(tmp_path, "rt.jsonl")
    save_jsonl(examples, path)
    reloaded = load_jsonl(path)
    assert [e.to_dict() for e in examples] == [e.to_dict() for e in reloaded]


def test_malformed_jsonl_raises(tmp_path):
    path = os.path.join(tmp_path, "bad.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"id": "x", "question": "q"}\n')  # missing required fields
    with pytest.raises(ValueError):
        load_jsonl(path)


def test_example_label_mapping():
    assert Example("a", "q", "c", "ans", faithful=True).label == 1
    assert Example("b", "q", "c", "ans", faithful=False).label == 0


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def test_run_detector_returns_metrics():
    examples = load_sample()
    result = run_detector(LexicalOverlapDetector(), examples)
    assert result.name == "lexical_overlap"
    assert "auroc" in result.metrics
    assert len(result.scores) == len(examples)


def test_run_benchmark_ranks_by_auroc():
    examples = load_sample()
    benchmark = run_benchmark(default_detectors(examples), examples)
    aurocs = [row["auroc"] for row in benchmark["leaderboard"]]
    assert aurocs == sorted(aurocs, reverse=True)


def test_benchmark_includes_all_detectors():
    examples = load_sample()
    benchmark = run_benchmark(default_detectors(examples), examples)
    names = {row["detector"] for row in benchmark["leaderboard"]}
    assert names == {
        "majority_constant", "lexical_overlap", "novel_token", "tfidf_cosine"
    }


def test_lexical_beats_majority_on_sample():
    # A content-aware detector must clear the 0.5 AUROC floor the sample is
    # designed to expose.
    examples = load_sample()
    benchmark = run_benchmark(default_detectors(examples), examples)
    results = benchmark["results"]
    assert results["majority_constant"]["auroc"] == 0.5
    assert results["lexical_overlap"]["auroc"] > 0.5


def test_format_leaderboard_renders():
    examples = load_sample()
    benchmark = run_benchmark(default_detectors(examples), examples)
    text = format_leaderboard(benchmark)
    assert "Tether leaderboard" in text
    assert "auroc" in text
    assert "tfidf_cosine" in text
