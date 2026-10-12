"""
Tests for difficulty / error-type stratified evaluation.

Verify error_type/support_type parsing on examples, per-category recall and
false-positive computation, dataset category coverage, and the breakdown table.
"""

import pytest

from tether import (
    load_sample,
    default_detectors,
    category_breakdown,
    run_category_breakdown,
    format_category_breakdown,
    LexicalOverlapDetector,
    MajorityDetector,
)
from tether.data import Example


# ---------------------------------------------------------------------------
# Example annotations
# ---------------------------------------------------------------------------

def test_hallucinated_has_error_type_not_support():
    ex = Example("x", "q", "c", "a", faithful=False, annotations={"error_type": "numeric"})
    assert ex.error_type == "numeric"
    assert ex.support_type is None


def test_faithful_has_support_type_not_error():
    ex = Example("x", "q", "c", "a", faithful=True, annotations={"support_type": "direct"})
    assert ex.support_type == "direct"
    assert ex.error_type is None


def test_missing_annotation_defaults_to_unspecified():
    assert Example("x", "q", "c", "a", faithful=False).error_type == "unspecified"
    assert Example("x", "q", "c", "a", faithful=True).support_type == "unspecified"


# ---------------------------------------------------------------------------
# Dataset coverage
# ---------------------------------------------------------------------------

def test_sample_covers_expected_error_types():
    examples = load_sample()
    error_types = {e.error_type for e in examples if not e.faithful}
    assert {"numeric", "entity_swap", "unsupported_addition", "contradiction"} <= error_types


def test_sample_every_hallucination_has_known_error_type():
    for e in load_sample():
        if not e.faithful:
            assert e.error_type != "unspecified", f"{e.id} missing error_type"


def test_sample_every_faithful_has_support_type():
    for e in load_sample():
        if e.faithful:
            assert e.support_type != "unspecified", f"{e.id} missing support_type"


# ---------------------------------------------------------------------------
# category_breakdown
# ---------------------------------------------------------------------------

def _ex(id, faithful, score_word, **ann):
    # context fixed; answer controls lexical overlap for deterministic scoring
    return Example(
        id=id,
        question="q",
        context="alpha beta gamma delta epsilon",
        answer=score_word,
        faithful=faithful,
        annotations=ann,
    )


def test_breakdown_recall_and_fpr_structure():
    examples = load_sample()
    result = category_breakdown(LexicalOverlapDetector(), examples)
    assert "threshold" in result
    assert "by_error_type" in result and "by_support_type" in result
    for bucket in result["by_error_type"].values():
        assert bucket["total"] > 0
        assert 0.0 <= bucket["recall"] <= 1.0
        assert bucket["caught"] <= bucket["total"]
    for bucket in result["by_support_type"].values():
        assert 0.0 <= bucket["fpr"] <= 1.0


def test_breakdown_counts_sum_to_class_totals():
    examples = load_sample()
    result = category_breakdown(LexicalOverlapDetector(), examples)
    hallu = sum(b["total"] for b in result["by_error_type"].values())
    faithful = sum(b["total"] for b in result["by_support_type"].values())
    assert hallu == sum(1 for e in examples if not e.faithful)
    assert faithful == sum(1 for e in examples if e.faithful)


def test_majority_detector_catches_all_or_nothing_per_category():
    # Majority emits a constant score; at its F1-optimal threshold it flags
    # everything as hallucinated, so recall is 1.0 for every error type.
    examples = load_sample()
    result = category_breakdown(MajorityDetector(), examples)
    for bucket in result["by_error_type"].values():
        assert bucket["recall"] == 1.0


def test_run_category_breakdown_covers_all_detectors():
    examples = load_sample()
    detectors = default_detectors(examples)
    breakdowns = run_category_breakdown(detectors, examples)
    assert set(breakdowns.keys()) == {d.name for d in detectors}


def test_format_category_breakdown_renders_rows():
    examples = load_sample()
    breakdowns = run_category_breakdown(default_detectors(examples), examples)
    text = format_category_breakdown(breakdowns)
    assert "Detection recall by error type" in text
    assert "numeric" in text
    assert "lexical_overlap" in text
