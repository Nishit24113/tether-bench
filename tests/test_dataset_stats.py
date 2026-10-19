"""
Tests for dataset statistics and validation.
"""

import pytest

from tether import load_sample, validate_dataset, dataset_report, format_dataset_report
from tether.data import Example


def _ok(id, faithful=True):
    typ = {"support_type": "direct"} if faithful else {"error_type": "numeric"}
    return Example(id, "q", "ctx", "ans", faithful=faithful, annotations=typ)


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------

def test_valid_dataset_has_no_issues():
    assert validate_dataset([_ok("a"), _ok("b", faithful=False)]) == []


def test_detects_duplicate_ids():
    issues = validate_dataset([_ok("a"), _ok("a")])
    assert any("duplicate id: a" in i for i in issues)


def test_detects_empty_fields():
    ex = Example("x", "", "ctx", "ans", faithful=True, annotations={"support_type": "direct"})
    issues = validate_dataset([ex])
    assert any("empty question" in i for i in issues)


def test_detects_missing_error_type():
    ex = Example("x", "q", "c", "a", faithful=False, annotations={})
    issues = validate_dataset([ex])
    assert any("missing error_type" in i for i in issues)


def test_detects_missing_support_type():
    ex = Example("x", "q", "c", "a", faithful=True, annotations={})
    issues = validate_dataset([ex])
    assert any("missing support_type" in i for i in issues)


# ---------------------------------------------------------------------------
# dataset_report
# ---------------------------------------------------------------------------

def test_report_counts():
    examples = [_ok("a"), _ok("b"), _ok("c", faithful=False)]
    report = dataset_report(examples)
    assert report["total"] == 3
    assert report["faithful"] == 2
    assert report["hallucinated"] == 1
    assert report["by_support_type"]["direct"] == 2
    assert report["by_error_type"]["numeric"] == 1


def test_report_overlap_separation_present():
    report = dataset_report(load_sample())
    ov = report["lexical_overlap"]
    assert "faithful_mean_overlap" in ov
    assert "hallucinated_mean_overlap" in ov
    # faithful answers should, on average, overlap the context more than
    # hallucinated ones, so separation is positive
    assert ov["separation"] > 0


# ---------------------------------------------------------------------------
# bundled sample integrity
# ---------------------------------------------------------------------------

def test_bundled_sample_is_valid():
    issues = validate_dataset(load_sample())
    assert issues == [], f"sample.jsonl has integrity issues: {issues}"


def test_bundled_sample_covers_all_error_types():
    report = dataset_report(load_sample())
    assert set(report["by_error_type"]) == {
        "numeric", "entity_swap", "unsupported_addition", "contradiction"
    }


def test_format_dataset_report_renders():
    text = format_dataset_report(dataset_report(load_sample()))
    assert "Tether dataset report" in text
    assert "Integrity: OK" in text
    assert "separation" in text
