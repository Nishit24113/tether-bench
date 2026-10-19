"""
Tests for error analysis — correct identification of false positives
(faithful flagged) and false negatives (hallucinations missed), ordering,
and rendering.
"""

from tether.error_analysis import misclassified, format_errors
from tether.data import Example
from tether.baselines.base import Detector


class ScriptedDetector(Detector):
    """Returns a preset score per example id, for deterministic tests."""
    name = "scripted"
    cost = "free"

    def __init__(self, score_map):
        self._scores = score_map

    def score(self, example):
        return self._scores[example.id]


def _ex(id, faithful, typ):
    key = "support_type" if faithful else "error_type"
    return Example(id, "q", "ctx", f"answer {id}", faithful=faithful, annotations={key: typ})


def test_identifies_false_negatives_and_positives():
    examples = [
        _ex("fn", faithful=False, typ="entity_swap"),  # hallucination...
        _ex("tp", faithful=False, typ="numeric"),
        _ex("fp", faithful=True, typ="paraphrase"),     # faithful...
        _ex("tn", faithful=True, typ="direct"),
    ]
    # threshold 0.5: score<=0.5 => predicted hallucinated
    scores = {
        "fn": 0.9,   # hallucination scored faithful -> false negative
        "tp": 0.1,   # hallucination caught -> true positive
        "fp": 0.2,   # faithful flagged -> false positive
        "tn": 0.8,   # faithful kept -> true negative
    }
    result = misclassified(ScriptedDetector(scores), examples, threshold=0.5)

    assert [r["id"] for r in result["false_negatives"]] == ["fn"]
    assert result["false_negatives"][0]["error_type"] == "entity_swap"
    assert [r["id"] for r in result["false_positives"]] == ["fp"]
    assert result["false_positives"][0]["support_type"] == "paraphrase"
    assert result["n_false_negative"] == 1
    assert result["n_false_positive"] == 1


def test_false_negatives_sorted_most_confident_first():
    examples = [
        _ex("a", faithful=False, typ="numeric"),
        _ex("b", faithful=False, typ="numeric"),
    ]
    scores = {"a": 0.7, "b": 0.95}  # both missed at threshold 0.5
    result = misclassified(ScriptedDetector(scores), examples, threshold=0.5)
    # highest score (most confidently "faithful") first
    assert [r["id"] for r in result["false_negatives"]] == ["b", "a"]


def test_false_positives_sorted_most_confident_first():
    examples = [
        _ex("a", faithful=True, typ="direct"),
        _ex("b", faithful=True, typ="paraphrase"),
    ]
    scores = {"a": 0.4, "b": 0.1}  # both flagged at threshold 0.5
    result = misclassified(ScriptedDetector(scores), examples, threshold=0.5)
    # lowest score (most confidently "hallucinated") first
    assert [r["id"] for r in result["false_positives"]] == ["b", "a"]


def test_auto_threshold_used_when_none():
    examples = [_ex("a", faithful=True, typ="direct"), _ex("b", faithful=False, typ="numeric")]
    result = misclassified(ScriptedDetector({"a": 0.9, "b": 0.1}), examples)
    assert "threshold" in result


def test_format_errors_renders():
    examples = [_ex("fp", faithful=True, typ="paraphrase"), _ex("fn", faithful=False, typ="entity_swap")]
    result = misclassified(ScriptedDetector({"fp": 0.1, "fn": 0.9}), examples, threshold=0.5)
    text = format_errors(result)
    assert "Error analysis: scripted" in text
    assert "False negatives" in text and "False positives" in text
    assert "entity_swap" in text and "paraphrase" in text


def test_on_real_sample_lexical_detector():
    from tether import load_sample
    from tether.baselines import LexicalOverlapDetector
    result = misclassified(LexicalOverlapDetector(), load_sample())
    # on the hard sample the lexical detector should miss some vocabulary-reusing
    # hallucinations and/or flag some low-overlap paraphrases
    assert result["n_false_negative"] + result["n_false_positive"] > 0
