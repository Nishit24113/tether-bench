"""
Tests for baseline detectors — scoring behavior, edge cases, and the
score ordering that makes them discriminative on clear examples.
"""

import pytest

from tether.data import Example
from tether.baselines import (
    MajorityDetector, LexicalOverlapDetector, NovelTokenDetector, TfidfCosineDetector,
)
from tether.baselines.base import tokenize


def _ex(context, answer, question="q?", faithful=True, id="t"):
    return Example(id=id, question=question, context=context, answer=answer, faithful=faithful)


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_tokenize_empty():
    assert tokenize("") == []


# ---------------------------------------------------------------------------
# Majority
# ---------------------------------------------------------------------------

def test_majority_constant():
    d = MajorityDetector(constant=0.5)
    assert d.score(_ex("anything", "whatever")) == 0.5
    assert d.score(_ex("different", "stuff")) == 0.5


# ---------------------------------------------------------------------------
# Lexical overlap
# ---------------------------------------------------------------------------

def test_lexical_full_overlap_scores_high():
    d = LexicalOverlapDetector()
    ex = _ex("Paris is the capital of France.", "Paris is the capital of France.")
    assert d.score(ex) == 1.0


def test_lexical_no_overlap_scores_low():
    d = LexicalOverlapDetector()
    ex = _ex("Paris is the capital of France.", "Quantum chromodynamics describes gluons.")
    assert d.score(ex) < 0.2


def test_lexical_empty_answer_scores_one():
    d = LexicalOverlapDetector()
    assert d.score(_ex("some context", "")) == 1.0


def test_lexical_discriminates_faithful_vs_hallucinated():
    d = LexicalOverlapDetector()
    context = "The Eiffel Tower was completed in 1889 and stands 330 metres tall."
    faithful = d.score(_ex(context, "The tower was completed in 1889."))
    hallucinated = d.score(_ex(context, "The tower contains a nuclear reactor."))
    assert faithful > hallucinated


# ---------------------------------------------------------------------------
# Novel token
# ---------------------------------------------------------------------------

def test_novel_token_allows_question_words():
    d = NovelTokenDetector()
    # "capital" appears only in the question, not the context -> not penalized
    ex = _ex(
        context="France is in Europe. Paris is a major city.",
        answer="Paris is the capital.",
        question="What is the capital of France?",
    )
    score_with_q = d.score(ex)
    # Same answer but question doesn't contain "capital" -> should score lower
    ex2 = _ex(
        context="France is in Europe. Paris is a major city.",
        answer="Paris is the capital.",
        question="Where is Paris?",
    )
    assert score_with_q > d.score(ex2)


def test_novel_token_empty_answer():
    d = NovelTokenDetector()
    assert d.score(_ex("ctx", "")) == 1.0


# ---------------------------------------------------------------------------
# TF-IDF cosine
# ---------------------------------------------------------------------------

def test_tfidf_identical_text_high_similarity():
    examples = [_ex("alpha beta gamma delta", "alpha beta gamma delta")]
    d = TfidfCosineDetector(examples)
    assert d.score(examples[0]) > 0.9


def test_tfidf_disjoint_text_zero():
    corpus = [
        _ex("alpha beta gamma", "alpha beta gamma"),
        _ex("delta epsilon zeta", "delta epsilon zeta"),
    ]
    d = TfidfCosineDetector(corpus)
    ex = _ex("alpha beta gamma", "delta epsilon zeta")
    assert d.score(ex) == 0.0


def test_tfidf_discriminates_on_rare_terms():
    corpus = [
        _ex("The patient was prescribed amoxicillin for infection.",
            "The patient received amoxicillin."),
        _ex("The patient was prescribed amoxicillin for infection.",
            "The patient received ibuprofen."),
    ]
    d = TfidfCosineDetector(corpus)
    faithful = d.score(corpus[0])
    hallucinated = d.score(corpus[1])
    assert faithful > hallucinated
