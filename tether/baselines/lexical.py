"""
Lexical-overlap faithfulness detectors.

The intuition: a grounded answer mostly restates tokens that appear in the
context, while a hallucination introduces tokens the context never mentions.
These detectors are free, instant, and surprisingly hard to beat on easy cases —
which is exactly why they make a good benchmark floor above "majority".
"""

from typing import List

from .base import Detector, tokenize
from ..data import Example

# Extremely common words carry no grounding signal; ignore them so the score
# reflects content-word overlap rather than shared function words.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to", "in",
    "on", "at", "by", "for", "with", "and", "or", "but", "it", "its", "that",
    "this", "as", "from", "into", "about", "there", "their", "has", "have", "had",
}


def _content_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in _STOPWORDS]


class LexicalOverlapDetector(Detector):
    """
    Faithfulness = fraction of the answer's content tokens that also appear in
    the context (token containment).

    An answer whose every content word is present in the context scores 1.0; an
    answer full of context-absent words scores near 0.
    """

    name = "lexical_overlap"
    cost = "free"

    def score(self, example: Example) -> float:
        answer_tokens = _content_tokens(example.answer)
        if not answer_tokens:
            return 1.0  # nothing to contradict the context

        context_tokens = set(_content_tokens(example.context))
        covered = sum(1 for t in answer_tokens if t in context_tokens)
        return covered / len(answer_tokens)


class NovelTokenDetector(Detector):
    """
    Faithfulness = 1 - (fraction of answer content tokens that are *novel*, i.e.
    absent from both the context and the question).

    Allowing question tokens as "grounded" avoids penalizing an answer for
    echoing the question. This variant isolates genuinely unsupported content.
    """

    name = "novel_token"
    cost = "free"

    def score(self, example: Example) -> float:
        answer_tokens = _content_tokens(example.answer)
        if not answer_tokens:
            return 1.0

        allowed = set(_content_tokens(example.context)) | set(_content_tokens(example.question))
        novel = sum(1 for t in answer_tokens if t not in allowed)
        return 1.0 - novel / len(answer_tokens)
