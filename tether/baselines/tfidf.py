"""
TF-IDF cosine faithfulness detector.

A step up from raw overlap: weight tokens by how informative they are. Rare,
content-bearing words (names, numbers, domain terms) count more than common
ones, so an answer that shares the context's *distinctive* vocabulary scores
higher than one that only shares filler. Implemented in pure Python (no sklearn)
to keep the benchmark reproducible anywhere.

IDF is fit over the dataset's contexts and answers at construction time, so the
detector is a function of the corpus — fit once, score many.
"""

import math
from collections import Counter
from typing import List, Dict

from .base import Detector, tokenize
from ..data import Example


def _term_freq(tokens: List[str]) -> Dict[str, float]:
    """Normalized term frequencies for one document."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


class TfidfCosineDetector(Detector):
    """Faithfulness = TF-IDF cosine similarity between answer and context."""

    name = "tfidf_cosine"
    cost = "free"

    def __init__(self, examples: List[Example]):
        self.idf = self._fit_idf(examples)

    @staticmethod
    def _fit_idf(examples: List[Example]) -> Dict[str, float]:
        """
        Smoothed inverse document frequency over the corpus. Each context and
        each answer counts as a document, so the IDF reflects the full benchmark
        vocabulary.
        """
        documents: List[set] = []
        for ex in examples:
            documents.append(set(tokenize(ex.context)))
            documents.append(set(tokenize(ex.answer)))

        n_docs = len(documents)
        df: Counter = Counter()
        for doc in documents:
            for term in doc:
                df[term] += 1

        # Smoothed idf: log((1 + N) / (1 + df)) + 1  (never zero, never negative)
        return {
            term: math.log((1 + n_docs) / (1 + df_count)) + 1.0
            for term, df_count in df.items()
        }

    def _vector(self, tokens: List[str]) -> Dict[str, float]:
        """TF-IDF vector for a document, using the fitted IDF."""
        tf = _term_freq(tokens)
        # Unseen terms (not in idf) get the max idf weight — they are maximally
        # rare from the corpus's perspective.
        default_idf = max(self.idf.values()) if self.idf else 1.0
        return {term: freq * self.idf.get(term, default_idf) for term, freq in tf.items()}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        shared = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in shared)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def score(self, example: Example) -> float:
        answer_vec = self._vector(tokenize(example.answer))
        context_vec = self._vector(tokenize(example.context))
        return self._cosine(answer_vec, context_vec)
