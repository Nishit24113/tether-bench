"""Baseline faithfulness detectors for Tether."""

from .base import Detector, tokenize
from .majority import MajorityDetector
from .lexical import LexicalOverlapDetector, NovelTokenDetector
from .tfidf import TfidfCosineDetector
from .llm_judge import LLMJudgeDetector, anthropic_judge, parse_verdict
from .nli import NLIEntailmentDetector, transformers_nli

__all__ = [
    "Detector",
    "tokenize",
    "MajorityDetector",
    "LexicalOverlapDetector",
    "NovelTokenDetector",
    "TfidfCosineDetector",
    "LLMJudgeDetector",
    "anthropic_judge",
    "parse_verdict",
    "NLIEntailmentDetector",
    "transformers_nli",
]
