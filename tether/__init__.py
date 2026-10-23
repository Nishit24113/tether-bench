"""
Tether — a reproducible benchmark for RAG answer faithfulness.

Given a (question, context, answer) triple, detectors score how faithful the
answer is to the context. Tether evaluates how well low-cost detectors separate
grounded answers from hallucinations, and (in later versions) how closely they
approximate an LLM judge.
"""

from .data import Example, load_jsonl, save_jsonl, load_sample, class_balance
from .metrics import evaluate, auroc, confusion_matrix, best_threshold, ConfusionMatrix
from .harness import (
    run_benchmark,
    run_detector,
    format_leaderboard,
    DetectorResult,
    category_breakdown,
    run_category_breakdown,
    format_category_breakdown,
)
from .baselines import (
    Detector,
    MajorityDetector,
    LexicalOverlapDetector,
    NovelTokenDetector,
    TfidfCosineDetector,
    LLMJudgeDetector,
    anthropic_judge,
    parse_verdict,
    NLIEntailmentDetector,
    transformers_nli,
)
from .agreement import agreement_rate, cohen_kappa, judge_agreement
from .frontier import (
    build_frontier,
    pareto_frontier,
    format_frontier,
    frontier_detectors,
    cost_of,
    COST_SCORE,
)
from .report import (
    generate_report,
    leaderboard_markdown,
    breakdown_markdown,
    frontier_markdown,
)
from .cache import (
    write_cache,
    read_cache,
    run_and_cache,
    PrecomputedDetector,
    precomputed_from_cache,
)
from .dataset_stats import (
    validate_dataset,
    dataset_report,
    format_dataset_report,
)
from .error_analysis import misclassified, format_errors
from .stats import bootstrap_ci, paired_bootstrap, detector_uncertainty
from .cost_model import (
    CostEstimate,
    estimate_api_cost,
    estimate_model_cost,
    free_cost,
    normalized_cost_scores,
)

__version__ = "0.1.0"

__all__ = [
    "Example", "load_jsonl", "save_jsonl", "load_sample", "class_balance",
    "evaluate", "auroc", "confusion_matrix", "best_threshold", "ConfusionMatrix",
    "run_benchmark", "run_detector", "format_leaderboard", "DetectorResult",
    "category_breakdown", "run_category_breakdown", "format_category_breakdown",
    "Detector", "MajorityDetector", "LexicalOverlapDetector",
    "NovelTokenDetector", "TfidfCosineDetector",
    "LLMJudgeDetector", "anthropic_judge", "parse_verdict",
    "NLIEntailmentDetector", "transformers_nli",
    "agreement_rate", "cohen_kappa", "judge_agreement",
    "build_frontier", "pareto_frontier", "format_frontier",
    "frontier_detectors", "cost_of", "COST_SCORE",
    "generate_report", "leaderboard_markdown", "breakdown_markdown",
    "frontier_markdown",
    "write_cache", "read_cache", "run_and_cache",
    "PrecomputedDetector", "precomputed_from_cache",
    "validate_dataset", "dataset_report", "format_dataset_report",
    "misclassified", "format_errors",
    "bootstrap_ci", "paired_bootstrap", "detector_uncertainty",
    "CostEstimate", "estimate_api_cost", "estimate_model_cost",
    "free_cost", "normalized_cost_scores",
]


def default_detectors(examples):
    """The standard baseline suite (TF-IDF fits its IDF on the given examples)."""
    return [
        MajorityDetector(),
        LexicalOverlapDetector(),
        NovelTokenDetector(),
        TfidfCosineDetector(examples),
    ]
