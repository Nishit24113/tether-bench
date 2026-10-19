"""
Dataset statistics and validation for Tether.

As the dataset grows, these utilities keep it honest: they validate structural
integrity (unique ids, required annotations, non-empty fields) and summarize
composition (class balance, per-error-type and per-support-type counts). They
also report the lexical-overlap separation between faithful and hallucinated
answers, which is a quick sanity check on how hard the dataset is for the
free detectors: the smaller the gap, the harder the benchmark.
"""

from typing import List, Dict, Any
from collections import Counter

from .data import Example
from .baselines.lexical import LexicalOverlapDetector


def validate_dataset(examples: List[Example]) -> List[str]:
    """
    Return a list of integrity problems (empty list means the dataset is valid).

    Checks: duplicate ids, empty question/context/answer, hallucinated examples
    missing an error_type, and faithful examples missing a support_type.
    """
    issues: List[str] = []

    seen = set()
    for ex in examples:
        if ex.id in seen:
            issues.append(f"duplicate id: {ex.id}")
        seen.add(ex.id)

        if not ex.question.strip():
            issues.append(f"{ex.id}: empty question")
        if not ex.context.strip():
            issues.append(f"{ex.id}: empty context")
        if not ex.answer.strip():
            issues.append(f"{ex.id}: empty answer")

        if ex.faithful and ex.support_type == "unspecified":
            issues.append(f"{ex.id}: faithful example missing support_type")
        if not ex.faithful and ex.error_type == "unspecified":
            issues.append(f"{ex.id}: hallucinated example missing error_type")

    return issues


def _overlap_separation(examples: List[Example]) -> Dict[str, float]:
    """
    Mean lexical overlap of faithful vs hallucinated answers, and the gap.

    A large gap means token overlap alone separates the classes well (an easy
    dataset for lexical detectors); a small gap means the dataset forces
    detectors to go beyond surface overlap.
    """
    det = LexicalOverlapDetector()
    faithful = [det.score(e) for e in examples if e.faithful]
    hallucinated = [det.score(e) for e in examples if not e.faithful]

    mean_f = sum(faithful) / len(faithful) if faithful else 0.0
    mean_h = sum(hallucinated) / len(hallucinated) if hallucinated else 0.0
    return {
        "faithful_mean_overlap": round(mean_f, 4),
        "hallucinated_mean_overlap": round(mean_h, 4),
        "separation": round(mean_f - mean_h, 4),
    }


def dataset_report(examples: List[Example]) -> Dict[str, Any]:
    """Full composition + integrity + difficulty summary for a dataset."""
    faithful = [e for e in examples if e.faithful]
    hallucinated = [e for e in examples if not e.faithful]

    return {
        "total": len(examples),
        "faithful": len(faithful),
        "hallucinated": len(hallucinated),
        "by_error_type": dict(Counter(e.error_type for e in hallucinated)),
        "by_support_type": dict(Counter(e.support_type for e in faithful)),
        "lexical_overlap": _overlap_separation(examples),
        "issues": validate_dataset(examples),
    }


def format_dataset_report(report: Dict[str, Any]) -> str:
    """Render the dataset report as plain text."""
    lines = [
        "Tether dataset report",
        "",
        f"Total: {report['total']}  "
        f"(faithful {report['faithful']}, hallucinated {report['hallucinated']})",
        "",
        "Hallucinations by error type:",
    ]
    for et, n in sorted(report["by_error_type"].items()):
        lines.append(f"  {et}: {n}")
    lines.append("")
    lines.append("Faithful by support type:")
    for st, n in sorted(report["by_support_type"].items()):
        lines.append(f"  {st}: {n}")

    ov = report["lexical_overlap"]
    lines += [
        "",
        "Lexical-overlap separation (faithful vs hallucinated):",
        f"  faithful mean:     {ov['faithful_mean_overlap']}",
        f"  hallucinated mean: {ov['hallucinated_mean_overlap']}",
        f"  separation:        {ov['separation']}",
    ]

    issues = report["issues"]
    lines.append("")
    lines.append(f"Integrity: {'OK' if not issues else str(len(issues)) + ' issue(s)'}")
    for issue in issues:
        lines.append(f"  - {issue}")
    return "\n".join(lines)
