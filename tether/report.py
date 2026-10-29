"""
Reproducible Markdown report generation for Tether.

Turns a benchmark run into Markdown tables (leaderboard, per-error-type
breakdown, cost/accuracy frontier) so the numbers in the paper and docs are
generated from a single run rather than copied by hand. Pure stdlib, so the
report is identical on any machine.
"""

from typing import List, Dict, Any

from .data import Example, class_balance
from .baselines.base import Detector
from .harness import run_benchmark, run_category_breakdown
from .frontier import build_frontier
from .stats import detector_uncertainty
from .calibration import detector_calibration, calibration_ci
from .validation import detector_validation
from .recalibration import cross_val_recalibration
from .ensemble import cross_val_stack
from .agreement import judge_agreement
from .metrics import best_threshold


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a GitHub-flavored Markdown table."""
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def leaderboard_markdown(benchmark: Dict[str, Any], cis: Dict[str, Any] = None) -> str:
    """
    Leaderboard as a Markdown table, ranked by AUROC (as the harness sorts it).
    If `cis` (from stats.detector_uncertainty) is given, adds a 95% CI column.
    """
    headers = ["Detector", "Cost", "AUROC"]
    if cis:
        headers.append("AUROC 95% CI")
    headers += ["F1", "Precision", "Recall", "Accuracy"]
    rows = []
    for row in benchmark["leaderboard"]:
        cells = [row["detector"], row["cost"], f"{row['auroc']:.3f}"]
        if cis:
            ci = cis.get(row["detector"])
            cells.append(f"[{ci['lo']:.3f}, {ci['hi']:.3f}]" if ci else "-")
        cells += [f"{row['f1']:.3f}", f"{row['precision']:.3f}",
                  f"{row['recall']:.3f}", f"{row['accuracy']:.3f}"]
        rows.append(cells)
    return _md_table(headers, rows)


def _fmt_p(p: float, n_boot: int) -> str:
    """A bootstrap p of 0 only means no resample crossed zero, so report it as a bound."""
    return f"< {2 / n_boot:.3g}" if p == 0 else f"{p:.3f}"


def significance_markdown(unc: Dict[str, Any]) -> str:
    """Paired bootstrap of the best detector vs each other detector (AUROC difference)."""
    headers = [f"{unc['best']} vs", "AUROC diff", "95% CI", "p-value", "Significant",
               "Examples scored differently"]
    rows = [
        [name, f"{r['diff']:+.3f}", f"[{r['lo']:+.3f}, {r['hi']:+.3f}]",
         _fmt_p(r["p_value"], r["n_boot"]), "yes" if r["significant"] else "no",
         f"{r['n_differ']}/{r['n']}"]
        for name, r in sorted(unc["vs_best"].items(), key=lambda kv: kv[1]["diff"])
    ]
    return _md_table(headers, rows)


def calibration_markdown(cal: Dict[str, Any], labels, scores) -> str:
    """
    Calibration table: ECE (with a bootstrap CI), MCE, and Brier per detector.
    `cal` is detector_calibration(...); `scores` maps detector name -> score list.
    Rows are ordered by ECE (best-calibrated first).
    """
    headers = ["Detector", "ECE", "ECE 95% CI", "MCE", "Brier"]
    rows = []
    for name, c in sorted(cal.items(), key=lambda kv: kv[1]["ece"]):
        ci = calibration_ci(labels, scores[name], "ece", n_bins=c["n_bins"])
        rows.append([name, f"{c['ece']:.3f}", f"[{ci['lo']:.3f}, {ci['hi']:.3f}]",
                     f"{c['mce']:.3f}", f"{c['brier']:.3f}"])
    return _md_table(headers, rows)


def ensemble_markdown(stack: Dict[str, Any]) -> str:
    """One-line comparison of the out-of-fold stacked combiner vs the best single."""
    headers = ["Combination", "AUROC", "95% CI"]
    lo, hi = stack["stack_ci"]
    rows = [
        ["stacked free detectors (logistic, out-of-fold)", f"{stack['stack_auroc']:.3f}", f"[{lo:.3f}, {hi:.3f}]"],
        [f"best single ({stack['best_single']})", f"{stack['best_single_auroc']:.3f}", "-"],
    ]
    return _md_table(headers, rows)


def recalibration_markdown(detectors: List[Detector], examples: List[Example]) -> str:
    """
    Raw vs post-calibration ECE/Brier per detector, cross-validated (k=5, seed 0).
    Shows whether a fitted calibration map turns the raw scores into probabilities.
    """
    labels = [e.label for e in examples]
    headers = ["Detector", "Raw ECE", "Isotonic ECE", "Platt ECE", "Raw Brier", "Isotonic Brier"]
    rows = []
    for d in detectors:
        s = d.score_all(examples)
        iso = cross_val_recalibration(labels, s, "isotonic")
        pla = cross_val_recalibration(labels, s, "platt")
        rows.append([
            d.name, f"{iso['raw_ece']:.3f}", f"{iso['cal_ece']:.3f}", f"{pla['cal_ece']:.3f}",
            f"{iso['raw_brier']:.3f}", f"{iso['cal_brier']:.3f}",
        ])
    return _md_table(headers, rows)


def validation_markdown(val: Dict[str, Any]) -> str:
    """
    Oracle vs cross-validated F1 per detector, ordered by cross-validated F1.
    `val` is detector_validation(...).
    """
    headers = ["Detector", "Oracle F1", "CV F1", "Optimism", "CV precision", "CV recall"]
    rows = [
        [name, f"{v['oracle_f1']:.3f}", f"{v['cv_f1']:.3f}", f"{v['optimism_gap']:+.3f}",
         f"{v['cv_precision']:.3f}", f"{v['cv_recall']:.3f}"]
        for name, v in sorted(val.items(), key=lambda kv: kv[1]["cv_f1"], reverse=True)
    ]
    return _md_table(headers, rows)


def agreement_markdown(detectors: List[Detector], examples: List[Example], judge: Detector) -> str:
    """
    Each non-judge detector's agreement with the judge on the binary
    hallucination call: raw agreement, Cohen's kappa, and where they diverge.
    Each cheap detector is thresholded at its own F1-optimal point; the judge at
    0.5. Rows ordered by kappa (closest-to-judge first).
    """
    labels = [e.label for e in examples]
    judge_scores = judge.score_all(examples)
    headers = ["Detector vs judge", "Agreement", "Cohen kappa",
               "Judge-only catches", "Cheap-only flags"]
    scored = []
    for d in detectors:
        if d.name == judge.name:
            continue
        s = d.score_all(examples)
        thr, _ = best_threshold(labels, s)
        ag = judge_agreement(s, judge_scores, cheap_threshold=thr)
        scored.append((ag["cohen_kappa"], [
            d.name, f"{ag['agreement_rate']:.3f}", f"{ag['cohen_kappa']:.3f}",
            str(ag["judge_only_flags"]), str(ag["cheap_only_flags"]),
        ]))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    return _md_table(headers, [row for _, row in scored])


def breakdown_markdown(breakdowns: Dict[str, Any]) -> str:
    """
    Per-error-type detection recall as a Markdown table (rows = error types,
    columns = detectors).
    """
    detectors = list(breakdowns.keys())
    error_types = sorted(
        {et for b in breakdowns.values() for et in b["by_error_type"]}
    )
    headers = ["Error type", *detectors]
    rows = []
    for et in error_types:
        cells = [et]
        for d in detectors:
            bucket = breakdowns[d]["by_error_type"].get(et)
            cells.append(f"{bucket['recall']:.2f}" if bucket else "-")
        rows.append(cells)
    return _md_table(headers, rows)


def frontier_markdown(points: List[Dict[str, Any]]) -> str:
    """Cost/accuracy frontier as a Markdown table; Pareto-optimal rows marked."""
    headers = ["Detector", "Cost tier", "Cost", "Quality", "Pareto"]
    rows = [
        [
            p["detector"],
            p["cost_tier"],
            f"{p['cost']:.2f}",
            f"{p['quality']:.3f}",
            "yes" if p["on_frontier"] else "",
        ]
        for p in points
    ]
    return _md_table(headers, rows)


def generate_report(
    detectors: List[Detector],
    examples: List[Example],
    metric: str = "auroc",
) -> str:
    """
    Run the benchmark and render a full Markdown results report.

    Deterministic for a given detector set and dataset, so it can be regenerated
    into RESULTS.md and cited by the paper.
    """
    benchmark = run_benchmark(detectors, examples)
    breakdowns = run_category_breakdown(detectors, examples)
    frontier = build_frontier(benchmark["leaderboard"], metric=metric)
    unc = detector_uncertainty(detectors, examples)
    cal = detector_calibration(detectors, examples)
    val = detector_validation(detectors, examples)
    labels = [e.label for e in examples]
    cal_scores = {d.name: d.score_all(examples) for d in detectors}
    bal = class_balance(examples)

    parts = [
        "# Tether — Results",
        "",
        f"Dataset: {bal['total']} examples "
        f"({bal['faithful']} faithful, {bal['hallucinated']} hallucinated).",
        "",
        "## Leaderboard",
        "",
        "Ranked by AUROC (threshold-free). Precision/recall/F1 are reported at "
        "each detector's F1-optimal threshold. AUROC 95% CIs come from a "
        f"stratified bootstrap ({unc['n_boot']} resamples, seed {unc['seed']}).",
        "",
        leaderboard_markdown(benchmark, unc["cis"]),
        "",
        "## Are the gaps real?",
        "",
        f"Paired stratified bootstrap of the best detector ({unc['best']}) against "
        "each other detector, scored on identical resamples. Two-sided p-values; "
        "significant means p < 0.05. The last column counts the examples the two "
        "detectors score differently, which is all the evidence a paired "
        "comparison has.",
        "",
        significance_markdown(unc),
        "",
    ]

    # Do the free signals combine to beat the best single free detector?
    free_dets = [d for d in detectors if getattr(d, "cost", None) == "free"]
    if len(free_dets) >= 2:
        stack = cross_val_stack(free_dets, examples)
        sig = "a significant" if stack["significant"] else "not a significant"
        dlo, dhi = stack["diff_ci"]
        parts += [
            "## Do the free signals combine to beat the best single detector?",
            "",
            "A logistic combiner is stacked on the free detectors' scores and "
            "evaluated out-of-fold (5-fold, seed 0), so its AUROC is not measured "
            "on the data the combiner was trained on.",
            "",
            ensemble_markdown(stack),
            "",
            f"The stacked combiner's AUROC differs from the best single detector "
            f"({stack['best_single']}) by {stack['diff']:+.3f} "
            f"(95% CI [{dlo:+.3f}, {dhi:+.3f}], p = {stack['p_value']:.3f}), which is "
            f"{sig} improvement. The free signals are largely redundant, so "
            "combining them does not clear the free-tier ceiling.",
            "",
        ]

    parts += [
        "## Are the scores trustworthy probabilities?",
        "",
        "AUROC only asks whether a detector *ranks* faithful answers above "
        "hallucinated ones. Calibration asks whether a faithfulness score of p "
        "means the answer is faithful about p of the time. ECE is the "
        "count-weighted mean gap between predicted score and observed faithful "
        "rate across 10 bins (lower is better); MCE is the worst single bin; "
        "Brier is the mean squared error of the scores as probabilities. The ECE "
        "CI is a stratified bootstrap (2000 resamples, seed 0).",
        "",
        calibration_markdown(cal, labels, cal_scores),
        "",
        "## Can the cheap scores be made into probabilities?",
        "",
        "The miscalibration above is a property of the raw scores, not a verdict "
        "on the detectors. This fits a post-hoc calibration map (isotonic, a "
        "monotone step fit, and Platt, a logistic fit) from score to observed "
        "faithful rate. The map is fit on training folds and applied to a "
        "held-out fold (5-fold, seed 0), so the post-calibration ECE/Brier is not "
        "measured on the data the map was tuned on. Lower is better.",
        "",
        recalibration_markdown(detectors, examples),
        "",
        "## How optimistic is the F1?",
        "",
        "The leaderboard F1 uses each detector's F1-optimal threshold, tuned on "
        "the same data it is scored on, so it is an oracle operating point. The "
        f"table below re-estimates F1 with {val['novel_token']['k']}-fold "
        "stratified cross-validation (seed "
        f"{val['novel_token']['seed']}): the threshold is chosen on the training "
        "folds and applied to the held-out fold, so every example is predicted "
        "by a threshold it did not help pick. The optimism column is how much the "
        "oracle F1 overstates the cross-validated one.",
        "",
        validation_markdown(val),
        "",
        "## Detection recall by error type",
        "",
        "Recall of hallucination detection, per annotated error type, at each "
        "detector's F1-optimal threshold.",
        "",
        breakdown_markdown(breakdowns),
        "",
        "## Cost/accuracy frontier",
        "",
        f"Quality axis = {metric}. Cost is a relative per-tier score "
        "(free 0.0, model 0.3, api 1.0).",
        "",
        frontier_markdown(frontier),
        "",
    ]

    # The founding question: how closely do the cheap detectors track the judge?
    # Only shown when a judge (api tier) is in the run.
    judge = next((d for d in detectors if getattr(d, "cost", None) == "api"), None)
    if judge is not None:
        parts += [
            "## How well do the cheap detectors agree with the judge?",
            "",
            f"Tether's thesis is that a cheap detector is useful insofar as it "
            f"agrees with the LLM judge ({judge.name}). Each cheap detector's "
            "hallucination calls (at its F1-optimal threshold) are compared with "
            "the judge's (at 0.5). Cohen's kappa corrects for chance agreement; "
            "`judge-only catches` are hallucinations the judge flagged and the "
            "cheap detector missed, and `cheap-only flags` are answers the cheap "
            "detector flagged that the judge called faithful.",
            "",
            agreement_markdown(detectors, examples, judge),
            "",
        ]

    return "\n".join(parts)
