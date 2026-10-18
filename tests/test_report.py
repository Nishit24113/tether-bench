"""
Tests for Markdown report generation — table structure, content, and
determinism.
"""

from tether import (
    load_sample, default_detectors, run_benchmark, run_category_breakdown,
    build_frontier, generate_report, leaderboard_markdown, breakdown_markdown,
    frontier_markdown,
)


def _bench():
    examples = load_sample()
    detectors = default_detectors(examples)
    return detectors, examples


# ---------------------------------------------------------------------------
# Individual tables
# ---------------------------------------------------------------------------

def test_leaderboard_markdown_is_a_table():
    detectors, examples = _bench()
    bench = run_benchmark(detectors, examples)
    md = leaderboard_markdown(bench)
    lines = md.splitlines()
    assert lines[0].startswith("| Detector |")
    assert set(lines[1].replace(" ", "")) <= set("|-")  # separator row
    # one row per detector
    assert len(lines) == 2 + len(bench["leaderboard"])
    assert "majority_constant" in md


def test_breakdown_markdown_has_error_type_rows():
    detectors, examples = _bench()
    breakdowns = run_category_breakdown(detectors, examples)
    md = breakdown_markdown(breakdowns)
    assert "| Error type |" in md
    for et in ("numeric", "entity_swap", "unsupported_addition", "contradiction"):
        assert et in md


def test_frontier_markdown_marks_pareto():
    detectors, examples = _bench()
    bench = run_benchmark(detectors, examples)
    pts = build_frontier(bench["leaderboard"], metric="auroc")
    md = frontier_markdown(pts)
    assert "| Detector | Cost tier |" in md
    assert "yes" in md  # at least one Pareto-optimal row


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def test_generate_report_sections_present():
    detectors, examples = _bench()
    report = generate_report(detectors, examples)
    assert "# Tether — Results" in report
    assert "## Leaderboard" in report
    assert "## Detection recall by error type" in report
    assert "## Cost/accuracy frontier" in report


def test_generate_report_is_deterministic():
    detectors, examples = _bench()
    # rebuild detectors each time (TF-IDF fits on the data) — output must match
    r1 = generate_report(default_detectors(examples), examples)
    r2 = generate_report(default_detectors(examples), examples)
    assert r1 == r2


def test_generate_report_reports_dataset_size():
    detectors, examples = _bench()
    report = generate_report(detectors, examples)
    assert f"{len(examples)} examples" in report
