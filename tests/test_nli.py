"""
Tests for the NLI-entailment detector.

Exercised with an injected fake entail_fn — no transformers/torch dependency,
no model download. Also covers the entailment-index lookup and frontier
integration (the NLI detector populating the mid-cost tier).
"""

import pytest

from tether.baselines.nli import NLIEntailmentDetector, _entailment_index
from tether.data import Example


def _ex(context="the pancreas makes insulin", answer="the pancreas makes insulin"):
    return Example("x", "q?", context, answer, faithful=True)


# ---------------------------------------------------------------------------
# _entailment_index
# ---------------------------------------------------------------------------

def test_entailment_index_standard_mnli_order():
    # bart-large-mnli: {0: contradiction, 1: neutral, 2: entailment}
    assert _entailment_index({0: "contradiction", 1: "neutral", 2: "entailment"}) == 2


def test_entailment_index_different_order():
    assert _entailment_index({0: "entailment", 1: "neutral", 2: "contradiction"}) == 0


def test_entailment_index_case_insensitive():
    assert _entailment_index({0: "CONTRADICTION", 1: "ENTAILMENT"}) == 1


def test_entailment_index_missing_raises():
    with pytest.raises(ValueError):
        _entailment_index({0: "contradiction", 1: "neutral"})


# ---------------------------------------------------------------------------
# NLIEntailmentDetector with an injected fake entail_fn
# ---------------------------------------------------------------------------

def test_detector_passes_context_as_premise_answer_as_hypothesis():
    seen = {}

    def fake_entail(premise, hypothesis):
        seen["premise"] = premise
        seen["hypothesis"] = hypothesis
        return 0.42

    detector = NLIEntailmentDetector(entail_fn=fake_entail)
    score = detector.score(_ex(context="CTX", answer="ANS"))

    assert score == 0.42
    assert seen["premise"] == "CTX"      # context is the premise/evidence
    assert seen["hypothesis"] == "ANS"   # answer is the hypothesis/claim


def test_detector_cost_tier_is_model():
    detector = NLIEntailmentDetector(entail_fn=lambda p, h: 1.0)
    assert detector.cost == "model"
    assert detector.name == "nli_entailment"


def test_detector_custom_name():
    detector = NLIEntailmentDetector(entail_fn=lambda p, h: 1.0, name="nli_bart")
    assert detector.name == "nli_bart"


def test_detector_score_all_order_preserved():
    detector = NLIEntailmentDetector(entail_fn=lambda p, h: len(h) / 10)
    examples = [_ex(answer="ab"), _ex(answer="abcd")]
    assert detector.score_all(examples) == [0.2, 0.4]


# ---------------------------------------------------------------------------
# Integration: NLI populates the mid-cost tier on the frontier
# ---------------------------------------------------------------------------

def test_nli_lands_on_model_cost_tier_in_frontier():
    from tether import load_sample, run_benchmark, build_frontier
    from tether.baselines import LexicalOverlapDetector, MajorityDetector

    examples = load_sample()

    # Fake NLI oracle: entailment prob tracks the gold label, so it scores well
    def oracle_entail(premise, hypothesis):
        match = next((e for e in examples if e.context == premise and e.answer == hypothesis), None)
        return 0.95 if (match and match.faithful) else 0.05

    nli = NLIEntailmentDetector(entail_fn=oracle_entail)
    detectors = [MajorityDetector(), LexicalOverlapDetector(), nli]

    bench = run_benchmark(detectors, examples)
    pts = build_frontier(bench["leaderboard"], metric="auroc")

    nli_pt = next(p for p in pts if p["detector"] == "nli_entailment")
    assert nli_pt["cost_tier"] == "model"
    assert nli_pt["cost"] == 0.3
    # a near-perfect oracle must be on the frontier
    assert nli_pt["on_frontier"] is True
