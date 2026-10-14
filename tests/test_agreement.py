"""
Tests for agreement analysis — agreement rate, Cohen's kappa, and the
cheap-detector-vs-judge comparison.
"""

import pytest

from tether.agreement import agreement_rate, cohen_kappa, judge_agreement


# ---------------------------------------------------------------------------
# agreement_rate
# ---------------------------------------------------------------------------

def test_agreement_perfect():
    assert agreement_rate([1, 0, 1], [1, 0, 1]) == 1.0


def test_agreement_none():
    assert agreement_rate([1, 1, 1], [0, 0, 0]) == 0.0


def test_agreement_partial():
    assert agreement_rate([1, 0, 1, 0], [1, 0, 0, 0]) == 0.75


def test_agreement_empty():
    assert agreement_rate([], []) == 0.0


def test_agreement_length_mismatch_raises():
    with pytest.raises(ValueError):
        agreement_rate([1, 0], [1])


# ---------------------------------------------------------------------------
# cohen_kappa
# ---------------------------------------------------------------------------

def test_kappa_perfect_agreement():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_kappa_chance_agreement_is_zero():
    # a is constant-1, b is split; expected agreement equals observed -> kappa 0
    a = [1, 1, 1, 1]
    b = [1, 1, 0, 0]
    # po = 0.5; pe: p_a(1)=1,p_b(1)=0.5 -> 0.5 ; p_a(0)=0 -> 0 ; pe=0.5
    # kappa = (0.5-0.5)/(1-0.5) = 0
    assert cohen_kappa(a, b) == 0.0


def test_kappa_both_constant_identical_is_one():
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_kappa_both_constant_opposite_is_zero():
    assert cohen_kappa([1, 1, 1], [0, 0, 0]) == 0.0


def test_kappa_worse_than_chance_negative():
    a = [1, 0, 1, 0]
    b = [0, 1, 0, 1]
    assert cohen_kappa(a, b) < 0


def test_kappa_empty():
    assert cohen_kappa([], []) == 0.0


# ---------------------------------------------------------------------------
# judge_agreement
# ---------------------------------------------------------------------------

def test_judge_agreement_identical_decisions():
    # faithfulness scores; threshold 0.5 -> <=0.5 means hallucinated
    cheap = [0.9, 0.1, 0.8, 0.2]
    judge = [0.9, 0.1, 0.8, 0.2]
    result = judge_agreement(cheap, judge, cheap_threshold=0.5)
    assert result["agreement_rate"] == 1.0
    assert result["cohen_kappa"] == 1.0
    assert result["n"] == 4


def test_judge_agreement_confusion_counts():
    # cheap flags idx0 and idx1 as hallucinated (<=0.5); judge flags only idx1
    cheap = [0.4, 0.3, 0.9]
    judge = [0.8, 0.2, 0.9]
    result = judge_agreement(cheap, judge, cheap_threshold=0.5, judge_threshold=0.5)
    assert result["both_flag_hallucinated"] == 1   # idx1
    assert result["both_flag_faithful"] == 1       # idx2
    assert result["cheap_only_flags"] == 1         # idx0: cheap hallucinated, judge faithful
    assert result["judge_only_flags"] == 0


def test_judge_agreement_length_mismatch_raises():
    with pytest.raises(ValueError):
        judge_agreement([0.5], [0.5, 0.5], cheap_threshold=0.5)
