"""
Tests for the LLM-judge detector.

The judge is exercised with an injected fake judge_fn and a mocked Anthropic
client — no real API calls, no dependency on the `anthropic` package.
"""

import pytest
from unittest.mock import MagicMock

from tether.baselines.llm_judge import (
    LLMJudgeDetector,
    anthropic_judge,
    parse_verdict,
    _build_prompt,
)
from tether.data import Example


def _ex(answer="a", faithful=True):
    return Example("x", "q?", "context text", answer, faithful=faithful)


# ---------------------------------------------------------------------------
# parse_verdict
# ---------------------------------------------------------------------------

def test_parse_faithful_high_confidence():
    assert parse_verdict('{"faithful": true, "confidence": 0.9}') == 0.9


def test_parse_hallucinated_maps_to_low_score():
    # confident hallucination -> low faithfulness score
    assert parse_verdict('{"faithful": false, "confidence": 0.8}') == pytest.approx(0.2)


def test_parse_tolerates_surrounding_text():
    text = 'Here is my verdict:\n{"faithful": false, "confidence": 1.0}\nDone.'
    assert parse_verdict(text) == 0.0


def test_parse_missing_confidence_defaults_to_one():
    assert parse_verdict('{"faithful": true}') == 1.0


def test_parse_clamps_confidence():
    assert parse_verdict('{"faithful": true, "confidence": 1.5}') == 1.0
    assert parse_verdict('{"faithful": true, "confidence": -0.3}') == 0.0


def test_parse_no_json_raises():
    with pytest.raises(ValueError):
        parse_verdict("I think it is faithful.")


def test_parse_missing_faithful_key_raises():
    with pytest.raises(ValueError):
        parse_verdict('{"confidence": 0.9}')


# ---------------------------------------------------------------------------
# LLMJudgeDetector with an injected fake judge
# ---------------------------------------------------------------------------

def test_detector_delegates_to_judge_fn():
    detector = LLMJudgeDetector(judge_fn=lambda ex: 0.3)
    assert detector.score(_ex()) == 0.3
    assert detector.cost == "api"
    assert detector.name == "llm_judge"


def test_detector_custom_name():
    detector = LLMJudgeDetector(judge_fn=lambda ex: 1.0, name="judge_opus")
    assert detector.name == "judge_opus"


def test_detector_score_all_order_preserved():
    detector = LLMJudgeDetector(judge_fn=lambda ex: len(ex.answer) / 10)
    examples = [_ex("ab"), _ex("abcd"), _ex("a")]
    assert detector.score_all(examples) == [0.2, 0.4, 0.1]


# ---------------------------------------------------------------------------
# anthropic_judge with a mocked client
# ---------------------------------------------------------------------------

def _fake_response(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_anthropic_judge_builds_request_and_parses():
    client = MagicMock()
    client.messages.create.return_value = _fake_response(
        '{"faithful": false, "confidence": 0.9}'
    )

    judge = anthropic_judge(model="claude-opus-5-5", client=client)
    score = judge(_ex("the liver makes insulin", faithful=False))

    assert score == pytest.approx(0.1)
    # verify the call used our model and passed the triple in the prompt
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["max_tokens"] >= 1
    user_content = kwargs["messages"][0]["content"]
    assert "CONTEXT:" in user_content and "ANSWER:" in user_content


def test_anthropic_judge_concatenates_text_blocks():
    client = MagicMock()
    # two text blocks that together form the JSON
    b1, b2 = MagicMock(), MagicMock()
    b1.type, b1.text = "text", '{"faithful": true,'
    b2.type, b2.text = "text", ' "confidence": 0.7}'
    resp = MagicMock()
    resp.content = [b1, b2]
    client.messages.create.return_value = resp

    judge = anthropic_judge(client=client)
    assert judge(_ex()) == pytest.approx(0.7)


def test_build_prompt_contains_fields():
    prompt = _build_prompt(_ex("some answer"))
    assert "q?" in prompt
    assert "context text" in prompt
    assert "some answer" in prompt
