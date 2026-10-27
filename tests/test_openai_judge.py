"""
Tests for the OpenAI-compatible judge adapter.

A fake transport stands in for the HTTP call, so these run offline with no
network, no API key, and no extra dependency.
"""

import pytest

from tether import LLMJudgeDetector, openai_compatible_judge
from tether.data import Example


EX = Example(id="1", question="q", context="c", answer="a", faithful=True)


def _fake_transport(reply_content, captured=None):
    """Return a transport that records the request and replies with fixed content."""
    def transport(url, payload, headers):
        if captured is not None:
            captured["url"] = url
            captured["payload"] = payload
            captured["headers"] = headers
        return {"choices": [{"message": {"content": reply_content}}]}
    return transport


def test_parses_faithful_verdict():
    judge = openai_compatible_judge(
        "some-model", base_url="https://gw.example/v1", api_key="k",
        transport=_fake_transport('{"faithful": true, "confidence": 0.9}'),
    )
    assert judge(EX) == pytest.approx(0.9)


def test_parses_hallucinated_verdict():
    judge = openai_compatible_judge(
        "some-model", base_url="https://gw.example/v1", api_key="k",
        transport=_fake_transport('{"faithful": false, "confidence": 0.8}'),
    )
    assert judge(EX) == pytest.approx(0.2)   # 1 - confidence


def test_builds_request_correctly():
    captured = {}
    judge = openai_compatible_judge(
        "my-model", base_url="https://gw.example/v1/", api_key="secret",
        transport=_fake_transport('{"faithful": true, "confidence": 1.0}', captured),
    )
    judge(EX)
    assert captured["url"] == "https://gw.example/v1/chat/completions"   # trailing slash trimmed
    assert captured["payload"]["model"] == "my-model"
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert captured["payload"]["messages"][1]["role"] == "user"
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_reads_env_when_args_missing(monkeypatch):
    monkeypatch.setenv("TETHER_JUDGE_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("TETHER_JUDGE_API_KEY", "envkey")
    captured = {}
    judge = openai_compatible_judge(
        "m", transport=_fake_transport('{"faithful": true, "confidence": 0.7}', captured),
    )
    assert judge(EX) == pytest.approx(0.7)
    assert captured["url"].startswith("https://env.example/v1")
    assert captured["headers"]["Authorization"] == "Bearer envkey"


def test_missing_base_url_raises(monkeypatch):
    monkeypatch.delenv("TETHER_JUDGE_BASE_URL", raising=False)
    monkeypatch.setenv("TETHER_JUDGE_API_KEY", "k")
    judge = openai_compatible_judge("m", transport=_fake_transport("{}"))
    with pytest.raises(ValueError):
        judge(EX)


def test_missing_key_raises(monkeypatch):
    monkeypatch.setenv("TETHER_JUDGE_BASE_URL", "https://x/v1")
    monkeypatch.delenv("TETHER_JUDGE_API_KEY", raising=False)
    judge = openai_compatible_judge("m", transport=_fake_transport("{}"))
    with pytest.raises(ValueError):
        judge(EX)


def test_detector_scores_all_examples():
    judge = openai_compatible_judge(
        "m", base_url="https://gw/v1", api_key="k",
        transport=_fake_transport('{"faithful": true, "confidence": 0.6}'),
    )
    detector = LLMJudgeDetector(judge)
    scores = detector.score_all([EX, EX, EX])
    assert scores == [pytest.approx(0.6)] * 3
    assert detector.cost == "api"
