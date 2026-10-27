"""
LLM-judge faithfulness detector for Tether.

This is the *reference* detector the cheap baselines are measured against: a
strong model reads the (question, context, answer) triple and judges whether the
answer is faithful. It is the "expensive" end of the cost/accuracy frontier.

The detector is deliberately decoupled from any provider: it takes a `judge_fn`
callable that maps an Example to a faithfulness score in [0, 1]. Tests inject a
deterministic fake judge, so the suite runs offline with no API calls and no
hard dependency on the `anthropic` package. `anthropic_judge()` builds a real
judge backed by the Anthropic API, imported lazily so it is only required when
actually used (`pip install tether-bench[llm]`).
"""

import json
import os
import re
from typing import Callable, Optional, Any

from .base import Detector
from ..data import Example


JudgeFn = Callable[[Example], float]

JUDGE_SYSTEM = (
    "You are a strict faithfulness judge for retrieval-augmented generation. "
    "Given a QUESTION, a CONTEXT, and an ANSWER, decide whether every claim in "
    "the ANSWER is supported by the CONTEXT. If the answer adds, changes, or "
    "contradicts any fact that is not in the CONTEXT, it is NOT faithful, even if "
    "the claim is true in general. Judge only against the CONTEXT.\n"
    'Respond with ONLY a JSON object: {"faithful": <true|false>, '
    '"confidence": <number between 0 and 1>}. '
    "confidence is how certain you are of the verdict."
)


def _build_prompt(example: Example) -> str:
    return (
        f"QUESTION:\n{example.question}\n\n"
        f"CONTEXT:\n{example.context}\n\n"
        f"ANSWER:\n{example.answer}\n\n"
        "Return only the JSON object."
    )


def parse_verdict(text: str) -> float:
    """
    Turn a judge's JSON verdict into a faithfulness score in [0, 1].

    A `faithful` verdict maps to its confidence (high = confidently faithful);
    a `hallucinated` verdict maps to 1 - confidence (low = confidently
    unfaithful). This keeps the output on the same "higher = more faithful"
    scale as every other detector, so the harness can threshold and rank it.

    Robust to reasoning-model output: any <think>...</think> preamble is stripped
    first, and when several JSON objects are present the LAST one carrying a
    `faithful` key wins (the final answer follows the reasoning).
    """
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    candidates = re.findall(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if not candidates:
        raise ValueError(f"No JSON object found in judge response: {text!r}")

    verdict = None
    for chunk in candidates:
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "faithful" in obj:
            verdict = obj  # keep scanning; the last valid verdict wins

    if verdict is None:
        raise ValueError(f"Judge response missing 'faithful': {text!r}")

    faithful = bool(verdict["faithful"])
    confidence = float(verdict.get("confidence", 1.0))
    confidence = max(0.0, min(1.0, confidence))

    return confidence if faithful else 1.0 - confidence


def anthropic_judge(
    model: str = "claude-opus-5-5",
    client: Optional[Any] = None,
    effort: str = "low",
    max_tokens: int = 1024,
) -> JudgeFn:
    """
    Build a judge_fn backed by the Anthropic API.

    Args:
        model: Claude model id for the judge (the reference is a strong model).
        client: an anthropic.Anthropic instance; constructed lazily if None.
        effort: reasoning effort for the judge call ("low" keeps the simple
            classification cheap and fast).
        max_tokens: output cap; small, since the judge returns a tiny JSON object.

    Requires `pip install tether-bench[llm]` and credentials in the environment
    (e.g. ANTHROPIC_API_KEY). The import is lazy so the rest of Tether needs
    neither.
    """
    _client = {"instance": client}

    def judge(example: Example) -> float:
        if _client["instance"] is None:
            import anthropic  # lazy: only needed when the judge actually runs
            _client["instance"] = anthropic.Anthropic()

        response = _client["instance"].messages.create(
            model=model,
            max_tokens=max_tokens,
            system=JUDGE_SYSTEM,
            output_config={"effort": effort},
            messages=[{"role": "user", "content": _build_prompt(example)}],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return parse_verdict(text)

    return judge


def openai_compatible_judge(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 60.0,
    extra_body: Optional[dict] = None,
    transport: Optional[Callable[[str, dict, dict], dict]] = None,
) -> JudgeFn:
    """
    Build a judge_fn backed by any OpenAI-compatible chat-completions endpoint.

    This talks to a gateway that implements the OpenAI `/chat/completions` schema
    (many self-hosted and institutional LLM gateways do, including ones that
    serve open-source models). Nothing provider-specific is hardcoded: point it
    at a gateway with `base_url` or the `TETHER_JUDGE_BASE_URL` environment
    variable, and authenticate with `api_key` or `TETHER_JUDGE_API_KEY`.

    Args:
        model: model id to request, exactly as the gateway expects it.
        base_url: OpenAI-compatible base (the part before `/chat/completions`,
            e.g. `.../v1`). Falls back to `TETHER_JUDGE_BASE_URL`.
        api_key: bearer token. Falls back to `TETHER_JUDGE_API_KEY`.
        temperature: sampling temperature; 0 keeps the verdict deterministic.
        max_tokens: output cap. Default 1024 leaves room for a reasoning model's
            think step before the small JSON verdict.
        timeout: per-request timeout in seconds.
        extra_body: optional dict merged into the request payload, for
            gateway-specific knobs (e.g. disabling a reasoning model's thinking).
        transport: optional `(url, payload, headers) -> response_dict` hook. The
            default uses the stdlib (`urllib`), so there is no extra dependency;
            tests inject a fake transport to run offline.

    The key and endpoint are read lazily on each call, so the environment can be
    set after the judge is built.
    """

    def _default_transport(url: str, payload: dict, headers: dict) -> dict:
        import urllib.request  # stdlib: no extra dependency

        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    send = transport or _default_transport

    def judge(example: Example) -> float:
        resolved_base = base_url or os.environ.get("TETHER_JUDGE_BASE_URL")
        resolved_key = api_key or os.environ.get("TETHER_JUDGE_API_KEY")
        if not resolved_base:
            raise ValueError(
                "openai_compatible_judge needs base_url or the TETHER_JUDGE_BASE_URL env var"
            )
        if not resolved_key:
            raise ValueError(
                "openai_compatible_judge needs api_key or the TETHER_JUDGE_API_KEY env var"
            )

        url = resolved_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": _build_prompt(example)},
            ],
        }
        if extra_body:
            payload.update(extra_body)
        headers = {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }
        data = send(url, payload, headers)
        text = data["choices"][0]["message"]["content"]
        return parse_verdict(text)

    return judge


class LLMJudgeDetector(Detector):
    """Faithfulness detector that delegates to an injected judge function."""

    name = "llm_judge"
    cost = "api"

    def __init__(self, judge_fn: JudgeFn, name: Optional[str] = None):
        self.judge_fn = judge_fn
        if name:
            self.name = name

    def score(self, example: Example) -> float:
        return self.judge_fn(example)
