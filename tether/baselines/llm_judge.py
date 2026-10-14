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
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text!r}")

    obj = json.loads(match.group(0))
    if "faithful" not in obj:
        raise ValueError(f"Judge response missing 'faithful': {obj!r}")

    faithful = bool(obj["faithful"])
    confidence = float(obj.get("confidence", 1.0))
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
