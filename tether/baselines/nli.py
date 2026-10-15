"""
NLI-entailment faithfulness detector for Tether.

The mid-cost tier between free lexical heuristics and a paid LLM judge. A
natural-language-inference model scores whether the context *entails* the
answer: premise = context, hypothesis = answer. The probability of the
`entailment` class is the faithfulness score (higher = more supported). This is
the standard semantic approach to factual-consistency checking, and it should
catch exactly the vocabulary-reusing errors (entity swaps, numeric changes) that
slip past token overlap.

Like the LLM judge, the detector is provider-decoupled: it takes an `entail_fn`
mapping (premise, hypothesis) to an entailment probability in [0, 1], so tests
run offline with a fake. `transformers_nli()` builds a real cross-encoder NLI
scorer, imported lazily (`pip install tether-bench[nli]`).
"""

from typing import Callable, Optional, Any, Dict

from .base import Detector
from ..data import Example


EntailFn = Callable[[str, str], float]


def _entailment_index(id2label: Dict[int, str]) -> int:
    """
    Find the model's `entailment` class index from its id2label map.

    MNLI head label order varies by checkpoint, so never hard-code index 2 —
    look up the label whose name contains "entail".
    """
    for idx, label in id2label.items():
        if "entail" in str(label).lower():
            return int(idx)
    raise ValueError(f"No entailment label found in id2label: {id2label}")


def transformers_nli(
    model: str = "facebook/bart-large-mnli",
    device: Optional[int] = None,
) -> EntailFn:
    """
    Build an entail_fn backed by a HuggingFace cross-encoder NLI model.

    Requires `pip install tether-bench[nli]` (transformers + torch). The model is
    loaded once, lazily, on first construction. Returns P(entailment) for the
    (premise, hypothesis) pair.
    """
    import torch  # lazy
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # lazy

    tokenizer = AutoTokenizer.from_pretrained(model)
    net = AutoModelForSequenceClassification.from_pretrained(model)
    net.eval()
    if device is not None:
        net.to(device)

    entail_idx = _entailment_index(net.config.id2label)

    def entail(premise: str, hypothesis: str) -> float:
        inputs = tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = net(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        return float(probs[entail_idx].item())

    return entail


class NLIEntailmentDetector(Detector):
    """Faithfulness = P(context entails answer) under an NLI model."""

    name = "nli_entailment"
    cost = "model"

    def __init__(self, entail_fn: EntailFn, name: Optional[str] = None):
        self.entail_fn = entail_fn
        if name:
            self.name = name

    def score(self, example: Example) -> float:
        # premise = context (the evidence), hypothesis = answer (the claim)
        return self.entail_fn(example.context, example.answer)
