"""
Dataset types and loading for Tether.

A Tether example is one (question, context, answer) triple with a gold label
saying whether the answer is *faithful* — i.e. fully supported by the context —
or a *hallucination* that asserts something the context does not. Detectors are
scored on how well they separate the two.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Example:
    """A single faithfulness-labeled RAG triple."""
    id: str
    question: str
    context: str
    answer: str
    faithful: bool                       # gold label: True = grounded, False = hallucinated
    annotations: Dict[str, Any] = field(default_factory=dict)   # optional span/notes

    @property
    def label(self) -> int:
        """1 if faithful, 0 if hallucinated."""
        return 1 if self.faithful else 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Example":
        return cls(
            id=str(d["id"]),
            question=d["question"],
            context=d["context"],
            answer=d["answer"],
            faithful=bool(d["faithful"]),
            annotations=d.get("annotations", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "answer": self.answer,
            "faithful": self.faithful,
            "annotations": self.annotations,
        }


def load_jsonl(path: str) -> List[Example]:
    """Load a dataset from a JSONL file (one example per line)."""
    examples: List[Example] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                examples.append(Example.from_dict(json.loads(raw)))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Malformed example on line {line_no} of {path}: {e}")
    return examples


def save_jsonl(examples: List[Example], path: str) -> None:
    """Write examples to a JSONL file."""
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict()) + "\n")


def _default_sample_path() -> str:
    """Path to the bundled sample dataset shipped with the package."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "data", "sample.jsonl")


def load_sample() -> List[Example]:
    """Load the bundled sample dataset — runs offline, no download."""
    return load_jsonl(_default_sample_path())


def class_balance(examples: List[Example]) -> Dict[str, int]:
    """Count faithful vs hallucinated examples."""
    faithful = sum(1 for e in examples if e.faithful)
    return {
        "total": len(examples),
        "faithful": faithful,
        "hallucinated": len(examples) - faithful,
    }
