"""
Score one detector over the dataset and cache its scores to JSON.

This is the entry point you submit as a cluster job for the expensive tiers:
score the NLI model or the LLM judge once, write the cache, then run the offline
analysis anywhere with the cached scores.

Usage:
    python scripts/score_detector.py lexical  out/lexical.json
    python scripts/score_detector.py nli      out/nli.json      # needs [nli] extra
    python scripts/score_detector.py judge    out/judge.json    # needs [llm] + API key
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tether import (
    load_sample, run_and_cache,
    MajorityDetector, LexicalOverlapDetector, NovelTokenDetector, TfidfCosineDetector,
)


def build_detector(name, examples):
    """Construct a detector by short name. Heavy tiers import their deps lazily."""
    if name == "majority":
        return MajorityDetector()
    if name == "lexical":
        return LexicalOverlapDetector()
    if name == "novel":
        return NovelTokenDetector()
    if name == "tfidf":
        return TfidfCosineDetector(examples)
    if name == "nli":
        from tether import NLIEntailmentDetector, transformers_nli
        return NLIEntailmentDetector(transformers_nli())
    if name == "judge":
        from tether import LLMJudgeDetector, anthropic_judge
        return LLMJudgeDetector(anthropic_judge())
    raise SystemExit(f"Unknown detector '{name}'. Choose: majority, lexical, novel, tfidf, nli, judge")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2:
        raise SystemExit("Usage: python scripts/score_detector.py <detector> <out.json>")

    name, out_path = argv[0], argv[1]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    examples = load_sample()
    detector = build_detector(name, examples)
    scores = run_and_cache(detector, examples, out_path)

    print(f"Scored {len(scores)} examples with '{detector.name}' -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
