"""
Run the Tether baseline benchmark and print the leaderboard.

Usage:
    python scripts/run_benchmark.py              # bundled sample dataset
    python scripts/run_benchmark.py path.jsonl   # custom dataset
"""

import sys
import os
import json

# Allow running directly (python scripts/run_benchmark.py) without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tether import load_sample, load_jsonl, default_detectors, run_benchmark, format_leaderboard


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]

    if argv:
        examples = load_jsonl(argv[0])
        source = argv[0]
    else:
        examples = load_sample()
        source = "bundled sample"

    detectors = default_detectors(examples)
    benchmark = run_benchmark(detectors, examples)

    print(f"Dataset: {source}\n")
    print(format_leaderboard(benchmark))
    print("\nFull metrics:")
    print(json.dumps(benchmark["results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
