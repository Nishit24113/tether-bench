"""
Print the misclassifications of each free detector on the dataset.

Usage:
    python scripts/error_analysis.py            # all default detectors
    python scripts/error_analysis.py lexical    # one detector by short name
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tether import load_sample, default_detectors, misclassified, format_errors


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    examples = load_sample()
    detectors = default_detectors(examples)

    if argv:
        detectors = [d for d in detectors if argv[0] in d.name]
        if not detectors:
            raise SystemExit(f"No default detector matching '{argv[0]}'")

    for det in detectors:
        if det.name == "majority_constant":
            continue  # flags everything; nothing to analyze
        print(format_errors(misclassified(det, examples), limit=8))
        print("\n" + "-" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
