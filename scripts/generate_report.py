"""
Generate the reproducible Markdown results report.

Usage:
    python scripts/generate_report.py              # writes RESULTS.md from the sample
    python scripts/generate_report.py out.md       # custom output path
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tether import load_sample, default_detectors, generate_report


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    out_path = argv[0] if argv else "RESULTS.md"

    examples = load_sample()
    report = generate_report(default_detectors(examples), examples)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"Wrote {out_path} ({len(report)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
