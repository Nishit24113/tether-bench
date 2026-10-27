"""
Generate the reproducible Markdown results report.

Usage:
    python scripts/generate_report.py              # writes RESULTS.md from the sample
    python scripts/generate_report.py out.md       # custom output path

If cached scores for the heavy tiers exist under out/ (written by
scripts/score_detector.py, e.g. on an HPC run), they are folded in as
precomputed detectors so the report includes the model/api tiers with no model
or API call. See COMPUTE.md.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tether import load_sample, default_detectors, generate_report, precomputed_from_cache

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cached heavy-tier scores to fold in when present: (cache path, cost tier).
CACHED_TIERS = [
    (os.path.join(HERE, "out", "nli.json"), "model"),
    (os.path.join(HERE, "out", "judge.json"), "api"),
]


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    out_path = argv[0] if argv else "RESULTS.md"

    examples = load_sample()
    detectors = default_detectors(examples)
    for path, cost in CACHED_TIERS:
        if os.path.exists(path):
            detectors.append(precomputed_from_cache(path, cost=cost))
            print(f"Folded in cached scores from {os.path.relpath(path, HERE)} (cost={cost})")

    report = generate_report(detectors, examples)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"Wrote {out_path} ({len(report)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
