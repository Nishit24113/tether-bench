# Producing the heavy tiers (GPU)

The free detectors run anywhere in milliseconds. The `model` tier (NLI) and the
`api` tier (LLM judge) are heavier: NLI wants a GPU to be fast, and the judge
costs money per call. The workflow below runs those once, caches their
per-example scores, and keeps all downstream analysis offline and reproducible.

The committed `model`-tier scores (`out/nli.json`,
`facebook/bart-large-mnli`) were produced on a GPU (an ASU SOL node). Only the
resulting per-example scores are stored; no environment or cluster details are
needed to reuse them.

## The idea: compute once, analyze anywhere

1. On a machine with the model or API access, score a detector once and cache
   its per-example scores to JSON. `scripts/score_detector.py` is the entry
   point; it writes `{detector, meta, scores: {example_id: score}}` and resumes,
   recomputing only example ids missing from the cache.

2. Anywhere else (laptop, CI), fold the cached scores back into the analysis
   with no model and no API key:
   ```python
   from tether import load_sample, default_detectors, run_benchmark, precomputed_from_cache

   examples = load_sample()
   detectors = default_detectors(examples) + [
       precomputed_from_cache("out/nli.json", cost="model"),
       precomputed_from_cache("out/judge.json", cost="api"),
   ]
   bench = run_benchmark(detectors, examples)
   ```

`scripts/generate_report.py` does this automatically: any cache files present
under `out/` are folded into `RESULTS.md`, so the model/api tiers appear with no
model download and the numbers are recomputed deterministically from the frozen
JSON.

## Running a heavy tier yourself

Run `score_detector.py` for the tier you want on any machine that has the
dependencies, then commit the resulting JSON:

- NLI (`model` tier): needs the `[nli]` extra (a transformers model) and is much
  faster on a GPU. The bundled result used `facebook/bart-large-mnli`.
- LLM judge (`api` tier): needs the `[llm]` extra and a provider API key.

Because detectors are just scorers, a fine-tuned faithfulness model drops in the
same way: expose it as an `entail_fn` (or a judge `judge_fn`), wrap it in the
matching detector, and it slots onto the cost/accuracy frontier next to the
off-the-shelf baselines. This is a natural way to ask whether a small fine-tuned
model can match the LLM judge at a fraction of the cost, which is the benchmark's
whole question.

## Cost note

The frontier uses relative per-tier cost scores by default. Once you have
measured dollar and latency numbers from the model and API runs, override
`COST_SCORE` so the frontier is priced in real units.
