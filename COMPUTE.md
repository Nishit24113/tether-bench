# Running the heavy tiers (HPC / ASU SOL)

The free detectors run anywhere in milliseconds. The `model` tier (NLI) and the
`api` tier (LLM judge) are heavier: NLI wants a GPU or accelerator to be fast,
and the judge costs money per call. The workflow below runs those once, caches
their per-example scores, and keeps all analysis offline and reproducible.

## The idea: compute once, analyze anywhere

1. On a machine with the model or API access, score a detector and cache it:
   ```bash
   python scripts/score_detector.py nli   out/nli.json
   python scripts/score_detector.py judge out/judge.json
   ```
   Each run writes `{detector, meta, scores: {example_id: score}}`. Runs resume:
   only example ids missing from the cache are recomputed.

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

This keeps the paper's numbers reproducible: the expensive scores are frozen in
JSON and version-controllable, so the leaderboard, breakdown, and frontier are
recomputed deterministically from them.

## ASU SOL (Slurm)

[ASU SOL](https://cores.research.asu.edu/research-computing/) is a Slurm cluster
with GPU and Intel Gaudi accelerator partitions. Submit the scoring job, then
pull the cache back for offline analysis.

```bash
#!/bin/bash
#SBATCH --job-name=tether-nli
#SBATCH --partition=general          # or a GPU/Gaudi partition
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

module load mamba/latest
source activate tether

pip install -e ".[nli]"              # transformers + torch
python scripts/score_detector.py nli out/nli.json
```

```bash
sbatch run_nli.slurm
# when it finishes, out/nli.json holds the cached scores
```

### Intel Gaudi

On a Gaudi partition, load the model through the Habana stack
(`habana-frameworks` / `optimum-habana`) and move it to the HPU device. The NLI
detector is provider-decoupled, so only the model-loading `entail_fn` changes;
the rest of the pipeline is unchanged. Sketch:

```python
import torch
from optimum.habana import ...  # Habana / HPU runtime
# load the NLI model onto the HPU, then wrap it in an entail_fn and pass it to
# NLIEntailmentDetector(entail_fn=...). Everything downstream is identical.
```

## Fine-tuning (optional)

Because detectors are just scorers, a fine-tuned faithfulness model drops in the
same way: train on SOL, expose it as an `entail_fn` (or a judge `judge_fn`), wrap
it in the matching detector, and it slots onto the cost/accuracy frontier next to
the off-the-shelf baselines. This is a natural way to ask whether a small
fine-tuned model can match the LLM judge at a fraction of the cost, which is the
benchmark's whole question.

## Cost note

The frontier uses relative per-tier cost scores by default. Once you have
measured dollar and latency numbers from the SOL and API runs, override
`COST_SCORE` so the frontier is priced in real units.
