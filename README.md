# Tether

**A reproducible benchmark for RAG answer faithfulness.**

Retrieval-augmented generation (RAG) systems ground an LLM's answer in retrieved
context — but the model can still *hallucinate*, asserting things the context
never supports. Catching that is critical in production. A strong LLM-judge can
do it, but running one on every answer is slow and expensive. Teams want **cheap
faithfulness detectors** they can run inline.

Tether asks a focused research question:

> **How well do low-cost, zero-API faithfulness detectors separate grounded
> answers from hallucinations — and how closely can they approximate an
> LLM judge?**

The repo provides a labeled dataset, a rigorous evaluation harness, and a
growing suite of baselines spanning the **cost/accuracy frontier** — from
free lexical heuristics to (in later versions) a Claude-based judge.

## Why this is a benchmark, not a demo

- **Reproducible by construction.** The v0 baselines and metrics are pure
  Python stdlib — no model downloads, no API keys, no GPU. `pytest` and the
  leaderboard produce identical numbers on any machine.
- **A clear task + protocol.** Binary faithfulness labels, a threshold-free
  primary metric (AUROC), and F1 at the oracle-optimal threshold as a secondary
  operating-point view.
- **A floor and a frontier.** A majority baseline pins AUROC at 0.5; every real
  detector is measured against it and tagged with a cost tier for the
  cost/accuracy analysis.
- **Error bars on every number.** Each AUROC comes with a stratified bootstrap
  95% CI, and the leader is tested against every other detector with a paired
  bootstrap, so a leaderboard gap only counts if it survives resampling.

## Task

Each example is a `(question, context, answer)` triple with a gold label:

- `faithful = true` — every claim in the answer is supported by the context
- `faithful = false` — the answer asserts something the context does not

A **detector** maps an example to a faithfulness score in `[0, 1]` (higher =
more grounded). The harness converts scores to hallucination predictions via a
threshold and reports precision / recall / F1 / accuracy / AUROC. The positive
class is *hallucinated* — the thing we want to catch.

## Baselines (v0)

| Detector | Cost | Idea |
|----------|------|------|
| `majority_constant` | free | Constant score — the 0.5-AUROC floor |
| `lexical_overlap` | free | Fraction of answer content tokens present in context |
| `novel_token` | free | Penalizes answer tokens absent from context *and* question |
| `tfidf_cosine` | free | TF-IDF cosine similarity between answer and context |
| `nli_entailment` | model | A local NLI model scores whether the context entails the answer — the mid-cost tier |
| `llm_judge` | api | A strong model (default `claude-opus-5-5`) judges faithfulness — the reference the cheap detectors are measured against |

The `free` and `model` tiers are populated with real numbers (the NLI scores
come from a cached run of `facebook/bart-large-mnli`; see
[COMPUTE.md](COMPUTE.md)). The `api` judge tier is wired but not run here, since
it needs a paid key.

## Results on the bundled sample (n=50, balanced)

```
detector           cost   auroc   f1      precision  recall  accuracy
-----------------  -----  ------  ------  ---------  ------  --------
nli_entailment     model  1.000   1.000   1.000      1.000   1.000
novel_token        free   0.779   0.774   0.649      0.960   0.720
lexical_overlap    free   0.763   0.762   0.632      0.960   0.700
tfidf_cosine       free   0.603   0.676   0.510      1.000   0.520
majority_constant  free   0.500   0.667   0.500      1.000   0.500
```

The sample is built to be hard *for surface-overlap detectors*: many
hallucinations reuse almost all of the context's vocabulary (an entity or number
swapped), and many faithful answers paraphrase with few shared tokens. On this
set the free detectors fall well short of the easy-case ceiling — `tfidf_cosine`
drops to 0.60 AUROC, barely above the 0.50 floor, and the best lexical detector
only reaches 0.78. The mean lexical overlap is 0.81 for faithful answers versus
0.65 for hallucinated ones, a gap of just 0.16, so surface overlap no longer
separates the classes cleanly.

The NLI model at the mid-cost tier, which reasons about entailment rather than
word overlap, separates the two classes **perfectly** on this sample: AUROC,
F1, and accuracy are all 1.000. That is the benchmark's core thesis made
concrete — the vocabulary-reusing errors that defeat the free detectors are
exactly the ones a semantic model catches, so the mid tier earns its cost. It
also carries a caution, stated plainly in the limitations: a perfect score means
this 50-example set is no longer hard enough to find where even an off-the-shelf
NLI model fails, so the next job is a larger, more adversarial dataset that can
map the real cost/accuracy frontier instead of saturating at the mid tier.

### Are the gaps real?

Fifty examples give wide error bars, so Tether reports them instead of letting a
small gap pass as a result. Each AUROC gets a 95% CI from a stratified bootstrap
(2000 resamples, seed 0), and detectors are compared by a paired bootstrap on
identical resamples. The interesting question is *within the free tier*, where
the gaps are small (the NLI tier wins everything significantly, p < 0.001, so it
is the trivial comparison). Taking `novel_token`, the best free detector, as the
reference:

| Detector | AUROC | AUROC 95% CI | Gap from `novel_token` | Gap 95% CI | p-value |
|----------|-------|--------------|------------------------|------------|---------|
| `novel_token` | 0.779 | [0.635, 0.902] | - | - | - |
| `lexical_overlap` | 0.763 | [0.614, 0.890] | +0.016 | [-0.013, +0.070] | 0.776 |
| `tfidf_cosine` | 0.603 | [0.435, 0.758] | +0.176 | [+0.040, +0.321] | 0.007 |
| `majority_constant` | 0.500 | [0.500, 0.500] | +0.279 | [+0.135, +0.402] | < 0.001 |

The top two free places are a statistical tie. `novel_token` and
`lexical_overlap` give identical scores on 47 of the 50 examples (they only
differ when an answer reuses a question word the context lacks), so their 0.016
gap rests on three answers. Both beat `tfidf_cosine` and the constant floor
significantly. With intervals 0.27 to 0.32 AUROC wide, separating two free
detectors a few points apart will need a much larger dataset; separating the
tiers does not, because the NLI model's advantage over every free detector is
large and significant.

```python
from tether import load_sample, default_detectors, detector_uncertainty

examples = load_sample()
unc = detector_uncertainty(default_detectors(examples), examples)
unc["cis"]["novel_token"]          # point, lo, hi of the AUROC
unc["vs_best"]["lexical_overlap"]  # diff, lo, hi, p_value, significant, n_differ
```

`bootstrap_ci(labels, scores)` and `paired_bootstrap(labels, scores_a, scores_b)`
work on raw score lists, so they apply just as well to cached NLI or judge scores.

### Are the scores trustworthy probabilities?

AUROC only measures *ranking*. A separate question is whether a faithfulness
score of 0.7 means the answer is faithful about 70% of the time — i.e. whether
the raw score can be read as a probability. The detector interface never
promises this, and the calibration numbers show it does not hold:

| Detector | AUROC | ECE | ECE 95% CI | MCE | Brier |
|----------|-------|-----|------------|-----|-------|
| `nli_entailment` | 1.000 | 0.009 | [0.000, 0.028] | 0.263 | 0.001 |
| `majority_constant` | 0.500 | 0.000 | [0.000, 0.000] | 0.000 | 0.250 |
| `tfidf_cosine` | 0.603 | 0.187 | [0.135, 0.344] | 0.928 | 0.265 |
| `lexical_overlap` | 0.763 | 0.304 | [0.242, 0.391] | 1.000 | 0.281 |
| `novel_token` | 0.779 | 0.321 | [0.258, 0.399] | 1.000 | 0.288 |

ECE is the bin-weighted gap between a detector's mean score and the observed
faithful rate (lower is better); Brier is the mean squared error of the scores
as probabilities. Three things stand out, and the first two are warnings against
reading the column you expect:

- **Calibration is not usefulness.** `majority_constant` has ECE 0.000 — the
  best-calibrated free detector — because on a balanced set a constant 0.5
  exactly matches the 50% base rate. Its AUROC is still 0.5. A score can be
  perfectly calibrated and carry no signal.
- **The useful free detectors are badly miscalibrated.** `novel_token` ranks
  best among the free detectors but has the worst ECE: in its 0.6-0.7 score band
  only about 14% of answers are actually faithful, not 65%. Reading its raw score
  as a confidence would be far off, so deploying it as a probability would need a
  calibration map (Platt or isotonic) fit on a held-out split.
- **The NLI model is the exception.** It is both best-ranked (AUROC 1.000) and
  essentially calibrated (ECE 0.009, Brier 0.001): its entailment probabilities
  track the true faithful rate, so unlike the free detectors its score can be
  read as a probability. That is the mid tier buying trustworthy confidence, not
  just a better ordering.

Among the free detectors, `tfidf_cosine` has the lowest point ECE despite the
worst AUROC, but its CI overlaps the others, so that is a hint rather than a
result. Best-ranked and best-calibrated are simply different axes.

```python
from tether import load_sample, default_detectors, detector_calibration

examples = load_sample()
cal = detector_calibration(default_detectors(examples), examples)
cal["novel_token"]["ece"]    # expected calibration error
cal["novel_token"]["bins"]   # reliability bins: mean_score vs frac_faithful
```

### How optimistic is the reported F1?

The F1 above is measured at each detector's F1-optimal threshold, picked on the
same 50 examples it is then scored on. That is an oracle operating point: a
deployment has to fix its threshold first. To measure the optimism, Tether
re-estimates F1 with 5-fold stratified cross-validation, choosing the threshold
on the training folds and scoring on the held-out fold:

| Detector | Oracle F1 | CV F1 | Optimism | CV precision | CV recall |
|----------|-----------|-------|----------|--------------|-----------|
| `nli_entailment` | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| `novel_token` | 0.774 | 0.733 | +0.041 | 0.629 | 0.880 |
| `lexical_overlap` | 0.762 | 0.721 | +0.041 | 0.611 | 0.880 |
| `majority_constant` | 0.667 | 0.667 | +0.000 | 0.500 | 1.000 |
| `tfidf_cosine` | 0.676 | 0.629 | +0.047 | 0.489 | 0.880 |

The oracle F1 overstates the honest, held-out number by about 0.04 for the
signal-carrying free detectors. `majority_constant` has no gap because it flags
every answer at any threshold, so there is nothing to tune, and `nli_entailment`
has none because it separates the classes perfectly at any threshold the folds
pick. The free-detector gap is modest here because the F1-optimal threshold is
stable across folds (~0.87 for the lexical detectors), but at this sample size
the cross-validated estimate is itself noisy: the gap moves between about 0.04
and 0.09 depending on the fold seed, which is one more reason the dataset needs
to grow.

```python
from tether import load_sample, default_detectors, detector_validation

examples = load_sample()
val = detector_validation(default_detectors(examples), examples, k=5, seed=0)
val["novel_token"]["cv_f1"]          # pooled held-out F1
val["novel_token"]["optimism_gap"]   # oracle F1 minus cross-validated F1
```

## Error-type breakdown

Every hallucinated example is annotated with the kind of error it contains, and
every faithful one with how it is supported. Tether reports per-category
**detection recall** at each detector's F1-optimal threshold:

```
Detection recall by error type (hallucinated examples)

error_type            majority_constant  lexical_overlap  novel_token  tfidf_cosine
--------------------  -----------------  ---------------  -----------  ------------
contradiction         1.00               1.00             1.00         1.00
entity_swap           1.00               0.86             0.86         1.00
numeric               1.00               1.00             1.00         1.00
unsupported_addition  1.00               1.00             1.00         1.00
```

On the harder set the free detectors reach these high per-type recalls only by
moving their threshold down and flagging almost everything (precision falls to
0.51-0.65), so recall at a single threshold is no longer the honest lens. AUROC
is. `entity_swap` still has the lowest recall for the token-overlap detectors,
since swapping one entity leaves the overlap high, and the AUROC drop shows the
free detectors lose separability across the board when hallucinations reuse the
context's words. This is the regime where a semantic NLI model or an LLM judge
should earn its cost — and the leaderboard above confirms it does: the NLI model
scores a perfect AUROC on exactly the errors the free detectors cannot separate.

Per-example **error analysis** (`tether/error_analysis.py`,
`python scripts/error_analysis.py`) makes this concrete: at its threshold the
lexical detector's false positives are almost all paraphrases (faithful answers
it flags for sharing few tokens), and its misses are the vocabulary-reusing
swaps. Both are failures of meaning, not surface overlap.

## The LLM judge & agreement analysis

The whole point of Tether is to ask *how closely a free detector tracks an
expensive LLM judge*. The judge is the reference detector; agreement with it is
the headline metric.

The judge is provider-decoupled — `LLMJudgeDetector` takes any `judge_fn`
mapping an example to a faithfulness score, so it's trivially mockable and the
test suite runs offline. The Anthropic-backed judge is opt-in
(`pip install tether-bench[llm]`, `ANTHROPIC_API_KEY` set):

```python
from tether import (
    load_sample, LexicalOverlapDetector, LLMJudgeDetector,
    anthropic_judge, best_threshold, judge_agreement,
)

examples = load_sample()

# Reference judge (a strong model). Any callable works; this one calls Claude.
judge = LLMJudgeDetector(anthropic_judge(model="claude-opus-5-5"))
judge_scores = judge.score_all(examples)

# A cheap detector at its F1-optimal threshold
cheap = LexicalOverlapDetector()
cheap_scores = cheap.score_all(examples)
labels = [e.label for e in examples]
thr, _ = best_threshold(labels, cheap_scores)

# How closely does the cheap detector track the judge?
print(judge_agreement(cheap_scores, judge_scores, cheap_threshold=thr))
# {'agreement_rate': ..., 'cohen_kappa': ..., 'cheap_only_flags': ..., 'judge_only_flags': ..., ...}
```

`cohen_kappa` corrects for chance agreement; `judge_only_flags` counts the
hallucinations the judge caught that the cheap detector missed — exactly the gap
a paid judge is buying.

The judge is not tied to one vendor. `anthropic_judge` calls Claude, but
`openai_compatible_judge` talks to **any** OpenAI-compatible chat-completions
gateway using only the standard library (no extra dependency), so a self-hosted
or institutional gateway serving open-source models can be the judge too. The
endpoint and key are read from the environment, so nothing provider-specific is
baked into the repo:

```python
from tether import LLMJudgeDetector, openai_compatible_judge

# set TETHER_JUDGE_BASE_URL (e.g. a gateway's .../v1) and TETHER_JUDGE_API_KEY
judge = LLMJudgeDetector(openai_compatible_judge(model="your-model-id"))
scores = judge.score_all(examples)
```

The **mid-cost tier** is the NLI detector — same decoupled design, so it's
mockable in tests and the model is an opt-in dependency
(`pip install tether-bench[nli]`):

```python
from tether import NLIEntailmentDetector, transformers_nli

# premise = context, hypothesis = answer; score = P(context entails answer)
nli = NLIEntailmentDetector(transformers_nli("facebook/bart-large-mnli"))
scores = nli.score_all(examples)
```

## Cost/accuracy frontier

The benchmark's headline artifact: each detector placed in (cost, quality) space,
with the Pareto-optimal ones marked. Cost is a relative per-tier score (free
compute = 0, local model = 0.3, paid API = 1.0) — override `COST_SCORE` with
measured costs when you have them.

```
Cost/accuracy frontier (quality = auroc; * = Pareto-optimal)

detector           cost_tier  cost  auroc  frontier
-----------------  ---------  ----  -----  --------
nli_entailment     model      0.30  1.000  *
novel_token        free       0.00  0.779  *
lexical_overlap    free       0.00  0.763
tfidf_cosine       free       0.00  0.603
majority_constant  free       0.00  0.500
```

Two detectors are Pareto-optimal: `novel_token` (cheapest useful option) and
`nli_entailment` (highest quality). Among the free detectors only `novel_token`
is on the frontier — the rest cost the same but detect less, though its edge over
`lexical_overlap` is within noise, as the paired test above shows. The NLI tier
sits at the top of the staircase: one step up in cost buys the jump from 0.78 to
1.00 AUROC on this sample. Adding the `llm_judge` row (cost tier `api`) would
extend the ladder further, once it is run with a key. The research payoff is the
shape of that staircase — *how much AUROC each step up the cost ladder actually
buys* — and here the mid step buys a lot.

```python
from tether import load_sample, default_detectors, run_benchmark, build_frontier, format_frontier

examples = load_sample()
bench = run_benchmark(default_detectors(examples), examples)
print(format_frontier(build_frontier(bench["leaderboard"], metric="auroc")))
```

### Pricing the frontier in real units

The default cost axis uses relative placeholders. The `cost_model` estimates the
actual per-tier cost so the frontier can be priced in dollars:

```python
from tether import (
    load_sample, estimate_api_cost, estimate_model_cost, free_cost,
    normalized_cost_scores, build_frontier,
)
examples = load_sample()
costs = {
    "free": free_cost(len(examples)).total_usd,
    "model": estimate_model_cost(len(examples), seconds_per_example=0.05, hourly_rate_usd=2.0).total_usd,
    "api": estimate_api_cost(examples, input_price_per_1m=4.0, output_price_per_1m=20.0).total_usd,
}
build_frontier(bench["leaderboard"], cost_scores=normalized_cost_scores(costs))
```

On the 50-example sample, judging every answer with an LLM judge at Opus-tier
pricing costs about **$0.053** (~$0.0011/example), the NLI tier about **$0.0014**
of GPU time, and the free tier **$0**. Normalized, that is `free 0.0, model 0.026,
api 1.0` — the measured mid tier is far cheaper than the placeholder 0.3, which is
exactly the kind of correction real cost numbers make to the frontier.

## Quickstart

```bash
pip install -e ".[dev]"
pytest                          # 196 tests, all stdlib
python scripts/run_benchmark.py # prints the leaderboard above
```

Run against your own data (JSONL, one example per line):

```json
{"id": "1", "question": "...", "context": "...", "answer": "...", "faithful": true}
```

```bash
python scripts/run_benchmark.py path/to/your.jsonl
```

## Layout

```
tether/
  data.py            # Example type + JSONL / sample loaders
  metrics.py         # confusion matrix, AUROC (Mann-Whitney), threshold search
  harness.py         # run detectors -> metrics -> ranked leaderboard
  agreement.py       # agreement rate + Cohen's kappa vs the judge
  frontier.py        # cost/accuracy Pareto frontier
  report.py          # reproducible Markdown report (leaderboard/breakdown/frontier)
  cache.py           # persist/replay expensive detector scores (compute once, analyze anywhere)
  dataset_stats.py   # dataset composition, integrity validation, difficulty summary
  error_analysis.py  # which examples each detector gets wrong (false pos/neg by type)
  cost_model.py      # estimate per-tier $ cost; price the frontier in real units
  stats.py           # stratified bootstrap CIs + paired significance test for leaderboard gaps
  calibration.py     # reliability bins, Brier, ECE/MCE: are the scores real probabilities?
  validation.py      # k-fold cross-validated F1 vs oracle F1: how optimistic is the threshold?
  baselines/         # majority, lexical, novel-token, tf-idf, nli, llm-judge
data/sample.jsonl    # bundled labeled dataset (50 examples, runs offline), error-type annotated
scripts/run_benchmark.py   # prints leaderboard + breakdown + frontier
scripts/generate_report.py # regenerates RESULTS.md
scripts/score_detector.py  # score one detector, cache to JSON (HPC job entry point)
out/nli.json         # cached NLI (bart-large-mnli) scores from a SOL run, folded into the report
PAPER.md                   # working paper draft
RESULTS.md                 # generated results, cited by the paper
COMPUTE.md                 # producing the model/api tiers on a GPU and caching them
tests/               # 196 unit tests
```

## Roadmap

- **Grow the dataset into a harder, more adversarial set** of hundreds of
  examples with annotated unsupported spans. The NLI tier scoring a perfect
  AUROC shows the current 50 examples no longer challenge a semantic model, so
  the frontier saturates at the mid tier; a harder set is needed to find where
  NLI fails and to map the real staircase.
- **Run the `api` judge tier** with a key and cache it alongside `out/nli.json`,
  so all three tiers are populated.
- **Wire live measured latency/tokens** from real NLI and judge runs into the cost model

_Done:_ 50-example error-type / support-type annotated dataset with integrity
validation and a difficulty (lexical-overlap separation) summary; per-category
recall breakdown; NLI-entailment mid-cost baseline (run on ASU SOL, real cached
scores in `out/nli.json`, perfect AUROC on the sample), LLM-judge reference
detector (both provider-decoupled) + agreement analysis (agreement rate, Cohen's
kappa); cost/accuracy frontier with Pareto marking; reproducible Markdown report
([RESULTS.md](RESULTS.md)) and a working paper draft ([PAPER.md](PAPER.md));
score caching so the heavy tiers run once on HPC and the analysis stays offline
([COMPUTE.md](COMPUTE.md)); a cost model that prices the frontier in real dollars
per tier; stratified bootstrap CIs on every AUROC and a paired bootstrap test for
leaderboard gaps; calibration analysis (reliability bins, ECE/MCE, Brier) showing
the free detectors rank well but are not calibrated probabilities while the NLI
tier is; k-fold cross-validated operating points that quantify how much the
oracle-threshold F1 overstates the honest held-out F1.

## Reproducibility

v0 is deterministic and dependency-free. Later baselines will pin model
versions and seeds, and record per-example scores so every leaderboard number is
traceable to its inputs.

---

**Built by Nishit Patel** | MS Computer Science, Arizona State University
