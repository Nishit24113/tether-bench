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

All three tiers are populated with real numbers. The NLI scores come from a
cached run of `facebook/bart-large-mnli`; the judge scores come from a cached
run of a Qwen3 model (`qwen38-27b`) served over an OpenAI-compatible gateway.
Both are stored as per-example scores and folded into the report offline (see
[COMPUTE.md](COMPUTE.md)).

## Results on the bundled sample (n=130, balanced)

```
detector           cost   auroc   f1      precision  recall  accuracy
-----------------  -----  ------  ------  ---------  ------  --------
llm_judge          api    1.000   1.000   1.000      1.000   1.000
nli_entailment     model  0.974   0.927   0.877      0.985   0.923
lexical_overlap    free   0.538   0.690   0.550      0.923   0.585
novel_token        free   0.534   0.667   0.500      1.000   0.500
majority_constant  free   0.500   0.667   0.500      1.000   0.500
tfidf_cosine       free   0.453   0.674   0.508      1.000   0.515
```

The 130 examples are built to be hard for surface overlap *and* to probe
reasoning: hallucinations reuse the context's vocabulary, and a third of them
require a reasoning step (multi-step numeric, long-context distractors,
counting/comparison). The mean lexical overlap is 0.67 for faithful answers
versus 0.66 for hallucinated ones, a gap of just **0.005**, so surface overlap
carries no signal.

Two results stand out. First, **every free detector sits at chance**:
`lexical_overlap` and `novel_token` reach only 0.54, `tfidf_cosine` falls to 0.45
(below the floor), and every free 95% CI spans 0.5, so none significantly beats
the majority baseline. Second, and this is the payoff of the reasoning batch,
**the two semantic tiers finally diverge**: the LLM judge still scores a perfect
1.000, but NLI drops to **0.974**, and the gap is significant (paired bootstrap
diff +0.026, 95% CI [+0.010, +0.051], p < 0.001). The cheap entailment model
wrongly rejects correct multi-step and numeric answers it cannot verify, whereas
the judge, a reasoning LLM, gets them right. This is the benchmark's founding
question made live: a cheap detector tracks the judge well on simple faithfulness
but measurably falls behind on reasoning, which is exactly where paying for the
judge starts to be worth it.

### Are the gaps real?

Each AUROC gets a 95% CI from a stratified bootstrap (2000 resamples, seed 0),
and detectors are compared by a paired bootstrap on identical resamples. The best
detector is now the judge; its paired comparison against each other detector:

| `llm_judge` vs | AUROC diff | 95% CI | p-value | Significant |
|----------------|------------|--------|---------|-------------|
| `nli_entailment` | +0.026 | [+0.009, +0.051] | < 0.001 | yes |
| `lexical_overlap` | +0.462 | [+0.359, +0.560] | < 0.001 | yes |
| `novel_token` | +0.466 | [+0.359, +0.564] | < 0.001 | yes |
| `tfidf_cosine` | +0.547 | [+0.446, +0.644] | < 0.001 | yes |

Three things are clear-cut. No free detector's AUROC CI excludes 0.5
(`lexical_overlap` [0.440, 0.640], `tfidf_cosine` [0.356, 0.554]), so **none
significantly beats the majority baseline**. The semantic tiers beat every free
detector by 0.46 to 0.55 AUROC. And, the new result, **the judge significantly
beats the NLI tier** (+0.026, p < 0.001): the small but real gap that the
reasoning batch opened up. The expensive tier is no longer interchangeable with
the cheap semantic one.

```python
from tether import load_sample, default_detectors, detector_uncertainty

examples = load_sample()
unc = detector_uncertainty(default_detectors(examples), examples)
unc["cis"]["novel_token"]          # point, lo, hi of the AUROC
unc["vs_best"]["lexical_overlap"]  # diff, lo, hi, p_value, significant, n_differ
```

`bootstrap_ci(labels, scores)` and `paired_bootstrap(labels, scores_a, scores_b)`
work on raw score lists, so they apply just as well to cached NLI or judge scores.

### Do the free signals combine to beat the best single detector?

The free detectors look at overlapping surface signals, so it is worth asking
whether combining them clears the free-tier ceiling. Tether stacks them with a
logistic combiner, fit out-of-fold (5-fold, seed 0) so the combiner's AUROC is
never measured on the data it was trained on:

| Combination | AUROC | 95% CI |
|-------------|-------|--------|
| stacked free detectors (logistic, out-of-fold) | 0.529 | [0.429, 0.628] |
| best single (`lexical_overlap`) | 0.538 | - |

The answer is no. The stack matches the best single detector (difference -0.009,
95% CI [-0.148, +0.127], p = 0.876), both at chance. The free signals are
redundant, so combining them cannot clear the free-tier ceiling. You cannot reach
the semantic tier by blending surface heuristics, which is the point of paying
for NLI.

```python
from tether import load_sample, default_detectors, cross_val_stack

examples = load_sample()
cross_val_stack(default_detectors(examples), examples)  # stack vs best single, out-of-fold
```

### Are the scores trustworthy probabilities?

AUROC only measures *ranking*. A separate question is whether a faithfulness
score of 0.7 means the answer is faithful about 70% of the time — i.e. whether
the raw score can be read as a probability. The detector interface never
promises this, and the calibration numbers show it does not hold:

| Detector | AUROC | ECE | ECE 95% CI | MCE | Brier |
|----------|-------|-----|------------|-----|-------|
| `llm_judge` | 1.000 | 0.022 | [0.014, 0.035] | 0.720 | 0.005 |
| `nli_entailment` | 0.974 | 0.087 | [0.057, 0.136] | 0.563 | 0.071 |
| `majority_constant` | 0.500 | 0.000 | [0.000, 0.000] | 0.000 | 0.250 |
| `tfidf_cosine` | 0.453 | 0.231 | [0.162, 0.321] | 0.936 | 0.315 |
| `lexical_overlap` | 0.538 | 0.296 | [0.239, 0.372] | 0.972 | 0.334 |
| `novel_token` | 0.534 | 0.356 | [0.300, 0.421] | 0.972 | 0.364 |

ECE is the bin-weighted gap between a detector's mean score and the observed
faithful rate (lower is better); Brier is the mean squared error of the scores
as probabilities. Three things stand out:

- **Calibration is not usefulness.** `majority_constant` has ECE 0.000 — the
  best-calibrated free detector — because on a balanced set a constant 0.5
  exactly matches the 50% base rate. Its AUROC is still 0.5. A score can be
  perfectly calibrated and carry no signal.
- **The free detectors are badly miscalibrated**: `novel_token` reaches ECE
  0.356, so its raw score is nowhere near a probability. Deploying it as a
  confidence would need a calibration map fit on a held-out split.
- **The tiers rank the same way on calibration as on AUROC.** The judge is best
  (ECE 0.022); NLI is close but visibly behind (0.087, up from 0.029 on the
  easier set), the same reasoning examples that cost it AUROC also cost it
  calibration. Both still read as probabilities where the free detectors do not.

Best-ranked and best-calibrated are different axes, but here they coincide: the
semantic tiers win both, and the free detectors lose both.

```python
from tether import load_sample, default_detectors, detector_calibration

examples = load_sample()
cal = detector_calibration(default_detectors(examples), examples)
cal["novel_token"]["ece"]    # expected calibration error
cal["novel_token"]["bins"]   # reliability bins: mean_score vs frac_faithful
```

### Can the cheap scores be fixed into probabilities?

Miscalibration is a property of the *raw* scores, not a verdict on the detectors.
A post-hoc calibration map can turn the scores into probabilities. Tether fits
two, isotonic (a monotone step map, via Pool Adjacent Violators) and Platt (a
logistic map), from score to observed faithful rate. The honest test is
cross-validated: the map is fit on training folds and applied to a held-out
fold, so the post-calibration numbers are never measured on the data the map was
tuned on.

| Detector | Raw ECE | Isotonic ECE | Platt ECE | Raw Brier | Isotonic Brier |
|----------|---------|--------------|-----------|-----------|----------------|
| `majority_constant` | 0.000 | 0.000 | 0.000 | 0.250 | 0.250 |
| `lexical_overlap` | 0.296 | 0.065 | 0.150 | 0.334 | 0.248 |
| `novel_token` | 0.356 | 0.082 | 0.136 | 0.364 | 0.268 |
| `tfidf_cosine` | 0.231 | 0.112 | 0.073 | 0.315 | 0.261 |
| `nli_entailment` | 0.087 | 0.058 | 0.054 | 0.071 | 0.062 |
| `llm_judge` | 0.022 | 0.006 | 0.008 | 0.005 | 0.004 |

A map cuts the free detectors' ECE by roughly two thirds (`lexical_overlap` 0.296
to 0.065 isotonic, `novel_token` 0.356 to 0.082) and lowers their Brier score too,
so the cheap scores *can* be made into usable probabilities even though they rank
poorly. It also tightens NLI's calibration (0.087 to ~0.055). Two honest caveats:
the already-calibrated detectors have little to gain, and a well-calibrated score
is still not a useful one (a recalibrated free detector reports honest ~0.5-ish
probabilities precisely because it cannot tell the classes apart). The practical
rule is that if you need a cheap detector's output as a confidence, fit a
calibration map on a held-out split rather than reading the raw score as a
probability.

```python
from tether import load_sample, default_detectors, detector_recalibration

examples = load_sample()
rc = detector_recalibration(default_detectors(examples), examples, method="isotonic")
rc["lexical_overlap"]["raw_ece"], rc["lexical_overlap"]["cal_ece"]  # before vs after
```

### How optimistic is the reported F1?

The F1 above is measured at each detector's F1-optimal threshold, picked on the
same 130 examples it is then scored on. That is an oracle operating point: a
deployment has to fix its threshold first. To measure the optimism, Tether
re-estimates F1 with 5-fold stratified cross-validation, choosing the threshold
on the training folds and scoring on the held-out fold:

| Detector | Oracle F1 | CV F1 | Optimism | CV precision | CV recall |
|----------|-----------|-------|----------|--------------|-----------|
| `llm_judge` | 1.000 | 0.992 | +0.008 | 0.985 | 1.000 |
| `nli_entailment` | 0.927 | 0.927 | +0.000 | 0.877 | 0.985 |
| `lexical_overlap` | 0.690 | 0.674 | +0.015 | 0.542 | 0.892 |
| `majority_constant` | 0.667 | 0.667 | +0.000 | 0.500 | 1.000 |
| `tfidf_cosine` | 0.674 | 0.667 | +0.007 | 0.504 | 0.985 |
| `novel_token` | 0.667 | 0.631 | +0.036 | 0.484 | 0.908 |

The oracle F1 overstates the honest, held-out number only slightly (under 0.015
for every detector except `novel_token` at 0.036), so the operating points are
not badly overfit. The free detectors' F1 still looks deceptively healthy (~0.67)
only because at a low threshold they flag almost everything; precision collapses
to ~0.50, and F1 at a single operating point hides what AUROC makes plain, that
they barely separate the classes.

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

error_type            majority  lexical  novel  tfidf  nli   judge
--------------------  --------  -------  -----  -----  ----  -----
comparison            1.00      0.75     1.00   1.00   1.00  1.00
contradiction         1.00      0.89     1.00   1.00   1.00  1.00
distractor            1.00      0.33     1.00   1.00   1.00  1.00
entity_swap           1.00      0.92     1.00   1.00   1.00  1.00
multi_hop             1.00      1.00     1.00   1.00   1.00  1.00
numeric               1.00      1.00     1.00   1.00   0.95  1.00
temporal              1.00      1.00     1.00   1.00   1.00  1.00
unsupported_addition  1.00      1.00     1.00   1.00   1.00  1.00
```

These recalls look high only because at their low F1-optimal threshold the free
detectors flag almost everything as hallucinated (precision collapses to
~0.50), so recall at a single threshold is not the honest lens; AUROC is, and
there the free detectors sit near 0.5. The per-type view still exposes specific
blind spots: `lexical_overlap` catches only a third of the `distractor` errors
(the right-looking value is lifted from elsewhere in the context, so overlap
stays high) and three quarters of the comparison errors. The one crack in a
semantic tier shows here too: `nli_entailment` misses one `numeric` hallucination
(recall 0.95), a computed answer it cannot verify, which the judge gets right.

Per-example **error analysis** (`tether/error_analysis.py`,
`python scripts/error_analysis.py`) makes this concrete: at its threshold the
lexical detector's false positives are almost all paraphrases and correct
reasoning answers (faithful answers it flags for sharing few tokens), and its
misses are the vocabulary-reusing swaps. Both are failures of meaning, not
surface overlap.

## The LLM judge & agreement analysis

The whole point of Tether is to ask *how closely a free detector tracks an
expensive LLM judge*. The judge is the reference detector; agreement with it is
the headline metric. With the judge tier now run (`qwen38-27b`), that question
has a real answer on this sample:

| Detector vs judge | Agreement | Cohen's kappa | Judge-only catches | Cheap-only flags |
|-------------------|-----------|---------------|--------------------|------------------|
| `nli_entailment` | 0.915 | 0.830 | 2 | 9 |
| `lexical_overlap` | 0.592 | 0.176 | 5 | 48 |
| `tfidf_cosine` | 0.523 | 0.032 | 0 | 62 |
| `majority_constant` | 0.508 | 0.000 | 0 | 64 |
| `novel_token` | 0.508 | 0.000 | 0 | 64 |

This is the benchmark's founding question answered, and the answer has two
levels. The free detectors agree with the judge barely above chance (kappa 0.03
to 0.18) and degenerate into flagging almost everything (48 to 64 false alarms
out of 130), so a surface detector is not a usable proxy for the judge on a hard
set. The NLI tier tracks the judge far better (kappa 0.83) but no longer *is* the
judge: its agreement fell from 0.98 on the earlier set to 0.83 here, because it
raises 9 false alarms on correct reasoning answers the judge accepts. So the
cheap semantic tier is a good but imperfect proxy, and the reasoning examples are
precisely where it and the judge part ways.

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
llm_judge          api        1.00  1.000  *
nli_entailment     model      0.30  0.974  *
lexical_overlap    free       0.00  0.538  *
novel_token        free       0.00  0.534
majority_constant  free       0.00  0.500
tfidf_cosine       free       0.00  0.453
```

Now **all three tiers are Pareto-optimal**, a clean staircase: `lexical_overlap`
(free, barely above chance), `nli_entailment` (0.974 at the model tier), and
`llm_judge` (1.000 at the api tier). This is the key change the reasoning batch
produced. On the earlier, easier sets NLI matched the judge exactly, so the paid
tier was dominated and off the frontier; now that NLI falls to 0.974 the judge is
strictly better and **earns its place on the frontier**. The research payoff is
the shape of that staircase — *how much AUROC each step up the cost ladder
actually buys* — and here it is finally a real staircase: free→model buys 0.44,
model→api buys the last 0.026 that only a reasoning model can.

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

On the 130-example sample, judging every answer with an LLM judge at Opus-tier
pricing costs about **$0.136** (~$0.0010/example), the NLI tier about **$0.0036**
of GPU time, and the free tier **$0**. Normalized, that is `free 0.0, model 0.026,
api 1.0` — the measured mid tier is far cheaper than the placeholder 0.3, which is
exactly the kind of correction real cost numbers make to the frontier.

## Quickstart

```bash
pip install -e ".[dev]"
pytest                          # 219 tests, all stdlib
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
  recalibration.py   # isotonic/Platt calibration maps: can the cheap scores become probabilities?
  ensemble.py        # out-of-fold logistic stack of the free detectors: do they combine to win?
  baselines/         # majority, lexical, novel-token, tf-idf, nli, llm-judge
data/sample.jsonl    # bundled labeled dataset (130 examples, runs offline), error-type annotated
scripts/run_benchmark.py   # prints leaderboard + breakdown + frontier
scripts/generate_report.py # regenerates RESULTS.md
scripts/score_detector.py  # score one detector, cache to JSON (HPC job entry point)
out/nli.json         # cached NLI (bart-large-mnli) scores from a GPU run, folded into the report
out/judge.json       # cached LLM-judge (qwen38-27b) scores, folded into the report
PAPER.md                   # working paper draft
RESULTS.md                 # generated results, cited by the paper
COMPUTE.md                 # producing the model/api tiers on a GPU and caching them
tests/               # 219 unit tests
```

## Roadmap

- **Widen the gap the reasoning batch opened.** The 130 examples drive the free
  tier to chance and pulled NLI down to 0.974 while the judge held at 1.000; the
  next set should press that further with harder multi-step numeric reasoning,
  long contexts with distractors, retrieval-style noise, and domain-specific
  claims (with annotated unsupported spans), to see how far the NLI-vs-judge gap
  widens and where the judge itself finally fails.
- **Wire live measured latency/tokens** from real NLI and judge runs into the cost model

_Done:_ 130-example error-type / support-type annotated dataset (surface-overlap
hard plus a reasoning-stress batch) with integrity validation and a difficulty
(lexical-overlap separation) summary; per-category recall breakdown; NLI-entailment
mid-cost baseline and an LLM-judge api tier, both run for real and cached
(`out/nli.json`, `out/judge.json`) and provider-decoupled, with the reasoning
batch opening a significant, Pareto-visible gap between them (judge 1.000 vs NLI
0.974, p < 0.001); a real agreement analysis (agreement rate, Cohen's kappa) of
the cheap detectors vs the judge;
cost/accuracy frontier with Pareto marking; reproducible Markdown report
([RESULTS.md](RESULTS.md)) and a working paper draft ([PAPER.md](PAPER.md));
score caching so the heavy tiers run once on HPC and the analysis stays offline
([COMPUTE.md](COMPUTE.md)); a cost model that prices the frontier in real dollars
per tier; stratified bootstrap CIs on every AUROC and a paired bootstrap test for
leaderboard gaps; calibration analysis (reliability bins, ECE/MCE, Brier) showing
the free detectors are not calibrated probabilities while the semantic tiers are;
k-fold cross-validated operating points that quantify how much the
oracle-threshold F1 overstates the honest held-out F1; post-hoc recalibration
(isotonic and Platt maps, cross-validated) that roughly halves the free
detectors' calibration error; an out-of-fold logistic stack showing the free
signals are redundant and do not combine to beat the best single detector.

## Reproducibility

v0 is deterministic and dependency-free. Later baselines will pin model
versions and seeds, and record per-example scores so every leaderboard number is
traceable to its inputs.

---

**Built by Nishit Patel** | MS Computer Science, Arizona State University
