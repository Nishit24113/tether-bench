# Tether: A Reproducible Benchmark for the Cost/Accuracy Frontier of RAG Faithfulness Detection

**Nishit Patel**
Arizona State University

> Working draft. All numbers in this paper are produced by
> `python scripts/generate_report.py` from the bundled dataset, so every figure
> is reproducible from a single command.

## Abstract

Retrieval-augmented generation (RAG) systems can produce answers that are not
supported by their retrieved context. Detecting these hallucinations is
important in production, and a strong large language model (LLM) judge can do it
well, but running such a judge on every answer is slow and expensive. In
practice teams want cheaper detectors. We present Tether, a small, fully
reproducible benchmark that measures how well low-cost faithfulness detectors
separate grounded answers from hallucinations, and how closely they approximate
an LLM judge. Tether contributes an error-type stratified dataset, a set of
baselines spanning three cost tiers (free lexical heuristics, a local NLI model,
and a paid LLM judge), agreement metrics against the judge, a cost/accuracy
frontier that makes the trade-off explicit, bootstrap confidence intervals with
paired significance tests so that small leaderboard gaps are not over-read, a
calibration analysis of whether the scores can be read as probabilities,
cross-validated operating points that measure how optimistic the reported F1 is,
and post-hoc calibration maps that recover usable probabilities from the cheap
scores.
On a deliberately hard sample of 130 examples, where the mean lexical overlap of
faithful and hallucinated answers differs by just 0.005, every free detector
sits at chance: the best reaches only 0.54 AUROC, a TF-IDF baseline falls to 0.45
below the floor, and no free detector's 95% interval excludes 0.5. The two
semantic tiers, a local NLI model (run on a GPU) and an LLM judge (a Qwen3 model
queried over an OpenAI-compatible gateway, each cached for offline reproduction),
beat every free detector by 0.46 to 0.55 AUROC at p < 0.001. The headline result
is that a reasoning-stress subset (multi-step numeric, long-context distractors,
counting) finally separates the two semantic tiers: the judge stays perfect at
1.000 AUROC while the NLI model drops to 0.974, a gap that is significant (paired
bootstrap p < 0.001), and their agreement falls from near-identical (Cohen's
kappa 0.98 on an easier set) to 0.83, because the cheap entailment model wrongly
rejects correct computed answers it cannot verify. The cheap tier therefore
tracks the judge well on simple faithfulness but measurably falls behind on
reasoning, which is exactly where paying for the judge begins to be worth it, and
the judge now earns a place on the cost/accuracy frontier that it did not hold
when the two tiers were tied.

## 1. Introduction

A RAG system retrieves context and asks a model to answer from it. When the
answer asserts something the context does not support, the system has
hallucinated. Catching that automatically is valuable, and the most reliable
automatic approach is to ask a strong LLM to judge faithfulness. That judge is
accurate but costly to run at scale, so teams reach for cheaper proxies such as
lexical overlap or embedding similarity without a clear picture of what accuracy
they give up.

Tether makes that picture concrete. It asks one question: how closely does a
cheap faithfulness detector track an LLM judge, and what does each step up the
cost ladder buy? The benchmark is deliberately small and dependency-light so
that its core baselines and metrics run offline and deterministically, which
makes the results easy to reproduce and extend.

## 2. Task

Each example is a triple of (question, context, answer) with a gold label:
`faithful` if every claim in the answer is supported by the context, and
`hallucinated` otherwise. A detector maps a triple to a faithfulness score in
[0, 1], where higher means more supported. The harness converts scores to
hallucination predictions with a threshold and reports precision, recall, F1,
accuracy, and AUROC. The positive class is the hallucinated answer, since that
is what we want to catch.

## 3. Dataset

The bundled dataset is 130 hand-written examples, balanced 65 faithful and 65
hallucinated, across general-knowledge domains. It is built to be hard for
surface overlap and to probe reasoning: hallucinations swap an entity, number, or
quantifier while reusing the context's wording, or require a reasoning step
(numeric, multi-hop, scope, temporal) to detect, and faithful answers paraphrase
the context or must be inferred from it. A 30-example reasoning-stress subset
targets the failure modes of a cheap entailment model rather than of surface
overlap: multi-step numeric reasoning, long-context distractors, and
counting/comparison. The mean lexical overlap is 0.67 for faithful answers and
0.66 for hallucinated ones, a gap of only 0.005, so surface overlap carries no
signal. Every hallucinated example is annotated with an error type, and every
faithful example with a support type:

- **numeric**: a wrong number, or a wrong result of numeric reasoning (an
  aggregation, rate, or scaling).
- **entity_swap**: a wrong entity that reuses the context vocabulary.
- **unsupported_addition**: an extra claim the context does not make.
- **contradiction**: a statement that contradicts the context, often by negation
  or a changed scope word (*only*, *all*).
- **multi_hop**: a wrong conclusion from combining two context facts.
- **temporal**: a wrong ordering of events the context dates or sequences.
- **distractor**: a wrong value lifted from elsewhere in a multi-fact context.
- **comparison**: a wrong count or comparison over the context.
- **direct** / **paraphrase** / **multi_hop** / **inference**: faithful answers
  that copy, restate, combine, or are entailed by the context.

The dataset is curated so it ships in the repository and the benchmark runs with
no download. The error-type annotations are what let the benchmark report where
detectors fail rather than only how often.

## 4. Detectors

Tether groups detectors into three cost tiers:

- **free**: `majority_constant` (a constant-score floor), `lexical_overlap`
  (fraction of answer content tokens present in the context), `novel_token`
  (penalizes answer tokens absent from both context and question), and
  `tfidf_cosine` (TF-IDF cosine between answer and context).
- **model**: `nli_entailment`, a cross-encoder natural-language-inference model
  scoring whether the context entails the answer.
- **api**: `llm_judge`, a strong model (default Claude) that judges faithfulness
  and serves as the reference detector.

The NLI and LLM detectors are provider-decoupled. Each takes an injected scoring
function, so the benchmark runs offline with deterministic fakes and the heavy
dependencies stay optional. For the results below both heavy tiers are run for
real. `facebook/bart-large-mnli` is scored once on a GPU for the model tier, and
the judge tier is a Qwen3 model (`qwen38-27b`) queried over an OpenAI-compatible
gateway using only the standard library. Both detectors' per-example scores are
cached to JSON and committed, so the model-tier and api-tier rows are
reproducible with no model download and no API call. The judge endpoint and
credentials are supplied through the environment, so nothing provider-specific
is embedded in the benchmark.

## 5. Metrics

AUROC is the primary metric because it is threshold-free and reflects a
detector's intrinsic ability to separate the two classes. We also report F1 at
each detector's F1-optimal threshold as an operating-point view. That threshold
is tuned on the same data it is scored on, so the F1 is an oracle operating
point and is optimistic. To measure the optimism we additionally report a
cross-validated F1: with 5-fold stratified cross-validation the threshold is
chosen on the training folds and applied to the held-out fold, so every example
is scored by a threshold it did not help select, and we pool the held-out
predictions into one honest F1. To answer the
central question we compute agreement between a cheap detector and the judge on
the binary hallucination decision, using both raw agreement rate and Cohen's
kappa, which corrects for chance agreement.

Because the dataset is small, every AUROC is reported with a 95% confidence
interval from a percentile bootstrap (2000 resamples, fixed seed). Resampling is
stratified by class, so each resample keeps 65 faithful and 65 hallucinated
examples and AUROC is always defined. To decide whether a leaderboard gap is
real, we run a paired bootstrap: both detectors are scored on the same
resamples, the interval is placed on their AUROC difference, and the two-sided
p-value is twice the smaller share of resampled differences on either side of
zero. Resamples where the two detectors tie count toward both sides, which keeps
the test conservative. We also report how many examples the two detectors score
differently, since a paired comparison can only draw evidence from those.

AUROC and the paired test both concern ranking. We additionally measure
calibration, which asks whether a faithfulness score of p corresponds to an
answer that is faithful about p of the time. Reading the score as a predicted
probability of being faithful, we report the expected calibration error (ECE,
the bin-count-weighted mean gap between mean score and observed faithful rate
over ten equal-width bins), the maximum single-bin gap (MCE), and the Brier
score (mean squared error of the scores as probabilities). ECE carries the same
stratified bootstrap interval as the AUROC numbers. Because miscalibration is a
property of the raw scores rather than of the detector, we also fit post-hoc
calibration maps (isotonic regression and Platt scaling) out-of-fold and report
the raw versus calibrated ECE and Brier, which measures how much of the gap a
practitioner could recover.

## 6. Results

Leaderboard on the bundled sample (ranked by AUROC, with stratified bootstrap
95% CIs):

| Detector | Cost | AUROC | AUROC 95% CI | F1 | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llm_judge | api | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 1.000 |
| nli_entailment | model | 0.974 | [0.949, 0.991] | 0.927 | 0.877 | 0.985 | 0.923 |
| lexical_overlap | free | 0.538 | [0.440, 0.640] | 0.690 | 0.550 | 0.923 | 0.585 |
| novel_token | free | 0.534 | [0.436, 0.641] | 0.667 | 0.500 | 1.000 | 0.500 |
| majority_constant | free | 0.500 | [0.500, 0.500] | 0.667 | 0.500 | 1.000 | 0.500 |
| tfidf_cosine | free | 0.453 | [0.356, 0.554] | 0.674 | 0.508 | 1.000 | 0.515 |

The NLI numbers come from a cached run of `facebook/bart-large-mnli` on a GPU;
the judge numbers from a cached run of a Qwen3 model (`qwen38-27b`) served over
an OpenAI-compatible gateway. Both detectors' per-example scores are committed so
their rows are reproducible offline. No free detector's interval excludes 0.5, so
none significantly beats the majority baseline. The best detector is now the
judge, and the key result is in its paired bootstrap against each other detector:
it significantly beats even the NLI tier.

| llm_judge vs | AUROC diff | 95% CI | p-value | Examples scored differently |
| --- | --- | --- | --- | --- |
| nli_entailment | +0.026 | [+0.009, +0.051] | < 0.001 | 130/130 |
| lexical_overlap | +0.462 | [+0.359, +0.560] | < 0.001 | 123/130 |
| novel_token | +0.466 | [+0.359, +0.564] | < 0.001 | 122/130 |
| majority_constant | +0.500 | [+0.500, +0.500] | < 0.001 | 130/130 |
| tfidf_cosine | +0.547 | [+0.446, +0.644] | < 0.001 | 129/130 |

A logistic combiner stacked on the free detectors and evaluated out-of-fold
(5-fold, seed 0) does not clear the free-tier ceiling: it reaches AUROC 0.529
(95% CI [0.429, 0.628]) versus lexical_overlap's 0.538, a difference of -0.009
(95% CI [-0.148, +0.127], p = 0.876, not significant).

Calibration of the faithfulness scores (ECE/MCE/Brier, lower is better):

| Detector | AUROC | ECE | ECE 95% CI | MCE | Brier |
| --- | --- | --- | --- | --- | --- |
| llm_judge | 1.000 | 0.022 | [0.014, 0.035] | 0.720 | 0.005 |
| nli_entailment | 0.974 | 0.087 | [0.057, 0.136] | 0.563 | 0.071 |
| majority_constant | 0.500 | 0.000 | [0.000, 0.000] | 0.000 | 0.250 |
| tfidf_cosine | 0.453 | 0.231 | [0.162, 0.321] | 0.936 | 0.315 |
| lexical_overlap | 0.538 | 0.296 | [0.239, 0.372] | 0.972 | 0.334 |
| novel_token | 0.534 | 0.356 | [0.300, 0.421] | 0.972 | 0.364 |

Raw vs post-calibration ECE/Brier, with the calibration map fit out-of-fold
(5-fold, seed 0) by isotonic regression and by Platt scaling:

| Detector | Raw ECE | Isotonic ECE | Platt ECE | Raw Brier | Isotonic Brier |
| --- | --- | --- | --- | --- | --- |
| majority_constant | 0.000 | 0.000 | 0.000 | 0.250 | 0.250 |
| lexical_overlap | 0.296 | 0.065 | 0.150 | 0.334 | 0.248 |
| novel_token | 0.356 | 0.082 | 0.136 | 0.364 | 0.268 |
| tfidf_cosine | 0.231 | 0.112 | 0.073 | 0.315 | 0.261 |
| nli_entailment | 0.087 | 0.058 | 0.054 | 0.071 | 0.062 |
| llm_judge | 0.022 | 0.006 | 0.008 | 0.005 | 0.004 |

Oracle vs 5-fold cross-validated F1 (seed 0), with the optimism the oracle
threshold hides:

| Detector | Oracle F1 | CV F1 | Optimism | CV precision | CV recall |
| --- | --- | --- | --- | --- | --- |
| llm_judge | 1.000 | 0.992 | +0.008 | 0.985 | 1.000 |
| nli_entailment | 0.927 | 0.927 | +0.000 | 0.877 | 0.985 |
| lexical_overlap | 0.690 | 0.674 | +0.015 | 0.542 | 0.892 |
| majority_constant | 0.667 | 0.667 | +0.000 | 0.500 | 1.000 |
| tfidf_cosine | 0.674 | 0.667 | +0.007 | 0.504 | 0.985 |
| novel_token | 0.667 | 0.631 | +0.036 | 0.484 | 0.908 |

Agreement of each detector with the judge on the binary hallucination call
(cheap detectors at their F1-optimal threshold, judge at 0.5), ordered by kappa:

| Detector vs judge | Agreement | Cohen's kappa | Judge-only catches | Cheap-only flags |
| --- | --- | --- | --- | --- |
| nli_entailment | 0.915 | 0.830 | 2 | 9 |
| lexical_overlap | 0.592 | 0.176 | 5 | 48 |
| tfidf_cosine | 0.523 | 0.032 | 0 | 62 |
| majority_constant | 0.508 | 0.000 | 0 | 64 |
| novel_token | 0.508 | 0.000 | 0 | 64 |

Detection recall by error type (at each detector's F1-optimal threshold):

| Error type | lexical_overlap | novel_token | tfidf_cosine | nli_entailment | llm_judge |
| --- | --- | --- | --- | --- | --- |
| comparison | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 |
| contradiction | 0.89 | 1.00 | 1.00 | 1.00 | 1.00 |
| distractor | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 |
| entity_swap | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| multi_hop | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| numeric | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 |
| temporal | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| unsupported_addition | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

(The free detectors' high per-type recalls are an artifact of a low threshold
that flags almost everything, as the collapsed precision in the leaderboard
shows; AUROC is the honest view. The one informative crack is `nli_entailment`
missing a single `numeric` hallucination, a computed answer it cannot verify,
which the judge gets right.)

## 7. Analysis

Five findings stand out.

First, on this dataset the free detectors sit at chance. The best,
`lexical_overlap`, reaches only 0.538 AUROC, `novel_token` 0.534, and
`tfidf_cosine` 0.453, below the 0.50 floor. Faithful and hallucinated answers
overlap the context by almost the same amount (0.67 versus 0.66, a 0.005 gap), so
surface token signals carry no information, and every free detector's 95% AUROC
interval includes 0.5, so none significantly beats the majority baseline. Their
healthy-looking F1 (around 0.67) is a threshold artifact: at a low threshold they
flag almost everything, recall stays high, and precision collapses to about 0.50.
AUROC, which is threshold-free, is the honest measure, and there the free tier is
indistinguishable from chance.

Second, ranking quality and calibration track together, and the free detectors
lose on both. `majority_constant` has ECE 0.000 only because a constant 0.5
matches the base rate while carrying no signal. The free detectors that vary are
badly miscalibrated (`novel_token` ECE 0.356). This is a property of the raw
scores and is largely fixable: an out-of-fold calibration map cuts the free
detectors' ECE by roughly two thirds (isotonic takes `lexical_overlap` from 0.296
to 0.065 and `novel_token` from 0.356 to 0.082), but a well-calibrated free score
is still a near-chance one, so calibration makes the number honest, not useful.
The reported F1 is only mildly optimistic (cross-validation lowers it by under
0.015 for all but `novel_token` at 0.036), and a logistic stack of the free
detectors evaluated out-of-fold does not clear the ceiling (0.529 versus 0.538,
p = 0.876): the surface signals are redundant, so blending them cannot reach the
semantic tier.

Third, the main event: the reasoning-stress subset separates the two semantic
tiers, which had been tied at a perfect score on easier data. The LLM judge
still scores 1.000, but the NLI model drops to 0.974, and the paired bootstrap
makes the gap significant (+0.026 AUROC, 95% CI [+0.009, +0.051], p < 0.001).
The per-type breakdown and the agreement analysis locate the cause: the NLI
model, a single-pass entailment classifier, cannot verify a computed or
multi-step answer, so it wrongly rejects correct faithful answers (its agreement
with the judge falls from Cohen's kappa 0.98 on the easier set to 0.83 here,
driven by nine false alarms on correct reasoning answers the judge accepts). Its
calibration degrades in step, ECE rising from about 0.03 to 0.087, worse than
the judge's 0.022. The cheap semantic tier is thus a strong but imperfect proxy
for the judge, and reasoning is exactly where the two part ways.

Fourth, the cost/accuracy frontier now has three Pareto-optimal rungs, which is
the practical consequence of that divergence. `lexical_overlap` (free, at
chance), `nli_entailment` (0.974 at the model tier), and `llm_judge` (1.000 at
the api tier) are all Pareto-optimal: the free-to-model step buys 0.44 AUROC and
the model-to-api step buys the final 0.026 that only a reasoning model supplies.
On the earlier, easier sets the NLI model matched the judge exactly, so the paid
tier was dominated and off the frontier; the reasoning batch is what puts the
judge back on it. The free detectors meanwhile collapse to near-chance agreement
with the judge (kappa 0.03 to 0.18) and flag 48 to 64 of 130 answers the judge
calls faithful, so a surface detector is not a usable proxy for the judge on a
hard set at all.

The one caveat is that the judge itself remains perfect, so the dataset opens a
significant gap between all three tiers but does not yet probe where an LLM judge
fails; that is the next challenge rather than a flaw in the present result.

## 8. Limitations

The dataset is modest in size (130 examples) and hand-written, which keeps it
reproducible but limits domain coverage. Growing it to 130 narrowed the bootstrap
intervals, but the free detectors' 95% AUROC intervals are still about 0.2 wide;
that no longer matters for ranking them, since they all sit at chance, but the
dataset cannot resolve small differences among weak detectors. The error types
are a coarse taxonomy. The clearest limitation is now the difficulty ceiling for
the *judge*: the reasoning batch pulled the NLI model down to 0.974, but the LLM
judge still reaches a perfect AUROC, so the sample does not yet probe where a
strong judge fails. The NLI and judge scores are produced once (on a GPU and over
a hosted gateway respectively) and cached rather than recomputed in continuous
integration; the committed JSON makes those rows reproducible offline, but the
live paths are not exercised in CI. Absolute cost is represented by relative
per-tier scores rather than measured dollars, so the judge-over-NLI result is a
statement about quality-per-tier rather than a precise dollar comparison.

## 9. Future Work

The most pressing next step is to press the gap the reasoning batch opened: more
and harder reasoning-stress examples (deeper multi-step numeric chains, longer
contexts with more distractors, retrieval-style noise, domain-specific claims,
with annotated unsupported spans) to widen the measured NLI-versus-judge gap and,
eventually, to find where the LLM judge itself fails. Beyond that: measured dollar
and latency costs for the model and API tiers (which would turn the per-tier
"judge beats NLI" result into a priced one), and more mid-cost-tier baselines
(embedding similarity, smaller or larger NLI models) to see whether a stronger
cheap model closes the gap to the judge.

## 10. Reproducibility

The free-tier results, metrics, frontier, bootstrap statistics, and calibration
analysis are pure-stdlib and deterministic. The bootstrap is seeded, so every
interval and p-value is identical across runs. The model-tier (NLI) and api-tier
(judge) scores are computed once and cached to `out/nli.json` and
`out/judge.json`, which are committed, so `python scripts/generate_report.py`
folds them into `RESULTS.md` with no model download and no API call, and the
model and api numbers are reproducible offline like the rest. The
test suite (`pytest`) covers the metrics, baselines, breakdown, frontier,
agreement, bootstrap statistics, calibration, post-hoc recalibration,
cross-validated operating points, free-detector stacking, and report generation
offline.
