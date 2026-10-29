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
On a deliberately hard sample of 100 examples, where the mean lexical overlap of
faithful and hallucinated answers differs by just 0.009, every free detector
collapses to near chance: the best reaches only 0.57 AUROC and a TF-IDF baseline
falls to 0.45, below the 0.50 floor, and no free detector's 95% interval excludes
0.5. Both semantic tiers, by contrast, a local NLI model (run on a GPU) and an
LLM judge (a Qwen3 model queried over an OpenAI-compatible gateway, each cached
for offline reproduction), still separate the classes perfectly (AUROC 1.000) and
are well calibrated, beating every free detector by 0.43 to 0.55 AUROC at
p < 0.001. The NLI model reproduces the judge (Cohen's kappa 0.98) while the free
detectors fall to near-chance agreement, and because the local model matches the
judge's quality at far lower cost the paid judge is not Pareto-optimal. The hard
set thus opens a wide, significant gap between the surface and semantic tiers;
that both semantic tiers remain perfect shows the next challenge is a set hard
enough to probe where they fail.

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

The bundled dataset is 100 hand-written examples, balanced 50 faithful and 50
hallucinated, across general-knowledge domains. It is built to be hard for
surface overlap and to probe reasoning: hallucinations swap an entity, number, or
quantifier while reusing the context's wording, or require a reasoning step
(numeric, multi-hop, scope, temporal) to detect, and faithful answers paraphrase
the context or must be inferred from it. The mean lexical overlap is 0.68 for
faithful answers and 0.67 for hallucinated ones, a gap of only 0.009, so surface
overlap carries almost no signal. Every hallucinated example is annotated with an
error type, and every faithful example with a support type:

- **numeric**: a wrong number, or a wrong result of numeric reasoning (an
  aggregation, rate, or scaling).
- **entity_swap**: a wrong entity that reuses the context vocabulary.
- **unsupported_addition**: an extra claim the context does not make.
- **contradiction**: a statement that contradicts the context, often by negation
  or a changed scope word (*only*, *all*).
- **multi_hop**: a wrong conclusion from combining two context facts.
- **temporal**: a wrong ordering of events the context dates or sequences.
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
stratified by class, so each resample keeps 50 faithful and 50 hallucinated
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
| nli_entailment | model | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 1.000 |
| llm_judge | api | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 1.000 |
| lexical_overlap | free | 0.572 | [0.460, 0.688] | 0.706 | 0.558 | 0.960 | 0.600 |
| novel_token | free | 0.562 | [0.443, 0.673] | 0.682 | 0.557 | 0.880 | 0.590 |
| majority_constant | free | 0.500 | [0.500, 0.500] | 0.667 | 0.500 | 1.000 | 0.500 |
| tfidf_cosine | free | 0.447 | [0.340, 0.566] | 0.676 | 0.510 | 1.000 | 0.520 |

The NLI numbers come from a cached run of `facebook/bart-large-mnli` on a GPU;
the judge numbers from a cached run of a Qwen3 model (`qwen38-27b`) served over
an OpenAI-compatible gateway. Both detectors' per-example scores are committed so
their rows are reproducible offline. No free detector's interval excludes 0.5, so
none significantly beats the majority baseline. Both semantic tiers beat every
free detector decisively; the paired bootstrap of the best detector
(`nli_entailment`) against each free one:

| nli_entailment vs | AUROC diff | 95% CI | p-value | Examples scored differently |
| --- | --- | --- | --- | --- |
| lexical_overlap | +0.428 | [+0.312, +0.540] | < 0.001 | 100/100 |
| novel_token | +0.438 | [+0.327, +0.557] | < 0.001 | 100/100 |
| tfidf_cosine | +0.553 | [+0.434, +0.660] | < 0.001 | 100/100 |
| majority_constant | +0.500 | [+0.500, +0.500] | < 0.001 | 100/100 |

A logistic combiner stacked on the free detectors and evaluated out-of-fold
(5-fold, seed 0) does not clear the free-tier ceiling: it reaches AUROC 0.647
(95% CI [0.540, 0.756]) versus lexical_overlap's 0.572, a difference of +0.074
(95% CI [-0.066, +0.231], p = 0.333, not significant).

Calibration of the faithfulness scores (ECE/MCE/Brier, lower is better):

| Detector | AUROC | ECE | ECE 95% CI | MCE | Brier |
| --- | --- | --- | --- | --- | --- |
| nli_entailment | 1.000 | 0.029 | [0.016, 0.044] | 0.370 | 0.006 |
| llm_judge | 1.000 | 0.024 | [0.014, 0.042] | 0.720 | 0.006 |
| majority_constant | 0.500 | 0.000 | [0.000, 0.000] | 0.000 | 0.250 |
| tfidf_cosine | 0.447 | 0.211 | [0.157, 0.320] | 0.919 | 0.319 |
| lexical_overlap | 0.572 | 0.321 | [0.262, 0.404] | 0.972 | 0.333 |
| novel_token | 0.562 | 0.378 | [0.314, 0.453] | 0.972 | 0.362 |

Raw vs post-calibration ECE/Brier, with the calibration map fit out-of-fold
(5-fold, seed 0) by isotonic regression and by Platt scaling:

| Detector | Raw ECE | Isotonic ECE | Platt ECE | Raw Brier | Isotonic Brier |
| --- | --- | --- | --- | --- | --- |
| majority_constant | 0.000 | 0.000 | 0.000 | 0.250 | 0.250 |
| lexical_overlap | 0.321 | 0.118 | 0.090 | 0.333 | 0.242 |
| novel_token | 0.378 | 0.117 | 0.092 | 0.362 | 0.262 |
| tfidf_cosine | 0.211 | 0.076 | 0.008 | 0.319 | 0.256 |
| nli_entailment | 0.029 | 0.007 | 0.000 | 0.006 | 0.001 |
| llm_judge | 0.024 | 0.007 | 0.010 | 0.006 | 0.004 |

Oracle vs 5-fold cross-validated F1 (seed 0), with the optimism the oracle
threshold hides:

| Detector | Oracle F1 | CV F1 | Optimism | CV precision | CV recall |
| --- | --- | --- | --- | --- | --- |
| nli_entailment | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| llm_judge | 1.000 | 0.990 | +0.010 | 0.980 | 1.000 |
| lexical_overlap | 0.706 | 0.696 | +0.010 | 0.553 | 0.940 |
| majority_constant | 0.667 | 0.667 | +0.000 | 0.500 | 1.000 |
| tfidf_cosine | 0.676 | 0.667 | +0.009 | 0.505 | 0.980 |
| novel_token | 0.682 | 0.638 | +0.044 | 0.500 | 0.880 |

Agreement of each detector with the judge on the binary hallucination call
(cheap detectors at their F1-optimal threshold, judge at 0.5), ordered by kappa:

| Detector vs judge | Agreement | Cohen's kappa | Judge-only catches | Cheap-only flags |
| --- | --- | --- | --- | --- |
| nli_entailment | 0.990 | 0.980 | 1 | 0 |
| lexical_overlap | 0.610 | 0.209 | 2 | 37 |
| novel_token | 0.600 | 0.191 | 6 | 34 |
| tfidf_cosine | 0.530 | 0.042 | 0 | 47 |
| majority_constant | 0.510 | 0.000 | 0 | 49 |

Detection recall by error type (at each detector's F1-optimal threshold):

| Error type | lexical_overlap | novel_token | tfidf_cosine | nli_entailment | llm_judge |
| --- | --- | --- | --- | --- | --- |
| contradiction | 0.89 | 0.89 | 1.00 | 1.00 | 1.00 |
| entity_swap | 0.92 | 0.92 | 1.00 | 1.00 | 1.00 |
| multi_hop | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 |
| numeric | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| temporal | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| unsupported_addition | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

(The free detectors' high per-type recalls are an artifact of a low threshold
that flags almost everything, as the collapsed precision in the leaderboard
shows; AUROC is the honest view. `majority_constant` is omitted from the table
for the same reason.)

## 7. Analysis

Five findings stand out.

First, on a hard dataset the free detectors collapse to chance. The best,
`lexical_overlap`, reaches only 0.572 AUROC, `novel_token` 0.562, and
`tfidf_cosine` 0.447, below the 0.50 floor. Faithful and hallucinated answers now
overlap the context by almost the same amount (0.68 versus 0.67, a 0.009 gap), so
surface token signals carry essentially no information. Crucially, every free
detector's 95% AUROC interval includes 0.5, so none significantly beats the
majority baseline. The free detectors' healthy-looking F1 (around 0.68 to 0.71)
is a threshold artifact: at a low threshold they flag almost everything, so
recall stays high and precision collapses to about 0.50 to 0.55. AUROC, which is
threshold-free, is the honest measure, and there the free tier is indistinguishable
from chance.

Second, the failures are failures of meaning, which the per-type view exposes.
`novel_token` misses every `temporal` hallucination (recall 0.00) and a quarter
of the `multi_hop` ones, because an answer assembled from words already in the
context looks grounded to a token-overlap detector even when its logic or
ordering is wrong. The free detectors do not fail by letting subtle
hallucinations through so much as by firing on everything, including faithful
paraphrases and inferences. These are precisely the cases a semantic model is
built to handle.

Third, ranking quality and calibration are different axes, and the free detectors
lose on both. `majority_constant` has ECE 0.000, the best in the table, only
because a constant 0.5 matches the base rate while carrying no signal. The free
detectors that do vary are badly miscalibrated and worse on the hard set:
`novel_token` reaches ECE 0.378. The miscalibration is a property of the raw
scores and is largely fixable, an out-of-fold calibration map cuts the free
detectors' ECE by roughly two thirds (Platt takes `tfidf_cosine` from 0.211 to
0.008, and both lexical detectors from about 0.33 and 0.38 down to about 0.09 to
0.12), but a well-calibrated free score is still a near-chance one: calibration
makes the number honest, not useful. The semantic tiers win both axes, ranking
perfectly and staying well calibrated (ECE 0.024 to 0.029).

Fourth, the reported F1 is only mildly optimistic at this sample size, and the
free signals do not combine. Cross-validation lowers F1 by under 0.01 for most
detectors and 0.044 for `novel_token`, so the oracle threshold is not badly
overfit, though the free detectors' operating-point F1 flatters them for the
threshold reason above. A logistic stack of the free detectors, evaluated
out-of-fold, does not clear the free-tier ceiling (0.647 versus 0.572,
p = 0.333): the surface signals are redundant, one shared axis of information
rather than complementary ones, so blending them cannot reach the semantic tier.
Doubling the dataset from 50 to 100 examples also roughly halved the optimism
gaps and narrowed every confidence interval, a direct gain in statistical power.

Fifth, the semantic tiers earn their cost emphatically, and the frontier and
agreement analyses make it sharp. Both the NLI model and the LLM judge separate
the classes perfectly (AUROC, F1, accuracy all 1.000), beating every free
detector by 0.43 to 0.55 AUROC with p < 0.001, and both are well calibrated so
their scores read as confidences. The NLI model reproduces the judge (Cohen's
kappa 0.98), while the free detectors fall to near-chance agreement (kappa 0.19
to 0.21, down from about 0.45 on the easier 50-example set) and raise 34 to 49
false alarms out of 100 that the judge rejects: on a hard set a surface detector
degenerates into flagging almost everything. On the cost/accuracy frontier
`lexical_overlap` (free, though barely above chance) and `nli_entailment` are
Pareto-optimal; the single free-to-model step now buys the jump from 0.57 to
1.00 AUROC. The paid `llm_judge` is pointedly *not* on the frontier, because it
matches the NLI model's perfect quality at higher cost, so a practitioner would
never pay for it on this data. The one caveat is that both semantic tiers remain
perfect: the hard set has opened a wide, significant gap above the free tier but
does not yet probe where a semantic model itself fails, which is the next
challenge rather than a flaw in the present result.

## 8. Limitations

The dataset is modest in size (100 examples) and hand-written, which keeps it
reproducible but limits domain coverage. Doubling it from 50 narrowed the
bootstrap intervals, but the free detectors' 95% AUROC intervals are still about
0.11 to 0.23 wide; that no longer matters for ranking them, since they all sit at
chance, but it does mean the dataset cannot resolve small differences. The error
types are a coarse taxonomy. The clearest limitation is now the dataset's
difficulty ceiling for the *semantic* tiers: both the NLI model and the LLM judge
reach a perfect AUROC, so the sample no longer discriminates among capable
semantic models and the measured frontier saturates above the free tier.
The NLI and judge scores are produced once (on a GPU and over a hosted gateway
respectively) and cached rather than recomputed in continuous integration; the
committed JSON makes those rows reproducible offline, but the live paths are not
exercised in CI. Absolute cost is represented by relative per-tier scores rather
than measured dollars, which is why the judge's dominance by the NLI model is a
statement about quality-per-tier rather than a precise dollar comparison.

## 9. Future Work

The most pressing next step is a dataset hard enough to challenge the semantic
tiers, not just surface overlap: multi-step numeric reasoning, long contexts with
distractors, retrieval-style noise, and domain-specific claims, with annotated
unsupported spans. The current 100 examples already drive the free tier to
chance, so the open question is where an entailment model or an LLM judge itself
fails, and whether the two diverge there. Beyond that: measured dollar and
latency costs for the model and API tiers (which would turn the "NLI dominates
the judge" result from a per-tier statement into a priced one), and more
mid-cost-tier baselines (embedding similarity, smaller NLI models).

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
