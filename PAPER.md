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
calibration analysis of whether the scores can be read as probabilities, and
cross-validated operating points that measure how optimistic the reported F1 is.
On a deliberately hard sample, where hallucinations reuse most of the context
vocabulary and faithful answers paraphrase with few shared tokens, the free
detectors fall well short: the best reaches only 0.78 AUROC (95% CI 0.64 to
0.90) and a TF-IDF baseline drops to 0.60, barely above the 0.50 floor. A paired
bootstrap shows the two best free detectors are statistically tied, while both
significantly outperform TF-IDF and the floor. A calibration analysis adds a
caution: the detectors that rank best are the worst calibrated, so their raw
scores order answers well but cannot be read as probabilities. The mean lexical overlap separates faithful from hallucinated answers
by just 0.16, so surface overlap is not enough, which is where a semantic model
or an LLM judge earns its cost.

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

The bundled dataset is 50 hand-written examples, balanced 25 faithful and 25
hallucinated, across general-knowledge domains. It is built to be hard: many
hallucinations swap a single entity or number while reusing the rest of the
context's wording, and many faithful answers paraphrase the context with few
shared tokens. The mean lexical overlap is 0.81 for faithful answers and 0.65 for
hallucinated ones, a gap of only 0.16. Every hallucinated example is annotated
with an error type, and every faithful example with a support type:

- **numeric**: a wrong number, date, or count.
- **entity_swap**: a wrong entity that reuses the context vocabulary.
- **unsupported_addition**: an extra claim the context does not make.
- **contradiction**: a statement that directly contradicts the context.
- **direct** / **paraphrase**: faithful answers that copy or restate the context.

The dataset is intentionally small and curated so it ships in the repository and
the benchmark runs with no download. The error-type annotations are what let the
benchmark report where detectors fail rather than only how often.

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
dependencies stay optional.

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
stratified by class, so each resample keeps 25 faithful and 25 hallucinated
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
stratified bootstrap interval as the AUROC numbers.

## 6. Results

Leaderboard on the bundled sample (ranked by AUROC, with stratified bootstrap
95% CIs):

| Detector | Cost | AUROC | AUROC 95% CI | F1 | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| novel_token | free | 0.779 | [0.635, 0.902] | 0.774 | 0.649 | 0.960 | 0.720 |
| lexical_overlap | free | 0.763 | [0.614, 0.890] | 0.762 | 0.632 | 0.960 | 0.700 |
| tfidf_cosine | free | 0.603 | [0.435, 0.758] | 0.676 | 0.510 | 1.000 | 0.520 |
| majority_constant | free | 0.500 | [0.500, 0.500] | 0.667 | 0.500 | 1.000 | 0.500 |

Paired bootstrap of the best detector against each of the others:

| novel_token vs | AUROC diff | 95% CI | p-value | Significant | Examples scored differently |
| --- | --- | --- | --- | --- | --- |
| lexical_overlap | +0.016 | [-0.013, +0.070] | 0.776 | no | 3/50 |
| tfidf_cosine | +0.176 | [+0.040, +0.321] | 0.007 | yes | 49/50 |
| majority_constant | +0.279 | [+0.135, +0.402] | < 0.001 | yes | 48/50 |

Calibration of the faithfulness scores (ECE/MCE/Brier, lower is better):

| Detector | AUROC | ECE | ECE 95% CI | MCE | Brier |
| --- | --- | --- | --- | --- | --- |
| majority_constant | 0.500 | 0.000 | [0.000, 0.000] | 0.000 | 0.250 |
| tfidf_cosine | 0.603 | 0.187 | [0.135, 0.344] | 0.928 | 0.265 |
| lexical_overlap | 0.763 | 0.304 | [0.242, 0.391] | 1.000 | 0.281 |
| novel_token | 0.779 | 0.321 | [0.258, 0.399] | 1.000 | 0.288 |

Oracle vs 5-fold cross-validated F1 (seed 0), with the optimism the oracle
threshold hides:

| Detector | Oracle F1 | CV F1 | Optimism | CV precision | CV recall |
| --- | --- | --- | --- | --- | --- |
| novel_token | 0.774 | 0.733 | +0.041 | 0.629 | 0.880 |
| lexical_overlap | 0.762 | 0.721 | +0.041 | 0.611 | 0.880 |
| majority_constant | 0.667 | 0.667 | +0.000 | 0.500 | 1.000 |
| tfidf_cosine | 0.676 | 0.629 | +0.047 | 0.489 | 0.880 |

Detection recall by error type (at each detector's F1-optimal threshold):

| Error type | lexical_overlap | novel_token | tfidf_cosine |
| --- | --- | --- | --- |
| contradiction | 1.00 | 1.00 | 1.00 |
| entity_swap | 0.86 | 0.86 | 1.00 |
| numeric | 1.00 | 1.00 | 1.00 |
| unsupported_addition | 1.00 | 1.00 | 1.00 |

(`majority_constant` is omitted from the breakdown discussion because it flags
every answer as hallucinated, so its per-type recall is trivially 1.00 while its
AUROC is 0.50.)

## 7. Analysis

Five findings stand out.

First, on a hard dataset the free detectors are not enough. The best, `novel_token`,
reaches only 0.78 AUROC, and `tfidf_cosine` falls to 0.60, barely above the 0.50
floor. Because faithful and hallucinated answers overlap the context by similar
amounts (0.81 versus 0.65, a 0.16 gap), surface token signals no longer separate
the classes. The high per-type recalls in the breakdown are achieved only by
pushing the threshold down and flagging almost everything, which collapses
precision to 0.51-0.65; AUROC, which is threshold-free, is the honest measure and
it is where the free detectors visibly degrade.

Second, the hardest errors are the ones that reuse the context's words.
`entity_swap` has the lowest recall for the token-overlap detectors, since
swapping a single entity leaves most tokens shared. These vocabulary-preserving
errors, together with the overall AUROC drop, are exactly the cases a semantic
model (NLI) or an LLM judge should catch, which is the argument for paying for a
higher tier.

Per-example error analysis makes the failure modes concrete. At its F1-optimal
threshold the lexical detector's false positives are dominated by paraphrases: it
flags faithful answers such as "They teach the body's defenses to identify and
combat particular germs" because they restate the context with few shared tokens.
Its false negatives are the vocabulary-reusing swaps, for example answering that
"the capital of Australia is Sydney" from a context that names Canberra. Both
failure modes are about meaning rather than surface words, which is precisely what
the higher tiers are meant to add.

Third, the ranking at the top of the free tier cannot be resolved at this sample
size. `novel_token` leads `lexical_overlap` by 0.016 AUROC, but the paired 95%
interval on that gap is [-0.013, +0.070] and p = 0.776. The two detectors give
identical scores on 47 of the 50 examples, because they differ only when an
answer reuses a question word that the context lacks, so the entire gap rests on
three answers. In contrast, `novel_token` beats `tfidf_cosine` by 0.176
(p = 0.007) and the constant floor by 0.279 (p < 0.001). The single-detector
intervals are wide too, 0.27 to 0.32 AUROC for each non-constant detector, so a
claim that one cheap detector beats another by a few points needs a larger
dataset than this one.

Fourth, ranking quality and calibration are different axes, and neither implies
the other here. `majority_constant` has an ECE of 0.000, the best in the table,
because a constant 0.5 on a balanced set matches the base rate exactly, yet its
AUROC is 0.5 and it carries no signal. The detectors that do rank well are the
worst calibrated: `novel_token` has an ECE of 0.321, and in its 0.6 to 0.7 score
band only about 14% of answers are actually faithful rather than the roughly 65%
the score implies. The scores are therefore usable as an ordering but not as
probabilities; a deployment that thresholds on a raw score and treats it as a
confidence would be badly miscalibrated and would first need a calibration map
(for example Platt scaling or isotonic regression) fit on a held-out split.
`tfidf_cosine` has the lowest point ECE among the signal-carrying detectors
despite the worst AUROC, but its interval overlaps the others, so it is a hint
rather than a result.

Fifth, the reported F1 is mildly optimistic. Because its threshold is tuned on
the same data it is scored on, the leaderboard F1 overstates the cross-validated
F1 by about 0.04 for the signal-carrying detectors (`novel_token` 0.774 to
0.733, `lexical_overlap` 0.762 to 0.721, `tfidf_cosine` 0.676 to 0.629).
`majority_constant` shows no gap because it flags every answer at any threshold,
so there is nothing to overfit. The gap is small here because the F1-optimal
threshold is stable across folds, but at this sample size the cross-validated
estimate is itself noisy, moving between roughly 0.04 and 0.09 with the fold
seed, so the honest operating-point number should be read as a range, not a
point. This is the operating-point analogue of the wide AUROC intervals: the
small dataset limits what any single number can claim.

The cost/accuracy frontier ties this together. Among the free detectors only
`novel_token` is Pareto-optimal, since the others cost the same and detect less,
although its edge over `lexical_overlap` is within noise.
Adding the NLI model populates the mid-cost tier, and the LLM judge sits at the
top of the frontier: most expensive, highest quality. The research payoff is the
shape of that staircase, which quantifies how much AUROC each step up the cost
ladder actually buys.

## 8. Limitations

The dataset is small and hand-written, which keeps it reproducible but limits
statistical power and domain coverage. The bootstrap makes the cost in power
explicit: the 95% AUROC intervals span 0.27 to 0.32, too wide to separate
detectors a few points apart, and the cross-validated F1 swings by several
points with the fold seed. The error types are a coarse taxonomy.
The live NLI and LLM paths are not exercised in continuous integration, since
they require a model download or API access, so the committed results cover the
free tier. Absolute cost is represented by relative per-tier scores rather than
measured dollars.

## 9. Future Work

Natural next steps are a larger and harder dataset with annotated unsupported
spans, large enough to resolve the gaps the current intervals leave open,
measured dollar and latency costs for the model and API tiers, more
baselines in the mid-cost tier (embedding similarity, smaller NLI models), and a
reported agreement study between the cheap detectors and the judge across the
full dataset.

## 10. Reproducibility

The free-tier results, metrics, frontier, bootstrap statistics, and calibration
analysis are pure-stdlib and deterministic. The bootstrap is seeded, so every
interval and p-value is identical across runs. Running
`python scripts/generate_report.py` regenerates `RESULTS.md`, and the test suite
(`pytest`) covers the metrics, baselines, breakdown, frontier, agreement,
bootstrap statistics, calibration, cross-validated operating points, and report
generation offline.
