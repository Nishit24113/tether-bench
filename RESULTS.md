# Tether — Results

Dataset: 160 examples (80 faithful, 80 hallucinated).

## Leaderboard

Ranked by AUROC (threshold-free). Precision/recall/F1 are reported at each detector's F1-optimal threshold. AUROC 95% CIs come from a stratified bootstrap (2000 resamples, seed 0).

| Detector | Cost | AUROC | AUROC 95% CI | F1 | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llm_judge | api | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 1.000 |
| nli_entailment | model | 0.939 | [0.901, 0.969] | 0.886 | 0.812 | 0.975 | 0.875 |
| lexical_overlap | free | 0.561 | [0.470, 0.653] | 0.691 | 0.547 | 0.938 | 0.581 |
| novel_token | free | 0.551 | [0.464, 0.641] | 0.667 | 0.500 | 1.000 | 0.500 |
| tfidf_cosine | free | 0.501 | [0.415, 0.591] | 0.678 | 0.513 | 1.000 | 0.525 |
| majority_constant | free | 0.500 | [0.500, 0.500] | 0.667 | 0.500 | 1.000 | 0.500 |

## Are the gaps real?

Paired stratified bootstrap of the best detector (llm_judge) against each other detector, scored on identical resamples. Two-sided p-values; significant means p < 0.05. The last column counts the examples the two detectors score differently, which is all the evidence a paired comparison has.

| llm_judge vs | AUROC diff | 95% CI | p-value | Significant | Examples scored differently |
| --- | --- | --- | --- | --- | --- |
| nli_entailment | +0.061 | [+0.031, +0.099] | < 0.001 | yes | 160/160 |
| lexical_overlap | +0.439 | [+0.347, +0.530] | < 0.001 | yes | 151/160 |
| novel_token | +0.449 | [+0.359, +0.536] | < 0.001 | yes | 150/160 |
| tfidf_cosine | +0.499 | [+0.409, +0.585] | < 0.001 | yes | 157/160 |
| majority_constant | +0.500 | [+0.500, +0.500] | < 0.001 | yes | 160/160 |

## Do the free signals combine to beat the best single detector?

A logistic combiner is stacked on the free detectors' scores and evaluated out-of-fold (5-fold, seed 0), so its AUROC is not measured on the data the combiner was trained on.

| Combination | AUROC | 95% CI |
| --- | --- | --- |
| stacked free detectors (logistic, out-of-fold) | 0.495 | [0.403, 0.586] |
| best single (lexical_overlap) | 0.561 | - |

The stacked combiner's AUROC differs from the best single detector (lexical_overlap) by -0.065 (95% CI [-0.160, +0.026], p = 0.180), which is not a significant improvement. The free signals are largely redundant, so combining them does not clear the free-tier ceiling.

## Are the scores trustworthy probabilities?

AUROC only asks whether a detector *ranks* faithful answers above hallucinated ones. Calibration asks whether a faithfulness score of p means the answer is faithful about p of the time. ECE is the count-weighted mean gap between predicted score and observed faithful rate across 10 bins (lower is better); MCE is the worst single bin; Brier is the mean squared error of the scores as probabilities. The ECE CI is a stratified bootstrap (2000 resamples, seed 0).

| Detector | ECE | ECE 95% CI | MCE | Brier |
| --- | --- | --- | --- | --- |
| majority_constant | 0.000 | [0.000, 0.000] | 0.000 | 0.250 |
| llm_judge | 0.019 | [0.012, 0.029] | 0.720 | 0.004 |
| nli_entailment | 0.126 | [0.089, 0.184] | 0.563 | 0.117 |
| tfidf_cosine | 0.188 | [0.133, 0.272] | 0.937 | 0.298 |
| lexical_overlap | 0.276 | [0.224, 0.341] | 0.972 | 0.318 |
| novel_token | 0.329 | [0.280, 0.385] | 0.972 | 0.350 |

## Can the cheap scores be made into probabilities?

The miscalibration above is a property of the raw scores, not a verdict on the detectors. This fits a post-hoc calibration map (isotonic, a monotone step fit, and Platt, a logistic fit) from score to observed faithful rate. The map is fit on training folds and applied to a held-out fold (5-fold, seed 0), so the post-calibration ECE/Brier is not measured on the data the map was tuned on. Lower is better.

| Detector | Raw ECE | Isotonic ECE | Platt ECE | Raw Brier | Isotonic Brier |
| --- | --- | --- | --- | --- | --- |
| majority_constant | 0.000 | 0.000 | 0.000 | 0.250 | 0.250 |
| lexical_overlap | 0.276 | 0.085 | 0.023 | 0.318 | 0.249 |
| novel_token | 0.329 | 0.053 | 0.065 | 0.350 | 0.255 |
| tfidf_cosine | 0.188 | 0.069 | 0.050 | 0.298 | 0.250 |
| nli_entailment | 0.126 | 0.064 | 0.048 | 0.117 | 0.102 |
| llm_judge | 0.019 | 0.004 | 0.006 | 0.004 | 0.003 |

## How optimistic is the F1?

The leaderboard F1 uses each detector's F1-optimal threshold, tuned on the same data it is scored on, so it is an oracle operating point. The table below re-estimates F1 with 5-fold stratified cross-validation (seed 0): the threshold is chosen on the training folds and applied to the held-out fold, so every example is predicted by a threshold it did not help pick. The optimism column is how much the oracle F1 overstates the cross-validated one.

| Detector | Oracle F1 | CV F1 | Optimism | CV precision | CV recall |
| --- | --- | --- | --- | --- | --- |
| llm_judge | 1.000 | 0.994 | +0.006 | 0.988 | 1.000 |
| nli_entailment | 0.886 | 0.875 | +0.011 | 0.802 | 0.963 |
| lexical_overlap | 0.691 | 0.688 | +0.003 | 0.543 | 0.938 |
| tfidf_cosine | 0.678 | 0.678 | +0.000 | 0.513 | 1.000 |
| majority_constant | 0.667 | 0.667 | +0.000 | 0.500 | 1.000 |
| novel_token | 0.667 | 0.655 | +0.011 | 0.497 | 0.963 |

## When can a detector abstain? (selective prediction)

A deployed detector need not answer on every example: it can answer the cases it is most confident about — score farthest from its threshold — and abstain on the rest, deferring them to a human or a costlier check. AURC is the area under the risk-coverage curve (mean selective risk across coverage levels, lower is better); `Risk @C%` is the error rate when answering only the most-confident C% and abstaining on the rest; `Cov @risk<=r` is the dual — the largest fraction answerable while keeping selective risk within r. Each detector uses its own F1-optimal threshold, matching the leaderboard operating point.

| Detector | AURC | Risk @100% | Risk @75% | Risk @50% | Cov @risk<=5% | Cov @risk<=10% |
| --- | --- | --- | --- | --- | --- | --- |
| llm_judge | 0.000 | 0.000 | 0.000 | 0.000 | 1.00 | 1.00 |
| nli_entailment | 0.094 | 0.125 | 0.133 | 0.175 | 0.29 | 0.40 |
| majority_constant | 0.471 | 0.500 | 0.458 | 0.375 | 0.00 | 0.00 |
| lexical_overlap | 0.489 | 0.419 | 0.400 | 0.463 | 0.00 | 0.00 |
| novel_token | 0.509 | 0.500 | 0.475 | 0.438 | 0.00 | 0.00 |
| tfidf_cosine | 0.519 | 0.475 | 0.508 | 0.487 | 0.00 | 0.00 |

## Detection recall by error type

Recall of hallucination detection, per annotated error type, at each detector's F1-optimal threshold.

| Error type | majority_constant | lexical_overlap | novel_token | tfidf_cosine | nli_entailment | llm_judge |
| --- | --- | --- | --- | --- | --- | --- |
| comparison | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 |
| contradiction | 1.00 | 0.91 | 1.00 | 1.00 | 1.00 | 1.00 |
| distractor | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 |
| entity_swap | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 |
| multi_hop | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| numeric | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 |
| temporal | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| unsupported_addition | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 |

## Cost/accuracy frontier

Quality axis = auroc. Cost is a relative per-tier score (free 0.0, model 0.3, api 1.0).

| Detector | Cost tier | Cost | Quality | Pareto |
| --- | --- | --- | --- | --- |
| lexical_overlap | free | 0.00 | 0.561 | yes |
| novel_token | free | 0.00 | 0.551 |  |
| tfidf_cosine | free | 0.00 | 0.501 |  |
| majority_constant | free | 0.00 | 0.500 |  |
| nli_entailment | model | 0.30 | 0.939 | yes |
| llm_judge | api | 1.00 | 1.000 | yes |

## How well do the cheap detectors agree with the judge?

Tether's thesis is that a cheap detector is useful insofar as it agrees with the LLM judge (llm_judge). Each cheap detector's hallucination calls (at its F1-optimal threshold) are compared with the judge's (at 0.5). Cohen's kappa corrects for chance agreement; `judge-only catches` are hallucinations the judge flagged and the cheap detector missed, and `cheap-only flags` are answers the cheap detector flagged that the judge called faithful.

| Detector vs judge | Agreement | Cohen kappa | Judge-only catches | Cheap-only flags |
| --- | --- | --- | --- | --- |
| nli_entailment | 0.869 | 0.737 | 3 | 18 |
| lexical_overlap | 0.588 | 0.168 | 5 | 61 |
| tfidf_cosine | 0.531 | 0.051 | 0 | 75 |
| majority_constant | 0.506 | 0.000 | 0 | 79 |
| novel_token | 0.506 | 0.000 | 0 | 79 |
