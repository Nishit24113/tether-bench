# Tether — Results

Dataset: 130 examples (65 faithful, 65 hallucinated).

## Leaderboard

Ranked by AUROC (threshold-free). Precision/recall/F1 are reported at each detector's F1-optimal threshold. AUROC 95% CIs come from a stratified bootstrap (2000 resamples, seed 0).

| Detector | Cost | AUROC | AUROC 95% CI | F1 | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llm_judge | api | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 1.000 |
| nli_entailment | model | 0.974 | [0.949, 0.991] | 0.927 | 0.877 | 0.985 | 0.923 |
| lexical_overlap | free | 0.538 | [0.440, 0.640] | 0.690 | 0.550 | 0.923 | 0.585 |
| novel_token | free | 0.534 | [0.436, 0.641] | 0.667 | 0.500 | 1.000 | 0.500 |
| majority_constant | free | 0.500 | [0.500, 0.500] | 0.667 | 0.500 | 1.000 | 0.500 |
| tfidf_cosine | free | 0.453 | [0.356, 0.554] | 0.674 | 0.508 | 1.000 | 0.515 |

## Are the gaps real?

Paired stratified bootstrap of the best detector (llm_judge) against each other detector, scored on identical resamples. Two-sided p-values; significant means p < 0.05. The last column counts the examples the two detectors score differently, which is all the evidence a paired comparison has.

| llm_judge vs | AUROC diff | 95% CI | p-value | Significant | Examples scored differently |
| --- | --- | --- | --- | --- | --- |
| nli_entailment | +0.026 | [+0.009, +0.051] | < 0.001 | yes | 130/130 |
| lexical_overlap | +0.462 | [+0.359, +0.560] | < 0.001 | yes | 123/130 |
| novel_token | +0.466 | [+0.359, +0.564] | < 0.001 | yes | 122/130 |
| majority_constant | +0.500 | [+0.500, +0.500] | < 0.001 | yes | 130/130 |
| tfidf_cosine | +0.547 | [+0.446, +0.644] | < 0.001 | yes | 129/130 |

## Do the free signals combine to beat the best single detector?

A logistic combiner is stacked on the free detectors' scores and evaluated out-of-fold (5-fold, seed 0), so its AUROC is not measured on the data the combiner was trained on.

| Combination | AUROC | 95% CI |
| --- | --- | --- |
| stacked free detectors (logistic, out-of-fold) | 0.529 | [0.429, 0.628] |
| best single (lexical_overlap) | 0.538 | - |

The stacked combiner's AUROC differs from the best single detector (lexical_overlap) by -0.009 (95% CI [-0.148, +0.127], p = 0.876), which is not a significant improvement. The free signals are largely redundant, so combining them does not clear the free-tier ceiling.

## Are the scores trustworthy probabilities?

AUROC only asks whether a detector *ranks* faithful answers above hallucinated ones. Calibration asks whether a faithfulness score of p means the answer is faithful about p of the time. ECE is the count-weighted mean gap between predicted score and observed faithful rate across 10 bins (lower is better); MCE is the worst single bin; Brier is the mean squared error of the scores as probabilities. The ECE CI is a stratified bootstrap (2000 resamples, seed 0).

| Detector | ECE | ECE 95% CI | MCE | Brier |
| --- | --- | --- | --- | --- |
| majority_constant | 0.000 | [0.000, 0.000] | 0.000 | 0.250 |
| llm_judge | 0.022 | [0.014, 0.035] | 0.720 | 0.005 |
| nli_entailment | 0.087 | [0.057, 0.136] | 0.563 | 0.071 |
| tfidf_cosine | 0.231 | [0.162, 0.321] | 0.936 | 0.315 |
| lexical_overlap | 0.296 | [0.239, 0.372] | 0.972 | 0.334 |
| novel_token | 0.356 | [0.300, 0.421] | 0.972 | 0.364 |

## Can the cheap scores be made into probabilities?

The miscalibration above is a property of the raw scores, not a verdict on the detectors. This fits a post-hoc calibration map (isotonic, a monotone step fit, and Platt, a logistic fit) from score to observed faithful rate. The map is fit on training folds and applied to a held-out fold (5-fold, seed 0), so the post-calibration ECE/Brier is not measured on the data the map was tuned on. Lower is better.

| Detector | Raw ECE | Isotonic ECE | Platt ECE | Raw Brier | Isotonic Brier |
| --- | --- | --- | --- | --- | --- |
| majority_constant | 0.000 | 0.000 | 0.000 | 0.250 | 0.250 |
| lexical_overlap | 0.296 | 0.065 | 0.150 | 0.334 | 0.248 |
| novel_token | 0.356 | 0.082 | 0.136 | 0.364 | 0.268 |
| tfidf_cosine | 0.231 | 0.112 | 0.073 | 0.315 | 0.261 |
| nli_entailment | 0.087 | 0.058 | 0.054 | 0.071 | 0.062 |
| llm_judge | 0.022 | 0.006 | 0.008 | 0.005 | 0.004 |

## How optimistic is the F1?

The leaderboard F1 uses each detector's F1-optimal threshold, tuned on the same data it is scored on, so it is an oracle operating point. The table below re-estimates F1 with 5-fold stratified cross-validation (seed 0): the threshold is chosen on the training folds and applied to the held-out fold, so every example is predicted by a threshold it did not help pick. The optimism column is how much the oracle F1 overstates the cross-validated one.

| Detector | Oracle F1 | CV F1 | Optimism | CV precision | CV recall |
| --- | --- | --- | --- | --- | --- |
| llm_judge | 1.000 | 0.992 | +0.008 | 0.985 | 1.000 |
| nli_entailment | 0.927 | 0.927 | +0.000 | 0.877 | 0.985 |
| lexical_overlap | 0.690 | 0.674 | +0.015 | 0.542 | 0.892 |
| majority_constant | 0.667 | 0.667 | +0.000 | 0.500 | 1.000 |
| tfidf_cosine | 0.674 | 0.667 | +0.007 | 0.504 | 0.985 |
| novel_token | 0.667 | 0.631 | +0.036 | 0.484 | 0.908 |

## Detection recall by error type

Recall of hallucination detection, per annotated error type, at each detector's F1-optimal threshold.

| Error type | majority_constant | lexical_overlap | novel_token | tfidf_cosine | nli_entailment | llm_judge |
| --- | --- | --- | --- | --- | --- | --- |
| comparison | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 |
| contradiction | 1.00 | 0.89 | 1.00 | 1.00 | 1.00 | 1.00 |
| distractor | 1.00 | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 |
| entity_swap | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| multi_hop | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| numeric | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 |
| temporal | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| unsupported_addition | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Cost/accuracy frontier

Quality axis = auroc. Cost is a relative per-tier score (free 0.0, model 0.3, api 1.0).

| Detector | Cost tier | Cost | Quality | Pareto |
| --- | --- | --- | --- | --- |
| lexical_overlap | free | 0.00 | 0.538 | yes |
| novel_token | free | 0.00 | 0.534 |  |
| majority_constant | free | 0.00 | 0.500 |  |
| tfidf_cosine | free | 0.00 | 0.453 |  |
| nli_entailment | model | 0.30 | 0.974 | yes |
| llm_judge | api | 1.00 | 1.000 | yes |

## How well do the cheap detectors agree with the judge?

Tether's thesis is that a cheap detector is useful insofar as it agrees with the LLM judge (llm_judge). Each cheap detector's hallucination calls (at its F1-optimal threshold) are compared with the judge's (at 0.5). Cohen's kappa corrects for chance agreement; `judge-only catches` are hallucinations the judge flagged and the cheap detector missed, and `cheap-only flags` are answers the cheap detector flagged that the judge called faithful.

| Detector vs judge | Agreement | Cohen kappa | Judge-only catches | Cheap-only flags |
| --- | --- | --- | --- | --- |
| nli_entailment | 0.915 | 0.830 | 2 | 9 |
| lexical_overlap | 0.592 | 0.176 | 5 | 48 |
| tfidf_cosine | 0.523 | 0.032 | 0 | 62 |
| majority_constant | 0.508 | 0.000 | 0 | 64 |
| novel_token | 0.508 | 0.000 | 0 | 64 |
