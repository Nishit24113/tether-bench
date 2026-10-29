# Tether — Results

Dataset: 100 examples (50 faithful, 50 hallucinated).

_Omitted this run: nli_entailment, llm_judge — their cached scores do not cover all 100 examples. Re-run `scripts/score_detector.py` on the new examples to restore these tiers._

## Leaderboard

Ranked by AUROC (threshold-free). Precision/recall/F1 are reported at each detector's F1-optimal threshold. AUROC 95% CIs come from a stratified bootstrap (2000 resamples, seed 0).

| Detector | Cost | AUROC | AUROC 95% CI | F1 | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lexical_overlap | free | 0.572 | [0.460, 0.688] | 0.706 | 0.558 | 0.960 | 0.600 |
| novel_token | free | 0.562 | [0.443, 0.673] | 0.682 | 0.557 | 0.880 | 0.590 |
| majority_constant | free | 0.500 | [0.500, 0.500] | 0.667 | 0.500 | 1.000 | 0.500 |
| tfidf_cosine | free | 0.447 | [0.340, 0.566] | 0.676 | 0.510 | 1.000 | 0.520 |

## Are the gaps real?

Paired stratified bootstrap of the best detector (lexical_overlap) against each other detector, scored on identical resamples. Two-sided p-values; significant means p < 0.05. The last column counts the examples the two detectors score differently, which is all the evidence a paired comparison has.

| lexical_overlap vs | AUROC diff | 95% CI | p-value | Significant | Examples scored differently |
| --- | --- | --- | --- | --- | --- |
| novel_token | +0.011 | [-0.044, +0.066] | 0.661 | no | 17/100 |
| majority_constant | +0.072 | [-0.040, +0.188] | 0.218 | no | 94/100 |
| tfidf_cosine | +0.126 | [+0.055, +0.207] | < 0.001 | yes | 99/100 |

## Do the free signals combine to beat the best single detector?

A logistic combiner is stacked on the free detectors' scores and evaluated out-of-fold (5-fold, seed 0), so its AUROC is not measured on the data the combiner was trained on.

| Combination | AUROC | 95% CI |
| --- | --- | --- |
| stacked free detectors (logistic, out-of-fold) | 0.647 | [0.540, 0.756] |
| best single (lexical_overlap) | 0.572 | - |

The stacked combiner's AUROC differs from the best single detector (lexical_overlap) by +0.074 (95% CI [-0.066, +0.231], p = 0.333), which is not a significant improvement. The free signals are largely redundant, so combining them does not clear the free-tier ceiling.

## Are the scores trustworthy probabilities?

AUROC only asks whether a detector *ranks* faithful answers above hallucinated ones. Calibration asks whether a faithfulness score of p means the answer is faithful about p of the time. ECE is the count-weighted mean gap between predicted score and observed faithful rate across 10 bins (lower is better); MCE is the worst single bin; Brier is the mean squared error of the scores as probabilities. The ECE CI is a stratified bootstrap (2000 resamples, seed 0).

| Detector | ECE | ECE 95% CI | MCE | Brier |
| --- | --- | --- | --- | --- |
| majority_constant | 0.000 | [0.000, 0.000] | 0.000 | 0.250 |
| tfidf_cosine | 0.211 | [0.157, 0.320] | 0.919 | 0.319 |
| lexical_overlap | 0.321 | [0.262, 0.404] | 0.972 | 0.333 |
| novel_token | 0.378 | [0.314, 0.453] | 0.972 | 0.362 |

## Can the cheap scores be made into probabilities?

The miscalibration above is a property of the raw scores, not a verdict on the detectors. This fits a post-hoc calibration map (isotonic, a monotone step fit, and Platt, a logistic fit) from score to observed faithful rate. The map is fit on training folds and applied to a held-out fold (5-fold, seed 0), so the post-calibration ECE/Brier is not measured on the data the map was tuned on. Lower is better.

| Detector | Raw ECE | Isotonic ECE | Platt ECE | Raw Brier | Isotonic Brier |
| --- | --- | --- | --- | --- | --- |
| majority_constant | 0.000 | 0.000 | 0.000 | 0.250 | 0.250 |
| lexical_overlap | 0.321 | 0.118 | 0.090 | 0.333 | 0.242 |
| novel_token | 0.378 | 0.117 | 0.092 | 0.362 | 0.262 |
| tfidf_cosine | 0.211 | 0.076 | 0.008 | 0.319 | 0.256 |

## How optimistic is the F1?

The leaderboard F1 uses each detector's F1-optimal threshold, tuned on the same data it is scored on, so it is an oracle operating point. The table below re-estimates F1 with 5-fold stratified cross-validation (seed 0): the threshold is chosen on the training folds and applied to the held-out fold, so every example is predicted by a threshold it did not help pick. The optimism column is how much the oracle F1 overstates the cross-validated one.

| Detector | Oracle F1 | CV F1 | Optimism | CV precision | CV recall |
| --- | --- | --- | --- | --- | --- |
| lexical_overlap | 0.706 | 0.696 | +0.010 | 0.553 | 0.940 |
| majority_constant | 0.667 | 0.667 | +0.000 | 0.500 | 1.000 |
| tfidf_cosine | 0.676 | 0.667 | +0.009 | 0.505 | 0.980 |
| novel_token | 0.682 | 0.638 | +0.044 | 0.500 | 0.880 |

## Detection recall by error type

Recall of hallucination detection, per annotated error type, at each detector's F1-optimal threshold.

| Error type | majority_constant | lexical_overlap | novel_token | tfidf_cosine |
| --- | --- | --- | --- | --- |
| contradiction | 1.00 | 0.89 | 0.89 | 1.00 |
| entity_swap | 1.00 | 0.92 | 0.92 | 1.00 |
| multi_hop | 1.00 | 1.00 | 0.75 | 1.00 |
| numeric | 1.00 | 1.00 | 1.00 | 1.00 |
| temporal | 1.00 | 1.00 | 0.00 | 1.00 |
| unsupported_addition | 1.00 | 1.00 | 1.00 | 1.00 |

## Cost/accuracy frontier

Quality axis = auroc. Cost is a relative per-tier score (free 0.0, model 0.3, api 1.0).

| Detector | Cost tier | Cost | Quality | Pareto |
| --- | --- | --- | --- | --- |
| lexical_overlap | free | 0.00 | 0.572 | yes |
| novel_token | free | 0.00 | 0.562 |  |
| majority_constant | free | 0.00 | 0.500 |  |
| tfidf_cosine | free | 0.00 | 0.447 |  |
