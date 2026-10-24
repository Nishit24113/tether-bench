# Tether — Results

Dataset: 50 examples (25 faithful, 25 hallucinated).

## Leaderboard

Ranked by AUROC (threshold-free). Precision/recall/F1 are reported at each detector's F1-optimal threshold. AUROC 95% CIs come from a stratified bootstrap (2000 resamples, seed 0).

| Detector | Cost | AUROC | AUROC 95% CI | F1 | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| novel_token | free | 0.779 | [0.635, 0.902] | 0.774 | 0.649 | 0.960 | 0.720 |
| lexical_overlap | free | 0.763 | [0.614, 0.890] | 0.762 | 0.632 | 0.960 | 0.700 |
| tfidf_cosine | free | 0.603 | [0.435, 0.758] | 0.676 | 0.510 | 1.000 | 0.520 |
| majority_constant | free | 0.500 | [0.500, 0.500] | 0.667 | 0.500 | 1.000 | 0.500 |

## Are the gaps real?

Paired stratified bootstrap of the best detector (novel_token) against each other detector, scored on identical resamples. Two-sided p-values; significant means p < 0.05. The last column counts the examples the two detectors score differently, which is all the evidence a paired comparison has.

| novel_token vs | AUROC diff | 95% CI | p-value | Significant | Examples scored differently |
| --- | --- | --- | --- | --- | --- |
| lexical_overlap | +0.016 | [-0.013, +0.070] | 0.776 | no | 3/50 |
| tfidf_cosine | +0.176 | [+0.040, +0.321] | 0.007 | yes | 49/50 |
| majority_constant | +0.279 | [+0.135, +0.402] | < 0.001 | yes | 48/50 |

## Are the scores trustworthy probabilities?

AUROC only asks whether a detector *ranks* faithful answers above hallucinated ones. Calibration asks whether a faithfulness score of p means the answer is faithful about p of the time. ECE is the count-weighted mean gap between predicted score and observed faithful rate across 10 bins (lower is better); MCE is the worst single bin; Brier is the mean squared error of the scores as probabilities. The ECE CI is a stratified bootstrap (2000 resamples, seed 0).

| Detector | ECE | ECE 95% CI | MCE | Brier |
| --- | --- | --- | --- | --- |
| majority_constant | 0.000 | [0.000, 0.000] | 0.000 | 0.250 |
| tfidf_cosine | 0.187 | [0.135, 0.344] | 0.928 | 0.265 |
| lexical_overlap | 0.304 | [0.242, 0.391] | 1.000 | 0.281 |
| novel_token | 0.321 | [0.258, 0.399] | 1.000 | 0.288 |

## Detection recall by error type

Recall of hallucination detection, per annotated error type, at each detector's F1-optimal threshold.

| Error type | majority_constant | lexical_overlap | novel_token | tfidf_cosine |
| --- | --- | --- | --- | --- |
| contradiction | 1.00 | 1.00 | 1.00 | 1.00 |
| entity_swap | 1.00 | 0.86 | 0.86 | 1.00 |
| numeric | 1.00 | 1.00 | 1.00 | 1.00 |
| unsupported_addition | 1.00 | 1.00 | 1.00 | 1.00 |

## Cost/accuracy frontier

Quality axis = auroc. Cost is a relative per-tier score (free 0.0, model 0.3, api 1.0).

| Detector | Cost tier | Cost | Quality | Pareto |
| --- | --- | --- | --- | --- |
| novel_token | free | 0.00 | 0.779 | yes |
| lexical_overlap | free | 0.00 | 0.763 |  |
| tfidf_cosine | free | 0.00 | 0.603 |  |
| majority_constant | free | 0.00 | 0.500 |  |
