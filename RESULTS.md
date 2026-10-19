# Tether — Results

Dataset: 50 examples (25 faithful, 25 hallucinated).

## Leaderboard

Ranked by AUROC (threshold-free). Precision/recall/F1 are reported at each detector's F1-optimal threshold.

| Detector | Cost | AUROC | F1 | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| novel_token | free | 0.779 | 0.774 | 0.649 | 0.960 | 0.720 |
| lexical_overlap | free | 0.763 | 0.762 | 0.632 | 0.960 | 0.700 |
| tfidf_cosine | free | 0.603 | 0.676 | 0.510 | 1.000 | 0.520 |
| majority_constant | free | 0.500 | 0.667 | 0.500 | 1.000 | 0.500 |

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
