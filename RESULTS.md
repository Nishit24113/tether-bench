# Tether — Results

Dataset: 36 examples (18 faithful, 18 hallucinated).

## Leaderboard

Ranked by AUROC (threshold-free). Precision/recall/F1 are reported at each detector's F1-optimal threshold.

| Detector | Cost | AUROC | F1 | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| novel_token | free | 0.912 | 0.850 | 0.773 | 0.944 | 0.833 |
| lexical_overlap | free | 0.887 | 0.829 | 0.739 | 0.944 | 0.806 |
| tfidf_cosine | free | 0.809 | 0.800 | 0.727 | 0.889 | 0.778 |
| majority_constant | free | 0.500 | 0.667 | 0.500 | 1.000 | 0.500 |

## Detection recall by error type

Recall of hallucination detection, per annotated error type, at each detector's F1-optimal threshold.

| Error type | majority_constant | lexical_overlap | novel_token | tfidf_cosine |
| --- | --- | --- | --- | --- |
| contradiction | 1.00 | 1.00 | 1.00 | 0.75 |
| entity_swap | 1.00 | 0.80 | 0.80 | 1.00 |
| numeric | 1.00 | 1.00 | 1.00 | 0.80 |
| unsupported_addition | 1.00 | 1.00 | 1.00 | 1.00 |

## Cost/accuracy frontier

Quality axis = auroc. Cost is a relative per-tier score (free 0.0, model 0.3, api 1.0).

| Detector | Cost tier | Cost | Quality | Pareto |
| --- | --- | --- | --- | --- |
| novel_token | free | 0.00 | 0.912 | yes |
| lexical_overlap | free | 0.00 | 0.887 |  |
| tfidf_cosine | free | 0.00 | 0.809 |  |
| majority_constant | free | 0.00 | 0.500 |  |
