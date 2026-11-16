# Recommendation model experiments

Same 50,000 customers, seed 42. Fixed settings; rolling validation on February, March, and April, with six preceding training target months per fold. The mean gives each validation month equal weight.

| Model | History | Mean validation MAP@7 | Change vs original classifier |
|---|---|---:|---:|
| LightGBM LambdaRank | Yes | 0.02705 | +9.97% |
| LightGBM LambdaRank | No | 0.02687 | +9.23% |
| CatBoostRanker | Yes | 0.02625 | +6.71% |
| CatBoostRanker | No | 0.02598 | +5.60% |
| CatBoost classifier | Yes | 0.02477 | +0.70% |
| CatBoost classifier | No | 0.02460 | +0.00% |
| Popularity | No | 0.02097 | -14.73% |

## Interpretation

LightGBM with history has the highest mean score. Both ranking methods outperform the original classifier in every validation month. History alone gives a smaller gain and slightly reduces classifier MAP@7 in April. The larger change is from the ranking approach; the comparison also changes training sampling, so it does not isolate the loss function alone.

Rankers train on customer-month groups with mixed relevance labels and all their eligible products. Constant-label groups give no ordering signal. Classifiers retain all acquisitions plus 3% of eligible negatives. Every method evaluates all eligible products and the same customers.

History uses only information observed at the feature snapshot. Rolling windows restart at gaps or unknown holdings. Tests check future changes cannot alter earlier features, that ranking groups stay together, and that classifier sampling matches the existing baseline.

## May reference for the validation-selected winner

| Population | Original classifier MAP@7 | LightGBM with history MAP@7 |
|---|---:|---:|
| All customers | 0.02352 | 0.02612 |
| Actual acquiring customers | 0.76833 | 0.85307 |
| Fixed-classifier-selected customers | 0.20996 | 0.26332 |

May all-customer MAP@7 improves by 11.03% relative to the original ranker. The buyer classifier and threshold are unchanged: precision 29.18%, recall 36.46%, coverage 3.82%. Gated MAP@7 over everyone is 0.01007.

**May is exploratory:** it was inspected in earlier project work and is not a new untouched test. These experiments use fixed settings, not an exhaustive hyperparameter search. No serving artifact is replaced. The current API accepts one snapshot, so deploying a history-based model also requires a way to supply or retrieve historical features.

Details: [fold scores](ranking_folds.csv), [comparison](ranking_comparison.csv), [metrics](ranking_metrics.json). Reproduce with `uv run python experiments/ranking.py`.

Algorithm documentation: [CatBoost ranking objectives](https://catboost.ai/docs/en/concepts/loss-functions-ranking), [LightGBM Ranker](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html).
