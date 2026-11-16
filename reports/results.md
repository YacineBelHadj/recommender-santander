# Model results

Fixed cohort: 50,000 customers, 714,263 snapshots; seed 42.

| Model | April validation MAP@7 | May test MAP@7 |
|---|---:|---:|
| Selected CatBoost (depth 6, 150 iterations) | 0.02138 | 0.02352 |
| Popularity baseline | See comparison CSV | 0.01969 |

All hyperparameters were selected on April. May evaluation contains 48,474 complete consecutive pairs out of 48,718 sampled May snapshots. Of evaluated customers, 1,484 acquired something. The score among these positive customers is 0.76833; this is not the overall MAP@7.

The pre-refit evaluation model uses October 2015–March 2016 target months. The serving artifact is subsequently refitted with the selected parameters through May, so its June predictions are not directly measured by this test score.

The strongest learned model's leading features are product identity, customer activity, relationship status, age, tenure, and joining channel. Product identity captures acquisition prevalence; these importances do not establish causality.

See `model_comparison.csv` for every tuning run, `metrics.json` for machine-readable results, and `feature_importance.csv` for all model features. Results cover one historical holdout month and a cohort selected for observed consecutive labels. Further rolling validation and separate cold-start analysis remain future work.

## First-stage buyer classifier

Fit on October–March customer-month pairs without negative subsampling. The decision threshold (0.17875) maximizes F1 on April validation. The following results use May only after that threshold was frozen.

| Ranking population | Customers | MAP@7 |
|---|---:|---:|
| All customers, before gating | 48,474 | 0.02352 |
| Actual acquiring customers, known afterward | 1,484 | 0.76833 |
| Classifier-selected customers, including false positives | 1,854 | 0.20996 |

Buyer precision is **29.18%**, recall **36.46%**, and coverage **3.82%**. Buyer F1 is 0.3241; average precision over the buyer PR curve is 0.2582.

**Gated MAP@7 over all 48,474 customers is 0.00803** when abstentions receive zero. Selecting a subgroup changes the denominator; its larger MAP@7 is not a whole-population improvement. The classifier reduces offers but misses 63.54% of actual acquiring customers. Acquisition prediction also does not establish that an offer caused a purchase.

Both models are refitted through May for serving; the April-selected threshold is kept. The API abstains below that threshold.
