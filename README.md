# Santander product recommendation

Predict whether a customer will acquire a new banking product next month. If the buyer classifier selects them, rank up to seven products they do not already own. A bank could use this to choose customers and products for offers. We predict acquisition, not whether an offer causes a purchase.

The [Santander dataset](https://www.kaggle.com/competitions/santander-product-recommendation/data) has 13.65 million monthly snapshots, 956,645 customers, and 24 product indicators. Training covers January 2015–May 2016. Kaggle notes that this sample does not contain real Santander Spain customers or represent its customer base.

## Files to start with

- `notebooks/`: data exploration and model results.
- `train.py`: train and save the two-stage CatBoost model.
- `predict.py`: serve that model with FastAPI.
- `src/recommender_santander/`: shared preparation, features, and scoring.
- `reports/`: saved training metrics and local deployment evidence.
- `experiments/`: optional history/learning-to-rank comparison and its results; not used by the API.
- `examples/` and `tests/`: example requests and checks.

`pyproject.toml` and `uv.lock` define the environment. `requirements.txt` is generated for the Docker build. The Beanstalk configuration is in `.elasticbeanstalk/`.

## Install and download

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
uv sync --locked
source .venv/bin/activate
cp .env.example .env
```

Accept the [competition rules](https://www.kaggle.com/competitions/santander-product-recommendation/rules). Generate a token in [Kaggle settings](https://www.kaggle.com/settings/api), put it in `.env` as `KAGGLE_API_TOKEN`, then run:

```sh
uv run python scripts/download_data.py
```

Alternatively, download `train_ver2.csv.zip` in your browser and put it in `data/raw/train/`. Existing files are reused. Keep any competition test data separate in `data/raw/competition_test/`; it has no product labels and is not used for local evaluation.

## Notebooks and training

```sh
uv run jupyter lab
uv run python train.py
```

Read the notebooks in this order:

1. [eda.ipynb](notebooks/eda.ipynb): missing values, customer histories, ownership, and acquisitions.
2. [classification_eda.ipynb](notebooks/classification_eda.ipynb): buyer profiles, class imbalance, precision–recall, and threshold tradeoffs.
3. [modeling.ipynb](notebooks/modeling.ipynb): reads the saved reports and displays model comparison, evaluation, and feature importance. It does not launch training.

Queries and plots are shown directly in the notebooks. Shared data cleaning and labels are in `src/recommender_santander/data.py`; model fitting and scoring are in `modeling.py`.

By default, each run selects 5,000 customer IDs with seed 42 from sorted unique IDs and retains all their months. The sample is prepared from the raw CSV in memory; no processed sample files are needed. Restart the kernel and run one notebook or training job at a time. The defaults use at most two model workers and limit Polars to two workers. `train.py` saves `artifacts/model.joblib` and reports under `reports/`.

The saved benchmark used 50,000 customers. Repeat it only explicitly with `uv run python train.py --customers 50000`. Training overwrites the model and reports with the selected sample size. Start with 5,000 after a crash; do not run training alongside other notebooks.

## Experiment

A new acquisition is a 0 → 1 transition between consecutive months. Features come from the earlier month. Exclude gaps and unknown product holdings. Invalid ages (outside 0–120) and negative income/tenure become missing. Logistic regression imputes numeric values using training data; CatBoost handles them natively. Missing categories become `Unknown`.

| Target months | Use |
|---|---|
| October 2015–March 2016 | Fit |
| April 2016 | Select ranker and buyer threshold |
| May 2016 | Final local evaluation |
| October 2015–May 2016 | Refit the selected models for June predictions |

Compare popularity, logistic regression (`C`: 0.1, 1.0), and CatBoost (depth: 4, 6; 150 iterations). Keep all positive product candidates and a uniform 3% of eligible negatives during ranker training; evaluate every eligible product. Select by April MAP@7.

A separate CatBoost buyer classifier (depth 4, 150 iterations) uses all training pairs. Its threshold maximizes April F1 and stays fixed for May evaluation and the serving refit. A real offer policy would need a budget or contact cost; F1 is a starting point.

### Results (completed 50,000-customer run)

CatBoost depth 6 won on April (MAP@7 **0.02138**). On May:

| Population | Customers | MAP@7 |
|---|---:|---:|
| All evaluated customers, before filtering | 48,474 | 0.02352 |
| Actual acquiring customers | 1,484 | 0.76833 |
| Classifier-selected customers | 1,854 | 0.20996 |

Popularity scores 0.01969 over all customers. Buyer precision is 29.18%, recall 36.46%, and coverage 3.82%. Counting abstentions as zero across everyone, gated MAP@7 is 0.00803. The subgroup scores have different denominators; they are not whole-population improvements. The modeling notebook explains MAP@7 with an example.

Full results: [comparison](reports/model_comparison.csv), [metrics](reports/metrics.json), [three scores](reports/three_scores.csv), and [feature importance](reports/feature_importance.csv). May scores refer to the fits trained through March, not the serving models refitted through May.

## API

Train first, then start the service:

```sh
uv run uvicorn predict:app --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
curl http://localhost:8000/health
curl -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' --data-binary @examples/request_selected.json
curl -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

The first profile is selected; the second demonstrates abstention. These are synthetic examples, not evaluation data. Request fields and validation are shown at <http://localhost:8000/docs>.

Provide the current profile, its month (`snapshot_month`), the complete `owned_products` list, and `k` (1–7). Unlisted products are treated as unowned. Missing profile fields are allowed. The response includes `should_recommend`, `buyer_score`, `buyer_threshold`, and ranked products with `ranking_score`. Abstention returns an empty list. Scores have not been shown to be calibrated probabilities. Product names are listed in `columns.py`. In the source data, sex is coded `H` for male and `V` for female; joining channels such as `KHE` are anonymized categories without a published meaning.

## Docker and tests

After training:

```sh
docker build -t santander-recommender .
docker run --rm --name santander-recommender -p 127.0.0.1:8000:8000 santander-recommender
```

Stop the local Uvicorn process before using the same port for Docker. Use the curl commands above. The image includes the model and locked runtime dependencies, excluding data and `.env`. [Recorded Docker requests and responses](reports/deployment/README.md) include both recommendation and abstention examples.

```sh
uv run pytest -q
```

Train before the API integration test. Tests cover previous-month features, gaps, unknown holdings, MAP@7 denominators, threshold selection, ownership filtering, and abstention. Dependencies are in `pyproject.toml` and `uv.lock`; `requirements.txt` is the locked runtime export used by Docker. Polars reads the data, scikit-learn fits logistic regression, CatBoost fits tree models, and FastAPI serves them.

The evaluation uses one historical holdout month and complete consecutive labels. Cold-start behavior and probability calibration remain untested; the optional ranking experiment below adds rolling validation. Only load model files produced by this project. Cloud deployment is optional and is not included. The course also requires three peer reviews.

## Additional recommendation experiments

```sh
uv run python experiments/ranking.py
```

This compares CatBoost classification, CatBoostRanker, and LightGBM LambdaRank, each with and without history features. The default is 5,000 customers and seed 42, with two workers. To reproduce the saved 50,000-customer benchmark explicitly, run `uv run python experiments/ranking.py --customers 50000`. LightGBM is a development dependency installed by `uv sync`.

Use rolling validation on February, March, and April 2016; each fit uses the previous six target months. All settings are fixed: 150 trees, depth 6 for CatBoost, 31 leaves for LightGBM. Select by the mean of the three monthly MAP@7 scores. May is only an exploratory reference because earlier project work already inspected it.

History features are computed at the feature snapshot, before its target: recent acquisitions and removals, observed history length, ownership count, time since an observed acquisition, and activity changes. Rolling windows restart after gaps or unknown holdings. Rankers train on groups with both positive and negative labels, using all eligible products in those groups; evaluation still includes all customers. The classifier keeps its original 3% negative sampling. CatBoostRanker uses a MAP-oriented YetiRank objective; LambdaRank optimizes NDCG. Both are evaluated with the same AP@7 code.

Results are saved under `experiments/results/` as `ranking_folds.csv`, `ranking_comparison.csv`, and `ranking_metrics.json`, and displayed at the end of the modeling notebook. This experiment does not overwrite the serving artifact: a history-based model would require historical inputs that the current single-snapshot API does not accept.

**Measured comparison:** LightGBM LambdaRank with history averaged **0.02705** MAP@7 across February–April, versus **0.02460** for the original classifier. See [experiment results](experiments/results/ranking_results.md) for every model and the exploratory May comparison.
