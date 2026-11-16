"""Compare history features and learning-to-rank on rolling monthly validation."""
import argparse
import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from datetime import date
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier, CatBoostRanker
from lightgbm import LGBMRanker
import numpy as np
import pandas as pd
import polars as pl

from recommender_santander.columns import PRODUCTS, CATEGORICAL_FEATURES
from recommender_santander.data import load_sample, monthly_pairs
from recommender_santander.history import HISTORY_FEATURES, history_features
from recommender_santander.modeling import (
    feature_frame, per_customer_ap, popularity_rates,
    fit_buyer_classifier, choose_buyer_threshold, classifier_metrics,
)

CATEGORIES = CATEGORICAL_FEATURES + ["product"]


def features(frame, history):
    result = feature_frame(frame)
    if history:
        for column in HISTORY_FEATURES:
            result[column] = frame[column].astype(float)
    return result


def candidates(pairs, history, ranking):
    frame = pairs.to_pandas()
    x = features(frame, history)
    rng = np.random.default_rng(42)
    buyers = frame[[f"new__{p}" for p in PRODUCTS]].sum(axis=1).to_numpy() > 0
    parts = []
    for product in PRODUCTS:
        y = frame[f"new__{product}"].to_numpy()
        eligible = frame[product].to_numpy() == 0
        keep = eligible & buyers if ranking else eligible & ((y == 1) | (rng.random(len(frame)) < .03))
        part = x.loc[keep].copy()
        part["product"] = product
        part["label"] = y[keep]
        part["group"] = frame.index.to_numpy()[keep]
        parts.append(part)
    data = pd.concat(parts, ignore_index=True)
    if ranking:
        data = data.sort_values("group", kind="stable").reset_index(drop=True)
        # Constant-label groups provide no ordering signal.
        data = data[data.groupby("group")["label"].transform("nunique") > 1]
    group = data.pop("group").to_numpy()
    labels = data.pop("label").to_numpy()
    return data, labels, group


def fit_model(kind, history, training):
    x, y, groups = candidates(training, history, ranking=kind != "classifier")
    category_values = {}
    if kind == "classifier":
        model = CatBoostClassifier(iterations=150, depth=6, learning_rate=.1,
            cat_features=CATEGORIES, random_seed=42, thread_count=2,
            loss_function="Logloss", verbose=False, allow_writing_files=False)
        model.fit(x, y)
    elif kind == "catboost_ranker":
        model = CatBoostRanker(iterations=150, depth=6, learning_rate=.1,
            cat_features=CATEGORIES, random_seed=42, thread_count=2,
            loss_function="YetiRank:mode=MAP;top=7", verbose=False, allow_writing_files=False)
        model.fit(x, y, group_id=groups)
    else:
        for column in CATEGORIES:
            category_values[column] = sorted(x[column].unique())
            x[column] = pd.Categorical(x[column], categories=category_values[column])
        model = LGBMRanker(objective="lambdarank", n_estimators=150, num_leaves=31,
            learning_rate=.1, lambdarank_truncation_level=10, random_state=42,
            n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)
        sizes = pd.Series(groups).groupby(groups, sort=False).size().to_numpy()
        model.fit(x, y, group=sizes, categorical_feature=CATEGORIES)
    return {"kind": kind, "history": history, "model": model, "categories": category_values,
            "training_candidates": len(y), "training_groups": len(np.unique(groups))}


def predictions(model, pairs):
    frame = pairs.to_pandas()
    x = features(frame, model.get("history", False))
    scores = np.full((len(frame), len(PRODUCTS)), -np.inf)
    for index, product in enumerate(PRODUCTS):
        eligible = frame[product].to_numpy() == 0
        if not eligible.any():
            continue
        if model["kind"] == "popularity":
            scores[eligible, index] = model["rates"][index]
            continue
        candidate = x.loc[eligible].copy()
        candidate["product"] = product
        if model["kind"] == "classifier":
            values = model["model"].predict_proba(candidate)[:, 1]
        else:
            for column, values in model["categories"].items():
                known_values = candidate[column].where(candidate[column].isin(values))
                candidate[column] = pd.Categorical(known_values, categories=values)
            values = model["model"].predict(candidate)
        scores[eligible, index] = values
    actual = frame[[f"new__{p}" for p in PRODUCTS]].to_numpy()
    return per_customer_ap(scores, actual), actual.sum(axis=1) > 0


def run(source, customers=5_000, report_dir=Path(__file__).resolve().parent / "results"):
    sample = load_sample(source, customers=customers, seed=42)
    pairs = monthly_pairs(sample).join(history_features(sample), on=["customer_id", "feature_date"], how="left", maintain_order="left")
    configs = [(kind, history) for kind in ["classifier", "catboost_ranker", "lightgbm_ranker"] for history in [False, True]]
    rows = []
    # Fixed settings; select only by the mean of February, March, and April MAP@7.
    for month in [2, 3, 4]:
        end = date(2016, month, 1)
        start = date(2015, month + 6, 1)  # Six target months before validation.
        training = pairs.filter(pl.col("target_date").is_between(start, end, closed="left"))
        validation = pairs.filter(pl.col("target_date") == date(2016, month, 28))
        baseline = {"kind": "popularity", "rates": popularity_rates(training)}
        ap, buyers = predictions(baseline, validation)
        rows.append({"validation_month": str(end)[:7], "model": "popularity", "history": False,
                     "map_at_7": ap.mean(), "map_at_7_buyers": ap[buyers].mean(),
                     "customers": len(ap), "fit_seconds": 0.})
        for kind, history in configs:
            started = time.perf_counter()
            fitted = fit_model(kind, history, training)
            elapsed = time.perf_counter() - started
            ap, buyers = predictions(fitted, validation)
            rows.append({"validation_month": str(end)[:7], "model": kind, "history": history,
                         "map_at_7": ap.mean(), "map_at_7_buyers": ap[buyers].mean(),
                         "customers": len(ap), "fit_seconds": elapsed,
                         "training_candidates": fitted["training_candidates"], "training_groups": fitted["training_groups"]})
            print(f"{end:%Y-%m} {kind} history={history}: MAP@7={ap.mean():.5f} ({elapsed:.1f}s fit)", flush=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(report_dir / "ranking_folds.csv", index=False)
    folds = pd.DataFrame(rows)
    summary = folds.groupby(["model", "history"], as_index=False).agg(
        mean_map_at_7=("map_at_7", "mean"), mean_map_at_7_buyers=("map_at_7_buyers", "mean"),
        fit_seconds=("fit_seconds", "sum"))
    summary = summary.sort_values("mean_map_at_7", ascending=False)
    summary.to_csv(report_dir / "ranking_comparison.csv", index=False)
    winner = summary.iloc[0]
    training = pairs.filter(pl.col("target_date").is_between(date(2015, 10, 1), date(2016, 4, 1), closed="left"))
    april = pairs.filter(pl.col("target_date") == date(2016, 4, 28))
    may = pairs.filter(pl.col("target_date") == date(2016, 5, 28))
    if winner["model"] == "popularity":
        fitted = {"kind": "popularity", "rates": popularity_rates(training)}
    else:
        fitted = fit_model(winner["model"], bool(winner["history"]), training)
    ap, buyers = predictions(fitted, may)
    # Keep the existing buyer classifier and April threshold policy fixed.
    buyer_model = fit_buyer_classifier(training)
    threshold = choose_buyer_threshold(buyer_model, april)
    buyer_scores = buyer_model.predict_proba(feature_frame(may.to_pandas()))[:, 1]
    selected = buyer_scores >= threshold
    gate = classifier_metrics(buyers, buyer_scores, threshold)
    metrics = {"selected_model": str(winner["model"]), "history": bool(winner["history"]),
        "selection": "mean MAP@7 over February, March, April; six training target months per fold",
        "mean_validation_map_at_7": float(winner["mean_map_at_7"]),
        "customers_sampled": customers, "seed": 42,
        "may_status": "exploratory reference: this month was inspected in earlier experiments",
        "may_map_all": float(ap.mean()), "may_map_actual_buyers": float(ap[buyers].mean()),
        "may_map_selected": float(ap[selected].mean()) if selected.any() else None,
        "may_map_gated_all": float((ap * selected).mean()), "classifier": gate}
    (report_dir / "ranking_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    return folds, summary, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/train/train_ver2.csv.zip"))
    parser.add_argument("--customers", type=int, default=5_000)
    args = parser.parse_args()
    run(args.data, args.customers)
