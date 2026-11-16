"""Compare models on April, evaluate on May, and save the fitted models."""
import argparse
import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from datetime import date
import json
from pathlib import Path
import time

import joblib
import pandas as pd
import polars as pl
import sklearn

from recommender_santander.data import load_sample, monthly_pairs
from recommender_santander.modeling import (
    candidate_training_data, evaluate, logistic_model, catboost_ranker, popularity_rates,
    fit_buyer_classifier, choose_buyer_threshold, evaluate_two_stage,
)


def run_training(source=Path("data/raw/train/train_ver2.csv.zip"), customers=5_000,
                 seed=42, negative_fraction=0.03, output_dir=Path("artifacts"),
                 report_dir=Path("reports")):
    sample = load_sample(Path(source), customers, seed)
    pairs = monthly_pairs(sample)
    # Six training target months; April tunes parameters; May is held out.
    training = pairs.filter(pl.col("target_date").is_between(date(2015, 10, 1), date(2016, 3, 31)))
    validation = pairs.filter(pl.col("target_date") == date(2016, 4, 28))
    test = pairs.filter(pl.col("target_date") == date(2016, 5, 28))
    if min(training.height, validation.height, test.height) == 0:
        raise ValueError("The sample must contain training, April validation, and May test pairs")
    print(f"Training pairs: {training.height:,}; April: {validation.height:,}; May: {test.height:,}", flush=True)
    x, y = candidate_training_data(training, negative_fraction, seed)
    print(f"Sampled candidate training rows: {len(y):,}; acquisitions: {int(y.sum()):,}", flush=True)
    baseline = {"kind": "popularity", "rates": popularity_rates(training), "params": {}}
    models = [
        baseline,
        {"kind": "logistic", "model": logistic_model(0.1, seed), "params": {"C": 0.1}},
        {"kind": "logistic", "model": logistic_model(1.0, seed), "params": {"C": 1.0}},
        {"kind": "catboost", "model": catboost_ranker(4, seed), "params": {"depth": 4, "iterations": 150}},
        {"kind": "catboost", "model": catboost_ranker(6, seed), "params": {"depth": 6, "iterations": 150}},
    ]
    rows = []
    for candidate in models:
        start = time.perf_counter()
        if candidate["kind"] != "popularity":
            candidate["model"].fit(x, y)
        fit_seconds = time.perf_counter() - start
        validation_scores = evaluate(candidate, validation)
        rows.append({
            "model": candidate["kind"], "params": json.dumps(candidate["params"]),
            **validation_scores, "fit_seconds": round(fit_seconds, 2),
        })
        print(f"{candidate['kind']} {candidate['params']}: April MAP@7={validation_scores['map_at_7']:.5f}", flush=True)

    comparison = pd.DataFrame(rows)
    best_index = comparison["map_at_7"].idxmax()
    best_bundle = models[best_index]
    best_name = best_bundle["kind"]
    best_params = best_bundle["params"]
    best_score = comparison.loc[best_index, "map_at_7"]
    # The strongest learned model supplies feature importance, even if popularity wins.
    best_learned_index = comparison.iloc[1:]["map_at_7"].idxmax()
    best_learned = models[best_learned_index]
    buyer_model = fit_buyer_classifier(training, seed)
    buyer_threshold = choose_buyer_threshold(buyer_model, validation)
    buyer_validation = evaluate_two_stage(best_bundle, buyer_model, buyer_threshold, validation)
    print(f"Buyer classifier: April threshold={buyer_threshold:.4f}; F1={buyer_validation['classifier']['f1']:.4f}", flush=True)
    # Freeze both the ranking model and buyer threshold before inspecting May.
    final_test = evaluate(best_bundle, test)
    baseline_test = evaluate(baseline, test)
    two_stage_test = evaluate_two_stage(best_bundle, buyer_model, buyer_threshold, test)
    output_dir, report_dir = Path(output_dir), Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    results = comparison.sort_values("map_at_7", ascending=False)
    results.to_csv(report_dir / "model_comparison.csv", index=False)
    learned_model = best_learned["model"]
    if best_learned["kind"] == "catboost":
        importance = pd.DataFrame({"feature": learned_model.feature_names_, "importance": learned_model.feature_importances_})
    else:
        importance = pd.DataFrame({
            "feature": learned_model.named_steps["features"].get_feature_names_out(),
            "importance": abs(learned_model.named_steps["classifier"].coef_[0]),
        })
    importance.sort_values("importance", ascending=False).to_csv(report_dir / "feature_importance.csv", index=False)
    metrics = {
        "selected_model": best_name, "selected_params": best_params,
        "seed": seed, "sample_customers": customers, "sample_rows": sample.height,
        "training_target_months": "2015-10 through 2016-03",
        "training_pairs": training.height, "sampled_candidate_rows": len(y),
        "negative_sampling_fraction": negative_fraction,
        "validation_month": "2016-04", "validation_map_at_7": best_score,
        "test_month": "2016-05", "selected_model_test": final_test,
        "popularity_test": baseline_test,
        "buyer_classifier": {"model": "catboost", "depth": 4, "iterations": 150,
                             "threshold_selection": "maximum F1 on April validation",
                             "threshold": buyer_threshold, "validation": buyer_validation["classifier"]},
        "two_stage_test": two_stage_test,
        "feature_importance_model": best_learned["kind"],
        "evaluated_population": "consecutive pairs with all 24 holdings observed at both dates",
        "test_month_sample_rows": sample.filter(pl.col("snapshot_date") == date(2016, 5, 28)).height,
        "sklearn_version": sklearn.__version__,
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame([
        {"population": "All customers (ungated)", "customers": test.height,
         "map_at_7": two_stage_test["map_at_7_all_customers"]},
        {"population": "Actual acquiring customers (known afterward)", "customers": two_stage_test["actual_buyers"],
         "map_at_7": two_stage_test["map_at_7_actual_buyers"]},
        {"population": "Classifier-selected customers", "customers": two_stage_test["classifier"]["selected_customers"],
         "map_at_7": two_stage_test["map_at_7_predicted_buyers"]},
    ]).to_csv(report_dir / "three_scores.csv", index=False)
    print(f"Selected: {best_name}; untouched May MAP@7={final_test['map_at_7']:.5f}", flush=True)
    # After evaluation, refit fixed parameters through May for June-serving use.
    refit = pairs.filter(pl.col("target_date").is_between(date(2015, 10, 1), date(2016, 5, 31)))
    if best_name == "popularity":
        serving = {"kind": "popularity", "rates": popularity_rates(refit)}
    else:
        final_x, final_y = candidate_training_data(refit, negative_fraction, seed)
        if best_name == "logistic":
            final_model = logistic_model(best_params["C"], seed)
        else:
            final_model = catboost_ranker(best_params["depth"], seed)
        final_model.fit(final_x, final_y)
        serving = {"kind": best_name, "model": final_model}
    serving["buyer_model"] = fit_buyer_classifier(refit, seed)
    serving["buyer_threshold"] = buyer_threshold
    serving["metadata"] = {**metrics, "refit_through": "2016-05-28", "artifact_version": 2}
    joblib.dump(serving, output_dir / "model.joblib")
    print(f"Saved {output_dir / 'model.joblib'}", flush=True)
    return results, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/train/train_ver2.csv.zip"))
    parser.add_argument("--customers", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-fraction", type=float, default=0.03)
    args = parser.parse_args()
    run_training(args.data, args.customers, args.seed, args.negative_fraction)
