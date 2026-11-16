"""Shared feature preparation, recommendation ranking, and evaluation."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .columns import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, PRODUCTS

MODEL_CATEGORIES = CATEGORICAL_FEATURES + ["product"]
MODEL_NUMERIC = NUMERIC_FEATURES + PRODUCTS


def feature_frame(frame):
    result = frame[FEATURES].copy()
    for c in MODEL_NUMERIC:
        result[c] = pd.to_numeric(result[c], errors="coerce").astype(float)
    for c in CATEGORICAL_FEATURES:
        result[c] = result[c].fillna("Unknown").astype(str)
    return result


def candidate_training_data(pairs, negative_fraction=0.03, seed=42):
    """Keep all acquisitions and a uniform fraction of eligible negatives."""
    if not 0 < negative_fraction <= 1:
        raise ValueError("negative_fraction must be in (0, 1]")
    frame = pairs.to_pandas()
    features = feature_frame(frame)
    rng = np.random.default_rng(seed)
    chunks, labels = [], []
    for product in PRODUCTS:
        y = frame[f"new__{product}"].to_numpy()
        eligible = frame[product].to_numpy() == 0
        keep = eligible & ((y == 1) | (rng.random(len(frame)) < negative_fraction))
        chunk = features.loc[keep].copy()
        chunk["product"] = product
        chunks.append(chunk)
        labels.append(y[keep])
    x, y = pd.concat(chunks, ignore_index=True), np.concatenate(labels)
    if len(np.unique(y)) != 2:
        raise ValueError("Training data needs both positive and negative examples; increase the customer sample")
    return x, y


def logistic_model(C, seed=42):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, MODEL_NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), MODEL_CATEGORIES),
    ])
    return Pipeline([
        ("features", preprocessing),
        ("classifier", LogisticRegression(C=C, solver="lbfgs", max_iter=500, random_state=seed)),
    ])


def catboost_ranker(depth, seed=42):
    return CatBoostClassifier(
        iterations=150, depth=depth, learning_rate=0.1,
        cat_features=MODEL_CATEGORIES, random_seed=seed, thread_count=2,
        loss_function="Logloss", verbose=False, allow_writing_files=False,
    )


def popularity_rates(pairs):
    frame = pairs.to_pandas()
    return np.array([
        frame[f"new__{c}"].sum() / max((frame[c] == 0).sum(), 1) for c in PRODUCTS
    ])


def score_products(bundle, frame):
    features = feature_frame(frame)
    scores = np.full((len(frame), len(PRODUCTS)), -np.inf)
    for index, product in enumerate(PRODUCTS):
        eligible = features[product].to_numpy() == 0
        if not eligible.any():
            continue
        if bundle["kind"] == "popularity":
            scores[eligible, index] = bundle["rates"][index]
        else:
            candidate = features.loc[eligible].copy()
            candidate["product"] = product
            scores[eligible, index] = bundle["model"].predict_proba(candidate)[:, 1]
    return scores


def ranked_indices(scores, k=7):
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def per_customer_ap(scores, actual):
    """One AP@7 per customer, including zero for customers with no acquisitions."""
    ranked = ranked_indices(scores)
    relevant = np.take_along_axis(actual, ranked, axis=1)
    eligible = np.isfinite(np.take_along_axis(scores, ranked, axis=1))
    hits = relevant * eligible

    ranks = np.arange(1, ranked.shape[1] + 1)
    precision_at_rank = hits.cumsum(axis=1) / ranks
    actual_acquisitions = actual.sum(axis=1)
    # Dividing by 1 for non-buyers leaves their zero score unchanged.
    denominator = np.maximum(np.minimum(actual_acquisitions, 7), 1)
    return (precision_at_rank * hits).sum(axis=1) / denominator


def evaluate(bundle, pairs):
    frame = pairs.to_pandas()
    scores = score_products(bundle, frame)
    actual = frame[[f"new__{c}" for c in PRODUCTS]].to_numpy()
    ap = per_customer_ap(scores, actual)
    positive = actual.sum(axis=1) > 0
    return {"map_at_7": float(ap.mean()), "customers": len(frame),
            "customers_with_additions": int(positive.sum()),
            "map_at_7_positive_customers": float(ap[positive].mean()) if positive.any() else 0.0}


def fit_buyer_classifier(training, seed=42):
    """Predict any acquisition using all customer pairs, without negative sampling."""
    frame = training.to_pandas()
    target = frame[[f"new__{c}" for c in PRODUCTS]].sum(axis=1).gt(0).astype(int)
    if target.nunique() != 2:
        raise ValueError("Buyer classification needs acquiring and non-acquiring customers")
    model = CatBoostClassifier(
        iterations=150, depth=4, learning_rate=0.1, cat_features=CATEGORICAL_FEATURES,
        random_seed=seed, thread_count=2, loss_function="Logloss", verbose=False,
        allow_writing_files=False,
    )
    model.fit(feature_frame(frame), target)
    return model


def choose_buyer_threshold(model, validation):
    """Choose the threshold by F1 on April only; never use May labels."""
    frame = validation.to_pandas()
    actual = frame[[f"new__{c}" for c in PRODUCTS]].sum(axis=1).gt(0).to_numpy()
    if len(np.unique(actual)) != 2:
        raise ValueError("Threshold validation needs acquiring and non-acquiring customers")
    scores = model.predict_proba(feature_frame(frame))[:, 1]
    precision, recall, thresholds = precision_recall_curve(actual, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.argmax(f1)])


def classifier_metrics(actual, scores, threshold):
    selected = scores >= threshold
    return {
        "threshold": threshold, "customers": len(actual), "selected_customers": int(selected.sum()),
        "selected_actual_buyers": int((selected & actual).sum()),
        "coverage": float(selected.mean()),
        "precision": float(precision_score(actual, selected, zero_division=0)),
        "recall": float(recall_score(actual, selected, zero_division=0)),
        "f1": float(f1_score(actual, selected, zero_division=0)),
        "average_precision": float(average_precision_score(actual, scores)),
    }


def evaluate_two_stage(bundle, buyer_model, threshold, pairs):
    frame = pairs.to_pandas()
    actual_products = frame[[f"new__{c}" for c in PRODUCTS]].to_numpy()
    actual_buyers = actual_products.sum(axis=1) > 0
    ap = per_customer_ap(score_products(bundle, frame), actual_products)
    buyer_scores = buyer_model.predict_proba(feature_frame(frame))[:, 1]
    selected = buyer_scores >= threshold
    return {
        "map_at_7_all_customers": float(ap.mean()),
        "map_at_7_actual_buyers": float(ap[actual_buyers].mean()) if actual_buyers.any() else None,
        "map_at_7_predicted_buyers": float(ap[selected].mean()) if selected.any() else None,
        # Abstention receives zero while the full-population denominator stays fixed.
        "map_at_7_gated_all_customers": float((ap * selected).mean()),
        "actual_buyers": int(actual_buyers.sum()),
        "classifier": classifier_metrics(actual_buyers, buyer_scores, threshold),
    }
