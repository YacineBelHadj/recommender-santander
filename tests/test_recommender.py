"""Check temporal labels, recommendation eligibility, and API validation."""
from datetime import date
import numpy as np
import pandas as pd
import polars as pl
from fastapi.testclient import TestClient

from recommender_santander.columns import FEATURES, PRODUCTS
from recommender_santander.data import monthly_pairs
from recommender_santander.modeling import evaluate, score_products, evaluate_two_stage, choose_buyer_threshold
from predict import app


def test_targets_use_previous_features_and_skip_gaps_and_unknowns():
    records = []
    for customer, months, holdings in [(1, [1, 2], [0, 1]), (2, [1, 3], [0, 1]), (3, [1, 2], [None, 1])]:
        for month, holding in zip(months, holdings):
            row = {c: None for c in FEATURES}
            row.update({p: 0. for p in PRODUCTS})
            row.update(customer_id=customer, snapshot_date=date(2016, month, 28), age=20.+month)
            row[PRODUCTS[0]] = holding
            records.append(row)
    pairs = monthly_pairs(pl.DataFrame(records))
    assert pairs.height == 1
    assert pairs['customer_id'].item() == 1
    assert pairs['age'].item() == 21  # January feature, February target.
    assert pairs[f'new__{PRODUCTS[0]}'].item() == 1


def test_map_and_owned_product_filter():
    row = {c: None for c in FEATURES}
    row.update({p: 0. for p in PRODUCTS})
    row[PRODUCTS[0]] = 1.
    row.update({f'new__{p}': int(p == PRODUCTS[1]) for p in PRODUCTS})
    bundle = {'kind': 'popularity', 'rates': np.arange(24, 0, -1)}
    scores = score_products(bundle, pd.DataFrame([row]))
    assert scores[0, 0] == -np.inf
    assert evaluate(bundle, pl.DataFrame([row]))['map_at_7'] == 1.
    row[f'new__{PRODUCTS[1]}'] = 0
    assert evaluate(bundle, pl.DataFrame([row]))['map_at_7'] == 0.


class FixedBuyerScores:
    def __init__(self, scores):
        self.scores = np.asarray(scores)

    def predict_proba(self, frame):
        assert set(frame.columns) == set(FEATURES)  # No acquisition labels as features.
        scores = np.repeat(self.scores, len(frame)) if len(self.scores) == 1 else self.scores
        return np.column_stack([1 - scores, scores])


def test_three_populations_and_abstention_denominator():
    rows = []
    for acquired in [0, 1, None]:
        row = {c: None for c in FEATURES}
        row.update({p: 0. for p in PRODUCTS})
        row.update({f'new__{p}': int(i == acquired) for i, p in enumerate(PRODUCTS)})
        rows.append(row)
    pairs = pl.DataFrame(rows)
    ranking = {'kind': 'popularity', 'rates': np.arange(24, 0, -1)}
    classifier = FixedBuyerScores([.9, .1, .8])
    result = evaluate_two_stage(ranking, classifier, .5, pairs)
    assert result['map_at_7_all_customers'] == .5
    assert result['map_at_7_actual_buyers'] == .75
    assert result['map_at_7_predicted_buyers'] == .5
    assert result['map_at_7_gated_all_customers'] == 1 / 3
    assert result['classifier']['precision'] == .5
    assert result['classifier']['recall'] == .5
    assert result['classifier']['coverage'] == 2 / 3
    assert choose_buyer_threshold(classifier, pairs) == .1
    empty = evaluate_two_stage(ranking, FixedBuyerScores([0.]), .5, pairs)
    assert empty['map_at_7_predicted_buyers'] is None
    assert empty['map_at_7_gated_all_customers'] == 0.


def test_service_rejects_unknown_products_and_never_recommends_owned_products(monkeypatch):
    with TestClient(app) as client:
        monkeypatch.setitem(app.state.bundle, 'buyer_model', FixedBuyerScores([1.]))
        assert client.get('/health').status_code == 200
        request = {'profile': {'snapshot_month': 5, 'age': 35}, 'owned_products': PRODUCTS[:3]}
        response = client.post('/recommend', json=request)
        assert response.status_code == 200
        recommendations = response.json()['recommendations']
        assert response.json()['should_recommend'] is True
        assert len(recommendations) == 7
        assert not {r['product'] for r in recommendations} & set(PRODUCTS[:3])
        request['owned_products'] = ['invalid']
        assert client.post('/recommend', json=request).status_code == 422
        request['owned_products'] = PRODUCTS
        assert client.post('/recommend', json=request).json()['recommendations'] == []
        request['owned_products'] = []
        monkeypatch.setitem(app.state.bundle, 'buyer_model', FixedBuyerScores([0.]))
        abstained = client.post('/recommend', json=request).json()
        assert abstained['should_recommend'] is False
        assert abstained['recommendations'] == []
