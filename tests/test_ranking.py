"""History must be known at prediction time; ranking groups must stay together."""
from datetime import date
import numpy as np
import polars as pl

from recommender_santander.columns import FEATURES, PRODUCTS
from recommender_santander.data import monthly_pairs
from recommender_santander.history import history_features
from recommender_santander.modeling import candidate_training_data
from experiments.ranking import candidates


def sample_histories():
    rows = []
    for customer in [1, 2]:
        for month in [1, 2, 3, 5, 6]:
            row = {c: None for c in FEATURES}
            row.update({p: 0. for p in PRODUCTS})
            row.update(customer_id=customer, snapshot_date=date(2016, month, 28), age=35.,
                       is_active_customer='1', snapshot_month=float(month))
            row[PRODUCTS[0]] = float(month in [2, 3, 6])
            rows.append(row)
    return pl.DataFrame(rows)


def test_history_does_not_see_future_and_resets_rolling_windows_at_gaps():
    sample = sample_histories()
    history = history_features(sample)
    february = history.filter(pl.col('feature_date') == date(2016, 2, 28))
    assert february['additions_last_month'].to_list() == [1, 1]
    assert february['months_since_observed_acquisition'].to_list() == [0., 0.]
    may = history.filter(pl.col('feature_date') == date(2016, 5, 28))
    assert may['additions_3m'].null_count() == 2
    assert may['observed_changes_3m'].to_list() == [0, 0]
    changed = sample.with_columns(
        pl.when(pl.col('snapshot_date') == date(2016, 6, 28))
        .then(1.).otherwise(pl.col(PRODUCTS[1])).alias(PRODUCTS[1])
    )
    before = pl.col('feature_date') < date(2016, 6, 28)
    assert history.filter(before).equals(history_features(changed).filter(before))


def test_ranking_groups_are_contiguous_and_classifier_sampling_matches_baseline():
    sample = sample_histories()
    pairs = monthly_pairs(sample).join(history_features(sample), on=['customer_id', 'feature_date'])
    x, y, groups = candidates(pairs, history=True, ranking=True)
    assert np.all(groups[:-1] <= groups[1:])
    for group in np.unique(groups):
        labels = y[groups == group]
        assert set(labels) == {0, 1}
        assert len(labels) == 24  # All 24 products were unowned in these positive groups.
    assert not {'group', 'label', 'customer_id', 'target_date'} & set(x.columns)
    x, y, _ = candidates(pairs, history=False, ranking=False)
    baseline_x, baseline_y = candidate_training_data(pairs, seed=42)
    assert x.equals(baseline_x)
    assert np.array_equal(y, baseline_y)


def test_both_rankers_fit_and_score_eligible_products():
    from experiments.ranking import fit_model, predictions
    sample = sample_histories()
    pairs = monthly_pairs(sample).join(history_features(sample), on=['customer_id', 'feature_date'])
    for kind in ['catboost_ranker', 'lightgbm_ranker']:
        model = fit_model(kind, history=True, training=pairs)
        ap, buyers = predictions(model, pairs)
        assert np.isfinite(ap).all()
        assert ((ap >= 0) & (ap <= 1)).all()
        assert buyers.sum() == 4
        if kind == 'lightgbm_ranker':
            import warnings
            unseen = pairs.with_columns(pl.lit('UNSEEN_COUNTRY').alias('residence_country'))
            with warnings.catch_warnings():
                warnings.simplefilter('error', FutureWarning)
                unseen_ap, _ = predictions(model, unseen)
            assert np.isfinite(unseen_ap).all()
