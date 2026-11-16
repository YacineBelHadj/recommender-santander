"""History features known at the end of each observed customer month."""
import polars as pl
from .columns import PRODUCTS

HISTORY_FEATURES = [
    "products_owned", "observed_history_months", "additions_last_month", "removals_last_month",
    "additions_3m", "removals_3m", "observed_changes_3m",
    "months_since_observed_acquisition", "activity_change",
]


def history_features(sample):
    frame = sample.sort(["customer_id", "snapshot_date"]).with_columns(
        (pl.col("snapshot_date").dt.year() * 12 + pl.col("snapshot_date").dt.month()).alias("month_index"),
        *[pl.col(p).shift(1).over("customer_id").alias(f"prior__{p}") for p in PRODUCTS],
    ).with_columns(
        pl.col("month_index").diff().over("customer_id").alias("month_gap"),
        pl.col("customer_id").cum_count().over("customer_id").cast(pl.Float64).alias("observed_history_months"),
        pl.sum_horizontal(PRODUCTS).alias("products_owned"),
    )
    comparable = (pl.col("month_gap") == 1) & pl.all_horizontal([
        pl.col(p).is_not_null() & pl.col(f"prior__{p}").is_not_null() for p in PRODUCTS
    ])
    additions = pl.sum_horizontal([((pl.col(f"prior__{p}") == 0) & (pl.col(p) == 1)).cast(pl.Int16) for p in PRODUCTS])
    removals = pl.sum_horizontal([((pl.col(f"prior__{p}") == 1) & (pl.col(p) == 0)).cast(pl.Int16) for p in PRODUCTS])
    activity = pl.col("is_active_customer").cast(pl.Float64, strict=False)
    frame = frame.with_columns(
        pl.when(comparable).then(additions).otherwise(None).alias("additions_last_month"),
        pl.when(comparable).then(removals).otherwise(None).alias("removals_last_month"),
        pl.when(pl.col("month_gap") == 1)
        .then(activity - activity.shift(1).over("customer_id")).otherwise(None).alias("activity_change"),
        (~comparable.fill_null(False)).cast(pl.Int32).cum_sum().over("customer_id").alias("history_segment"),
    )
    # Rolling windows restart after gaps or unknown holdings. A partial window is counted explicitly.
    groups = ["customer_id", "history_segment"]
    frame = frame.with_columns(
        pl.col("additions_last_month").rolling_sum(3, min_samples=1).over(groups).alias("additions_3m"),
        pl.col("removals_last_month").rolling_sum(3, min_samples=1).over(groups).alias("removals_3m"),
        pl.col("additions_last_month").is_not_null().cast(pl.Int16)
        .rolling_sum(3, min_samples=1).over(groups).alias("observed_changes_3m"),
        pl.when(pl.col("additions_last_month") > 0).then(pl.col("month_index"))
        .otherwise(None).forward_fill().over("customer_id").alias("last_observed_acquisition"),
    ).with_columns(
        (pl.col("month_index") - pl.col("last_observed_acquisition")).cast(pl.Float64)
        .alias("months_since_observed_acquisition"),
    )
    return frame.select("customer_id", pl.col("snapshot_date").alias("feature_date"), *HISTORY_FEATURES)
