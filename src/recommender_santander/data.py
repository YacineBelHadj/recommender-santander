"""Read customer histories and create next-month acquisition labels."""
from pathlib import Path
from zipfile import ZipFile
import shutil

import polars as pl

from .columns import COLUMN_NAMES, FEATURES, PRODUCTS


def training_csv(path: Path) -> Path:
    # Use a CSV directly; only ZIP inputs need extraction.
    if path.suffix != ".zip":
        if not path.is_file():
            raise FileNotFoundError(f"Training file not found: {path}. See README download instructions.")
        return path
    # train_ver2.csv.zip becomes train_ver2.csv. Reuse it if already extracted.
    csv_path = path.with_suffix("")
    if not csv_path.exists():
        with ZipFile(path) as archive:
            # Match the filename even when the archive stores it inside a folder.
            name = next(n for n in archive.namelist() if Path(n).name == csv_path.name)
            with archive.open(name) as source, csv_path.open("wb") as target:
                # Copy in chunks rather than loading the entire CSV into memory.
                shutil.copyfileobj(source, target)
    return csv_path


def load_sample(source: Path, customers=5_000, seed=42) -> pl.DataFrame:
    """Select fixed customer IDs from the raw CSV and clean their histories in memory."""
    source = training_csv(source)
    # Scan lazily and start with text columns: this CSV contains mixed types.
    raw = pl.scan_csv(source, infer_schema=False).with_columns(pl.col("ncodpers").str.strip_chars())
    # Sort all unique IDs before sampling so the same seed gives the same cohort.
    # Only the ID list is collected here, not the full training table.
    ids = raw.select(pl.col("ncodpers").cast(pl.Int64)).unique().sort("ncodpers").collect(engine="streaming")
    if not 1 <= customers <= ids.height:
        raise ValueError(f"customers must be between 1 and {ids.height}")
    selected = ids.sample(n=customers, seed=seed).sort("ncodpers")
    # Keep every month for the selected customers, rather than sampling rows.
    customer_rows = raw.with_columns(pl.col("ncodpers").cast(pl.Int64))
    # A semi join filters on matching IDs without adding columns from selected.
    customer_rows = customer_rows.join(selected.lazy(), on="ncodpers", how="semi")
    # Materialize the smaller sample and apply the English column names.
    sample = customer_rows.collect(engine="streaming").rename(COLUMN_NAMES)
    # Strip spaces first so values such as " NA" match the missing markers.
    text_columns = [c for c, dtype in sample.schema.items() if dtype == pl.String]
    sample = sample.with_columns([pl.col(c).str.strip_chars() for c in text_columns])
    sample = sample.with_columns([
        pl.when(pl.col(c).is_in(["", "NA", "N/A", "null", "None", "nan"]))
        .then(None).otherwise(pl.col(c)).alias(c) for c in text_columns
    ])
    # strict=False converts unparseable numeric text to null.
    # snapshot_month supplies the model with the month number (1–12).
    sample = sample.with_columns(
        pl.col("snapshot_date").str.to_date("%Y-%m-%d"),
        *[pl.col(c).cast(pl.Float64, strict=False).alias(c)
          for c in ["age", "tenure_months", "household_income"] + PRODUCTS],
    ).with_columns(pl.col("snapshot_date").dt.month().cast(pl.Float64).alias("snapshot_month"))
    # Check non-null numeric holdings are binary; missing holdings stay unknown.
    invalid = sample.select([
        (~pl.col(c).is_in([0., 1.])).fill_null(False).sum().alias(c) for c in PRODUCTS
    ]).row(0)
    if any(invalid):
        raise ValueError("Non-binary product holdings found; inspect the source data")
    # Replace implausible values with null, without discarding customer histories.
    sample = sample.with_columns(
        *[pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c)
          for c in ["tenure_months", "household_income"]],
        pl.when(pl.col("age").is_between(0, 120)).then(pl.col("age")).otherwise(None).alias("age"),
    ).sort(["customer_id", "snapshot_date"])
    # Each customer should have at most one record for a given snapshot date.
    if sample.select(pl.struct("customer_id", "snapshot_date").n_unique()).item() != sample.height:
        raise ValueError("Duplicate customer-month records")
    return sample


def monthly_pairs(sample: pl.DataFrame) -> pl.DataFrame:
    """Use last month's profile to predict this month's new products."""
    ordered = sample.sort(["customer_id", "snapshot_date"])
    # shift(1) takes the previous row; over() prevents crossing customer boundaries.
    previous_values = [
        pl.col(column).shift(1).over("customer_id").alias(f"previous__{column}")
        for column in FEATURES
    ]
    previous_date = pl.col("snapshot_date").shift(1).over("customer_id").alias("feature_date")
    shifted = ordered.with_columns(previous_values + [previous_date])

    # Compare calendar months, including December → January.
    # Keep only consecutive months; the first observation has no previous date.
    current_month = pl.col("snapshot_date").dt.year() * 12 + pl.col("snapshot_date").dt.month()
    previous_month = pl.col("feature_date").dt.year() * 12 + pl.col("feature_date").dt.month()
    shifted = shifted.filter(current_month - previous_month == 1)

    # Require all product holdings at both dates: null does not mean unowned.
    observed_holdings = [
        pl.col(product).is_not_null() & pl.col(f"previous__{product}").is_not_null()
        for product in PRODUCTS
    ]
    shifted = shifted.filter(pl.all_horizontal(observed_holdings))

    # Model inputs come from the earlier month, renamed back to their usual names.
    features = [pl.col(f"previous__{column}").alias(column) for column in FEATURES]
    targets = []
    # A 0 → 1 change is a new acquisition. Other observed transitions get label 0.
    # For example: January holdings/profile → February acquisition labels.
    for product in PRODUCTS:
        previously_unowned = pl.col(f"previous__{product}") == 0
        now_owned = pl.col(product) == 1
        targets.append((previously_unowned & now_owned).cast(pl.Int8).alias(f"new__{product}"))

    # Keep both dates so training can split by the target month without leaking it.
    return shifted.select(
        "customer_id", "feature_date", pl.col("snapshot_date").alias("target_date"),
        *features, *targets,
    )
