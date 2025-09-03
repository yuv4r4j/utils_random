"""
pip install polars pyarrow zstandard

One-shot size reduction with POLARS (loads CSV once, ~128GB RAM available)

What it does:
- Read CSV once with Polars (fast, memory-efficient).
- Drop mostly-missing and near-constant columns (configurable).
- Downcast integers to the smallest safe width; cast floats -> Float32.
- Convert low-cardinality strings to Categorical with rare values -> "OTHER".
- Write a single ZSTD-compressed Parquet.

Tune the CONFIG section for your dataset.
"""

from __future__ import annotations
import os
from typing import Dict, Iterable, Optional, Tuple

import polars as pl

# ====================== CONFIG ======================
CSV_PATH = "/path/to/huge.csv"                   # <-- change
OUT_PARQUET = "/path/to/reduced_dataset.parquet" # <-- change
USECOLS: Optional[Iterable[str]] = None          # list of columns to read; None = all

# Drop rules
DROP_MISSING_FRAC = 0.98        # drop columns with >= 98% missing
DROP_SINGLE_VALUE_FRAC = 0.995  # drop columns where modal value covers >= 99.5% of non-nulls

# Categorical rules
LOW_CARD_FRACTION = 0.01        # if nunique / nrows <= 1% -> treat as low-card candidate
CAT_TOP_K = 500                 # keep top-K categories
RARE_MIN_COUNT = 50             # bucket cats with < 50 occurrences to "OTHER"

# Parquet compression
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 6
# ====================================================

def bytes_readable(num: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if num < 1024:
            return f"{num:,.2f} {unit}"
        num /= 1024
    return f"{num:.2f} PB"

def load_csv_once(path: str, usecols=None) -> pl.DataFrame:
    read_kwargs = dict(
        infer_schema_length=10_000, # more robust inference
        ignore_errors=True,
        try_parse_dates=True,
        rechunk=True,
    )
    if usecols is not None:
        read_kwargs["columns"] = list(usecols)

    print("Reading CSV with Polars …")
    df = pl.read_csv(path, **read_kwargs)
    try:
        print("Loaded shape:", df.shape, "| approx memory:", bytes_readable(df.estimated_size()))
    except Exception:
        print("Loaded shape:", df.shape)
    return df

def apply_missing_and_constant_drops(df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, str]]:
    dropped: Dict[str, str] = {}
    n = df.height
    # Drop mostly-missing
    if DROP_MISSING_FRAC is not None and DROP_MISSING_FRAC > 0:
        miss = df.null_count().row(0)  # dict-like of counts
        to_drop = [c for c, cnt in miss.items() if n and (cnt / n) >= DROP_MISSING_FRAC]
        if to_drop:
            df = df.drop(to_drop)
            for c in to_drop: dropped[c] = "mostly_missing"
            print(f"Dropped {len(to_drop)} columns for missingness ≥ {DROP_MISSING_FRAC:.3f}")

    # Drop near-constant (modal frequency)
    if DROP_SINGLE_VALUE_FRAC is not None and DROP_SINGLE_VALUE_FRAC > 0:
        const_drop = []
        for c in df.columns:
            s_non_null = df.select(pl.col(c).drop_nulls())
            if s_non_null.height == 0:
                continue
            # top frequency
            top_cnt = (
                df.select(pl.col(c).value_counts(sort=True))
                  .unnest(c)                             # -> columns: c, counts
                  .select(pl.col("counts").first())
                  .item()
            )
            frac = top_cnt / s_non_null.height
            if frac >= DROP_SINGLE_VALUE_FRAC:
                const_drop.append(c)
        if const_drop:
            df = df.drop(const_drop)
            for c in const_drop: dropped[c] = "near_constant"
            print(f"Dropped {len(const_drop)} near-constant columns (modal freq ≥ {DROP_SINGLE_VALUE_FRAC:.3f})")
    return df, dropped

def choose_int_dtype(min_val: int, max_val: int, unsigned_ok: bool) -> pl.DataType:
    if unsigned_ok and min_val >= 0:
        if max_val <= 255: return pl.UInt8
        if max_val <= 65_535: return pl.UInt16
        if max_val <= 4_294_967_295: return pl.UInt32
        return pl.UInt64
    else:
        if min_val >= -128 and max_val <= 127: return pl.Int8
        if min_val >= -32_768 and max_val <= 32_767: return pl.Int16
        if min_val >= -2_147_483_648 and max_val <= 2_147_483_647: return pl.Int32
        return pl.Int64

def downcast_numerics(df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, str]]:
    casts: Dict[str, str] = {}
    numeric_cols = [c for c, dt in df.schema.items() if pl.datatypes.is_numeric(dt)]
    if not numeric_cols:
        return df, casts

    # compute min/max for all numeric cols in one shot
    stats = df.select(
        *[pl.col(c).min().alias(f"{c}__min") for c in numeric_cols],
        *[pl.col(c).max().alias(f"{c}__max") for c in numeric_cols],
    ).to_dicts()[0]

    exprs = []
    for c in numeric_cols:
        dt = df.schema[c]
        mn = stats.get(f"{c}__min")
        mx = stats.get(f"{c}__max")
        if mn is None or mx is None:
            continue
        if pl.datatypes.is_integer(dt):
            unsigned_ok = pl.datatypes.is_unsigned_integer(dt) or (mn >= 0)
            target = choose_int_dtype(int(mn), int(mx), unsigned_ok=unsigned_ok)
            if target != dt:
                exprs.append(pl.col(c).cast(target).alias(c))
                casts[c] = str(target)
        elif pl.datatypes.is_float(dt):
            # prefer Float32 for training
            if dt != pl.Float32:
                exprs.append(pl.col(c).cast(pl.Float32).alias(c))
                casts[c] = "Float32"
    if exprs:
        df = df.with_columns(exprs).rechunk()
    return df, casts

def normalize_booleanish_strings(df: pl.DataFrame) -> Tuple[pl.DataFrame, int]:
    changed = 0
    for c, dt in df.schema.items():
        if dt == pl.Utf8:
            # If all non-null values are in {"true","false","0","1"} (case-insensitive), map to bool
            bad = df.filter(
                pl.col(c).is_not_null()
                & (~pl.col(c).str.to_lowercase().is_in(["true","false","0","1"]))
            ).height
            if bad == 0:
                expr = (
                    pl.when(pl.col(c).str.to_lowercase().is_in(["true", "1"])).then(pl.lit(True))
                      .when(pl.col(c).str.to_lowercase().is_in(["false", "0"])).then(pl.lit(False))
                      .otherwise(None)
                      .cast(pl.Boolean)
                      .alias(c)
                )
                df = df.with_columns(expr)
                changed += 1
    if changed:
        df = df.rechunk()
    return df, changed

def handle_categoricals(df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, int]]:
    """
    Convert low-cardinality Utf8 columns to Categorical with rare bucketing.
    High-card columns remain Utf8.
    """
    n = df.height
    changed: Dict[str, int] = {}
    cat_exprs = []

    # Make category mapping consistent
    with pl.StringCache():
        for c, dt in df.schema.items():
            if dt != pl.Utf8:
                continue

            nun = df.select(pl.col(c).n_unique()).item()
            if nun == 0:
                continue
            frac = nun / max(n, 1)
            if frac <= LOW_CARD_FRACTION:
                # Determine which categories to keep (top-K & min count)
                vc = (
                    df.select(pl.col(c).value_counts(sort=True))
                      .unnest(c)  # -> [c, counts]
                )
                kept = (
                    vc.filter(pl.col("counts") >= RARE_MIN_COUNT)
                      .head(CAT_TOP_K)
                      .select(pl.col(c))
                      .to_series()
                      .to_list()
                )
                if not kept:
                    # keep the single most common category if nothing passes thresholds
                    kept = vc.select(pl.col(c)).head(1).to_series().to_list()

                if kept:
                    cat_exprs.append(
                        pl.when(pl.col(c).is_in(kept))
                          .then(pl.col(c))
                          .otherwise(pl.lit("OTHER"))
                          .cast(pl.Categorical)
                          .alias(c)
                    )
                    changed[c] = nun
            else:
                # leave as Utf8
                pass

        if cat_exprs:
            df = df.with_columns(cat_exprs).rechunk()

    return df, changed

def save_parquet(df: pl.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.write_parquet(
        path,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        statistics=True,
    )
    print("Saved Parquet ->", path)

def main():
    df = load_csv_once(CSV_PATH, USECOLS)

    df, dropped = apply_missing_and_constant_drops(df)

    df, bool_changes = normalize_booleanish_strings(df)

    df, num_casts = downcast_numerics(df)

    df, cats_changed = handle_categoricals(df)

    try:
        mem_after = bytes_readable(df.estimated_size())
    except Exception:
        mem_after = "n/a"

    print("\n=== Summary ===")
    print("Shape:", df.shape, "| approx memory after:", mem_after)
    print("Dropped:", len(dropped), "columns")
    print("Downcast numerics:", len(num_casts), list(num_casts.items())[:6], "…")
    print("Categoricals made:", len(cats_changed), list(cats_changed.items())[:6], "…")
    print("Boolean normalized:", bool_changes, "columns")

    save_parquet(df, OUT_PARQUET)

    # Example: training-friendly matrices (Polars -> one-hots)
    # target = "charge_off_flag"
    # X = df.drop(target).to_dummies(drop_first=True).with_columns(pl.all().cast(pl.Float32))
    # y = df.get_column(target).cast(pl.Int8)
    # X_np, y_np = X.to_numpy(), y.to_numpy()

if __name__ == "__main__":
    main()