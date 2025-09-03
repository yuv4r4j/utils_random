"""
One-shot size reduction for a large CSV when you have plenty of RAM (~128GB).

Actions:
- Load CSV once with pandas (pyarrow engine).
- Downcast numeric columns aggressively but safely.
- Convert low-cardinality text to 'category' with rare values -> 'OTHER'.
- Optionally drop mostly-missing and near-constant columns.
- Save a single compressed Parquet.

Requires: pandas>=2.0, pyarrow
"""

from __future__ import annotations
import os
from typing import Dict, Iterable, Optional, Tuple
import pandas as pd

# =================== CONFIG ===================
CSV_PATH = "/path/to/huge.csv"                   # <-- change
OUT_PARQUET = "/path/to/reduced_dataset.parquet" # <-- change
USECOLS: Optional[Iterable[str]] = None          # keep None to read all

# Dropping rules (set to None/0.0 to disable)
DROP_MISSING_FRAC = 0.98        # drop columns with >=98% missing
DROP_SINGLE_VALUE_FRAC = 0.995  # drop columns where modal value ≥99.5% of non-nulls

# Categorical bucketing rules
LOW_CARD_FRACTION = 0.01        # if nunique / nrows ≤ 1% -> treat as low-cardinal
CAT_TOP_K = 500                 # keep top-K categories; others -> "OTHER"
RARE_MIN_COUNT = 50             # also bucket categories with fewer than this count

# Parquet options
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 6
# ==============================================

def have_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False

def bytes_readable(num: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if num < 1024.0: return f"{num:,.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PB"

# ---------- Numeric helpers ----------
_INT_BOUNDS = {
    "Int8": (-128, 127),
    "Int16": (-32768, 32767),
    "Int32": (-2147483648, 2147483647),
    "Int64": (-(2**63), 2**63 - 1),
}
def choose_int_dtype(s: pd.Series) -> str:
    # Work with numeric coercion for robustness
    x = pd.to_numeric(s, errors="coerce")
    mn, mx = x.min(skipna=True), x.max(skipna=True)
    for dtype, (lo, hi) in _INT_BOUNDS.items():
        if pd.notna(mn) and pd.notna(mx) and mn >= lo and mx <= hi:
            return dtype
    return "Int64"

# ---------- Load ----------
def load_csv_once(path: str, usecols=None) -> pd.DataFrame:
    kwargs = dict(usecols=usecols)
    if have_pyarrow():
        kwargs["engine"] = "pyarrow"
        try:
            kwargs["dtype_backend"] = "pyarrow"  # Arrow memory layout
        except TypeError:
            pass
    print("Reading CSV …")
    df = pd.read_csv(path, **kwargs)
    # Normalize pandas dtypes where convenient
    df = df.convert_dtypes()
    print("Loaded shape:", df.shape)
    print("Approx memory:", bytes_readable(df.memory_usage(deep=True).sum()))
    return df

# ---------- Drops ----------
def apply_drops(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    dropped = {}
    n = len(df)
    # Drop mostly-missing
    if DROP_MISSING_FRAC is not None and DROP_MISSING_FRAC > 0:
        miss_frac = df.isna().sum() / max(n, 1)
        to_drop = miss_frac[miss_frac >= DROP_MISSING_FRAC].index.tolist()
        if to_drop:
            df = df.drop(columns=to_drop)
            for c in to_drop: dropped[c] = 1.0  # mark as dropped
            print(f"Dropped {len(to_drop)} columns for missingness ≥ {DROP_MISSING_FRAC:.3f}")
    # Drop near-constant (based on top frequency among non-nulls)
    if DROP_SINGLE_VALUE_FRAC is not None and DROP_SINGLE_VALUE_FRAC > 0:
        const_drop = []
        for c in df.columns:
            s = df[c].dropna()
            if s.empty: 
                continue
            top = s.value_counts().iloc[0] / len(s)
            if top >= DROP_SINGLE_VALUE_FRAC:
                const_drop.append(c)
        if const_drop:
            df = df.drop(columns=const_drop)
            for c in const_drop: dropped[c] = 1.0
            print(f"Dropped {len(const_drop)} near-constant columns (modal freq ≥ {DROP_SINGLE_VALUE_FRAC:.3f})")
    return df, dropped

# ---------- Numerics ----------
def downcast_numerics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    casts: Dict[str, str] = {}
    num_cols = df.select_dtypes(include=["number","integer","floating"]).columns
    for c in num_cols:
        s = df[c]
        if pd.api.types.is_integer_dtype(s):
            dtype = choose_int_dtype(s)
            df[c] = pd.to_numeric(s, errors="coerce").astype(dtype)
            casts[c] = dtype
        elif pd.api.types.is_float_dtype(s):
            # prefer float32 (usually fine for training; change if you need float64)
            df[c] = pd.to_numeric(s, errors="coerce").astype("float32")
            casts[c] = "float32"
    return df, casts

# ---------- Booleans ----------
def normalize_booleans(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    # Map common string forms to boolean dtype
    normalized = 0
    for c in df.columns:
        if str(df[c].dtype) == "string":
            vals = set(df[c].dropna().unique().tolist())
            if vals <= {"True","False","true","false","0","1"}:
                df[c] = df[c].map({"True": True, "true": True, "1": True,
                                   "False": False, "false": False, "0": False}).astype("boolean")
                normalized += 1
    return df, normalized

# ---------- Categoricals ----------
def handle_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Choose low-card columns and convert to category with rare bucketing.
    High-card text remains as efficient string (Arrow-backed when available).
    """
    n = len(df)
    changed: Dict[str, int] = {}
    # consider object/string columns only
    text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or str(df[c].dtype).startswith("string")]
    for c in text_cols:
        s = df[c].astype("string")
        nun = s.nunique(dropna=True)
        if nun == 0:
            continue
        frac = nun / max(n, 1)
        if frac <= LOW_CARD_FRACTION:
            # low-card: bucket rare to OTHER, use pandas "category"
            counts = s.value_counts(dropna=True)
            keep = counts.nlargest(CAT_TOP_K)
            kept = set(keep.index[keep.values >= RARE_MIN_COUNT])
            if not kept:
                kept = {keep.index[0]} if len(keep) else set()
            if kept:
                df[c] = s.where(s.isin(kept), other="OTHER").astype("category")
                changed[c] = nun
            else:
                # fallback: leave as string
                pass
        else:
            # high-card: leave as string; Arrow-backed 'string[pyarrow]' if available
            if have_pyarrow():
                try:
                    df[c] = s.astype("string[pyarrow]")
                except TypeError:
                    df[c] = s.astype("string")
            else:
                df[c] = s.astype("string")
    return df, changed

# ---------- Save ----------
def save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pq_kwargs = dict(engine="pyarrow", compression=PARQUET_COMPRESSION)
    try:
        pq_kwargs["compression_level"] = PARQUET_COMPRESSION_LEVEL
    except TypeError:
        pass
    df.to_parquet(path, index=False, **pq_kwargs)
    print("Saved Parquet ->", path)

# ---------- Driver ----------
def main():
    df = load_csv_once(CSV_PATH, USECOLS)

    # Optional: drop junk columns first (saves work)
    df, dropped = apply_drops(df)

    # Normalize booleans from string forms
    df, norm_bools = normalize_booleans(df)

    # Downcast numeric columns
    df, num_casts = downcast_numerics(df)

    # Handle categoricals (low-card -> category + OTHER; high-card -> string[pyarrow]/string)
    df, cats_changed = handle_categoricals(df)

    print("\n=== Summary ===")
    print(f"Shape after transforms: {df.shape}")
    print(f"Memory after: {bytes_readable(df.memory_usage(deep=True).sum())}")
    print(f"Dropped columns: {len(dropped)}")
    print(f"Downcast numerics: {len(num_casts)} (e.g., {list(num_casts.items())[:5]} …)")
    print(f"Categoricals made: {len(cats_changed)} (e.g., {list(cats_changed.items())[:5]} …)")
    print(f"Boolean normalized: {norm_bools}")

    save_parquet(df, OUT_PARQUET)

if __name__ == "__main__":
    assert have_pyarrow(), "Please: pip install pyarrow"
    main()