"""
Reduce huge CSV training data size on low RAM.

WHAT IT DOES
- Pass 1 (profile): stream the CSV to learn numeric ranges, null rates,
  and frequent categories for "object"/string columns.
- Pass 2 (transform): stream again to:
    * downcast integers to Int8/Int16/Int32/Int64 (nullable)
    * cast floats to float32 when safe
    * bucket rare categories to "OTHER" and use 'category' dtype
    * drop columns that are constant or too-missing
    * write compact Parquet chunks (zstd compression + dictionary encoding)

TUNE THE CONFIG below, especially:
- CAT_TOP_K, RARE_MIN_COUNT, LOW_CARD_FRACTION
- DROP_MISSING_FRAC, DROP_SINGLE_VALUE_FRAC
- CHUNK_SIZE

Requires: pandas>=2.0; pyarrow recommended.
"""

from __future__ import annotations
import os
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Iterable, Tuple

import pandas as pd

# ====================== CONFIG ======================
CSV_PATH = "/path/to/huge.csv"                # <- change
OUT_DIR = "/path/to/parquet_reduced"          # <- change (created if missing)
CHUNK_SIZE = 5_000                            # smaller is safer for 1400 cols on 8GB
SAMPLE_ROWS = 25_000                          # for quick initial heuristics
USECOLS: Optional[Iterable[str]] = None       # e.g., subset of columns for faster profiling

# Drop rules
DROP_MISSING_FRAC = 0.98       # drop columns with >= 98% missing
DROP_SINGLE_VALUE_FRAC = 0.995 # drop columns where the modal value covers >= 99.5% rows (near-constant)

# Categorical handling
LOW_CARD_FRACTION = 0.01       # if sample unique fraction <= 1% -> consider low-card candidate
CAT_TOP_K = 500                # keep top-K categories; others -> "OTHER"
RARE_MIN_COUNT = 50            # even within top-K, if a cat has <50 occurrences overall, bucket to "OTHER"
MAX_TRACKED_CATS = 10_000      # if a col exceeds this unique count while profiling, treat it as high-card

# Output parquet compression
PARQUET_COMPRESSION = "zstd"   # good size/CPU trade-off
PARQUET_COMPRESSION_LEVEL = 6  # tweak 1..22 for zstd, 6 is balanced

# ====================================================

def have_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ---------- Numeric downcast helpers ----------
_INT_BOUNDS = {
    "Int8": (-128, 127),
    "Int16": (-32768, 32767),
    "Int32": (-2147483648, 2147483647),
    "Int64": (-(2**63), 2**63 - 1),
}
def choose_int_dtype(mn, mx) -> str:
    for dtype, (lo, hi) in _INT_BOUNDS.items():
        if pd.notna(mn) and pd.notna(mx) and mn >= lo and mx <= hi:
            return dtype
    return "Int64"

def can_float32(min_val, max_val) -> bool:
    # float32 range is huge; this just gates absurd infinities/NaNs scenario
    return True

# ---------- Pass 0: quick sample-based hints ----------
def quick_infer_hints(path: str, nrows: int, usecols=None) -> Dict[str, str]:
    """Use a small sample to decide initial types + low-card candidates."""
    kwargs = dict(nrows=nrows, usecols=usecols)
    if have_pyarrow():
        kwargs["engine"] = "pyarrow"
    df = pd.read_csv(path, **kwargs).convert_dtypes()

    hints: Dict[str, str] = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            hints[c] = "bool"
        elif pd.api.types.is_integer_dtype(s):
            hints[c] = "int"
        elif pd.api.types.is_float_dtype(s):
            hints[c] = "float"
        elif pd.api.types.is_datetime64_any_dtype(s):
            hints[c] = "datetime"
        else:
            # text/mixed -> check low-card candidacy
            nun = s.astype("string").nunique(dropna=True)
            frac = nun / max(len(s), 1)
            hints[c] = "text_lowcard" if frac <= LOW_CARD_FRACTION else "text_highcard"
    return hints

# ---------- Pass 1: profile full file ----------
def profile_csv(path: str, hints: Dict[str, str], chunk_size: int, usecols=None):
    """
    Collect per-column stats:
      - rows, nulls
      - for ints/floats: global min/max
      - for candidate low-card text: global top category counts (Counter)
      - track modal frequency approx (via Counter for text; for numerics, treat NaN/None separately)
    """
    # initialize structures
    col_rows = defaultdict(int)
    col_nulls = defaultdict(int)
    col_min = {}
    col_max = {}
    col_mode_counts = defaultdict(int)  # track max value count seen (approx via top categories)
    cat_counters: Dict[str, Counter] = {c: Counter() for c, t in hints.items() if t.startswith("text")}
    blocked_cats: Dict[str, bool] = {c: False for c, t in hints.items() if t.startswith("text")}

    read_kwargs = dict(chunksize=chunk_size, usecols=usecols)
    if have_pyarrow():
        read_kwargs["engine"] = "pyarrow"
        try:
            read_kwargs["dtype_backend"] = "pyarrow"
        except TypeError:
            pass
    read_kwargs.setdefault("low_memory", False)

    for chunk in pd.read_csv(path, **read_kwargs):
        chunk = chunk.convert_dtypes()
        # update row/nulls
        for c in chunk.columns:
            s = chunk[c]
            n = len(s)
            col_rows[c] += n
            col_nulls[c] += s.isna().sum()

        # numerics
        num_cols = chunk.select_dtypes(include="number").columns
        for c in num_cols:
            s = pd.to_numeric(chunk[c], errors="coerce")
            mn = s.min(skipna=True)
            mx = s.max(skipna=True)
            if c not in col_min or (pd.notna(mn) and mn < col_min[c]):
                col_min[c] = mn
            if c not in col_max or (pd.notna(mx) and mx > col_max[c]):
                col_max[c] = mx

        # text counters
        text_cols = [c for c in chunk.columns if hints.get(c, "").startswith("text")]
        for c in text_cols:
            if blocked_cats[c]:
                continue
            s = chunk[c].astype("string")
            vc = s.value_counts(dropna=True)
            # update counter
            cat_counters[c].update(dict(vc))
            # trim to manageable size: keep only top (MAX_TRACKED_CATS * 1.5) temporarily
            if len(cat_counters[c]) > MAX_TRACKED_CATS * 3 // 2:
                # keep only top MAX_TRACKED_CATS by count
                for k, _ in cat_counters[c].most_common()[MAX_TRACKED_CATS:]:
                    del cat_counters[c][k]
            # if we’re already huge, mark as high-cardinal (won't category-encode)
            if len(cat_counters[c]) > MAX_TRACKED_CATS:
                blocked_cats[c] = True

        # approx modal counts
        for c in text_cols:
            if cat_counters[c]:
                top_val, top_cnt = cat_counters[c].most_common(1)[0]
                col_mode_counts[c] = max(col_mode_counts[c], top_cnt)

        # for numerics, estimate modal via rounding bins (cheap): not strictly needed for drop logic

    # Build transform plan
    plan = {
        "drop_cols": [],
        "num_int_cols": [],
        "num_float_cols": [],
        "text_lowcard_cols": [],
        "text_highcard_cols": [],
        "cat_maps": {},  # col -> set of kept categories
        "int_dtype": {}, # col -> pandas nullable int dtype name
        "float_dtype": {}, # col -> 'float32' when applicable
        "rows_per_col": dict(col_rows),
        "nulls_per_col": dict(col_nulls),
    }

    for c, total in col_rows.items():
        nulls = col_nulls.get(c, 0)
        miss_frac = (nulls / total) if total else 1.0

        # decide drop for missingness
        if miss_frac >= DROP_MISSING_FRAC:
            plan["drop_cols"].append(c)
            continue

        t = hints.get(c, "")
        if t in ("int", "float", "bool"):
            # low-variance drop check for numerics is optional; we keep numerics by default
            if t == "int":
                mn, mx = col_min.get(c, None), col_max.get(c, None)
                plan["int_dtype"][c] = choose_int_dtype(mn, mx)
                plan["num_int_cols"].append(c)
            elif t == "float":
                # prefer float32 for size (unless you know you need float64)
                plan["float_dtype"][c] = "float32" if can_float32(col_min.get(c), col_max.get(c)) else "float64"
                plan["num_float_cols"].append(c)
            else:
                # bool stays 'boolean'
                pass
        elif t.startswith("text"):
            if blocked_cats.get(c, False) or t == "text_highcard":
                plan["text_highcard_cols"].append(c)
            else:
                # choose top-K and apply rare threshold
                counts = cat_counters.get(c, Counter())
                top = counts.most_common(CAT_TOP_K)
                kept = {k for k, v in top if v >= RARE_MIN_COUNT}
                # guard against empty kept set
                if not kept and top:
                    kept = {top[0][0]}
                if kept:
                    plan["cat_maps"][c] = kept
                    plan["text_lowcard_cols"].append(c)
                else:
                    plan["text_highcard_cols"].append(c)

    # near-constant columns (mostly one value)
    for c, total in col_rows.items():
        if c in plan["drop_cols"]:
            continue
        if hints.get(c, "").startswith("text"):
            # approximate modal via counters
            counts = cat_counters.get(c, Counter())
            if not counts:
                continue
            top_cnt = counts.most_common(1)[0][1]
            if total > 0 and (top_cnt / total) >= DROP_SINGLE_VALUE_FRAC:
                plan["drop_cols"].append(c)

    return plan

# ---------- Pass 2: transform & write ----------
def apply_transform_and_write(path: str, plan: dict, chunk_size: int, out_dir: str, usecols=None):
    ensure_dir(out_dir)
    read_kwargs = dict(chunksize=chunk_size, usecols=usecols)
    if have_pyarrow():
        read_kwargs["engine"] = "pyarrow"
        try:
            read_kwargs["dtype_backend"] = "pyarrow"
        except TypeError:
            pass
    read_kwargs.setdefault("low_memory", False)

    i = 0
    total_rows = 0
    for chunk in pd.read_csv(path, **read_kwargs):
        # Drop junk columns
        keep_cols = [c for c in chunk.columns if c not in plan["drop_cols"]]
        chunk = chunk[keep_cols].convert_dtypes()

        # Numerics downcast
        for c, dtype in plan["int_dtype"].items():
            if c in chunk.columns:
                # Coerce non-numeric to NA then cast to nullable IntX
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype(dtype)
        for c, dtype in plan["float_dtype"].items():
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype(dtype)

        # Booleans stay 'boolean' (convert if they came as strings)
        for c in chunk.columns:
            if chunk[c].dtype == "string" and set(chunk[c].dropna().unique().tolist()) <= {"True", "False", "true", "false", "0", "1"}:
                chunk[c] = chunk[c].map({"True": True, "true": True, "1": True,
                                         "False": False, "false": False, "0": False}).astype("boolean")

        # Low-card text -> bucket + category
        for c, kept in plan["cat_maps"].items():
            if c in chunk.columns:
                s = chunk[c].astype("string")
                chunk[c] = s.where(s.isin(kept), other="OTHER").astype("category")

        # High-card text -> leave as efficient string (Arrow-backed if available)
        # (When written to Parquet, pyarrow will compress well; you may also hash if desired.)

        # Write Parquet chunk
        out_path = os.path.join(out_dir, f"part-{i:05d}.parquet")
        # Configure compression
        pq_kwargs = dict(engine="pyarrow", compression=PARQUET_COMPRESSION)
        # compression_level supported for zstd/snappy in newer pyarrow
        try:
            pq_kwargs["compression_level"] = PARQUET_COMPRESSION_LEVEL
        except TypeError:
            pass
        chunk.to_parquet(out_path, index=False, **pq_kwargs)

        i += 1
        total_rows += len(chunk)
        if i % max(1, 1000 // max(1, CHUNK_SIZE // 5000)) == 0:
            print(f"wrote {i} chunks, ~{total_rows:,} rows")

    print(f"\nDone. Wrote {i} Parquet files to: {out_dir}")
    print("Tip: read selectively for training, e.g.:")
    print("  import glob, pandas as pd")
    print("  files = sorted(glob.glob(r'%s/*.parquet'))" % out_dir)
    print("  df = pd.read_parquet(files, columns=['feature1','feature2','target'])")

# ----------------- Main driver -----------------
def reduce_dataset():
    print("Pass 0: quick sampling for hints…")
    hints = quick_infer_hints(CSV_PATH, SAMPLE_ROWS, USECOLS)
    print("  sampled columns:", len(hints))

    print("Pass 1: profiling full CSV (streamed)…")
    plan = profile_csv(CSV_PATH, hints, CHUNK_SIZE, USECOLS)
    print("\n=== Transform Plan Summary ===")
    print(f"Drop columns: {len(plan['drop_cols'])}")
    print(f"Int columns (downcast): {len(plan['num_int_cols'])}")
    print(f"Float columns (-> float32 when safe): {len(plan['num_float_cols'])}")
    print(f"Low-card text (-> category w/ OTHER): {len(plan['text_lowcard_cols'])}")
    print(f"High-card text (left as string): {len(plan['text_highcard_cols'])}")

    print("\nPass 2: transforming + writing Parquet…")
    apply_transform_and_write(CSV_PATH, plan, CHUNK_SIZE, OUT_DIR, USECOLS)

if __name__ == "__main__":
    reduce_dataset()

####### Usage
import glob, pandas as pd
files = sorted(glob.glob("/path/to/parquet_reduced/*.parquet"))
cols = ["target", "num_var1", "num_var2", "cat_varA", "cat_varB"]
df = pd.read_parquet(files, columns=cols)

# For scikit-learn logistic regression:
# - numeric columns already downcast
# - categorical columns are pandas 'category' -> get one-hots on the fly
X = pd.get_dummies(df.drop(columns=["target"]), drop_first=True, dtype="float32", sparse=True)
y = df["target"].astype("int8")