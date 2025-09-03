import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional

def infer_feature_types(
    df: pd.DataFrame,
    target: Optional[str] = None,
    low_card_max_unique: int = 20,      # numeric features with <= this many uniques -> treat as categorical
    high_card_min_unique: int = 50,     # categorical with >= this many uniques -> high cardinality
    high_card_min_ratio: float = 0.20,  # or uniques / non-null rows >= this ratio -> high cardinality
    min_cont_unique: int = 15,          # numeric uniques > this -> continuous candidate
    drop_constant: bool = True,
) -> Dict[str, object]:
    """
    Classify columns for logistic regression modeling.

    Returns a dict with:
      - 'categorical': list[str]
      - 'high_cardinality': list[str]
      - 'continuous': list[str]
      - 'id_like_or_constant': list[str]
      - 'summary': pd.DataFrame with per-column stats and suggested_kind
    """
    X = df.copy()
    if target is not None and target in X.columns:
        X = X.drop(columns=[target])

    n = len(X)
    nunique = X.nunique(dropna=True)

    # Drop constant/empty columns (if requested)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if drop_constant and constant_cols:
        X = X.drop(columns=constant_cols)
        nunique = nunique.drop(index=constant_cols, errors="ignore")

    # Heuristic: ID-like columns (name patterns or nearly-all-unique)
    id_like: List[str] = []
    for c in X.columns:
        name = c.lower()
        if re.search(r'\b(id|uuid|guid|ssn|account|email)\b', name):
            id_like.append(c)
            continue
        non_null = X[c].notna().sum()
        if non_null > 0 and nunique[c] >= 0.98 * non_null:
            id_like.append(c)

    # Separate broad dtype buckets
    numeric_cols = X.select_dtypes(include=['number']).columns.difference(id_like).tolist()
    stringy_cols = X.select_dtypes(include=['object', 'category', 'string', 'bool']).columns.difference(id_like).tolist()
    datetime_cols = X.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns.difference(id_like).tolist()
    bool_cols = X.select_dtypes(include=['bool']).columns.difference(id_like).tolist()

    # Numeric but low-card -> treat as categorical (e.g., 0/1, 1/2/3)
    num_as_cat = []
    for c in numeric_cols:
        u = int(nunique[c])
        non_null = X[c].notna().sum()
        if u <= low_card_max_unique or (non_null > 0 and (u / non_null) <= high_card_min_ratio / 2):
            num_as_cat.append(c)

    # Candidate categoricals: stringy + booleans + numeric-as-categorical
    candidate_cat = list(dict.fromkeys(stringy_cols + bool_cols + num_as_cat))

    # Split candidate categoricals into low-card vs high-card
    high_card, low_card = [], []
    for c in candidate_cat:
        u = int(nunique[c])
        non_null = X[c].notna().sum()
        if u >= high_card_min_unique or (non_null > 0 and (u / non_null) >= high_card_min_ratio):
            high_card.append(c)
        else:
            low_card.append(c)

    # Continuous = numeric not treated as categorical; also exclude ID-like
    continuous = sorted(set(numeric_cols) - set(num_as_cat))

    # Build a readable summary
    rows = []
    for c in X.columns:
        non_null = X[c].notna().sum()
        u = int(nunique.get(c, 0))
        if c in high_card:
            kind = 'high_card_cat'
        elif c in low_card:
            kind = 'categorical'
        elif c in continuous:
            kind = 'continuous'
        elif c in datetime_cols:
            kind = 'datetime'
        elif c in id_like or c in constant_cols:
            kind = 'drop'
        else:
            kind = 'other'

        reason = 'id-like' if c in id_like else ('constant' if c in constant_cols else '')
        rows.append({
            'column': c,
            'dtype': str(X[c].dtype),
            'n_unique': u,
            'pct_unique': (u / non_null) if non_null else np.nan,
            'suggested_kind': kind,
            'reason': reason
        })

    summary = pd.DataFrame(rows).sort_values(['suggested_kind', 'column']).reset_index(drop=True)

    return {
        'categorical': sorted(low_card),
        'high_cardinality': sorted(high_card),
        'continuous': sorted(continuous),
        'id_like_or_constant': sorted(set(id_like + constant_cols)),
        'summary': summary
    }


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

info = infer_feature_types(df, target='y')

# Features we’ll actually use (drop high-card & id-like by default)
use_cols = info['continuous'] + info['categorical']
X = df[use_cols]
y = df['y']  # your binary target

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), info['continuous']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), info['categorical'])
    ],
    remainder='drop'
)

clf = Pipeline(steps=[
    ('prep', preprocess),
    ('logreg', LogisticRegression(max_iter=1000))
])

clf.fit(X, y)