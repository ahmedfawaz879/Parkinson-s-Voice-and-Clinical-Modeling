"""Feature engineering, preprocessing, and class-imbalance handling.

``compute_updrs_total`` is reconstructed from notebook cell 9.
``prepare_ml_data`` is reconstructed from notebook cell 13.

``apply_imbalance_handling`` is NEW code, not present in the original
notebook: the notebook's markdown-only install cell listed
``imbalanced-learn`` as a dependency, but no cell ever imported or called
it. Per the task of reconstructing the notebook's *intent* (imbalance
handling was clearly planned - the library is in the required-packages
list), this module implements a real, toggleable SMOTE step. This is a
genuine addition, not a whitespace-recovery inference - flagged here and in
the PR description.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def compute_updrs_total(df_part: pd.DataFrame | None, id_col: str = "PATNO") -> pd.DataFrame | None:
    """Compute a per-row TOTAL score across UPDRS item columns.

    Score columns are identified by the common PPMI naming convention
    (columns starting with 'NP', 'MDS', or 'P'); falls back to "all numeric
    columns except the id column" if none match. Handles the PPMI convention
    where part III uses 101 to mean "unable to rate" by coercing non-numeric
    / sentinel values to NaN before summing (skipna=True), matching the
    original notebook's approach.

    BUG FIX (not a whitespace-recovery inference): the original notebook's
    score-column filter was ``c.upper().startswith('P')``, which also
    matches the identifier column ``PATNO`` (it starts with "P"). Run
    literally, that would silently sum the participant ID into the UPDRS
    TOTAL score, corrupting it. This was caught by a test comparing TOTAL
    against a hand-computed sum of just the known score columns. The id
    column is now explicitly excluded from score_cols regardless of its
    name.
    """
    if df_part is None:
        logger.warning("UPDRS table not provided")
        return None

    df = df_part.copy()

    if "INFODT" in df.columns:
        df["INFODT"] = pd.to_datetime(df["INFODT"], errors="coerce")

    score_cols = [
        c
        for c in df.columns
        if c != id_col and (c.startswith("NP") or c.startswith("MDS") or c.upper().startswith("P"))
    ]
    if not score_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        score_cols = [c for c in numeric_cols if c != id_col]

    if not score_cols:
        logger.warning("No candidate score columns found to compute TOTAL")
        return df

    df[score_cols] = df[score_cols].apply(pd.to_numeric, errors="coerce")
    df["TOTAL"] = df[score_cols].sum(axis=1, skipna=True)
    return df


def prepare_ml_data(
    clinical_df: pd.DataFrame | None,
    target_col: str = "TARGET",
    id_col: str = "PATNO",
    test_size: float = 0.2,
    random_state: int = 42,
    impute_strategy: str = "median",
    scale: bool = True,
):
    """Impute, scale, and split a clinical table into train/test sets.

    Returns (X_train, X_test, y_train, y_test) or None if required columns
    are missing / clinical_df is None.
    """
    if clinical_df is None:
        logger.error("Clinical dataframe required for ML prep")
        return None
    df = clinical_df.copy()
    if id_col not in df.columns or target_col not in df.columns:
        logger.error("Required columns '%s' or '%s' missing in clinical data", id_col, target_col)
        return None

    X = df.drop(columns=[id_col, target_col])
    y = df[target_col].astype(int)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    imp = SimpleImputer(strategy=impute_strategy)
    X[num_cols] = imp.fit_transform(X[num_cols])

    if scale:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def apply_imbalance_handling(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "smote",
    random_state: int = 42,
):
    """Resample (X, y) to address class imbalance.

    method="smote" applies imbalanced-learn's SMOTE (requires >= 2 samples
    in the minority class, i.e. k_neighbors is capped by the smallest class
    size). method="none" returns the inputs unchanged. Should be applied to
    the training split only, never to the held-out test set.
    """
    if method == "none":
        return X, y
    if method != "smote":
        raise ValueError(f"Unknown imbalance handling method: {method!r}")

    from imblearn.over_sampling import SMOTE

    counts = y.value_counts()
    minority_count = counts.min()
    if minority_count < 2:
        logger.warning(
            "Minority class has %d sample(s); SMOTE requires >= 2. Skipping resampling.",
            minority_count,
        )
        return X, y

    k_neighbors = min(5, minority_count - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(
        "SMOTE resampled %d -> %d rows (class counts: %s -> %s)",
        len(X), len(X_res), counts.to_dict(), pd.Series(y_res).value_counts().to_dict(),
    )
    return X_res, y_res
