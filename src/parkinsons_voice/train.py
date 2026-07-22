"""Training orchestration: fit the PD classifier and the UPDRS regressor,
evaluate on the held-out split, and persist artifacts.

Reconstructed from the training portion of notebook cell 14
(``baseline_train_eval``); imbalance handling (features.py) and the UPDRS
regressor (models.py) are wired in per the config, both flagged elsewhere as
additions beyond the literal notebook code.
"""

from __future__ import annotations

import logging
import os

import joblib
import pandas as pd

from .config import Config
from .evaluate import compute_classification_metrics, compute_regression_metrics
from .features import apply_imbalance_handling, prepare_ml_data
from .models import build_pd_classifier, build_updrs_regressor

logger = logging.getLogger(__name__)


def train_classifier(clinical_df: pd.DataFrame, cfg: Config):
    """Fit the PD classifier on `clinical_df` per the config and evaluate it.

    Returns (clf, metrics, (X_train, X_test, y_train, y_test)).
    """
    split = prepare_ml_data(
        clinical_df,
        target_col=cfg.data.target_col,
        id_col=cfg.data.id_col,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        impute_strategy=cfg.features.impute_strategy,
        scale=cfg.features.scale,
    )
    if split is None:
        raise ValueError("Could not prepare ML data - check target_col/id_col against clinical_df columns")
    X_train, X_test, y_train, y_test = split

    X_train, y_train = apply_imbalance_handling(
        X_train, y_train, method=cfg.imbalance.method, random_state=cfg.imbalance.random_state
    )

    clf = build_pd_classifier(**cfg.model.classifier.to_dict())
    clf.fit(X_train, y_train)
    logger.info("Fitted PD classifier on %d training rows", len(X_train))

    metrics = compute_classification_metrics(clf, X_test, y_test)
    return clf, metrics, (X_train, X_test, y_train, y_test)


def train_updrs_regressor(updrs_df: pd.DataFrame, cfg: Config, feature_cols: list[str] | None = None):
    """Fit the UPDRS TOTAL-score regressor on an UPDRS table with a TOTAL column.

    ``feature_cols`` defaults to all numeric columns except PATNO and TOTAL.
    Returns (reg, metrics, (X_train, X_test, y_train, y_test)).
    """
    from sklearn.model_selection import train_test_split

    if "TOTAL" not in updrs_df.columns:
        raise ValueError("updrs_df must have a TOTAL column (see features.compute_updrs_total)")

    df = updrs_df.dropna(subset=["TOTAL"]).copy()
    if feature_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ("PATNO", "TOTAL")]
    if not feature_cols:
        raise ValueError("No numeric feature columns available to fit the UPDRS regressor")

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["TOTAL"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.split.test_size, random_state=cfg.split.random_state
    )

    reg = build_updrs_regressor(**cfg.model.updrs_regressor.to_dict())
    reg.fit(X_train, y_train)
    logger.info("Fitted UPDRS regressor on %d training rows, %d features", len(X_train), len(feature_cols))

    metrics = compute_regression_metrics(reg, X_test, y_test)
    return reg, metrics, (X_train, X_test, y_train, y_test)


def save_artifacts(artifacts: dict, artifacts_dir: str) -> dict[str, str]:
    """Persist fitted estimators (and other picklable artifacts) via joblib.

    ``artifacts`` maps a name (e.g. "classifier") to an object; each is
    written to ``artifacts_dir/<name>.joblib``. Returns the map of name ->
    saved path.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    saved = {}
    for name, obj in artifacts.items():
        path = os.path.join(artifacts_dir, f"{name}.joblib")
        joblib.dump(obj, path)
        logger.info("Saved artifact '%s' -> %s", name, path)
        saved[name] = path
    return saved
