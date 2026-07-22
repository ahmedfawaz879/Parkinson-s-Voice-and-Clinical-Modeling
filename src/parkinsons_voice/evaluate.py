"""Metrics computation and result visualization.

``compute_classification_metrics`` is reconstructed from the metrics portion
of notebook cell 14 (accuracy / ROC AUC / precision / recall / F1 /
confusion matrix).

``compute_regression_metrics`` is NEW - added as the counterpart needed to
evaluate the UPDRS regressor introduced in models.py (the original notebook
never fit or evaluated a regression model).

``plot_updrs_progression`` is reconstructed from notebook cell 10.

Every function here only reports numbers it actually computes from the
data/model passed in - nothing here prints a fabricated or hardcoded
performance number.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(clf, X_test, y_test) -> dict:
    """Compute accuracy / ROC AUC / precision / recall / F1 / confusion matrix.

    ROC AUC is only computed (and included in the result) if the estimator
    exposes predict_proba and both classes are present in y_test.
    """
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    auc = None
    if y_proba is not None and len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": acc,
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
    logger.info("Classification metrics: %s", {k: v for k, v in metrics.items() if k != "confusion_matrix"})
    return metrics


def compute_regression_metrics(reg, X_test, y_test) -> dict:
    """Compute MAE / RMSE / R^2 for the UPDRS progression regressor."""
    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    logger.info("UPDRS regression metrics: %s", metrics)
    return metrics


def plot_updrs_progression(updrs_df: pd.DataFrame | None, patno=None, title_prefix: str = "UPDRS"):
    """Build a line plot of a single participant's UPDRS TOTAL score over visits.

    Returns the matplotlib Figure (does not call plt.show(), so this is safe
    to use non-interactively / in tests). Returns None if data is unusable.
    """
    import matplotlib.pyplot as plt

    if updrs_df is None:
        logger.warning("No UPDRS data passed")
        return None
    df = updrs_df.copy()
    if "PATNO" not in df.columns:
        logger.error("PATNO not in UPDRS table")
        return None
    if patno is None:
        patno = df["PATNO"].iloc[0]

    sel = df[df["PATNO"] == patno].copy()
    if "INFODT" in sel.columns:
        sel = sel.sort_values("INFODT")
        x = sel["INFODT"]
        xlabel = "Visit Date"
    else:
        sel = sel.sort_values("EVENT_ID") if "EVENT_ID" in sel.columns else sel
        x = range(len(sel))
        xlabel = "Visit Index"

    if "TOTAL" not in sel.columns:
        logger.warning("TOTAL not computed; caller should run compute_updrs_total first")
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, sel["TOTAL"], marker="o")
    ax.set_title(f"{title_prefix} progression for PATNO {patno}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Total Score")
    fig.tight_layout()
    return fig
