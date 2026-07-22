"""SHAP and LIME explainability for the PD classifier.

``compute_shap_values`` / ``save_shap_summary_plot`` are reconstructed from
notebook cell 15 (``shap.TreeExplainer`` on the RandomForest baseline).

The LIME functions are NEW code: LIME appears only in the notebook's
markdown ``!pip install ... lime`` line - no cell ever imports or calls it,
despite being one of the two explainability methods this repo is meant to
demonstrate. ``explain_instance_lime`` implements a real
``lime.lime_tabular.LimeTabularExplainer`` wrapper following the same
pattern as the SHAP cell (fit on training data statistics, explain a single
test instance). Flagged as a genuine addition, not a whitespace-recovery
inference.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def compute_shap_values(clf, X_background, max_background: int = 100):
    """Compute SHAP values for a fitted tree-based classifier.

    ``X_background`` is capped at ``max_background`` rows for tractability
    (SHAP's TreeExplainer is exact for trees but still scales with rows x
    features x trees).
    """
    import shap

    X_sample = X_background.iloc[:max_background] if len(X_background) > max_background else X_background
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values, X_sample


def save_shap_summary_plot(shap_values, X_sample, output_dir: str, filename: str = "shap_summary.png") -> str:
    """Save a SHAP summary (beeswarm) plot to output_dir/filename.

    For binary classifiers, shap_values may be a list [class0, class1] (older
    SHAP API) or a single 2D/3D array (newer SHAP API); this handles both and
    uses the positive-class contributions.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import shap

    values = shap_values
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    elif isinstance(values, np.ndarray) and values.ndim == 3:
        # (n_samples, n_features, n_classes) -> positive class
        values = values[:, :, 1]

    os.makedirs(output_dir, exist_ok=True)
    out_fp = os.path.join(output_dir, filename)
    shap.summary_plot(values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(out_fp)
    plt.close()
    logger.info("Saved SHAP summary plot to %s", out_fp)
    return out_fp


def build_lime_explainer(X_train, class_names=("Control", "PD"), mode: str = "classification"):
    """Construct a LimeTabularExplainer fit on the training feature distribution."""
    from lime.lime_tabular import LimeTabularExplainer

    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        class_names=list(class_names),
        mode=mode,
        discretize_continuous=True,
    )


def explain_instance_lime(
    explainer,
    clf,
    instance,
    num_features: int = 10,
    num_samples: int = 500,
):
    """Explain a single row's prediction with LIME.

    ``instance`` is a 1D array-like of feature values (e.g. X_test.iloc[i].values).
    Returns the LIME Explanation object.
    """
    return explainer.explain_instance(
        instance,
        clf.predict_proba,
        num_features=num_features,
        num_samples=num_samples,
    )
