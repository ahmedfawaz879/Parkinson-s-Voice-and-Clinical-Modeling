"""Model constructors: PD classifier and UPDRS progression regressor.

``build_pd_classifier`` mirrors the ``RandomForestClassifier`` used in
notebook cell 14 (``n_estimators=200, random_state=0``).

``build_updrs_regressor`` is NEW: the original notebook computes a UPDRS
``TOTAL`` score (cell 9) and plots its progression per participant
(cell 10), but never fits a model to predict/forecast it - "UPDRS
progression modeling" was implementation-only intent, not code, in the
source notebook. A RandomForestRegressor is added here as the natural
counterpart to the classifier (same estimator family, config-driven
hyperparameters) to predict the UPDRS TOTAL score from clinical/demographic
features. Flagged as a genuine addition, not a whitespace-recovery
inference.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def build_pd_classifier(
    n_estimators: int = 200,
    random_state: int = 0,
    n_jobs: int = -1,
    **kwargs,
) -> RandomForestClassifier:
    """Construct the baseline PD-vs-control RandomForest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )


def build_updrs_regressor(
    n_estimators: int = 200,
    random_state: int = 0,
    n_jobs: int = -1,
    **kwargs,
) -> RandomForestRegressor:
    """Construct the UPDRS TOTAL-score progression regressor."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )
