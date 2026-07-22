import pandas as pd

from parkinsons_voice.evaluate import compute_classification_metrics, compute_regression_metrics, plot_updrs_progression
from parkinsons_voice.features import compute_updrs_total, prepare_ml_data
from parkinsons_voice.models import build_pd_classifier, build_updrs_regressor


def test_compute_classification_metrics_keys(synthetic_clinical_df):
    split = prepare_ml_data(synthetic_clinical_df, target_col="TARGET", id_col="PATNO", random_state=0)
    X_train, X_test, y_train, y_test = split
    clf = build_pd_classifier(n_estimators=50, random_state=0)
    clf.fit(X_train, y_train)

    metrics = compute_classification_metrics(clf, X_test, y_test)
    assert set(metrics) == {"accuracy", "roc_auc", "precision", "recall", "f1", "confusion_matrix"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_compute_regression_metrics_keys(synthetic_updrs_df):
    updrs_total = compute_updrs_total(synthetic_updrs_df)
    feature_cols = ["PATNO"]
    X = updrs_total[feature_cols]
    y = updrs_total["TOTAL"]
    reg = build_updrs_regressor(n_estimators=20, random_state=0)
    reg.fit(X, y)

    metrics = compute_regression_metrics(reg, X, y)
    assert set(metrics) == {"mae", "rmse", "r2"}


def test_plot_updrs_progression_returns_figure(synthetic_updrs_df):
    updrs_total = compute_updrs_total(synthetic_updrs_df)
    fig = plot_updrs_progression(updrs_total, patno=0)
    assert fig is not None


def test_plot_updrs_progression_none_without_total():
    df = pd.DataFrame({"PATNO": [1, 2], "EVENT_ID": ["V0", "V1"]})
    assert plot_updrs_progression(df) is None
