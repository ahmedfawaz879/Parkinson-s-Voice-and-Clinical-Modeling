from parkinsons_voice.config import load_config
from parkinsons_voice.features import compute_updrs_total
from parkinsons_voice.train import save_artifacts, train_classifier, train_updrs_regressor


def test_train_classifier_fits_and_predicts_without_error(synthetic_clinical_df):
    cfg = load_config()
    clf, metrics, split = train_classifier(synthetic_clinical_df, cfg)
    X_train, X_test, y_train, y_test = split

    preds = clf.predict(X_test)
    assert len(preds) == len(X_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    # class-separable synthetic data should be easy for a RandomForest
    assert metrics["accuracy"] > 0.7


def test_train_updrs_regressor_fits_and_predicts(synthetic_updrs_df):
    cfg = load_config()
    updrs_total = compute_updrs_total(synthetic_updrs_df)
    reg, metrics, split = train_updrs_regressor(updrs_total, cfg)
    _X_train, X_test, _y_train, _y_test = split

    preds = reg.predict(X_test)
    assert len(preds) == len(X_test)
    assert "mae" in metrics and "rmse" in metrics and "r2" in metrics


def test_save_artifacts_writes_joblib_files(tmp_path, synthetic_clinical_df):
    cfg = load_config()
    clf, _metrics, _split = train_classifier(synthetic_clinical_df, cfg)

    saved = save_artifacts({"classifier": clf}, str(tmp_path))
    assert "classifier" in saved
    assert (tmp_path / "classifier.joblib").exists()
