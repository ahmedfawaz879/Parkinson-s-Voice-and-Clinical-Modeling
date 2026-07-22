from parkinsons_voice.config import load_config
from parkinsons_voice.features import prepare_ml_data
from parkinsons_voice.models import build_pd_classifier
from parkinsons_voice.robustness import add_gaussian_noise, evaluate_robustness_numeric, robustness_sweep


def _fit_clf(synthetic_clinical_df):
    split = prepare_ml_data(synthetic_clinical_df, target_col="TARGET", id_col="PATNO", random_state=0)
    X_train, X_test, y_train, y_test = split
    clf = build_pd_classifier(n_estimators=50, random_state=0)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test


def test_add_gaussian_noise_changes_values(synthetic_clinical_df):
    _clf, X_test, _y_test = _fit_clf(synthetic_clinical_df)
    noisy = add_gaussian_noise(X_test, snr_db=0, random_state=0)
    assert noisy.shape == X_test.shape
    assert not noisy.equals(X_test)


def test_evaluate_robustness_numeric_returns_accuracy_in_range(synthetic_clinical_df):
    clf, X_test, y_test = _fit_clf(synthetic_clinical_df)
    acc = evaluate_robustness_numeric(clf, X_test, y_test, noise_snr_db=10, random_state=0)
    assert 0.0 <= acc <= 1.0


def test_robustness_degrades_with_more_noise(synthetic_clinical_df):
    """Accuracy at a very low (noisy) SNR should be no better than at a high
    (clean) SNR, on average, for class-separable synthetic data. A single
    noise draw can be noisy itself, so we compare the mean accuracy across
    several seeds rather than asserting strict monotonicity SNR-by-SNR."""
    clf, X_test, y_test = _fit_clf(synthetic_clinical_df)

    high_snr_accs = [evaluate_robustness_numeric(clf, X_test, y_test, noise_snr_db=40, random_state=s) for s in range(5)]
    low_snr_accs = [evaluate_robustness_numeric(clf, X_test, y_test, noise_snr_db=-10, random_state=s) for s in range(5)]

    assert sum(high_snr_accs) / len(high_snr_accs) >= sum(low_snr_accs) / len(low_snr_accs)


def test_robustness_sweep_returns_expected_shape(synthetic_clinical_df):
    cfg = load_config()
    clf, X_test, y_test = _fit_clf(synthetic_clinical_df)

    sweep = robustness_sweep(clf, X_test, y_test, cfg.robustness.snr_levels_db, random_state=0)
    assert list(sweep.columns) == ["snr_db", "accuracy"]
    assert len(sweep) == len(cfg.robustness.snr_levels_db)
    # sorted by descending SNR (ascending noise)
    assert list(sweep["snr_db"]) == sorted(cfg.robustness.snr_levels_db, reverse=True)
