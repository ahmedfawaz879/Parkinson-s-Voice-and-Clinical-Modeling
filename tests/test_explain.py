from parkinsons_voice.explain import build_lime_explainer, compute_shap_values, explain_instance_lime, save_shap_summary_plot
from parkinsons_voice.features import prepare_ml_data
from parkinsons_voice.models import build_pd_classifier


def _fit_clf(synthetic_clinical_df):
    split = prepare_ml_data(synthetic_clinical_df, target_col="TARGET", id_col="PATNO", random_state=0)
    X_train, X_test, y_train, y_test = split
    clf = build_pd_classifier(n_estimators=20, random_state=0)
    clf.fit(X_train, y_train)
    return clf, X_train, X_test


def test_compute_shap_values_shape(synthetic_clinical_df):
    clf, _X_train, X_test = _fit_clf(synthetic_clinical_df)
    explainer, shap_values, X_sample = compute_shap_values(clf, X_test, max_background=10)
    assert explainer is not None
    assert len(X_sample) <= 10


def test_save_shap_summary_plot_writes_file(tmp_path, synthetic_clinical_df):
    clf, _X_train, X_test = _fit_clf(synthetic_clinical_df)
    _explainer, shap_values, X_sample = compute_shap_values(clf, X_test, max_background=10)
    out_fp = save_shap_summary_plot(shap_values, X_sample, str(tmp_path))
    assert (tmp_path / "shap_summary.png").exists()
    assert out_fp.endswith("shap_summary.png")


def test_lime_explain_instance_runs(synthetic_clinical_df):
    clf, X_train, X_test = _fit_clf(synthetic_clinical_df)
    explainer = build_lime_explainer(X_train, class_names=("Control", "PD"))
    explanation = explain_instance_lime(explainer, clf, X_test.iloc[0].values, num_features=3, num_samples=50)
    assert explanation is not None
    assert len(explanation.as_list()) > 0
