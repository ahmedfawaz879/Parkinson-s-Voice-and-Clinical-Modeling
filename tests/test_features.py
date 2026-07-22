import numpy as np
import pandas as pd

from parkinsons_voice.features import apply_imbalance_handling, compute_updrs_total, prepare_ml_data


def test_prepare_ml_data_shapes(synthetic_clinical_df):
    result = prepare_ml_data(synthetic_clinical_df, target_col="TARGET", id_col="PATNO", test_size=0.25, random_state=0)
    assert result is not None
    X_train, X_test, y_train, y_test = result

    n_total = len(synthetic_clinical_df)
    assert len(X_train) + len(X_test) == n_total
    assert len(y_train) + len(y_test) == n_total
    # id/target columns dropped from the feature matrix
    assert "PATNO" not in X_train.columns
    assert "TARGET" not in X_train.columns
    # stratified split keeps both classes in the test set
    assert set(y_test.unique()) == {0, 1}


def test_prepare_ml_data_missing_columns_returns_none(synthetic_clinical_df):
    df = synthetic_clinical_df.drop(columns=["TARGET"])
    assert prepare_ml_data(df, target_col="TARGET") is None


def test_prepare_ml_data_none_input():
    assert prepare_ml_data(None) is None


def test_apply_imbalance_handling_smote_balances_classes(synthetic_imbalanced_df):
    X = synthetic_imbalanced_df.drop(columns=["TARGET", "PATNO"])
    y = synthetic_imbalanced_df["TARGET"]

    X_res, y_res = apply_imbalance_handling(X, y, method="smote", random_state=0)
    counts = pd.Series(y_res).value_counts()
    assert counts.min() == counts.max()  # SMOTE balances to majority count
    assert len(X_res) == len(y_res)


def test_apply_imbalance_handling_none_passthrough(synthetic_imbalanced_df):
    X = synthetic_imbalanced_df.drop(columns=["TARGET", "PATNO"])
    y = synthetic_imbalanced_df["TARGET"]

    X_res, y_res = apply_imbalance_handling(X, y, method="none")
    assert X_res is X
    assert y_res is y


def test_apply_imbalance_handling_invalid_method_raises(synthetic_imbalanced_df):
    X = synthetic_imbalanced_df.drop(columns=["TARGET", "PATNO"])
    y = synthetic_imbalanced_df["TARGET"]
    try:
        apply_imbalance_handling(X, y, method="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_compute_updrs_total_sums_score_columns(synthetic_updrs_df):
    result = compute_updrs_total(synthetic_updrs_df)
    assert result is not None
    assert "TOTAL" in result.columns
    expected = synthetic_updrs_df[["NP1_1", "NP1_2", "NP1_3"]].sum(axis=1)
    assert np.allclose(result["TOTAL"].to_numpy(), expected.to_numpy())


def test_compute_updrs_total_none_input():
    assert compute_updrs_total(None) is None
