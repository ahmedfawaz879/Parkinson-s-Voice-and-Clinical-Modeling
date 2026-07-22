from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from parkinsons_voice.models import build_pd_classifier, build_updrs_regressor


def test_build_pd_classifier_defaults():
    clf = build_pd_classifier()
    assert isinstance(clf, RandomForestClassifier)
    assert clf.n_estimators == 200
    assert clf.random_state == 0


def test_build_pd_classifier_overrides():
    clf = build_pd_classifier(n_estimators=10, random_state=7)
    assert clf.n_estimators == 10
    assert clf.random_state == 7


def test_build_updrs_regressor_defaults():
    reg = build_updrs_regressor()
    assert isinstance(reg, RandomForestRegressor)
    assert reg.n_estimators == 200
    assert reg.random_state == 0
