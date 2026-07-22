from parkinsons_voice.config import load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.model.classifier.n_estimators == 200
    assert cfg.model.classifier.random_state == 0
    assert cfg.split.test_size == 0.20
    assert cfg.imbalance.method == "smote"
    assert list(cfg.robustness.snr_levels_db) == [30, 20, 10, 5, 0]


def test_config_override(tmp_path):
    override_path = tmp_path / "override.yaml"
    override_path.write_text("model:\n  classifier:\n    n_estimators: 5\n", encoding="utf-8")

    cfg = load_config(override_path)
    assert cfg.model.classifier.n_estimators == 5
    # untouched keys still come from defaults
    assert cfg.model.classifier.random_state == 0
    assert cfg.seed == 42


def test_config_missing_override_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    try:
        load_config(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_config_to_dict_roundtrip():
    cfg = load_config()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["model"]["classifier"]["n_estimators"] == 200
