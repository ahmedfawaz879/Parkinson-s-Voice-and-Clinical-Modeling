import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_runs_without_data(tmp_path):
    # Run from an empty tmp_path (not the repo root) so this test can't
    # accidentally depend on - or create - PPMI_raw/processed/output dirs
    # inside the actual repo working tree. main.py locates the package via
    # its own __file__, so cwd does not affect importability.
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0
    assert "train" in result.stdout
    assert "robustness" in result.stdout


def test_cli_train_missing_data_dir_gives_clear_error(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "train", "--processed-dir", str(tmp_path / "nope")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 1
    assert "ppmi-info.org" in result.stderr.lower() or "ppmi-info.org" in result.stdout.lower()


def test_cli_train_end_to_end_on_synthetic_processed_data(tmp_path):
    """Exercises the full `python main.py train` path against a small
    synthetic (not real PPMI) processed_clinical CSV, to confirm the
    documented reproduce-in-3-commands workflow actually runs end to end."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    rng = np.random.default_rng(0)
    n_per_class = 30
    class0 = rng.normal(0.0, 1.0, size=(n_per_class, 4))
    class1 = rng.normal(2.5, 1.0, size=(n_per_class, 4))
    df = pd.DataFrame(np.vstack([class0, class1]), columns=[f"feat_{i}" for i in range(4)])
    df["TARGET"] = [0] * n_per_class + [1] * n_per_class
    df["PATNO"] = range(len(df))
    df.to_csv(processed_dir / "clinical_clean.csv", index=False)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "train", "--processed-dir", str(processed_dir)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "Classification metrics" in result.stdout
    assert (tmp_path / "output" / "artifacts" / "classifier.joblib").exists()
