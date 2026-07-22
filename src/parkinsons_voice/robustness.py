"""Feature-level noise-robustness testing.

Reconstructed from notebook cell 16 (``evaluate_robustness_numeric``). The
original cell's own docstring called this "a placeholder", explicitly
noting that real audio-level robustness testing would need to perturb audio
files *before* feature extraction. That caveat is preserved below - this
module still only perturbs already-extracted numeric feature vectors; it
does not add an audio pipeline, which does not exist anywhere in this
repo.

What changed from the notebook: the function is implemented for real (not a
stub) and made to run over a *sweep* of SNR levels, matching the config's
``robustness.snr_levels_db`` list, so callers can see degradation trends
rather than a single print statement.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


def add_gaussian_noise(X: pd.DataFrame, snr_db: float, random_state: int | None = None) -> pd.DataFrame:
    """Add Gaussian noise to a numeric feature matrix, scaled to a target SNR (dB).

    LIMITATION (preserved from the original notebook): this perturbs
    already-extracted numeric features, not raw audio. For a true
    audio-level robustness test you would need to perturb source audio
    waveforms before feature extraction - that pipeline does not exist in
    this repo. This is a feature-space proxy only.
    """
    rng = np.random.default_rng(random_state)
    Xn = X.copy()
    values = Xn.to_numpy(dtype=float)

    rms_signal = np.sqrt(np.mean(values ** 2))
    snr_lin = 10 ** (snr_db / 10.0)
    noise_std = np.sqrt(rms_signal / snr_lin) if rms_signal > 0 else 0.0
    noise = rng.normal(0, noise_std, size=values.shape)

    Xn.loc[:, :] = values + noise
    return Xn


def evaluate_robustness_numeric(clf, X_test: pd.DataFrame, y_test, noise_snr_db: float = 20, random_state: int | None = None) -> float:
    """Add Gaussian noise (scaled by SNR) to X_test and report accuracy.

    Feature-level perturbation only; see module docstring for the
    audio-level-noise caveat inherited from the original notebook.
    """
    Xn = add_gaussian_noise(X_test, noise_snr_db, random_state=random_state)
    y_pred = clf.predict(Xn)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Robustness test at SNR=%s dB: Accuracy = %.4f", noise_snr_db, acc)
    return acc


def robustness_sweep(clf, X_test: pd.DataFrame, y_test, snr_levels_db: list[float], random_state: int | None = None) -> pd.DataFrame:
    """Evaluate accuracy across a sweep of SNR levels (higher SNR = less noise).

    Returns a DataFrame with columns ['snr_db', 'accuracy'], sorted by
    descending SNR (i.e. ascending noise).
    """
    rows = []
    for snr_db in sorted(snr_levels_db, reverse=True):
        acc = evaluate_robustness_numeric(clf, X_test, y_test, noise_snr_db=snr_db, random_state=random_state)
        rows.append({"snr_db": snr_db, "accuracy": acc})
    return pd.DataFrame(rows)
