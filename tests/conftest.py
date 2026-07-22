"""Shared synthetic-data fixtures for tests.

None of these fixtures use real PPMI data (which is gated behind a Data Use
Agreement); everything is generated locally with numpy for structural
testing only. Any numbers produced by tests using these fixtures are
synthetic-data test results, not performance claims about the model on real
clinical data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_clinical_df():
    """A small, class-separable synthetic clinical table with a TARGET column."""
    rng = np.random.default_rng(0)
    n_per_class = 60
    n_features = 6

    class0 = rng.normal(loc=0.0, scale=1.0, size=(n_per_class, n_features))
    class1 = rng.normal(loc=2.5, scale=1.0, size=(n_per_class, n_features))
    X = np.vstack([class0, class1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    df["TARGET"] = y
    df["PATNO"] = np.arange(len(df))
    return df


@pytest.fixture
def synthetic_imbalanced_df():
    """A synthetic clinical table with a 9:1 class imbalance."""
    rng = np.random.default_rng(1)
    n_majority, n_minority = 90, 10
    n_features = 5

    majority = rng.normal(loc=0.0, scale=1.0, size=(n_majority, n_features))
    minority = rng.normal(loc=2.0, scale=1.0, size=(n_minority, n_features))
    X = np.vstack([majority, minority])
    y = np.array([0] * n_majority + [1] * n_minority)

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    df["TARGET"] = y
    df["PATNO"] = np.arange(len(df))
    return df


@pytest.fixture
def synthetic_updrs_df():
    """A synthetic longitudinal UPDRS-like table with NP* item columns."""
    rng = np.random.default_rng(2)
    n_participants = 8
    n_visits = 4
    rows = []
    for patno in range(n_participants):
        base = rng.integers(5, 20)
        for visit in range(n_visits):
            row = {
                "PATNO": patno,
                "EVENT_ID": f"V{visit:02d}",
                "NP1_1": max(0, base + visit + rng.integers(-1, 2)),
                "NP1_2": max(0, base + visit + rng.integers(-1, 2)),
                "NP1_3": max(0, base + visit + rng.integers(-1, 2)),
            }
            rows.append(row)
    return pd.DataFrame(rows)
