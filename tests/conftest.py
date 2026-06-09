"""
Shared fixtures for VendorWatch test suite.

Tests use an in-memory SQLite-backed mock OR a real PostgreSQL instance
depending on the TEST_DATABASE_URL environment variable.

For the ML tests that require trained models, fixtures train minimal
Isolation Forest models on synthetic data so tests stay fast and offline.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import psycopg2
import pytest

# ---------------------------------------------------------------------------
# Minimal synthetic feature matrix (n=200, 8 features) for model training
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

def _normal_feature_matrix(n: int = 200) -> np.ndarray:
    """Generate a feature matrix that looks like normal vendor behaviour."""
    return np.column_stack([
        RNG.lognormal(np.log(25_000), 0.5, n),  # amount
        RNG.normal(7, 2, n).clip(1, 30),          # days_to_approval
        RNG.normal(37, 8, n).clip(5, 90),          # days_to_payment
        RNG.poisson(0.7, n).astype(float),         # freq_7d
        RNG.poisson(3.0, n).astype(float),         # freq_30d
        RNG.normal(0, 1, n),                       # amount_dev
        RNG.normal(0, 1, n),                       # approval_z
        RNG.normal(0, 1, n),                       # payment_z
    ]).astype(np.float64)


@pytest.fixture(scope="session")
def normal_X() -> np.ndarray:
    return _normal_feature_matrix(200)


@pytest.fixture(scope="session")
def trained_if_bundle(normal_X):
    """Returns a dict matching what model_registry.ModelBundle expects."""
    from sklearn.ensemble import IsolationForest
    from src.detection.isolation_forest import train

    model, score_min, score_max, background = train(
        normal_X, n_estimators=20, contamination=0.05
    )
    return {
        "model":         model,
        "score_min":     score_min,
        "score_max":     score_max,
        "background":    background,
    }


@pytest.fixture(scope="session")
def trained_kmeans(normal_X):
    """Returns fitted (kmeans, scaler, peer_norm_p95) on 2-feature subset."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    X2 = normal_X[:, :2].copy()
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X2)
    kmeans  = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_sc)
    dists   = np.linalg.norm(X_sc - kmeans.cluster_centers_[kmeans.labels_], axis=1)
    p95     = float(np.percentile(dists, 95))
    return kmeans, scaler, p95
