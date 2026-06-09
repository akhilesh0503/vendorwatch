"""
Layer 1: Isolation Forest ensemble — one model per vendor category.

Normalization: -score_samples() (negate so higher = more anomalous), then
min-max scaled using training-set extremes stored alongside the model.
IF scores near 0 = normal, near 1 = highly anomalous.
"""

import logging
import os
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

log = logging.getLogger(__name__)


def train(
    X: np.ndarray,
    n_estimators: int = 200,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Tuple[IsolationForest, float, float, np.ndarray]:
    """
    Train an Isolation Forest.

    Returns:
        model:       fitted IsolationForest
        score_min:   min raw anomaly score from training set (for normalisation)
        score_max:   max raw anomaly score from training set
        background:  random subsample of X (≤100 rows) for SHAP background data
    """
    model = IsolationForest(
        n_estimators  = n_estimators,
        contamination = contamination,
        random_state  = random_state,
        n_jobs        = -1,
    )
    model.fit(X)

    raw = -model.score_samples(X)  # negate: higher = more anomalous
    score_min = float(raw.min())
    score_max = float(raw.max())

    n_bg = min(100, len(X))
    idx = np.random.default_rng(42).choice(len(X), n_bg, replace=False)
    background = X[idx].copy()

    return model, score_min, score_max, background


def score(
    model: IsolationForest,
    features: np.ndarray,
    score_min: float,
    score_max: float,
) -> float:
    """Score a single feature vector. Returns float in [0, 1], 1 = most anomalous."""
    raw = float(-model.score_samples(features.reshape(1, -1))[0])
    span = score_max - score_min
    if span < 1e-9:
        return 0.0
    return float(np.clip((raw - score_min) / span, 0.0, 1.0))


def save(bundle: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)


def load(path: str) -> dict:
    return joblib.load(path)
