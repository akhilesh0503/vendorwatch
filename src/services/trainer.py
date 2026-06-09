"""
VendorWatch training pipeline.

Responsibilities:
  - train_all()         full retrain for all categories (scheduled + manual trigger)
  - bootstrap()         initial training if no models exist on disk

Called from:
  - retrainer.py        (APScheduler container)
  - FastAPI router      (POST /admin/retrain)
"""

import json
import logging
import os
import time
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import psycopg2
import psycopg2.extras
from sklearn.metrics import f1_score

from src.config import get_settings
from src.detection import features as feat_mod
from src.detection import isolation_forest as if_mod
from src.detection import peer_groups as pg_mod

log = logging.getLogger(__name__)
settings = get_settings()

CATEGORIES = ("construction", "IT", "logistics", "facilities")


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------

def _model_path(category: str) -> str:
    return os.path.join(settings.MODELS_DIR, category, "model.joblib")


def _pending_path(category: str) -> str:
    return os.path.join(settings.MODELS_DIR, category, "model.joblib.pending")


def _meta_path(category: str) -> str:
    return os.path.join(settings.MODELS_DIR, category, "meta.json")


def models_exist() -> bool:
    return all(os.path.exists(_model_path(c)) for c in CATEGORIES)


# ---------------------------------------------------------------------------
# feedback-weighted contamination
# ---------------------------------------------------------------------------

def _adjusted_contamination(base: float, category: str, cur) -> float:
    cur.execute("""
        SELECT f.label, COUNT(*)
        FROM analyst_feedback f
        JOIN anomaly_flags af ON af.id = f.flag_id
        JOIN vendors v        ON v.id  = af.vendor_id
        WHERE v.category = %s
        GROUP BY f.label
    """, (category,))
    counts = {r[0]: int(r[1]) for r in cur.fetchall()}
    tp  = counts.get("true_positive", 0) + counts.get("escalated", 0)
    fp  = counts.get("false_positive", 0)
    total = tp + fp
    if total < 10:
        return base
    fp_rate    = fp / total
    adjustment = (0.5 - fp_rate) * 0.02   # ±0.01 max
    return float(np.clip(base + adjustment, 0.01, 0.15))


# ---------------------------------------------------------------------------
# F1 on labeled feedback
# ---------------------------------------------------------------------------

def _compute_f1(category: str, model, score_min: float, score_max: float, cur) -> Optional[float]:
    from datetime import datetime

    cur.execute("""
        SELECT af.vendor_id, f.label
        FROM analyst_feedback f
        JOIN anomaly_flags af ON af.id = f.flag_id
        JOIN vendors v        ON v.id  = af.vendor_id
        WHERE v.category = %s
    """, (category,))
    rows = cur.fetchall()
    if len(rows) < 5:
        return None

    preds, trues = [], []
    for vendor_id, label in rows:
        vec = feat_mod.compute_vendor_features(vendor_id, cur, datetime.now())
        if vec is None:
            continue
        s = if_mod.score(model, vec, score_min, score_max)
        preds.append(1 if s >= 0.5 else 0)
        trues.append(1 if label in ("true_positive", "escalated") else 0)

    return float(f1_score(trues, preds, zero_division=0)) if preds else None


# ---------------------------------------------------------------------------
# per-category training
# ---------------------------------------------------------------------------

def _train_category(
    category: str,
    conn,
    contamination: float,
    n_estimators: int,
    version: str,
) -> bool:
    cur = conn.cursor()
    log.info("Training category=%s version=%s contamination=%.3f", category, version, contamination)

    X, _vendor_ids = feat_mod.compute_training_matrix(category, cur)
    if len(X) < 10:
        log.warning("Skipping category=%s: only %d training samples", category, len(X))
        return False

    model, score_min, score_max, background = if_mod.train(X, n_estimators, contamination)

    # Peer groups
    peer_ids, peer_X = pg_mod.get_vendor_peer_features(category, cur)
    kmeans = scaler = None
    peer_norm_p95 = 1.0

    if len(peer_ids) >= 3:
        kmeans, scaler, labels, distances = pg_mod.fit(peer_ids, peer_X)
        norm_dist    = pg_mod.normalize_distances(distances)
        peer_norm_p95 = float(np.percentile(distances, 95)) if len(distances) else 1.0

        # Persist peer group assignments
        cur.execute("""
            DELETE FROM peer_groups
            WHERE vendor_id IN (SELECT id FROM vendors WHERE category = %s)
        """, (category,))
        psycopg2.extras.execute_values(cur, """
            INSERT INTO peer_groups (vendor_id, cluster_id, centroid_distance, features)
            VALUES %s
        """, [
            (peer_ids[i], int(labels[i]), float(distances[i]),
             json.dumps({"norm_distance": float(norm_dist[i])}))
            for i in range(len(peer_ids))
        ])
        conn.commit()

    f1 = _compute_f1(category, model, score_min, score_max, cur)

    # Write model atomically: pending → final
    os.makedirs(os.path.dirname(_pending_path(category)), exist_ok=True)
    bundle = {
        "model": model, "score_min": score_min, "score_max": score_max,
        "background": background,
        "kmeans": kmeans, "scaler": scaler, "peer_norm_p95": peer_norm_p95,
    }
    if_mod.save(bundle, _pending_path(category))
    os.replace(_pending_path(category), _model_path(category))

    with open(_meta_path(category), "w") as f:
        json.dump({
            "version": version, "category": category,
            "score_min": score_min, "score_max": score_max,
            "peer_norm_p95": peer_norm_p95,
        }, f)

    # Register version in DB
    cur.execute(
        "UPDATE model_versions SET is_active = false WHERE vendor_category = %s",
        (category,)
    )
    cur.execute("""
        INSERT INTO model_versions
            (version, vendor_category, sample_count, contamination_used,
             f1_on_labeled_subset, model_path, is_active)
        VALUES (%s, %s, %s, %s, %s, %s, true)
    """, (version, category, len(X), contamination, f1, _model_path(category)))
    conn.commit()

    log.info(
        "Trained category=%s samples=%d f1=%s path=%s",
        category, len(X), f"{f1:.3f}" if f1 is not None else "n/a",
        _model_path(category),
    )
    return True


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def train_all(reason: str = "scheduled") -> Dict:
    """Retrain all categories. Returns {version, results{category: ok/failed}}."""
    conn    = psycopg2.connect(settings.SYNC_DATABASE_URL)
    version = f"v{int(time.time())}"
    results = {}

    for category in CATEGORIES:
        contamination = _adjusted_contamination(
            settings.IF_CONTAMINATION, category, conn.cursor()
        )
        ok = _train_category(
            category, conn,
            contamination, settings.IF_ESTIMATORS, version,
        )
        results[category] = "ok" if ok else "failed"

    conn.close()
    log.info("Retraining done reason=%s version=%s results=%s", reason, version, results)
    return {"version": version, "reason": reason, "results": results}


def feedback_since_last_retrain(conn) -> int:
    """How many feedback labels exist since the last active model was trained."""
    cur = conn.cursor()
    cur.execute("SELECT MAX(training_date) FROM model_versions WHERE is_active = true")
    row       = cur.fetchone()
    last_ts   = row[0] if row and row[0] else None
    if last_ts is None:
        cur.execute("SELECT COUNT(*) FROM analyst_feedback")
    else:
        cur.execute(
            "SELECT COUNT(*) FROM analyst_feedback WHERE created_at > %s",
            (last_ts,)
        )
    return int(cur.fetchone()[0])


def bootstrap() -> None:
    """
    Train initial models if no model files exist.
    Waits for invoice data to be available (data-generator may still be running).
    """
    if models_exist():
        log.info("Models already exist — skipping bootstrap.")
        return

    conn = None
    for attempt in range(30):
        try:
            conn = psycopg2.connect(settings.SYNC_DATABASE_URL)
            cur  = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM invoices")
            count = int(cur.fetchone()[0])
            if count > 1000:
                log.info("Invoice count=%d, proceeding with bootstrap.", count)
                break
            log.info("Waiting for data... invoice count=%d (attempt %d/30)", count, attempt + 1)
        except Exception as exc:
            log.info("DB not ready yet (attempt %d/30): %s", attempt + 1, exc)
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
        time.sleep(10)

    if not models_exist():
        log.info("Running bootstrap training...")
        train_all(reason="bootstrap")
