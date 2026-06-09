"""
GET /dashboard/summary
GET /health
POST /admin/retrain
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import psycopg2
from fastapi import APIRouter, BackgroundTasks

from src.api.schemas import DashboardSummary, HealthResponse, RetrainRequest, RetrainResponse
from src.config import get_settings
from src.detection.model_registry import CATEGORIES, get_registry
from src.services.trainer import feedback_since_last_retrain, train_all

log      = logging.getLogger(__name__)
router   = APIRouter(tags=["dashboard"])
settings = get_settings()


def _sync_conn():
    return psycopg2.connect(settings.SYNC_DATABASE_URL)


@router.get("/dashboard/summary", response_model=DashboardSummary)
async def dashboard_summary():
    conn = _sync_conn()
    cur  = conn.cursor()

    # Active flag counts by tier
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE risk_score >= 0.70)  AS high,
            COUNT(*) FILTER (WHERE risk_score >= 0.40 AND risk_score < 0.70) AS medium,
            COUNT(*) FILTER (WHERE risk_score < 0.40)   AS low,
            COUNT(*)                                     AS total
        FROM anomaly_flags
        WHERE flag_status = 'active'
    """)
    tiers = cur.fetchone()
    high_count   = int(tiers[0])
    medium_count = int(tiers[1])
    low_count    = int(tiers[2])
    total_active = int(tiers[3])

    # Flags by category
    cur.execute("""
        SELECT vendor_category, COUNT(*)
        FROM anomaly_flags
        WHERE flag_status = 'active'
        GROUP BY vendor_category
    """)
    by_cat = {r[0]: int(r[1]) for r in cur.fetchall()}

    # 30-day avg risk score
    cur.execute("""
        SELECT COALESCE(AVG(risk_score), 0.0)
        FROM anomaly_flags
        WHERE detected_at >= NOW() - INTERVAL '30 days'
    """)
    avg_30d = float(cur.fetchone()[0])

    # Daily flag counts over last 30 days
    cur.execute("""
        SELECT detected_at::date AS day, COUNT(*)
        FROM anomaly_flags
        WHERE detected_at >= NOW() - INTERVAL '30 days'
        GROUP BY day
        ORDER BY day
    """)
    daily = [{"date": r[0].isoformat(), "count": int(r[1])} for r in cur.fetchall()]

    # Model versions
    cur.execute("""
        SELECT vendor_category, version, training_date, is_active
        FROM model_versions
        ORDER BY training_date DESC
    """)
    model_versions = [
        {"category": r[0], "version": r[1],
         "training_date": r[2].isoformat(), "is_active": r[3]}
        for r in cur.fetchall()
    ]

    # Feedback distribution
    cur.execute("SELECT label, COUNT(*) FROM analyst_feedback GROUP BY label")
    feedback_dist = {r[0]: int(r[1]) for r in cur.fetchall()}

    conn.close()
    return DashboardSummary(
        total_active_flags    = total_active,
        high_risk_count       = high_count,
        medium_risk_count     = medium_count,
        low_risk_count        = low_count,
        flags_by_category     = by_cat,
        avg_risk_score_30d    = round(avg_30d, 4),
        daily_flag_counts     = daily,
        model_versions        = model_versions,
        feedback_distribution = feedback_dist,
    )


@router.post("/admin/retrain", response_model=RetrainResponse)
async def manual_retrain(body: RetrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(train_all, reason=body.reason)
    return RetrainResponse(
        version = "pending",
        reason  = body.reason,
        results = {c: "queued" for c in CATEGORIES},
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    conn = _sync_conn()
    cur  = conn.cursor()

    db_ok = False
    try:
        cur.execute("SELECT 1")
        db_ok = True
    except Exception:
        pass

    # Model versions per category
    cur.execute("""
        SELECT vendor_category, version
        FROM model_versions
        WHERE is_active = true
    """)
    mv = {r[0]: r[1] for r in cur.fetchall()}

    # Last retrain timestamp
    cur.execute("SELECT MAX(training_date) FROM model_versions WHERE is_active = true")
    row       = cur.fetchone()
    last_ts   = row[0] if row else None

    feedback_depth = feedback_since_last_retrain(conn)
    conn.close()

    registry = get_registry()
    model_status = {}
    for cat in CATEGORIES:
        bundle = registry._bundles.get(cat)
        model_status[cat] = bundle.version if bundle else "not_loaded"

    return HealthResponse(
        status               = "ok" if db_ok else "degraded",
        model_versions       = model_status,
        last_retrain         = last_ts,
        feedback_queue_depth = feedback_depth,
        db_ok                = db_ok,
    )
