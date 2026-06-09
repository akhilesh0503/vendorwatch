"""
GET   /flags             — paginated list with filters
GET   /flags/{id}        — full flag detail + SHAP waterfall + CUSUM + peer scatter
PATCH /flags/{id}/feedback — analyst feedback + retraining trigger check
"""

import json
import logging
from datetime import datetime
from math import ceil
from typing import List, Optional

import psycopg2
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import FeedbackRequest, FeedbackResponse, FlagDetail, FlagListItem
from src.config import get_settings
from src.detection import composite, cusum, peer_groups
from src.detection.model_registry import get_registry
from src.services.trainer import feedback_since_last_retrain, train_all

log      = logging.getLogger(__name__)
router   = APIRouter(prefix="/flags", tags=["flags"])
settings = get_settings()


def _sync_conn():
    return psycopg2.connect(settings.SYNC_DATABASE_URL)


@router.get("", response_model=List[FlagListItem])
async def list_flags(
    risk_score_min:  float     = Query(default=0.0,   ge=0.0, le=1.0),
    vendor_category: Optional[str] = Query(default=None),
    flag_status:     Optional[str] = Query(default="active"),
    date_from:       Optional[str] = Query(default=None),
    date_to:         Optional[str] = Query(default=None),
    page:            int       = Query(default=1,    ge=1),
    page_size:       int       = Query(default=50,   ge=1, le=200),
):
    conn = _sync_conn()
    cur  = conn.cursor()

    conditions = ["af.risk_score >= %s"]
    params: list = [risk_score_min]

    if vendor_category:
        conditions.append("af.vendor_category = %s")
        params.append(vendor_category)
    if flag_status:
        conditions.append("af.flag_status = %s")
        params.append(flag_status)
    if date_from:
        conditions.append("af.detected_at >= %s")
        params.append(date_from)
    if date_to:
        conditions.append("af.detected_at <= %s")
        params.append(date_to)

    where = " AND ".join(conditions)
    offset = (page - 1) * page_size

    cur.execute(f"""
        SELECT
            af.id, af.vendor_id, v.name, af.vendor_category,
            af.risk_score, af.flag_status, af.detected_at,
            af.primary_signal,
            af.isolation_forest_score, af.cusum_breach_severity, af.peer_deviation_score,
            DATE_PART('day', NOW() - af.detected_at)::int AS days_since
        FROM anomaly_flags af
        JOIN vendors v ON v.id = af.vendor_id
        WHERE {where}
        ORDER BY af.risk_score DESC
        LIMIT %s OFFSET %s
    """, params + [page_size, offset])

    rows = cur.fetchall()
    conn.close()

    return [
        FlagListItem(
            id                     = r[0],
            vendor_id              = r[1],
            vendor_name            = r[2],
            vendor_category        = r[3],
            risk_score             = r[4],
            risk_tier              = composite.risk_tier(r[4]),
            flag_status            = r[5],
            detected_at            = r[6],
            primary_signal         = r[7],
            isolation_forest_score = r[8],
            cusum_breach_severity  = r[9],
            peer_deviation_score   = r[10],
            days_since_first_flag  = r[11] or 0,
        )
        for r in rows
    ]


@router.get("/{flag_id}", response_model=FlagDetail)
async def get_flag(flag_id: int):
    conn = _sync_conn()
    cur  = conn.cursor()

    cur.execute("""
        SELECT
            af.id, af.vendor_id, v.name, af.vendor_category,
            af.risk_score, af.flag_status, af.detected_at,
            af.primary_signal,
            af.isolation_forest_score, af.cusum_breach_severity, af.peer_deviation_score,
            af.shap_values, af.shap_explanation, af.layers_fired
        FROM anomaly_flags af
        JOIN vendors v ON v.id = af.vendor_id
        WHERE af.id = %s
    """, (flag_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Flag {flag_id} not found")

    (fid, vid, vname, vcat, rscore, fstatus, detected_at,
     primary_signal, if_sc, cusum_sc, peer_sc,
     shap_vals, shap_expl, layers) = row

    # Analyst feedback for this flag
    cur.execute("""
        SELECT id, analyst_id, label, notes, created_at
        FROM analyst_feedback WHERE flag_id = %s ORDER BY created_at DESC
    """, (flag_id,))
    feedback = [
        {"id": r[0], "analyst_id": r[1], "label": r[2],
         "notes": r[3], "created_at": r[4].isoformat()}
        for r in cur.fetchall()
    ]

    # CUSUM chart data
    cusum_chart = cusum.cusum_chart_data(vid, cur)

    # Peer scatter (if model loaded)
    peer_scatter = None
    bundle = await get_registry().get(vcat)
    if bundle and bundle.kmeans and bundle.scaler:
        try:
            peer_scatter = peer_groups.scatter_data(vid, cur, bundle.kmeans, bundle.scaler)
        except Exception as exc:
            log.warning("Peer scatter failed for vendor %d: %s", vid, exc)

    conn.close()
    return FlagDetail(
        id                     = fid,
        vendor_id              = vid,
        vendor_name            = vname,
        vendor_category        = vcat,
        risk_score             = rscore,
        risk_tier              = composite.risk_tier(rscore),
        flag_status            = fstatus,
        detected_at            = detected_at,
        primary_signal         = primary_signal,
        isolation_forest_score = if_sc,
        cusum_breach_severity  = cusum_sc,
        peer_deviation_score   = peer_sc,
        shap_values            = shap_vals,
        shap_explanation       = shap_expl,
        layers_fired           = layers,
        cusum_chart            = cusum_chart,
        peer_scatter           = peer_scatter,
        feedback               = feedback,
    )


@router.patch("/{flag_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(flag_id: int, body: FeedbackRequest):
    valid_labels = ("true_positive", "false_positive", "escalated")
    if body.label not in valid_labels:
        raise HTTPException(
            status_code=422,
            detail=f"label must be one of {valid_labels}",
        )

    conn = _sync_conn()
    cur  = conn.cursor()

    cur.execute("SELECT id FROM anomaly_flags WHERE id = %s", (flag_id,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail=f"Flag {flag_id} not found")

    cur.execute("""
        INSERT INTO analyst_feedback (flag_id, analyst_id, label, notes)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """, (flag_id, body.analyst_id, body.label, body.notes))
    feedback_id = cur.fetchone()[0]

    # Update flag status if escalated
    if body.label == "escalated":
        cur.execute(
            "UPDATE anomaly_flags SET flag_status = 'escalated' WHERE id = %s",
            (flag_id,)
        )
    conn.commit()

    feedback_count = feedback_since_last_retrain(conn)
    retrain_queued = False

    if feedback_count >= settings.FEEDBACK_RETRAIN_THRESHOLD:
        log.info("Feedback threshold reached (%d) — triggering async retrain.", feedback_count)
        import asyncio
        asyncio.get_event_loop().run_in_executor(None, lambda: train_all(reason="feedback_threshold"))
        retrain_queued = True

    conn.close()
    return FeedbackResponse(
        flag_id        = flag_id,
        label          = body.label,
        feedback_id    = feedback_id,
        retrain_queued = retrain_queued,
        feedback_count = feedback_count,
    )
