"""
POST /vendors/{id}/analyze
GET  /vendors/{id}/history
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import psycopg2
from fastapi import APIRouter, HTTPException

from src.api.schemas import AnalyzeResponse, VendorHistoryResponse
from src.config import get_settings
from src.detection import composite, cusum, features, peer_groups, shap_explainer
from src.detection import isolation_forest as if_mod
from src.detection.model_registry import get_registry

log      = logging.getLogger(__name__)
router   = APIRouter(prefix="/vendors", tags=["vendors"])
settings = get_settings()


def _sync_conn():
    return psycopg2.connect(settings.SYNC_DATABASE_URL)


@router.post("/{vendor_id}/analyze", response_model=AnalyzeResponse)
async def analyze_vendor(vendor_id: int):
    conn = _sync_conn()
    cur  = conn.cursor()

    # Vendor metadata
    cur.execute("SELECT id, name, category FROM vendors WHERE id = %s", (vendor_id,))
    vendor = cur.fetchone()
    if not vendor:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")
    _vid, vname, vcategory = vendor

    # ── Layer 1: Isolation Forest ──────────────────────────────────────
    bundle = await get_registry().get(vcategory)
    if bundle is None:
        conn.close()
        raise HTTPException(
            status_code=503,
            detail=f"No trained model for category '{vcategory}'. Trigger POST /admin/retrain first.",
        )

    as_of      = datetime.now()
    feat_vec   = features.compute_vendor_features(vendor_id, cur, as_of)
    if_score   = 0.0
    shap_dict  = None
    explanation = None

    if feat_vec is not None:
        if_score = if_mod.score(bundle.model, feat_vec, bundle.score_min, bundle.score_max)
        shap_dict   = shap_explainer.compute(bundle.model, feat_vec, bundle.background)
        explanation = shap_explainer.generate_explanation(shap_dict, feat_vec)

    # ── Layer 2: CUSUM ─────────────────────────────────────────────────
    cusum_result   = cusum.update_and_detect(vendor_id, cur, settings.CUSUM_K, settings.CUSUM_H)
    cusum_severity = cusum.combined_severity(cusum_result)
    conn.commit()  # persist CUSUM state updates

    # ── Layer 3: Peer group ────────────────────────────────────────────
    peer_score = 0.0
    if bundle.kmeans and bundle.scaler:
        peer_score = peer_groups.deviation_score(
            vendor_id, cur, bundle.kmeans, bundle.scaler, bundle.peer_norm_p95
        )

    # ── Composite score ────────────────────────────────────────────────
    rscore  = composite.risk_score(if_score, cusum_severity, peer_score)
    fired   = composite.layers_fired(if_score, cusum_result, peer_score)
    tier    = composite.risk_tier(rscore)

    # Primary signal: top SHAP feature name, human-readable
    primary_signal = None
    if shap_dict and feat_vec is not None:
        import numpy as np
        vals = shap_dict["shap_values"]
        top  = shap_dict["feature_names"][int(np.argmax(np.abs(vals)))]
        primary_signal = top.replace("_", " ")

    # ── Persist flag if score exceeds threshold ────────────────────────
    flag_id      = None
    flag_created = False
    if rscore >= composite.FLAG_THRESHOLD:
        cur.execute("""
            INSERT INTO anomaly_flags
                (vendor_id, detected_at, risk_score,
                 isolation_forest_score, cusum_breach_severity, peer_deviation_score,
                 shap_values, shap_explanation, layers_fired, flag_status,
                 primary_signal, vendor_category)
            VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, 'active', %s, %s)
            RETURNING id
        """, (
            vendor_id, rscore,
            if_score, cusum_severity, peer_score,
            json.dumps(shap_dict) if shap_dict else None,
            explanation,
            json.dumps(fired),
            primary_signal, vcategory,
        ))
        flag_id      = cur.fetchone()[0]
        flag_created = True
        conn.commit()

    conn.close()
    return AnalyzeResponse(
        vendor_id               = vendor_id,
        vendor_name             = vname,
        vendor_category         = vcategory,
        risk_score              = round(rscore, 4),
        risk_tier               = tier,
        isolation_forest_score  = round(if_score, 4),
        cusum_breach_severity   = round(cusum_severity, 4),
        peer_deviation_score    = round(peer_score, 4),
        layers_fired            = fired,
        shap_values             = shap_dict,
        shap_explanation        = explanation,
        flag_id                 = flag_id,
        flag_created            = flag_created,
    )


@router.get("/{vendor_id}/history", response_model=VendorHistoryResponse)
async def vendor_history(vendor_id: int):
    conn = _sync_conn()
    cur  = conn.cursor()

    cur.execute("SELECT id, name, category FROM vendors WHERE id = %s", (vendor_id,))
    vendor = cur.fetchone()
    if not vendor:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")
    _vid, vname, vcategory = vendor

    # All invoices
    cur.execute("""
        SELECT i.id, i.invoice_number, i.amount, i.submitted_at,
               i.paid_at, i.days_to_payment, i.status,
               a.days_to_approval
        FROM invoices i
        LEFT JOIN approvals a ON a.invoice_id = i.id
        WHERE i.vendor_id = %s
        ORDER BY i.submitted_at
    """, (vendor_id,))
    invoices = [
        {
            "id":               r[0],
            "invoice_number":   r[1],
            "amount":           float(r[2]),
            "submitted_at":     r[3].isoformat(),
            "paid_at":          r[4].isoformat() if r[4] else None,
            "days_to_payment":  r[5],
            "status":           r[6],
            "days_to_approval": r[7],
        }
        for r in cur.fetchall()
    ]

    # Anomaly score history from flags
    cur.execute("""
        SELECT detected_at, risk_score, isolation_forest_score,
               cusum_breach_severity, peer_deviation_score
        FROM anomaly_flags
        WHERE vendor_id = %s
        ORDER BY detected_at
    """, (vendor_id,))
    score_history = [
        {
            "date":           r[0].isoformat(),
            "risk_score":     r[1],
            "if_score":       r[2],
            "cusum":          r[3],
            "peer_deviation": r[4],
        }
        for r in cur.fetchall()
    ]

    # All historical flags with outcomes
    cur.execute("""
        SELECT af.id, af.detected_at, af.risk_score, af.flag_status,
               af.primary_signal,
               (SELECT f.label FROM analyst_feedback f WHERE f.flag_id = af.id
                ORDER BY f.created_at DESC LIMIT 1) AS final_label
        FROM anomaly_flags af
        WHERE af.vendor_id = %s
        ORDER BY af.detected_at DESC
    """, (vendor_id,))
    all_flags = [
        {
            "id":            r[0],
            "detected_at":   r[1].isoformat(),
            "risk_score":    r[2],
            "status":        r[3],
            "primary_signal": r[4],
            "outcome":       r[5],
        }
        for r in cur.fetchall()
    ]

    # CUSUM chart data
    cusum_chart = cusum.cusum_chart_data(vendor_id, cur)

    conn.close()
    return VendorHistoryResponse(
        vendor_id             = vendor_id,
        vendor_name           = vname,
        vendor_category       = vcategory,
        invoices              = invoices,
        anomaly_score_history = score_history,
        all_flags             = all_flags,
        cusum_chart           = cusum_chart,
    )
