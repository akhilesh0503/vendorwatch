"""
Layer 2: CUSUM (Cumulative Sum Control Chart) — stateful per-vendor detection.

Detects sustained shifts in vendor payment amounts and approval cycle times.
State persists across runs in the cusum_state table (incremental, not replay).

Two-sided CUSUM:
  x_n       = standardized observation: (raw - target_mean) / target_std
  C_pos_n   = max(0,  C_pos_{n-1} + x_n - k)   detects upward shift
  C_neg_n   = min(0,  C_neg_{n-1} + x_n + k)   detects downward shift

Flag when C_pos > h  or  |C_neg| > h.

Default parameters per spec: k=0.5, h=5.0
Severity normalisation (agreed before build):
  severity = min(cusum_stat / (h * 4.0), 1.0)
  → 0.25 at threshold, 1.0 at 4× threshold
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)

CUSUM_FEATURES = ("amount", "days_to_approval")


def _standardize(value: float, mean: float, std: float) -> float:
    return (value - mean) / max(std, 1e-6)


def breach_severity(cusum_stat: float, h: float) -> float:
    """Map CUSUM statistic to [0, 1]. 0.25 at threshold, saturates at 4× threshold."""
    return min(cusum_stat / (h * 4.0), 1.0)


def update_and_detect(
    vendor_id: int,
    cur,
    k: float = 0.5,
    h: float = 5.0,
    as_of: Optional[datetime] = None,
) -> Dict:
    """
    Update CUSUM for a vendor and return breach results.

    When as_of is provided: simulation mode — replays ALL invoices up to that
    date from a clean slate (c_pos=0, c_neg=0), does NOT write to DB.

    When as_of is None: incremental mode — processes only invoices since
    last_updated, writes updated state back to DB.

    In both modes, breach=True if the stat exceeded h at ANY point during
    processing (not just the final value), so sustained-drift patterns are
    caught even if later invoices partially recover the stat.
    """
    result = {}

    for feature in CUSUM_FEATURES:
        cur.execute("""
            SELECT cusum_pos, cusum_neg, target_mean, target_std, last_updated
            FROM cusum_state
            WHERE vendor_id = %s AND feature_name = %s
        """, (vendor_id, feature))
        row = cur.fetchone()
        if row is None:
            result[feature] = {
                "cusum_pos": 0.0, "cusum_neg": 0.0,
                "n_new_obs": 0, "breach": False, "severity": 0.0,
            }
            continue

        stored_pos, stored_neg, mean, std, last_updated = row
        mean = float(mean) if mean is not None else 0.0
        std  = float(std)  if std  is not None else 1.0

        if as_of is not None:
            # Simulation: replay full history up to as_of from clean slate
            c_pos, c_neg = 0.0, 0.0
            if feature == "amount":
                cur.execute("""
                    SELECT i.amount FROM invoices i
                    WHERE i.vendor_id = %s AND i.submitted_at <= %s
                      AND i.amount IS NOT NULL
                    ORDER BY i.submitted_at ASC
                """, (vendor_id, as_of))
            else:
                cur.execute("""
                    SELECT a.days_to_approval
                    FROM approvals a JOIN invoices i ON i.id = a.invoice_id
                    WHERE i.vendor_id = %s AND i.submitted_at <= %s
                      AND a.days_to_approval IS NOT NULL
                    ORDER BY i.submitted_at ASC
                """, (vendor_id, as_of))
            update_db = False
        else:
            # Incremental: start from stored state, only new invoices
            c_pos, c_neg = float(stored_pos), float(stored_neg)
            if feature == "amount":
                cur.execute("""
                    SELECT i.amount FROM invoices i
                    WHERE i.vendor_id = %s AND i.submitted_at > %s
                      AND i.amount IS NOT NULL
                    ORDER BY i.submitted_at ASC
                """, (vendor_id, last_updated))
            else:
                cur.execute("""
                    SELECT a.days_to_approval
                    FROM approvals a JOIN invoices i ON i.id = a.invoice_id
                    WHERE i.vendor_id = %s AND i.submitted_at > %s
                      AND a.days_to_approval IS NOT NULL
                    ORDER BY i.submitted_at ASC
                """, (vendor_id, last_updated))
            update_db = True

        observations = [float(r[0]) for r in cur.fetchall()]

        # Track peak and ever-breached across all observations
        ever_breached = False
        peak_stat = 0.0
        for obs in observations:
            x     = _standardize(obs, mean, std)
            c_pos = max(0.0, c_pos + x - k)
            c_neg = min(0.0, c_neg + x + k)
            stat  = max(c_pos, abs(c_neg))
            if stat > peak_stat:
                peak_stat = stat
            if stat > h:
                ever_breached = True

        if update_db:
            cur.execute("""
                UPDATE cusum_state
                SET cusum_pos = %s, cusum_neg = %s, last_updated = NOW()
                WHERE vendor_id = %s AND feature_name = %s
            """, (c_pos, c_neg, vendor_id, feature))

        severity = breach_severity(peak_stat, h) if ever_breached else 0.0
        result[feature] = {
            "cusum_pos":  c_pos,
            "cusum_neg":  c_neg,
            "n_new_obs":  len(observations),
            "breach":     ever_breached,
            "severity":   severity,
        }

    return result


def combined_severity(cusum_result: Dict) -> float:
    """Max severity across all tracked features."""
    return max(
        cusum_result.get(f, {}).get("severity", 0.0)
        for f in CUSUM_FEATURES
    )


def cusum_chart_data(vendor_id: int, cur) -> Dict:
    """
    Return CUSUM time-series data for dashboard charting.
    Replays the full history to reconstruct the trajectory.
    """
    cur.execute("""
        SELECT cs.feature_name, cs.target_mean, cs.target_std
        FROM cusum_state cs
        WHERE cs.vendor_id = %s
    """, (vendor_id,))
    states = {r[0]: {"mean": float(r[1] or 0), "std": float(r[2] or 1)} for r in cur.fetchall()}

    chart = {}
    for feature, baseline in states.items():
        if feature == "amount":
            cur.execute("""
                SELECT submitted_at, amount
                FROM invoices WHERE vendor_id = %s AND amount IS NOT NULL
                ORDER BY submitted_at
            """, (vendor_id,))
        else:
            cur.execute("""
                SELECT i.submitted_at, a.days_to_approval
                FROM approvals a JOIN invoices i ON i.id = a.invoice_id
                WHERE i.vendor_id = %s AND a.days_to_approval IS NOT NULL
                ORDER BY i.submitted_at
            """, (vendor_id,))

        rows = cur.fetchall()
        k_val, h_val = 0.5, 5.0
        c_pos = c_neg = 0.0
        series = []
        for dt, val in rows:
            x     = _standardize(float(val), baseline["mean"], baseline["std"])
            c_pos = max(0.0, c_pos + x - k_val)
            c_neg = min(0.0, c_neg + x + k_val)
            series.append({
                "date":    dt.isoformat(),
                "c_pos":   round(c_pos, 3),
                "c_neg":   round(c_neg, 3),
                "breached": max(c_pos, abs(c_neg)) > h_val,
            })
        chart[feature] = {"series": series, "h": h_val}

    return chart
