"""
Feature engineering for VendorWatch anomaly detection.

All 8 features are computed per-vendor over a rolling window.
The same feature set is used for training (monthly snapshots) and
inference (rolling 30-day window at analysis time).

Feature vector order (must remain stable — models are trained on this):
  0: invoice_amount                    — mean amount in window
  1: days_to_approval                  — mean approval cycle in window
  2: days_to_payment                   — mean total payment time in window
  3: invoice_frequency_7d              — invoice count in last 7 days
  4: invoice_frequency_30d             — invoice count in last 30 days
  5: amount_deviation_from_vendor_mean — (window_mean - hist_mean) / hist_std
  6: approval_cycle_z_score            — (window_mean_ap - hist_mean_ap) / hist_std_ap
  7: payment_timing_z_score            — (window_mean_pmt - hist_mean_pmt) / hist_std_pmt
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

FEATURE_NAMES: List[str] = [
    "invoice_amount",
    "days_to_approval",
    "days_to_payment",
    "invoice_frequency_7d",
    "invoice_frequency_30d",
    "amount_deviation_from_vendor_mean",
    "approval_cycle_z_score",
    "payment_timing_z_score",
]

WINDOW_DAYS = 30
MIN_WINDOW_INVOICES = 2
MIN_HISTORY_INVOICES = 5


def compute_vendor_features(
    vendor_id: int,
    cur,
    as_of_date: datetime,
    window_days: int = WINDOW_DAYS,
) -> Optional[np.ndarray]:
    """
    Compute the 8-feature vector for a vendor as of as_of_date.
    Uses a rolling window for current stats; all prior history for baselines.
    Returns None if the vendor has fewer than MIN_WINDOW_INVOICES in the window.
    """
    window_start   = as_of_date - timedelta(days=window_days)
    window_7d_start = as_of_date - timedelta(days=7)

    cur.execute("""
        SELECT
            AVG(i.amount)                                                 AS avg_amount,
            AVG(a.days_to_approval)                                       AS avg_approval,
            AVG(i.days_to_payment)                                        AS avg_payment,
            COUNT(CASE WHEN i.submitted_at >= %(w7)s THEN 1 END)          AS freq_7d,
            COUNT(*)                                                      AS freq_30d
        FROM invoices i
        JOIN approvals a ON a.invoice_id = i.id
        WHERE i.vendor_id = %(vid)s
          AND i.submitted_at >= %(ws)s
          AND i.submitted_at < %(aod)s
    """, {"vid": vendor_id, "ws": window_start, "w7": window_7d_start, "aod": as_of_date})
    row = cur.fetchone()
    if row is None or row[4] is None or int(row[4]) < MIN_WINDOW_INVOICES:
        return None

    avg_amount, avg_approval, avg_payment, freq_7d, freq_30d = (
        float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
    )

    # Historical baseline — excludes current window so baselines stay stable
    cur.execute("""
        SELECT
            AVG(i.amount)                AS hist_mean_amt,
            STDDEV(i.amount)             AS hist_std_amt,
            AVG(a.days_to_approval)      AS hist_mean_ap,
            STDDEV(a.days_to_approval)   AS hist_std_ap,
            AVG(i.days_to_payment)       AS hist_mean_pmt,
            STDDEV(i.days_to_payment)    AS hist_std_pmt,
            COUNT(*)                     AS hist_count
        FROM invoices i
        JOIN approvals a ON a.invoice_id = i.id
        WHERE i.vendor_id = %(vid)s
          AND i.submitted_at < %(ws)s
    """, {"vid": vendor_id, "ws": window_start})
    hist = cur.fetchone()

    hist_mean_amt  = float(hist[0]) if hist[0] else avg_amount
    hist_std_amt   = float(hist[1]) if hist[1] and float(hist[1]) > 0 else max(avg_amount * 0.1, 1.0)
    hist_mean_ap   = float(hist[2]) if hist[2] else avg_approval
    hist_std_ap    = float(hist[3]) if hist[3] and float(hist[3]) > 0 else 1.0
    hist_mean_pmt  = float(hist[4]) if hist[4] else avg_payment
    hist_std_pmt   = float(hist[5]) if hist[5] and float(hist[5]) > 0 else 1.0

    features = np.array([
        avg_amount,
        avg_approval,
        avg_payment,
        freq_7d,
        freq_30d,
        (avg_amount  - hist_mean_amt) / hist_std_amt,
        (avg_approval - hist_mean_ap) / hist_std_ap,
        (avg_payment  - hist_mean_pmt) / hist_std_pmt,
    ], dtype=np.float64)

    return np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)


def compute_vendor_stats(vendor_id: int, cur) -> dict:
    """Return historical mean/std for each feature — used by SHAP sentence generator."""
    cur.execute("""
        SELECT
            AVG(i.amount)                AS mean_amt,
            STDDEV(i.amount)             AS std_amt,
            AVG(a.days_to_approval)      AS mean_ap,
            STDDEV(a.days_to_approval)   AS std_ap,
            AVG(i.days_to_payment)       AS mean_pmt,
            STDDEV(i.days_to_payment)    AS std_pmt
        FROM invoices i
        JOIN approvals a ON a.invoice_id = i.id
        WHERE i.vendor_id = %s
    """, (vendor_id,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return {}
    return {
        "mean_amount":   float(row[0]),
        "std_amount":    float(row[1]) if row[1] else 1.0,
        "mean_approval": float(row[2]) if row[2] else 7.0,
        "std_approval":  float(row[3]) if row[3] else 1.0,
        "mean_payment":  float(row[4]) if row[4] else 37.0,
        "std_payment":   float(row[5]) if row[5] else 5.0,
    }


def compute_training_matrix(
    category: str,
    cur,
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute feature matrix for all vendor-month snapshots in a category.
    Each row is one vendor-month with enough prior history for z-scores.
    Returns (X, vendor_ids) where X has shape (n_samples, 8).
    """
    cur.execute("""
        WITH monthly AS (
            SELECT
                i.vendor_id,
                DATE_TRUNC('month', i.submitted_at)::date   AS month,
                AVG(i.amount)                               AS avg_amount,
                AVG(a.days_to_approval)                     AS avg_approval,
                AVG(i.days_to_payment)                      AS avg_payment,
                COUNT(*)                                    AS invoice_count
            FROM invoices i
            JOIN approvals a ON a.invoice_id = i.id
            JOIN vendors v   ON v.id = i.vendor_id
            WHERE v.category = %(cat)s
            GROUP BY i.vendor_id, DATE_TRUNC('month', i.submitted_at)
            HAVING COUNT(*) >= 1
        ),
        vendor_hist AS (
            SELECT
                m1.vendor_id,
                m1.month,
                AVG(m2.avg_amount)          AS hist_mean_amt,
                STDDEV(m2.avg_amount)       AS hist_std_amt,
                AVG(m2.avg_approval)        AS hist_mean_ap,
                STDDEV(m2.avg_approval)     AS hist_std_ap,
                AVG(m2.avg_payment)         AS hist_mean_pmt,
                STDDEV(m2.avg_payment)      AS hist_std_pmt,
                SUM(m2.invoice_count)       AS hist_total
            FROM monthly m1
            JOIN monthly m2
              ON m2.vendor_id = m1.vendor_id AND m2.month < m1.month
            GROUP BY m1.vendor_id, m1.month
            HAVING SUM(m2.invoice_count) >= %(min_hist)s
        )
        SELECT
            m.vendor_id,
            m.avg_amount,
            m.avg_approval,
            m.avg_payment,
            m.invoice_count / 4.3                                                AS freq_7d_approx,
            m.invoice_count::float                                               AS freq_30d,
            COALESCE((m.avg_amount   - h.hist_mean_amt) / NULLIF(h.hist_std_amt, 0), 0) AS amt_dev,
            COALESCE((m.avg_approval - h.hist_mean_ap)  / NULLIF(h.hist_std_ap,  0), 0) AS ap_z,
            COALESCE((m.avg_payment  - h.hist_mean_pmt) / NULLIF(h.hist_std_pmt, 0), 0) AS pmt_z
        FROM monthly m
        JOIN vendor_hist h ON h.vendor_id = m.vendor_id AND h.month = m.month
        ORDER BY m.vendor_id, m.month
    """, {"cat": category, "min_hist": MIN_HISTORY_INVOICES})

    rows = cur.fetchall()
    if not rows:
        return np.empty((0, len(FEATURE_NAMES))), []

    vendor_ids = [int(r[0]) for r in rows]
    X = np.array([
        [float(r[1]), float(r[2]), float(r[3]),
         float(r[4]), float(r[5]),
         float(r[6]), float(r[7]), float(r[8])]
        for r in rows
    ], dtype=np.float64)

    return np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0), vendor_ids
