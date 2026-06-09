#!/usr/bin/env python3
"""
VendorWatch synthetic data generator.

Produces 18 months of transaction history for 500 vendors across 4 categories
and 50 projects. Injects 5 detectable anomaly patterns at designated vendor IDs:

  Slot 1 -> first  DB vendor ID  (IT category):           invoice_splitter
  Slot 2 -> second DB vendor ID  (construction category): approval_bypasser
  Slot 3 -> third  DB vendor ID  (logistics category):    amount_drifter
  Slot 4 -> fourth DB vendor ID  (facilities category):   freq_spiker
  Slot 5 -> fifth  DB vendor ID  (facilities category):   peer_outlier

Distributions:
  - Invoice amounts: lognormal per category (approximates power-law tail behaviour)
  - Payment timing:  80% N(37, 8) days; 20% slow-payer cohort N(65, 10) days
  - Approval cycles: N(3,1) / N(7,2) / N(14,3) gated on amount thresholds
  - Invoice frequency: Poisson(lambda) per category, high-freq cohort at 2x lambda
"""

import os
import sys
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from faker import Faker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

fake = Faker()
rng = np.random.default_rng(42)
random.seed(42)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://vendorwatch:vendorwatch@localhost:5432/vendorwatch",
)

# 18-month window ending 2026-06-01 (fixed for reproducibility)
END_DATE = datetime(2026, 6, 1)
START_DATE = END_DATE - timedelta(days=547)  # ~18 months

MONTHS = 18
MONTH_BOUNDS: List[Tuple[datetime, datetime]] = [
    (START_DATE + timedelta(days=m * 30), START_DATE + timedelta(days=(m + 1) * 30))
    for m in range(MONTHS)
]

# (lognormal_mean, lognormal_sigma, min_amount, max_amount, monthly_freq_lambda)
CATEGORY_CONFIG: Dict[str, Tuple] = {
    "construction": (np.log(50_000), 0.80, 5_000,   500_000, 2.5),
    "IT":           (np.log(25_000), 0.90, 2_000,   200_000, 3.0),
    "logistics":    (np.log(8_000),  0.70,   500,    50_000, 4.0),
    "facilities":   (np.log(5_000),  0.60,   200,    30_000, 2.0),
}

VENDOR_COUNTS = {"construction": 150, "IT": 125, "logistics": 125, "facilities": 100}

# (amount_ceiling, mean_days, std_days)
APPROVAL_TIERS: List[Tuple] = [
    (10_000,       3, 1.0),
    (50_000,       7, 2.0),
    (float("inf"), 14, 3.0),
]

# Anomaly vendor slots: inserted first so they receive the lowest IDs
ANOMALY_SLOTS = [
    {"category": "IT",           "pattern": "invoice_splitter"},
    {"category": "construction", "pattern": "approval_bypasser"},
    {"category": "logistics",    "pattern": "amount_drifter"},
    {"category": "facilities",   "pattern": "freq_spiker"},
    {"category": "facilities",   "pattern": "peer_outlier"},
]

BATCH_SIZE = 1_000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def approval_cycle_params(amount: float) -> Tuple[float, float]:
    for ceiling, mean, std in APPROVAL_TIERS:
        if amount < ceiling:
            return mean, std
    return 14.0, 3.0


def sample_amount(category: str) -> float:
    mu, sigma, lo, hi, _ = CATEGORY_CONFIG[category]
    return float(np.clip(rng.lognormal(mu, sigma), lo, hi))


def sample_invoice_count(category: str, freq_multiplier: float = 1.0) -> int:
    _, _, _, _, lam = CATEGORY_CONFIG[category]
    return max(0, int(rng.poisson(lam * freq_multiplier)))


def sample_dates(start: datetime, end: datetime, n: int) -> List[datetime]:
    span = (end - start).total_seconds()
    return sorted(
        start + timedelta(seconds=float(rng.uniform(0, span)))
        for _ in range(n)
    )


def days_to_payment_sample(is_slow_payer: bool) -> float:
    if is_slow_payer:
        return float(np.clip(rng.normal(65.0, 10.0), 30.0, 90.0))
    return float(np.clip(rng.normal(37.0, 8.0), 5.0, 60.0))


# ---------------------------------------------------------------------------
# invoice record builder
# ---------------------------------------------------------------------------

def build_invoice_record(
    vendor_id: int,
    project_id: int,
    amount: float,
    submitted_at: datetime,
    days_approval: float,
    days_after_approval: float,
    counter: int,
) -> Dict:
    approved_at = submitted_at + timedelta(days=days_approval)
    paid_at = approved_at + timedelta(days=days_after_approval)
    return {
        "vendor_id":        vendor_id,
        "project_id":       project_id,
        "invoice_number":   f"INV-{vendor_id:05d}-{counter:07d}",
        "amount":           round(amount, 2),
        "submitted_at":     submitted_at,
        "approved_at":      approved_at,
        "paid_at":          paid_at,
        "days_to_approval": round(days_approval, 1),
        "days_to_payment":  round(days_approval + days_after_approval, 1),
    }


# ---------------------------------------------------------------------------
# per-vendor invoice generation
# ---------------------------------------------------------------------------

def generate_vendor_month(
    vendor_id: int,
    category: str,
    pattern: Optional[str],
    month_idx: int,
    month_start: datetime,
    month_end: datetime,
    project_ids: List[int],
    is_slow_payer: bool,
    freq_multiplier: float,
    invoice_counter: List[int],  # mutable single-element list used as counter
) -> List[Dict]:
    records = []

    # ------------------------------------------------------------------ #
    # Pattern: invoice_splitter — month 16 only                           #
    # Normal invoices in month 16 PLUS 8 invoices of $9,800 in one week  #
    # ------------------------------------------------------------------ #
    if pattern == "invoice_splitter" and month_idx == 16:
        # Regular invoices this month
        n = sample_invoice_count(category, freq_multiplier)
        for dt in sample_dates(month_start, month_end, n):
            invoice_counter[0] += 1
            amt = sample_amount(category)
            ap_mean, ap_std = approval_cycle_params(amt)
            days_ap = float(np.clip(rng.normal(ap_mean, ap_std), 1.0, 30.0))
            records.append(build_invoice_record(
                vendor_id, int(rng.choice(project_ids)), amt, dt,
                days_ap, days_to_payment_sample(is_slow_payer), invoice_counter[0],
            ))
        # Split batch: 8 invoices of $9,800 within a 7-day window
        split_start = month_start + timedelta(days=5)
        split_end   = split_start + timedelta(days=7)
        for dt in sample_dates(split_start, split_end, 8):
            invoice_counter[0] += 1
            amt = 9_800.0
            days_ap = float(np.clip(rng.normal(3.0, 0.5), 1.0, 5.0))
            records.append(build_invoice_record(
                vendor_id, int(rng.choice(project_ids)), amt, dt,
                days_ap, days_to_payment_sample(is_slow_payer), invoice_counter[0],
            ))
        return records

    # ------------------------------------------------------------------ #
    # Pattern: approval_bypasser — month 14 only                          #
    # 3 invoices (~$75 K each) with days_to_approval forced to 1         #
    # ------------------------------------------------------------------ #
    if pattern == "approval_bypasser" and month_idx == 14:
        for dt in sample_dates(month_start, month_end, 3):
            invoice_counter[0] += 1
            amt = float(np.clip(rng.lognormal(np.log(75_000), 0.3), 40_000, 200_000))
            records.append(build_invoice_record(
                vendor_id, int(rng.choice(project_ids)), amt, dt,
                1.0,  # bypass: 1 day instead of ~14
                days_to_payment_sample(is_slow_payer), invoice_counter[0],
            ))
        return records

    # ------------------------------------------------------------------ #
    # Pattern: freq_spiker — month 16 only (2-week window)               #
    # ------------------------------------------------------------------ #
    if pattern == "freq_spiker" and month_idx == 16:
        window_end = month_start + timedelta(days=14)
        normal_count = max(1, sample_invoice_count(category, freq_multiplier))
        n = normal_count * 3  # 3× normal in 2 weeks
        for dt in sample_dates(month_start, window_end, n):
            invoice_counter[0] += 1
            amt = sample_amount(category)
            ap_mean, ap_std = approval_cycle_params(amt)
            days_ap = float(np.clip(rng.normal(ap_mean, ap_std), 1.0, 30.0))
            records.append(build_invoice_record(
                vendor_id, int(rng.choice(project_ids)), amt, dt,
                days_ap, days_to_payment_sample(is_slow_payer), invoice_counter[0],
            ))
        return records

    # ------------------------------------------------------------------ #
    # Normal month generation (also used for anomaly vendors in           #
    # months where their pattern is inactive)                             #
    # ------------------------------------------------------------------ #
    n = sample_invoice_count(category, freq_multiplier)
    if n == 0:
        return records

    for dt in sample_dates(month_start, month_end, n):
        invoice_counter[0] += 1

        # peer_outlier: facilities vendor invoicing at IT rates always
        if pattern == "peer_outlier":
            amt = float(np.clip(rng.lognormal(np.log(35_000), 0.85), 10_000, 120_000))
        # amount_drifter: 15% MoM drift in months 11-14
        elif pattern == "amount_drifter" and 11 <= month_idx <= 14:
            drift_exp = month_idx - 10  # 1..4
            base = sample_amount(category)
            amt = float(np.clip(base * (1.15 ** drift_exp), 500.0, 150_000.0))
        else:
            amt = sample_amount(category)

        ap_mean, ap_std = approval_cycle_params(amt)
        days_ap = float(np.clip(rng.normal(ap_mean, ap_std), 1.0, 30.0))
        records.append(build_invoice_record(
            vendor_id, int(rng.choice(project_ids)), amt, dt,
            days_ap, days_to_payment_sample(is_slow_payer), invoice_counter[0],
        ))

    return records


# ---------------------------------------------------------------------------
# database operations
# ---------------------------------------------------------------------------

def insert_projects(cur) -> List[int]:
    rows = [
        (fake.bs().title()[:195], float(rng.uniform(500_000, 50_000_000)))
        for _ in range(50)
    ]
    execute_values(cur, "INSERT INTO projects (name, budget) VALUES %s RETURNING id", rows)
    return [r[0] for r in cur.fetchall()]


def insert_vendors(cur) -> Tuple[List[int], Dict[int, str], Dict[int, str]]:
    """Returns (all_ids, id->category, id->pattern_or_None)."""
    vendor_rows = []

    # Anomaly vendors first (they receive the 5 lowest IDs)
    for slot in ANOMALY_SLOTS:
        vendor_rows.append((
            f"VW-Anomaly-{slot['pattern'].replace('_', '-').title()}",
            slot["category"],
        ))

    # Normal vendors (495, distributed to keep category totals correct)
    anomaly_per_cat = {}
    for slot in ANOMALY_SLOTS:
        anomaly_per_cat[slot["category"]] = anomaly_per_cat.get(slot["category"], 0) + 1

    for cat, total in VENDOR_COUNTS.items():
        for _ in range(total - anomaly_per_cat.get(cat, 0)):
            vendor_rows.append((fake.company()[:195], cat))

    execute_values(cur, "INSERT INTO vendors (name, category) VALUES %s RETURNING id", vendor_rows)
    ids = [r[0] for r in cur.fetchall()]

    id_to_cat = {}
    id_to_pattern = {}
    for i, vid in enumerate(ids):
        if i < len(ANOMALY_SLOTS):
            id_to_cat[vid]     = ANOMALY_SLOTS[i]["category"]
            id_to_pattern[vid] = ANOMALY_SLOTS[i]["pattern"]
        else:
            id_to_cat[vid]     = vendor_rows[i][1]
            id_to_pattern[vid] = None

    return ids, id_to_cat, id_to_pattern


def batch_insert(cur, table: str, columns: List[str], rows: List[Tuple], returning: str = "id") -> List[int]:
    if not rows:
        return []
    placeholders = ", ".join(["%s"] * len(columns))
    col_str = ", ".join(columns)
    result_ids = []
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        execute_values(
            cur,
            f"INSERT INTO {table} ({col_str}) VALUES %s RETURNING {returning}",
            batch,
        )
        result_ids.extend(r[0] for r in cur.fetchall())
    return result_ids


# ---------------------------------------------------------------------------
# CUSUM baseline initialisation
# ---------------------------------------------------------------------------

def compute_baselines(
    vendor_invoice_data: Dict[int, List[Dict]],
    baseline_months: int = 9,
) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """Compute mean/std of amount and days_to_approval over first N months."""
    baselines = {}
    cutoff = START_DATE + timedelta(days=baseline_months * 30)

    for vendor_id, records in vendor_invoice_data.items():
        baseline_records = [r for r in records if r["submitted_at"] < cutoff]
        if len(baseline_records) < 3:
            baseline_records = records  # fall back to full history

        amounts   = [r["amount"]           for r in baseline_records]
        approvals = [r["days_to_approval"] for r in baseline_records]

        amt_mean = float(np.mean(amounts))   if amounts   else 0.0
        amt_std  = float(np.std(amounts))    if len(amounts) > 1   else 1.0
        ap_mean  = float(np.mean(approvals)) if approvals else 0.0
        ap_std   = float(np.std(approvals))  if len(approvals) > 1 else 1.0

        baselines[vendor_id] = {
            "amount":           (amt_mean, max(amt_std,  1.0)),
            "days_to_approval": (ap_mean,  max(ap_std,   0.5)),
        }
    return baselines


def insert_cusum_states(cur, baselines: Dict[int, Dict[str, Tuple[float, float]]]) -> None:
    rows = []
    for vendor_id, features in baselines.items():
        for feature_name, (mean, std) in features.items():
            rows.append((vendor_id, feature_name, 0.0, 0.0, mean, std))

    execute_values(
        cur,
        """
        INSERT INTO cusum_state (vendor_id, feature_name, cusum_pos, cusum_neg, target_mean, target_std)
        VALUES %s
        ON CONFLICT (vendor_id, feature_name) DO NOTHING
        """,
        rows,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    # Idempotency guard
    cur.execute("SELECT COUNT(*) FROM vendors")
    if cur.fetchone()[0] > 0:
        log.info("Database already populated — skipping generation.")
        cur.close()
        conn.close()
        return

    log.info("Starting data generation …")

    # Projects
    project_ids = insert_projects(cur)
    conn.commit()
    log.info("Inserted %d projects.", len(project_ids))

    # Vendors
    vendor_ids, id_to_cat, id_to_pattern = insert_vendors(cur)
    conn.commit()
    log.info("Inserted %d vendors.", len(vendor_ids))

    # Assign per-vendor behavioural properties deterministically
    approvers = [f"approver_{i:03d}" for i in range(1, 31)]
    vendor_props: Dict[int, Dict] = {}
    for vid in vendor_ids:
        vendor_props[vid] = {
            "is_slow_payer":    rng.random() < 0.20,
            "freq_multiplier":  2.0 if rng.random() < 0.20 else 1.0,
        }

    # Generate all invoices in memory
    invoice_counter = [0]
    vendor_invoice_data: Dict[int, List[Dict]] = {v: [] for v in vendor_ids}

    for vid in vendor_ids:
        cat     = id_to_cat[vid]
        pattern = id_to_pattern[vid]
        props   = vendor_props[vid]
        for m_idx, (m_start, m_end) in enumerate(MONTH_BOUNDS):
            records = generate_vendor_month(
                vendor_id       = vid,
                category        = cat,
                pattern         = pattern,
                month_idx       = m_idx,
                month_start     = m_start,
                month_end       = m_end,
                project_ids     = project_ids,
                is_slow_payer   = props["is_slow_payer"],
                freq_multiplier = props["freq_multiplier"],
                invoice_counter = invoice_counter,
            )
            vendor_invoice_data[vid].extend(records)

    total_invoices = sum(len(v) for v in vendor_invoice_data.values())
    log.info("Generated %d invoice records in memory. Inserting …", total_invoices)

    # Flatten and insert invoices
    invoice_tuples = []
    all_records_flat: List[Dict] = []
    for vid in vendor_ids:
        for rec in vendor_invoice_data[vid]:
            all_records_flat.append(rec)
            invoice_tuples.append((
                rec["vendor_id"],
                rec["project_id"],
                rec["invoice_number"],
                rec["amount"],
                rec["submitted_at"],
                rec["paid_at"],
                rec["days_to_payment"],
                "paid",
            ))

    inserted_ids = batch_insert(
        cur,
        "invoices",
        ["vendor_id", "project_id", "invoice_number", "amount",
         "submitted_at", "paid_at", "days_to_payment", "status"],
        invoice_tuples,
    )
    conn.commit()
    log.info("Inserted %d invoices.", len(inserted_ids))

    # Insert approvals
    approval_tuples = []
    for inv_id, rec in zip(inserted_ids, all_records_flat):
        approval_tuples.append((
            inv_id,
            str(rng.choice(approvers)),
            rec["submitted_at"],
            rec["approved_at"],
            rec["days_to_approval"],
            "approved",
        ))

    batch_insert(
        cur,
        "approvals",
        ["invoice_id", "approver_id", "submitted_at", "approved_at", "days_to_approval", "status"],
        approval_tuples,
    )
    conn.commit()
    log.info("Inserted %d approvals.", len(approval_tuples))

    # CUSUM state initialisation
    baselines = compute_baselines(vendor_invoice_data, baseline_months=9)
    insert_cusum_states(cur, baselines)
    conn.commit()
    log.info("Initialised CUSUM state for %d vendors × 2 features.", len(baselines))

    # Summary for sanity check
    cur.execute("SELECT category, COUNT(*) FROM vendors GROUP BY category ORDER BY category")
    log.info("Vendor counts: %s", dict(cur.fetchall()))

    cur.execute("SELECT COUNT(*) FROM invoices")
    log.info("Total invoices: %d", cur.fetchone()[0])

    cur.execute("SELECT COUNT(*) FROM cusum_state")
    log.info("CUSUM state rows: %d", cur.fetchone()[0])

    # Log anomaly vendor IDs for test reference
    cur.execute(
        "SELECT id, name, category FROM vendors WHERE name LIKE 'VW-Anomaly-%' ORDER BY id"
    )
    log.info("Anomaly vendors:")
    for row in cur.fetchall():
        log.info("  id=%d  name=%s  category=%s", *row)

    cur.close()
    conn.close()
    log.info("Data generation complete.")


if __name__ == "__main__":
    main()
