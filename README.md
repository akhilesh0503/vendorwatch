# VendorWatch

Production-grade supply chain anomaly detection system. Three-layer ML pipeline (Isolation Forest + CUSUM + Peer Group Analysis) with SHAP explainability, analyst feedback loop, and automatic model retraining.

---

## Architecture

```
Invoices / Approvals (PostgreSQL)
        │
        ▼
┌─────────────────────────────────────────────┐
│              Detection Engine                │
│                                              │
│  Layer 1: Isolation Forest (per category)   │
│  Layer 2: CUSUM (stateful, incremental)      │
│  Layer 3: KMeans Peer Group Analysis         │
│                                              │
│  risk_score = 0.40 × IF                      │
│             + 0.35 × CUSUM_severity          │
│             + 0.25 × peer_deviation          │
└─────────────────────────────────────────────┘
        │
        ▼
   SHAP Explanation (computed at flag time)
        │
        ▼
  anomaly_flags table (PostgreSQL JSONB)
        │
        ▼
  FastAPI (8 endpoints) ← Streamlit Dashboard
        │
        ▼
  Analyst Feedback → APScheduler → Retrain (if ≥50 new labels)
```

### Why three layers instead of one?

Each layer catches a different anomaly shape:

| Layer | Anomaly Type | What it catches |
|---|---|---|
| Isolation Forest | Point anomalies | Single invoices or windows with unusual feature combinations |
| CUSUM | Sustained drift | Gradual increases in amount or accelerating approval cycles over weeks/months |
| Peer Group | Structural outliers | Vendors that look normal individually but belong to the wrong peer group |

A vendor that flags on only one layer gets a lower composite score than one that triggers all three.

---

## Injected Anomaly Patterns

| # | Pattern | Vendor | Primary Detection Layer | Expected Score |
|---|---|---|---|---|
| 1 | **Invoice splitting** — 8 invoices of $9,800 in one week, just under the $10K approval threshold | Vendor 1 (IT) | Isolation Forest (`invoice_frequency_7d` spike) | > 0.6 |
| 2 | **Approval bypass** — approval cycle drops from 14 → 1 day for 3 consecutive invoices | Vendor 2 (construction) | CUSUM breach on `days_to_approval` | > 0.5 |
| 3 | **Amount drift** — monthly invoice total increases 15% MoM for 4 consecutive months | Vendor 3 (logistics) | CUSUM breach on `amount` | > 0.4 |
| 4 | **Frequency spike** — 3× normal invoice count in a 2-week window | Vendor 4 (facilities) | Isolation Forest (`invoice_frequency_7d/30d`) | > 0.5 |
| 5 | **Peer outlier** — facilities vendor invoicing at IT rates (~$35K vs ~$5K typical) | Vendor 5 (facilities) | Peer group deviation (distance from facilities centroid) | > 0.4 |

---

## Detection Results (on generated data)

| Pattern | Layer that catches it | Also catches |
|---|---|---|
| Invoice splitting | Isolation Forest ✓ | Peer group (if IT-category vendors don't split) |
| Approval bypass | CUSUM ✓ | Isolation Forest (approval_z_score feature) |
| Amount drift | CUSUM ✓ | Not Isolation Forest (any single invoice is 1.75× mean — not extreme) |
| Frequency spike | Isolation Forest ✓ | CUSUM (indirectly via amount accumulation) |
| Peer outlier | Peer group ✓ | Isolation Forest (if IT amounts are in training set for facilities) |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/vendors/{id}/analyze` | Run 3-layer detection, compute SHAP, persist flag if score ≥ threshold |
| `GET`  | `/flags` | Paginated list, filterable by score/category/date/status |
| `GET`  | `/flags/{id}` | Full flag detail: SHAP waterfall data, CUSUM chart, peer scatter |
| `PATCH`| `/flags/{id}/feedback` | Submit analyst label; triggers retrain if ≥ 50 new labels |
| `GET`  | `/dashboard/summary` | Aggregate stats: tier counts, trends, model versions, feedback distribution |
| `GET`  | `/vendors/{id}/history` | 18-month invoice history, anomaly score timeline, CUSUM chart |
| `POST` | `/admin/retrain` | Manual retrain trigger with reason logging |
| `GET`  | `/health` | Model version per category, last retrain timestamp, feedback queue depth |

---

## Data Model

Key tables:

- **vendors** — 500 vendors across 4 categories (construction, IT, logistics, facilities)
- **invoices** — ~27,000 invoice records over 18 months
- **approvals** — one approval row per invoice with `days_to_approval`
- **anomaly_flags** — `shap_values JSONB`, `layers_fired JSONB`, composite `risk_score`
- **analyst_feedback** — analyst labels with timestamp for retraining trigger
- **model_versions** — training metadata, contamination used, F1 on labeled subset
- **peer_groups** — KMeans cluster assignments per vendor
- **cusum_state** — running `(cusum_pos, cusum_neg, target_mean, target_std)` per vendor per feature

---

## SHAP Explainability

Every flag has a full SHAP explanation computed at flag time:

```
The primary signal is approval cycle Z-score: 4.33σ below this vendor's
90-day baseline (contributing +0.28 to anomaly score). The secondary signal
is mean invoice amount of $74,800 — elevated relative to this vendor category's
typical range (contributing +0.19 to anomaly score). Rapid approval of a
high-value invoice is a strong indicator that standard review controls were
bypassed — verify the approval chain manually.
```

SHAP values are stored as JSONB in `anomaly_flags.shap_values` and returned by `GET /flags/{id}` for Plotly waterfall rendering in the dashboard.

---

## Retraining Pipeline

- **Bootstrap**: retrainer container trains initial models if `/models` volume is empty
- **Scheduled**: APScheduler checks hourly — retrains if `analyst_feedback` has ≥ 50 new labels since last retrain
- **Manual**: `POST /admin/retrain`
- **Feedback-weighted contamination**: false positive rate adjusts the `contamination` parameter ±0.01
- **Atomic swap**: new model written to `model.joblib.pending` → `os.replace()` → `model.joblib`
- **Model registry**: FastAPI polls for file mtime changes every 60 s; holds requests off with `asyncio.Lock` during reload

---

## Running Locally

```bash
# Start all services
docker compose up --build

# Services:
#   PostgreSQL:  localhost:5432
#   FastAPI:     http://localhost:8000
#   Streamlit:   http://localhost:8501
#   API docs:    http://localhost:8000/docs
```

First startup runs migrations → data generation (~27K invoices) → initial model training automatically.

---

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Test coverage:
1. Invoice splitting detected by Isolation Forest with `risk_score > 0.5`
2. Approval bypass triggers CUSUM breach on `days_to_approval`
3. Amount drift caught by CUSUM but not necessarily Isolation Forest (layer independence)
4. Peer outlier produces elevated peer deviation score
5. Feedback endpoint stores label and increments counter
6. Retraining triggered when feedback count crosses 50
7. Model swap is atomic — no partial reads during swap
8. SHAP explanation sentences contain actual numeric values, not placeholder text

---

## Architectural Decisions

**Three detection layers**: Point anomalies (IF) miss sustained drift; CUSUM misses structural outliers; peer group analysis misses both but catches category mismatch. The composite score weights each layer and requires all three to run on every analysis.

**Stateful CUSUM**: Running state persisted in PostgreSQL so CUSUM is truly incremental. Each `analyze` call processes only new transactions since `last_updated`. First call after data generation processes all history in one pass — this is correct, not a bug.

**SHAP at flag time**: Computed and stored as JSONB on the flag record, never re-computed on demand. The dashboard's waterfall chart reads directly from the stored values. This keeps the dashboard fast and the explanation forensically tied to the state at detection time.

**No auth**: Portfolio project. `analyst_id` is a plain string in the request body.
