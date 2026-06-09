"""
Layer 3: KMeans peer group analysis.

Vendors are clustered within their category using lifetime aggregate features:
  [avg_invoice_amount, avg_monthly_invoice_frequency]

At training time: fit KMeans, store assignments in peer_groups table.
At inference time: project vendor's current lifetime features onto stored
centroids and return a normalized deviation score.

Score normalisation: 95th-percentile of centroid distances within category = 1.0.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

PEER_FEATURES = ["avg_invoice_amount", "avg_monthly_frequency"]


def auto_n_clusters(n_vendors: int) -> int:
    return max(3, n_vendors // 25)


def get_vendor_peer_features(
    category: str,
    cur,
) -> Tuple[List[int], np.ndarray]:
    """
    Compute clustering features for all vendors in a category.
    Requires at least 3 invoices per vendor.
    """
    cur.execute("""
        SELECT
            v.id,
            AVG(i.amount)                                         AS avg_amount,
            COUNT(*) / GREATEST(
                EXTRACT(MONTH FROM AGE(MAX(i.submitted_at), MIN(i.submitted_at))) + 1,
                1.0
            )                                                     AS avg_monthly_freq
        FROM vendors v
        JOIN invoices i ON i.vendor_id = v.id
        WHERE v.category = %s
        GROUP BY v.id
        HAVING COUNT(*) >= 3
    """, (category,))
    rows = cur.fetchall()
    if not rows:
        return [], np.empty((0, 2))

    ids = [int(r[0]) for r in rows]
    X   = np.array([[float(r[1]), float(r[2])] for r in rows], dtype=np.float64)
    return ids, np.nan_to_num(X, nan=0.0)


def fit(
    vendor_ids: List[int],
    X: np.ndarray,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[KMeans, StandardScaler, np.ndarray, np.ndarray]:
    """
    Fit KMeans on vendor peer features.

    Returns:
        kmeans:    fitted KMeans (operates on scaled space)
        scaler:    fitted StandardScaler
        labels:    cluster label per vendor
        distances: L2 distance to assigned centroid per vendor (in scaled space)
    """
    n_clusters = n_clusters or auto_n_clusters(len(vendor_ids))
    n_clusters  = min(n_clusters, len(vendor_ids))

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    distances = np.array([
        np.linalg.norm(X_scaled[i] - kmeans.cluster_centers_[labels[i]])
        for i in range(len(vendor_ids))
    ])
    return kmeans, scaler, labels, distances


def normalize_distances(distances: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] using 95th-percentile saturation."""
    p95 = np.percentile(distances, 95) if len(distances) > 0 else 1.0
    if p95 < 1e-9:
        return np.zeros_like(distances)
    return np.clip(distances / p95, 0.0, 1.0)


def deviation_score(
    vendor_id: int,
    cur,
    kmeans: KMeans,
    scaler: StandardScaler,
    peer_norm_p95: float,
) -> float:
    """
    Compute the normalized peer deviation score for one vendor at inference time.
    Uses lifetime averages (stable for peer group comparison).
    """
    cur.execute("""
        SELECT
            AVG(i.amount) AS avg_amount,
            COUNT(*) / GREATEST(
                EXTRACT(MONTH FROM AGE(MAX(i.submitted_at), MIN(i.submitted_at))) + 1,
                1.0
            ) AS avg_monthly_freq
        FROM invoices i
        WHERE i.vendor_id = %s
        HAVING COUNT(*) >= 3
    """, (vendor_id,))
    row = cur.fetchone()
    if row is None:
        return 0.0

    X    = np.array([[float(row[0]), float(row[1])]], dtype=np.float64)
    X_sc = scaler.transform(np.nan_to_num(X))
    lbl  = int(kmeans.predict(X_sc)[0])
    dist = float(np.linalg.norm(X_sc[0] - kmeans.cluster_centers_[lbl]))

    return float(np.clip(dist / max(peer_norm_p95, 1e-9), 0.0, 1.0))


def scatter_data(vendor_id: int, cur, kmeans: KMeans, scaler: StandardScaler) -> Dict:
    """Return peer group scatter data for dashboard chart."""
    category = None
    cur.execute("SELECT category FROM vendors WHERE id = %s", (vendor_id,))
    row = cur.fetchone()
    if row:
        category = row[0]

    cur.execute("""
        SELECT v.id, v.name,
               AVG(i.amount)                                         AS avg_amount,
               COUNT(*) / GREATEST(
                   EXTRACT(MONTH FROM AGE(MAX(i.submitted_at), MIN(i.submitted_at))) + 1, 1.0
               )                                                     AS avg_monthly_freq
        FROM vendors v
        JOIN invoices i ON i.vendor_id = v.id
        WHERE v.category = %s
        GROUP BY v.id
        HAVING COUNT(*) >= 3
    """, (category,))
    rows = cur.fetchall()
    if not rows:
        return {}

    ids   = [r[0] for r in rows]
    names = [r[1] for r in rows]
    X     = np.array([[float(r[2]), float(r[3])] for r in rows])
    X_sc  = scaler.transform(np.nan_to_num(X))
    labels = kmeans.predict(X_sc)
    centroids_sc = kmeans.cluster_centers_

    return {
        "vendors": [
            {
                "id":             ids[i],
                "name":           names[i],
                "cluster":        int(labels[i]),
                "avg_amount":     float(X[i][0]),
                "avg_freq":       float(X[i][1]),
                "is_target":      ids[i] == vendor_id,
            }
            for i in range(len(ids))
        ],
        "centroids": centroids_sc.tolist(),
    }
