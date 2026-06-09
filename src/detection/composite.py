"""
Composite risk score and flag decision logic.

risk_score = 0.40 * isolation_forest_score
           + 0.35 * cusum_breach_severity
           + 0.25 * peer_deviation_score

All input scores must be in [0, 1].
"""

from typing import Dict, Optional

IF_WEIGHT    = 0.40
CUSUM_WEIGHT = 0.35
PEER_WEIGHT  = 0.25

FLAG_THRESHOLD   = 0.35  # minimum risk_score to create a flag
IF_FIRE_THRESHOLD   = 0.50  # IF score above which the layer "fires"
PEER_FIRE_THRESHOLD = 0.60  # peer deviation score above which the layer fires


def risk_score(
    if_score: float,
    cusum_severity: float,
    peer_deviation: float,
) -> float:
    return (
        IF_WEIGHT    * max(0.0, min(1.0, if_score))
        + CUSUM_WEIGHT * max(0.0, min(1.0, cusum_severity))
        + PEER_WEIGHT  * max(0.0, min(1.0, peer_deviation))
    )


def layers_fired(
    if_score: float,
    cusum_result: Optional[Dict],
    peer_deviation: float,
) -> Dict:
    cusum_breach = False
    if cusum_result:
        cusum_breach = any(v.get("breach", False) for v in cusum_result.values())

    return {
        "isolation_forest": if_score    >= IF_FIRE_THRESHOLD,
        "cusum":            cusum_breach,
        "peer_group":       peer_deviation >= PEER_FIRE_THRESHOLD,
    }


def risk_tier(score: float) -> str:
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"
