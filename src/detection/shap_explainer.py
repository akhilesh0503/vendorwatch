"""
SHAP explainability — computed at flag time, never on-demand.

Pipeline:
  1. TreeExplainer on the Isolation Forest using stored background data.
  2. Negate raw SHAP values: IF score_samples is inverted (higher = more normal),
     so we negate so that positive SHAP contribution = pushes toward anomaly.
  3. Generate a deterministic 3-sentence natural language explanation using
     string templates that embed actual numeric values from the data.

Sentence structure:
  S1: primary signal   — feature with highest |negated_shap|
  S2: secondary signal — feature with second-highest |negated_shap|
  S3: combined pattern — what the combination of S1+S2 suggests
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from src.detection.features import FEATURE_NAMES

log = logging.getLogger(__name__)

FEATURE_LABELS = {
    "invoice_amount":                    "mean invoice amount",
    "days_to_approval":                  "approval cycle time",
    "days_to_payment":                   "total payment time",
    "invoice_frequency_7d":              "invoice frequency (7-day window)",
    "invoice_frequency_30d":             "invoice frequency (30-day window)",
    "amount_deviation_from_vendor_mean": "invoice amount deviation from vendor baseline",
    "approval_cycle_z_score":            "approval cycle Z-score",
    "payment_timing_z_score":            "payment timing Z-score",
}

# Specific pattern hints keyed by (primary_feature, secondary_feature) pair
_PATTERN_HINTS = {
    frozenset(["invoice_frequency_7d",             "invoice_amount"]):                    "This combination is consistent with invoice splitting — multiple near-threshold invoices submitted in a compressed window to circumvent approval controls.",
    frozenset(["invoice_frequency_7d",             "invoice_frequency_30d"]):             "Concentrated invoice bursts within both 7-day and 30-day windows suggest fabricated or pre-dated activity rather than legitimate workload.",
    frozenset(["invoice_frequency_30d",            "amount_deviation_from_vendor_mean"]): "Elevated volume combined with above-baseline amounts suggests either a genuine workload surge or a systematic budget extraction pattern.",
    frozenset(["approval_cycle_z_score",           "invoice_amount"]):                    "Rapid approval of a high-value invoice is a strong indicator that standard review controls were bypassed — verify the approval chain manually.",
    frozenset(["days_to_approval",                 "invoice_amount"]):                    "A high-value invoice with unusually short approval time warrants verification of approver identity and delegation authority.",
    frozenset(["amount_deviation_from_vendor_mean","invoice_frequency_30d"]):             "Both amount and frequency are elevated relative to this vendor's history, consistent with a gradual cost-inflation or budget-extraction scheme.",
    frozenset(["amount_deviation_from_vendor_mean","approval_cycle_z_score"]):            "Amount inflation paired with accelerated approvals suggests coordination between the vendor and an internal approver.",
    frozenset(["payment_timing_z_score",           "invoice_amount"]):                    "Unusually fast payment for a high-value invoice may indicate preferential treatment or a fictitious vendor relationship.",
}


def compute(
    model,
    features: np.ndarray,
    background: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute SHAP values for one feature vector.

    Returns dict with:
        shap_values  — list[float], negated so positive = contributes to anomaly
        base_value   — float, negated expected model output
        feature_names — list[str]
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model, data=background)
        raw = explainer.shap_values(features.reshape(1, -1))[0]
        anomaly_shap = -np.array(raw, dtype=np.float64)  # negate
        base_value   = float(-explainer.expected_value)
    except Exception as exc:
        log.warning("SHAP computation failed (%s); using zero values.", exc)
        anomaly_shap = np.zeros(len(FEATURE_NAMES))
        base_value   = 0.0

    return {
        "shap_values":   anomaly_shap.tolist(),
        "base_value":    base_value,
        "feature_names": list(FEATURE_NAMES),
    }


def generate_explanation(
    shap_dict: Dict,
    feature_values: np.ndarray,
) -> str:
    """
    Build a 3-sentence human-readable explanation. Uses actual numeric values —
    no generic descriptions, no placeholder text.
    """
    shap_vals = np.array(shap_dict["shap_values"])
    names     = shap_dict["feature_names"]

    ranked = np.argsort(np.abs(shap_vals))[::-1]
    top_idx    = int(ranked[0])
    second_idx = int(ranked[1]) if len(ranked) > 1 else top_idx

    top_name    = names[top_idx]
    top_val     = float(feature_values[top_idx])
    top_contrib = float(shap_vals[top_idx])

    sec_name    = names[second_idx]
    sec_val     = float(feature_values[second_idx])
    sec_contrib = float(shap_vals[second_idx])

    s1 = _sentence(top_name,    top_val,    top_contrib,  ordinal="primary")
    s2 = _sentence(sec_name,    sec_val,    sec_contrib,  ordinal="secondary")
    s3 = _combined(top_name, sec_name, top_val, sec_val)
    return f"{s1} {s2} {s3}"


def _contrib_str(v: float) -> str:
    return f"+{v:.2f}" if v >= 0 else f"{v:.2f}"


def _sentence(feature: str, value: float, contrib: float, ordinal: str) -> str:
    label  = FEATURE_LABELS.get(feature, feature.replace("_", " "))
    c_str  = _contrib_str(contrib)

    if feature in ("approval_cycle_z_score", "payment_timing_z_score",
                   "amount_deviation_from_vendor_mean"):
        direction = "above" if value >= 0 else "below"
        return (
            f"The {ordinal} signal is {label}: {abs(value):.2f}σ {direction} this "
            f"vendor's 90-day baseline (contributing {c_str} to anomaly score)."
        )
    if feature == "invoice_amount":
        return (
            f"The {ordinal} signal is {label} of ${value:,.0f} — "
            f"{'elevated' if contrib > 0 else 'suppressed'} relative to this vendor "
            f"category's typical range (contributing {c_str} to anomaly score)."
        )
    if feature in ("days_to_approval", "days_to_payment"):
        direction = "elevated" if contrib > 0 else "suppressed"
        return (
            f"The {ordinal} signal is {label} of {value:.1f} days — "
            f"this is {direction} vs this vendor's historical pattern "
            f"(contributing {c_str} to anomaly score)."
        )
    if feature in ("invoice_frequency_7d", "invoice_frequency_30d"):
        window = "7-day" if "7d" in feature else "30-day"
        return (
            f"The {ordinal} signal is {label}: {value:.1f} invoices in the "
            f"{window} window — {'above' if contrib > 0 else 'below'} peer-group norms "
            f"(contributing {c_str} to anomaly score)."
        )
    return (
        f"The {ordinal} signal is {label} (value: {value:.2f}), "
        f"contributing {c_str} to the anomaly score."
    )


def _combined(top_name: str, sec_name: str, top_val: float, sec_val: float) -> str:
    key = frozenset([top_name, sec_name])
    if key in _PATTERN_HINTS:
        return _PATTERN_HINTS[key]
    lbl1 = FEATURE_LABELS.get(top_name, top_name.replace("_", " "))
    lbl2 = FEATURE_LABELS.get(sec_name, sec_name.replace("_", " "))
    return (
        f"The combination of anomalous {lbl1} and {lbl2} warrants manual "
        f"review to determine whether this represents legitimate business activity."
    )
