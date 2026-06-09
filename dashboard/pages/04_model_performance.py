"""Page 4 — Model Performance."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

import api_client as api

st.title("Model Performance")

try:
    summary = api.get_summary()
    h_data  = api.health()
except Exception as exc:
    st.error(f"Cannot reach API: {exc}")
    st.stop()

# ── Model version table ────────────────────────────────────────────────────
st.subheader("Active Model Versions")
mvs = summary.get("model_versions", [])
if mvs:
    df_mv = pd.DataFrame(mvs)
    active_df = df_mv[df_mv["is_active"] == True]
    st.dataframe(active_df[["category", "version", "training_date"]], use_container_width=True)
else:
    st.warning("No model versions found.")

# ── Feedback distribution & confusion proxy ────────────────────────────────
st.subheader("Feedback Label Distribution")
fb_dist = summary.get("feedback_distribution", {})
if fb_dist:
    labels = list(fb_dist.keys())
    values = list(fb_dist.values())
    fig = px.bar(x=labels, y=values,
                 labels={"x": "Label", "y": "Count"},
                 color=labels,
                 color_discrete_map={
                     "true_positive":  "#2ca02c",
                     "false_positive": "#d62728",
                     "escalated":      "#ff7f0e",
                 })
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

    tp = fb_dist.get("true_positive", 0)
    fp = fb_dist.get("false_positive", 0)
    if tp + fp > 0:
        precision = tp / (tp + fp)
        st.metric("Precision (TP / (TP + FP))", f"{precision:.1%}")

# ── SHAP feature importance (global) ──────────────────────────────────────
st.subheader("SHAP Feature Importance — Global (from recent flags)")
try:
    flags = api.list_flags(risk_min=0.0, status="active")
    if flags:
        feature_names = [
            "invoice_amount", "days_to_approval", "days_to_payment",
            "invoice_frequency_7d", "invoice_frequency_30d",
            "amount_deviation_from_vendor_mean",
            "approval_cycle_z_score", "payment_timing_z_score",
        ]
        importance_sums = {f: 0.0 for f in feature_names}
        count = 0

        for flag in flags[:50]:  # sample up to 50 flags
            try:
                detail = api.get_flag(flag["id"])
                shap_d = detail.get("shap_values")
                if shap_d and "shap_values" in shap_d:
                    for name, val in zip(shap_d["feature_names"], shap_d["shap_values"]):
                        if name in importance_sums:
                            importance_sums[name] += abs(val)
                    count += 1
            except Exception:
                continue

        if count > 0:
            avg_imp = {k: v / count for k, v in importance_sums.items()}
            df_imp  = pd.DataFrame(
                sorted(avg_imp.items(), key=lambda x: x[1], reverse=True),
                columns=["Feature", "Mean |SHAP|"]
            )
            fig_imp = px.bar(
                df_imp,
                x     = "Mean |SHAP|",
                y     = "Feature",
                orientation = "h",
                labels = {"Feature": "", "Mean |SHAP|": "Mean |SHAP value|"},
                title  = f"Average absolute SHAP contributions across {count} flags",
            )
            fig_imp.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("No SHAP data in current flags to aggregate.")
    else:
        st.info("No active flags to compute feature importance.")
except Exception as exc:
    st.warning(f"Could not compute global SHAP importance: {exc}")

# ── Feedback queue depth ────────────────────────────────────────────────────
st.subheader("Retraining Queue")
depth     = h_data.get("feedback_queue_depth", 0)
threshold = 50
st.progress(min(depth / threshold, 1.0), text=f"{depth}/{threshold} new labels since last retrain")
if depth >= threshold:
    st.success("Threshold reached — retraining will trigger automatically on next hourly check.")
