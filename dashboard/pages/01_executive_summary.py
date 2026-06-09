"""Page 1 — Executive Summary."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import api_client as api

st.title("Executive Summary")

try:
    data = api.get_summary()
except Exception as exc:
    st.error(f"Cannot reach API: {exc}")
    st.stop()

# ── KPI tiles ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Flags",  data["total_active_flags"])
c2.metric("🔴 HIGH (≥0.7)", data["high_risk_count"])
c3.metric("🟡 MEDIUM",      data["medium_risk_count"])
c4.metric("🟢 LOW (<0.4)",  data["low_risk_count"])

st.divider()

col_left, col_right = st.columns(2)

# ── Flags by category bar chart ─────────────────────────────────────────────
with col_left:
    st.subheader("Active Flags by Vendor Category")
    by_cat = data.get("flags_by_category", {})
    if by_cat:
        fig = px.bar(
            x=list(by_cat.keys()),
            y=list(by_cat.values()),
            labels={"x": "Category", "y": "Flag Count"},
            color=list(by_cat.values()),
            color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No active flags.")

# ── 30-day trend line ────────────────────────────────────────────────────────
with col_right:
    st.subheader("30-Day Flag Trend")
    daily = data.get("daily_flag_counts", [])
    if daily:
        df = pd.DataFrame(daily)
        fig = px.line(df, x="date", y="count", markers=True,
                      labels={"date": "Date", "count": "Flags"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No flag data for the last 30 days.")

st.divider()

# ── Model health ─────────────────────────────────────────────────────────────
st.subheader("Model Health")
mvs = data.get("model_versions", [])
active_mvs = [m for m in mvs if m.get("is_active")]

if active_mvs:
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Active Model Version", active_mvs[0]["version"] if active_mvs else "—")
    mc2.metric("Last Retrain", active_mvs[0]["training_date"][:10] if active_mvs else "—")
    fb_dist = data.get("feedback_distribution", {})
    total_fb = sum(fb_dist.values())
    mc3.metric("Total Labeled Samples", total_fb)
else:
    st.warning("No active model version found. Trigger POST /admin/retrain.")

# Feedback distribution donut
fb_dist = data.get("feedback_distribution", {})
if fb_dist:
    fig = go.Figure(go.Pie(
        labels=list(fb_dist.keys()),
        values=list(fb_dist.values()),
        hole=0.55,
    ))
    fig.update_layout(title="Feedback Label Distribution", height=280)
    st.plotly_chart(fig, use_container_width=True)
