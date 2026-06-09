"""Page 3 — Vendor Deep Dive."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import api_client as api

st.title("Vendor Deep Dive")

# Vendor selector (by ID — extend with name search if desired)
vendor_id = st.number_input("Vendor ID", min_value=1, value=1, step=1)

if st.button("Load Vendor History"):
    st.session_state["vd_vendor_id"] = vendor_id

vd_id = st.session_state.get("vd_vendor_id")
if not vd_id:
    st.info("Enter a vendor ID and click 'Load Vendor History'.")
    st.stop()

try:
    data = api.vendor_history(vd_id)
except Exception as exc:
    st.error(f"Cannot load vendor {vd_id}: {exc}")
    st.stop()

st.subheader(f"{data['vendor_name']} — {data['vendor_category']}")

invoices = data.get("invoices", [])
if not invoices:
    st.info("No invoice history for this vendor.")
    st.stop()

df_inv = pd.DataFrame(invoices)
df_inv["submitted_at"] = pd.to_datetime(df_inv["submitted_at"])
df_inv["amount"]       = df_inv["amount"].astype(float)

# ── Invoice amount timeline ────────────────────────────────────────────────
st.subheader("Invoice Amount History (18 months)")
fig = px.scatter(
    df_inv,
    x       = "submitted_at",
    y       = "amount",
    hover_data = ["invoice_number", "days_to_approval", "days_to_payment"],
    labels  = {"submitted_at": "Submission Date", "amount": "Amount ($)"},
)
# Rolling 30-day mean
df_roll = df_inv.set_index("submitted_at").sort_index()
df_roll["rolling_mean"] = df_roll["amount"].rolling("30D").mean()
fig.add_trace(go.Scatter(
    x    = df_roll.index,
    y    = df_roll["rolling_mean"],
    name = "30-day rolling mean",
    line = {"color": "orange", "width": 2},
))
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)

# ── Approval cycle time series ─────────────────────────────────────────────
st.subheader("Approval Cycle Time")
df_ap = df_inv[df_inv["days_to_approval"].notna()].copy()
if not df_ap.empty:
    fig2 = px.line(
        df_ap.sort_values("submitted_at"),
        x      = "submitted_at",
        y      = "days_to_approval",
        markers = True,
        labels  = {"submitted_at": "Date", "days_to_approval": "Days to Approval"},
    )
    fig2.update_layout(height=280)
    st.plotly_chart(fig2, use_container_width=True)

# ── Anomaly score history ──────────────────────────────────────────────────
score_hist = data.get("anomaly_score_history", [])
if score_hist:
    st.subheader("Anomaly Score History")
    df_sc = pd.DataFrame(score_hist)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_sc["date"], y=df_sc["risk_score"],
                              name="Composite Risk Score", line={"color": "#d62728"}))
    fig3.add_trace(go.Scatter(x=df_sc["date"], y=df_sc["if_score"],
                              name="IF Score", line={"dash": "dash"}))
    fig3.add_hline(y=0.70, line_dash="dot", line_color="red",
                   annotation_text="HIGH threshold")
    fig3.add_hline(y=0.40, line_dash="dot", line_color="orange",
                   annotation_text="MEDIUM threshold")
    fig3.update_layout(height=320)
    st.plotly_chart(fig3, use_container_width=True)

# ── CUSUM chart ────────────────────────────────────────────────────────────
cusum_data = data.get("cusum_chart", {})
if cusum_data:
    st.subheader("CUSUM Control Chart")
    for feature, cdata in cusum_data.items():
        series = cdata.get("series", [])
        if not series:
            continue
        df_c = pd.DataFrame(series)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df_c["date"], y=df_c["c_pos"],
                                  name="C_pos", line={"color": "#d62728"}))
        fig4.add_hline(y=cdata["h"], line_dash="dot", line_color="red",
                       annotation_text=f"h={cdata['h']}")
        fig4.update_layout(title=f"CUSUM — {feature}", height=250)
        st.plotly_chart(fig4, use_container_width=True)

# ── Historical flags table ─────────────────────────────────────────────────
all_flags = data.get("all_flags", [])
if all_flags:
    st.subheader("All Historical Flags")
    st.dataframe(pd.DataFrame(all_flags), use_container_width=True)
