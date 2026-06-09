"""Page 2 — Flag Investigation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import api_client as api

st.title("Flag Investigation")

# ── Filters ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Filters")
    min_score = st.slider("Min Risk Score", 0.0, 1.0, 0.0, 0.05)
    category  = st.selectbox("Category", ["", "construction", "IT", "logistics", "facilities"])
    status    = st.selectbox("Status", ["active", "escalated", "resolved", ""])

try:
    flags = api.list_flags(risk_min=min_score, category=category, status=status)
except Exception as exc:
    st.error(f"Cannot reach API: {exc}")
    st.stop()

if not flags:
    st.info("No flags matching the current filters.")
    st.stop()

# ── Flag table ────────────────────────────────────────────────────────────────
df = pd.DataFrame([{
    "ID":           f["id"],
    "Vendor":       f["vendor_name"],
    "Category":     f["vendor_category"],
    "Risk Score":   round(f["risk_score"], 3),
    "Tier":         f["risk_tier"],
    "Status":       f["flag_status"],
    "Days Active":  f["days_since_first_flag"],
    "Primary Signal": f.get("primary_signal") or "—",
    "Detected":     f["detected_at"][:10],
} for f in flags])

st.dataframe(df, use_container_width=True, height=300)

# ── Flag detail expander ──────────────────────────────────────────────────────
selected_id = st.selectbox("Select Flag ID to Investigate", [f["id"] for f in flags])

if selected_id:
    try:
        detail = api.get_flag(selected_id)
    except Exception as exc:
        st.error(f"Failed to load flag: {exc}")
        st.stop()

    st.subheader(f"Flag #{detail['id']} — {detail['vendor_name']} ({detail['vendor_category']})")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Risk Score", round(detail["risk_score"], 3))
    k2.metric("Tier",       detail["risk_tier"])
    k3.metric("IF Score",   round(detail.get("isolation_forest_score") or 0, 3))
    k4.metric("CUSUM",      round(detail.get("cusum_breach_severity")  or 0, 3))

    # Natural language explanation
    if detail.get("shap_explanation"):
        st.info(f"**Explanation:** {detail['shap_explanation']}")

    tab1, tab2, tab3 = st.tabs(["SHAP Waterfall", "CUSUM Chart", "Peer Group"])

    # ── SHAP Waterfall ────────────────────────────────────────────────────
    with tab1:
        shap_data = detail.get("shap_values")
        if shap_data and "shap_values" in shap_data:
            vals  = shap_data["shap_values"]
            names = shap_data["feature_names"]
            base  = shap_data.get("base_value", 0)
            pairs = sorted(zip(names, vals), key=lambda x: x[1])
            fig   = go.Figure(go.Waterfall(
                name        = "SHAP",
                orientation = "h",
                measure     = ["relative"] * len(pairs),
                x           = [p[1] for p in pairs],
                y           = [p[0].replace("_", " ") for p in pairs],
                connector   = {"line": {"color": "rgb(63, 63, 63)"}},
                decreasing  = {"marker": {"color": "#1f77b4"}},
                increasing  = {"marker": {"color": "#d62728"}},
            ))
            fig.update_layout(
                title  = "SHAP Feature Contributions (red = pushes toward anomaly)",
                height = 420,
                xaxis_title = "SHAP value",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SHAP data available for this flag.")

    # ── CUSUM Chart ───────────────────────────────────────────────────────
    with tab2:
        cusum_data = detail.get("cusum_chart", {})
        if cusum_data:
            for feature, cdata in cusum_data.items():
                series = cdata.get("series", [])
                h_val  = cdata.get("h", 5.0)
                if not series:
                    continue
                df_c = pd.DataFrame(series)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_c["date"], y=df_c["c_pos"],
                    name="C_pos (upward shift)", line={"color": "#d62728"},
                ))
                fig.add_trace(go.Scatter(
                    x=df_c["date"], y=df_c["c_neg"].abs() if hasattr(df_c["c_neg"], "abs") else [-v for v in df_c["c_neg"]],
                    name="|C_neg| (downward shift)", line={"color": "#ff7f0e", "dash": "dash"},
                ))
                fig.add_hline(y=h_val, line_dash="dot", line_color="red",
                              annotation_text=f"Threshold h={h_val}")
                fig.update_layout(
                    title  = f"CUSUM — {feature.replace('_', ' ')}",
                    height = 320,
                    xaxis_title = "Date",
                    yaxis_title = "CUSUM statistic",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CUSUM data available.")

    # ── Peer Group Scatter ────────────────────────────────────────────────
    with tab3:
        peer_data = detail.get("peer_scatter", {})
        if peer_data and "vendors" in peer_data:
            df_p = pd.DataFrame(peer_data["vendors"])
            df_p["marker_size"]  = df_p["is_target"].apply(lambda x: 18 if x else 8)
            df_p["marker_color"] = df_p["is_target"].apply(lambda x: "red" if x else "steelblue")
            fig = px.scatter(
                df_p,
                x      = "avg_amount",
                y      = "avg_freq",
                color  = "cluster",
                size   = "marker_size",
                hover_name = "name",
                labels = {"avg_amount": "Avg Invoice Amount ($)", "avg_freq": "Avg Monthly Invoices"},
                title  = "Peer Group Scatter — flagged vendor highlighted in red",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No peer group data available.")

    # ── Feedback buttons ──────────────────────────────────────────────────
    st.subheader("Submit Feedback")
    analyst_id = st.text_input("Analyst ID", value="analyst_001")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("✅ True Positive", use_container_width=True):
            result = api.submit_feedback(selected_id, analyst_id, "true_positive")
            st.success(f"Labeled as True Positive. Feedback #{result['feedback_id']}")
            st.cache_data.clear()
    with col_b:
        if st.button("❌ False Positive", use_container_width=True):
            result = api.submit_feedback(selected_id, analyst_id, "false_positive")
            st.success(f"Labeled as False Positive. Feedback #{result['feedback_id']}")
            st.cache_data.clear()
    with col_c:
        if st.button("🔺 Escalate", use_container_width=True):
            result = api.submit_feedback(selected_id, analyst_id, "escalated")
            st.success(f"Escalated. Feedback #{result['feedback_id']}")
            st.cache_data.clear()

    if detail.get("feedback"):
        st.subheader("Existing Feedback")
        st.dataframe(pd.DataFrame(detail["feedback"]), use_container_width=True)
