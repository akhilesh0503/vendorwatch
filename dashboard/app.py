"""
VendorWatch Streamlit Dashboard — 4 pages.

All data comes from the FastAPI backend (never direct DB).
"""

import os
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title = "VendorWatch",
    page_icon  = "🔍",
    layout     = "wide",
)

pages = {
    "Executive Summary":   "pages/01_executive_summary.py",
    "Flag Investigation":  "pages/02_flag_investigation.py",
    "Vendor Deep Dive":    "pages/03_vendor_deep_dive.py",
    "Model Performance":   "pages/04_model_performance.py",
}

st.sidebar.title("VendorWatch")
st.sidebar.caption("Supply Chain Anomaly Detection")

page = st.sidebar.radio("Navigate", list(pages.keys()))

# Store API base in session so pages can read it
st.session_state["api_base"] = API_BASE

# Route to selected page module
import importlib.util, sys

spec = importlib.util.spec_from_file_location(
    "page_module",
    os.path.join(os.path.dirname(__file__), pages[page]),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
