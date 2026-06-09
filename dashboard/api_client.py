"""Thin HTTP client — all dashboard calls go through here."""

import streamlit as st
import httpx

_TIMEOUT = 30.0


def _base() -> str:
    return st.session_state.get("api_base", "http://localhost:8000")


@st.cache_data(ttl=30)
def get_summary() -> dict:
    r = httpx.get(f"{_base()}/dashboard/summary", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30)
def list_flags(
    risk_min: float = 0.0,
    category: str   = "",
    status:   str   = "active",
    page:     int   = 1,
) -> list:
    params = {"risk_score_min": risk_min, "flag_status": status, "page": page}
    if category:
        params["vendor_category"] = category
    r = httpx.get(f"{_base()}/flags", params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=10)
def get_flag(flag_id: int) -> dict:
    r = httpx.get(f"{_base()}/flags/{flag_id}", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def submit_feedback(flag_id: int, analyst_id: str, label: str, notes: str = "") -> dict:
    r = httpx.patch(
        f"{_base()}/flags/{flag_id}/feedback",
        json={"analyst_id": analyst_id, "label": label, "notes": notes},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=20)
def vendor_history(vendor_id: int) -> dict:
    r = httpx.get(f"{_base()}/vendors/{vendor_id}/history", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def health() -> dict:
    r = httpx.get(f"{_base()}/health", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()
