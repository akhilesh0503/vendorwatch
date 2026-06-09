"""Pydantic request / response schemas for the VendorWatch API."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Vendor analysis
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    vendor_id:               int
    vendor_name:             str
    vendor_category:         str
    risk_score:              float
    risk_tier:               str
    isolation_forest_score:  float
    cusum_breach_severity:   float
    peer_deviation_score:    float
    layers_fired:            Dict[str, bool]
    shap_values:             Optional[Dict[str, Any]] = None
    shap_explanation:        Optional[str]            = None
    flag_id:                 Optional[int]            = None
    flag_created:            bool                     = False


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

class FlagListItem(BaseModel):
    id:                     int
    vendor_id:              int
    vendor_name:            str
    vendor_category:        str
    risk_score:             float
    risk_tier:              str
    flag_status:            str
    detected_at:            datetime
    primary_signal:         Optional[str] = None
    days_since_first_flag:  int
    isolation_forest_score: Optional[float] = None
    cusum_breach_severity:  Optional[float] = None
    peer_deviation_score:   Optional[float] = None


class FlagDetail(BaseModel):
    id:                     int
    vendor_id:              int
    vendor_name:            str
    vendor_category:        str
    risk_score:             float
    risk_tier:              str
    flag_status:            str
    detected_at:            datetime
    primary_signal:         Optional[str]       = None
    isolation_forest_score: Optional[float]     = None
    cusum_breach_severity:  Optional[float]     = None
    peer_deviation_score:   Optional[float]     = None
    shap_values:            Optional[Dict]      = None
    shap_explanation:       Optional[str]       = None
    layers_fired:           Optional[Dict]      = None
    cusum_chart:            Optional[Dict]      = None
    peer_scatter:           Optional[Dict]      = None
    feedback:               List[Dict]          = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    analyst_id: str
    label:      str  # true_positive | false_positive | escalated
    notes:      Optional[str] = None


class FeedbackResponse(BaseModel):
    flag_id:          int
    label:            str
    feedback_id:      int
    retrain_queued:   bool
    feedback_count:   int


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class DashboardSummary(BaseModel):
    total_active_flags:     int
    high_risk_count:        int
    medium_risk_count:      int
    low_risk_count:         int
    flags_by_category:      Dict[str, int]
    avg_risk_score_30d:     float
    daily_flag_counts:      List[Dict]       # [{date, count}]
    model_versions:         List[Dict]       # [{category, version, training_date, is_active}]
    feedback_distribution:  Dict[str, int]


# ---------------------------------------------------------------------------
# Vendor history
# ---------------------------------------------------------------------------

class VendorHistoryResponse(BaseModel):
    vendor_id:           int
    vendor_name:         str
    vendor_category:     str
    invoices:            List[Dict]
    anomaly_score_history: List[Dict]
    all_flags:           List[Dict]
    cusum_chart:         Optional[Dict] = None


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

class RetrainRequest(BaseModel):
    reason: str = "manual"


class RetrainResponse(BaseModel):
    version:  str
    reason:   str
    results:  Dict[str, str]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:               str
    model_versions:       Dict[str, str]
    last_retrain:         Optional[datetime]
    feedback_queue_depth: int
    db_ok:                bool
