from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String, Integer, Float, Boolean, Text, DateTime, Numeric,
    ForeignKey, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Vendor(Base):
    __tablename__ = "vendors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    invoices: Mapped[list["Invoice"]] = relationship(back_populates="vendor")
    flags: Mapped[list["AnomalyFlag"]] = relationship(back_populates="vendor")
    peer_groups: Mapped[list["PeerGroup"]] = relationship(back_populates="vendor")
    cusum_states: Mapped[list["CusumState"]] = relationship(back_populates="vendor")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    budget: Mapped[Optional[float]] = mapped_column(Numeric(15, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    invoices: Mapped[list["Invoice"]] = relationship(back_populates="project")


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vendor_id: Mapped[int] = mapped_column(Integer, ForeignKey("vendors.id"), nullable=False)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    invoice_number: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    amount: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    paid_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    days_to_payment: Mapped[Optional[float]] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), server_default="paid", nullable=False)

    vendor: Mapped["Vendor"] = relationship(back_populates="invoices")
    project: Mapped["Project"] = relationship(back_populates="invoices")
    approval: Mapped[Optional["Approval"]] = relationship(back_populates="invoice", uselist=False)
    flags: Mapped[list["AnomalyFlag"]] = relationship(back_populates="invoice")


class Approval(Base):
    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[int] = mapped_column(Integer, ForeignKey("invoices.id"), nullable=False)
    approver_id: Mapped[Optional[str]] = mapped_column(String(50))
    submitted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    days_to_approval: Mapped[Optional[float]] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), server_default="approved", nullable=False)

    invoice: Mapped["Invoice"] = relationship(back_populates="approval")


class AnomalyFlag(Base):
    __tablename__ = "anomaly_flags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vendor_id: Mapped[int] = mapped_column(Integer, ForeignKey("vendors.id"), nullable=False)
    invoice_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("invoices.id"))
    detected_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    isolation_forest_score: Mapped[Optional[float]] = mapped_column(Float)
    cusum_breach_severity: Mapped[Optional[float]] = mapped_column(Float)
    peer_deviation_score: Mapped[Optional[float]] = mapped_column(Float)
    shap_values: Mapped[Optional[dict]] = mapped_column(JSONB)
    shap_explanation: Mapped[Optional[str]] = mapped_column(Text)
    layers_fired: Mapped[Optional[dict]] = mapped_column(JSONB)
    flag_status: Mapped[str] = mapped_column(String(20), server_default="active", nullable=False)
    primary_signal: Mapped[Optional[str]] = mapped_column(String(500))
    vendor_category: Mapped[Optional[str]] = mapped_column(String(50))

    vendor: Mapped["Vendor"] = relationship(back_populates="flags")
    invoice: Mapped[Optional["Invoice"]] = relationship(back_populates="flags")
    feedback: Mapped[list["AnalystFeedback"]] = relationship(back_populates="flag")


class AnalystFeedback(Base):
    __tablename__ = "analyst_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    flag_id: Mapped[int] = mapped_column(Integer, ForeignKey("anomaly_flags.id"), nullable=False)
    analyst_id: Mapped[str] = mapped_column(String(100), nullable=False)
    label: Mapped[str] = mapped_column(String(20), nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    flag: Mapped["AnomalyFlag"] = relationship(back_populates="feedback")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    vendor_category: Mapped[str] = mapped_column(String(50), nullable=False)
    training_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    sample_count: Mapped[Optional[int]] = mapped_column(Integer)
    contamination_used: Mapped[Optional[float]] = mapped_column(Float)
    f1_on_labeled_subset: Mapped[Optional[float]] = mapped_column(Float)
    model_path: Mapped[Optional[str]] = mapped_column(String(500))
    is_active: Mapped[bool] = mapped_column(Boolean, server_default="false", nullable=False)


class PeerGroup(Base):
    __tablename__ = "peer_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vendor_id: Mapped[int] = mapped_column(Integer, ForeignKey("vendors.id"), nullable=False)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    centroid_distance: Mapped[Optional[float]] = mapped_column(Float)
    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    features: Mapped[Optional[dict]] = mapped_column(JSONB)

    vendor: Mapped["Vendor"] = relationship(back_populates="peer_groups")


class CusumState(Base):
    __tablename__ = "cusum_state"
    __table_args__ = (
        UniqueConstraint("vendor_id", "feature_name", name="uq_cusum_vendor_feature"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vendor_id: Mapped[int] = mapped_column(Integer, ForeignKey("vendors.id"), nullable=False)
    feature_name: Mapped[str] = mapped_column(String(50), nullable=False)
    cusum_pos: Mapped[float] = mapped_column(Float, server_default="0", nullable=False)
    cusum_neg: Mapped[float] = mapped_column(Float, server_default="0", nullable=False)
    target_mean: Mapped[Optional[float]] = mapped_column(Float)
    target_std: Mapped[Optional[float]] = mapped_column(Float)
    last_updated: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    vendor: Mapped["Vendor"] = relationship(back_populates="cusum_states")
