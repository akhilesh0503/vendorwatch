"""Initial schema

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-06-08
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "vendors",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_vendors_category", "vendors", ["category"])

    op.create_table(
        "projects",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("budget", sa.Numeric(15, 2)),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "invoices",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("invoice_number", sa.String(50), nullable=False),
        sa.Column("amount", sa.Numeric(15, 2), nullable=False),
        sa.Column("submitted_at", sa.DateTime(), nullable=False),
        sa.Column("paid_at", sa.DateTime()),
        sa.Column("days_to_payment", sa.Float()),
        sa.Column("status", sa.String(20), server_default="paid", nullable=False),
        sa.ForeignKeyConstraint(["vendor_id"], ["vendors.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("invoice_number"),
    )
    op.create_index("ix_invoices_vendor_id", "invoices", ["vendor_id"])
    op.create_index("ix_invoices_submitted_at", "invoices", ["submitted_at"])
    op.create_index("ix_invoices_vendor_submitted", "invoices", ["vendor_id", "submitted_at"])

    op.create_table(
        "approvals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("invoice_id", sa.Integer(), nullable=False),
        sa.Column("approver_id", sa.String(50)),
        sa.Column("submitted_at", sa.DateTime(), nullable=False),
        sa.Column("approved_at", sa.DateTime()),
        sa.Column("days_to_approval", sa.Float()),
        sa.Column("status", sa.String(20), server_default="approved", nullable=False),
        sa.ForeignKeyConstraint(["invoice_id"], ["invoices.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_approvals_invoice_id", "approvals", ["invoice_id"])

    op.create_table(
        "anomaly_flags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.Integer(), nullable=False),
        sa.Column("invoice_id", sa.Integer()),
        sa.Column("detected_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("isolation_forest_score", sa.Float()),
        sa.Column("cusum_breach_severity", sa.Float()),
        sa.Column("peer_deviation_score", sa.Float()),
        sa.Column("shap_values", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("shap_explanation", sa.Text()),
        sa.Column("layers_fired", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("flag_status", sa.String(20), server_default="active", nullable=False),
        sa.Column("primary_signal", sa.String(500)),
        sa.Column("vendor_category", sa.String(50)),
        sa.ForeignKeyConstraint(["vendor_id"], ["vendors.id"]),
        sa.ForeignKeyConstraint(["invoice_id"], ["invoices.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_anomaly_flags_vendor_id", "anomaly_flags", ["vendor_id"])
    op.create_index("ix_anomaly_flags_risk_score", "anomaly_flags", ["risk_score"])
    op.create_index("ix_anomaly_flags_flag_status", "anomaly_flags", ["flag_status"])
    op.create_index("ix_anomaly_flags_detected_at", "anomaly_flags", ["detected_at"])

    op.create_table(
        "analyst_feedback",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("flag_id", sa.Integer(), nullable=False),
        sa.Column("analyst_id", sa.String(100), nullable=False),
        sa.Column("label", sa.String(20), nullable=False),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False),
        sa.ForeignKeyConstraint(["flag_id"], ["anomaly_flags.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analyst_feedback_flag_id", "analyst_feedback", ["flag_id"])
    op.create_index("ix_analyst_feedback_created_at", "analyst_feedback", ["created_at"])

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("vendor_category", sa.String(50), nullable=False),
        sa.Column("training_date", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("sample_count", sa.Integer()),
        sa.Column("contamination_used", sa.Float()),
        sa.Column("f1_on_labeled_subset", sa.Float()),
        sa.Column("model_path", sa.String(500)),
        sa.Column("is_active", sa.Boolean(), server_default="false", nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_versions_category_active", "model_versions", ["vendor_category", "is_active"])

    op.create_table(
        "peer_groups",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.Integer(), nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=False),
        sa.Column("centroid_distance", sa.Float()),
        sa.Column("computed_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text())),
        sa.ForeignKeyConstraint(["vendor_id"], ["vendors.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_peer_groups_vendor_id", "peer_groups", ["vendor_id"])

    op.create_table(
        "cusum_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.Integer(), nullable=False),
        sa.Column("feature_name", sa.String(50), nullable=False),
        sa.Column("cusum_pos", sa.Float(), server_default="0", nullable=False),
        sa.Column("cusum_neg", sa.Float(), server_default="0", nullable=False),
        sa.Column("target_mean", sa.Float()),
        sa.Column("target_std", sa.Float()),
        sa.Column("last_updated", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False),
        sa.ForeignKeyConstraint(["vendor_id"], ["vendors.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("vendor_id", "feature_name", name="uq_cusum_vendor_feature"),
    )
    op.create_index("ix_cusum_state_vendor_id", "cusum_state", ["vendor_id"])


def downgrade() -> None:
    op.drop_table("cusum_state")
    op.drop_table("peer_groups")
    op.drop_table("model_versions")
    op.drop_table("analyst_feedback")
    op.drop_table("anomaly_flags")
    op.drop_table("approvals")
    op.drop_table("invoices")
    op.drop_table("projects")
    op.drop_table("vendors")
