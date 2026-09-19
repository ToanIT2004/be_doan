"""add leave requests

Revision ID: e3f6a9b4c5d7
Revises: d2e5f8a9b3c4
Create Date: 2026-09-12
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e3f6a9b4c5d7"
down_revision: Union[str, Sequence[str], None] = "d2e5f8a9b3c4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "leaveRequests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("leaveType", sa.String(length=30), nullable=False),
        sa.Column("startDate", sa.Date(), nullable=False),
        sa.Column("endDate", sa.Date(), nullable=False),
        sa.Column("reason", sa.String(length=1000), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="pending", nullable=False),
        sa.Column("rejectReason", sa.String(length=1000), nullable=True),
        sa.Column("employeeId", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            '"leaveType" IN (\'annual_leave\', \'sick_leave\', \'unpaid_leave\', \'personal_leave\', \'other\')',
            name="ck_leaveRequests_leaveType",
        ),
        sa.CheckConstraint(
            'status IN (\'pending\', \'approved\', \'rejected\')',
            name="ck_leaveRequests_status",
        ),
        sa.CheckConstraint(
            '"endDate" >= "startDate"',
            name="ck_leaveRequests_date_range",
        ),
        sa.ForeignKeyConstraint(["employeeId"], ["employees.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_leaveRequests_id"), "leaveRequests", ["id"], unique=False)
    op.create_index(
        op.f("ix_leaveRequests_employeeId"),
        "leaveRequests",
        ["employeeId"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_leaveRequests_employeeId"), table_name="leaveRequests")
    op.drop_index(op.f("ix_leaveRequests_id"), table_name="leaveRequests")
    op.drop_table("leaveRequests")
