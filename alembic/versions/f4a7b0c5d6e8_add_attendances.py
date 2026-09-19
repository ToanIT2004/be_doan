"""add attendances

Revision ID: f4a7b0c5d6e8
Revises: e3f6a9b4c5d7
Create Date: 2026-09-12
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f4a7b0c5d6e8"
down_revision: Union[str, Sequence[str], None] = "e3f6a9b4c5d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "attendances",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("employeeId", sa.Integer(), nullable=False),
        sa.Column("attendanceDate", sa.Date(), nullable=False),
        sa.Column("checkIn", sa.String(length=40), nullable=False),
        sa.Column("checkOut", sa.String(length=40), nullable=True),
        sa.ForeignKeyConstraint(["employeeId"], ["employees.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "employeeId",
            "attendanceDate",
            name="uq_attendances_employee_date",
        ),
    )
    op.create_index(op.f("ix_attendances_id"), "attendances", ["id"], unique=False)
    op.create_index(
        op.f("ix_attendances_employeeId"),
        "attendances",
        ["employeeId"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_attendances_employeeId"), table_name="attendances")
    op.drop_index(op.f("ix_attendances_id"), table_name="attendances")
    op.drop_table("attendances")
