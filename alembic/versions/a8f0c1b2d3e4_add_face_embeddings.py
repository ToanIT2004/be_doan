"""add face embeddings

Revision ID: a8f0c1b2d3e4
Revises: 67f514c47028
Create Date: 2026-08-23 17:31:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a8f0c1b2d3e4"
down_revision: Union[str, Sequence[str], None] = "67f514c47028"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "employeeFaceEmbeddings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("employeeId", sa.Integer(), nullable=False),
        sa.Column("imagePath", sa.String(length=500), nullable=True),
        sa.Column("embedding", sa.JSON(), nullable=False),
        sa.Column("modelName", sa.String(length=100), nullable=False),
        sa.Column("createdAt", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updatedAt", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["employeeId"], ["employees.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_employeeFaceEmbeddings_employeeId"), "employeeFaceEmbeddings", ["employeeId"], unique=False)
    op.create_index(op.f("ix_employeeFaceEmbeddings_id"), "employeeFaceEmbeddings", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_employeeFaceEmbeddings_id"), table_name="employeeFaceEmbeddings")
    op.drop_index(op.f("ix_employeeFaceEmbeddings_employeeId"), table_name="employeeFaceEmbeddings")
    op.drop_table("employeeFaceEmbeddings")
