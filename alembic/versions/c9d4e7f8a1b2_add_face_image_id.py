"""add face image id

Revision ID: c9d4e7f8a1b2
Revises: a8f0c1b2d3e4
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c9d4e7f8a1b2"
down_revision: Union[str, Sequence[str], None] = "a8f0c1b2d3e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "faceImage",
        sa.Column("id", sa.Integer(), sa.Identity(), nullable=False),
    )
    op.drop_constraint("faceImage_pkey", "faceImage", type_="primary")
    op.create_primary_key("pk_faceImage", "faceImage", ["id"])
    op.create_unique_constraint(
        "uq_faceImage_employeeId_imagePath",
        "faceImage",
        ["employeeId", "imagePath"],
    )
    op.create_index(
        "ix_faceImage_employeeId",
        "faceImage",
        ["employeeId"],
        unique=False,
    )
    op.create_index("ix_faceImage_id", "faceImage", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_faceImage_id", table_name="faceImage")
    op.drop_index("ix_faceImage_employeeId", table_name="faceImage")
    op.drop_constraint(
        "uq_faceImage_employeeId_imagePath",
        "faceImage",
        type_="unique",
    )
    op.drop_constraint("pk_faceImage", "faceImage", type_="primary")
    op.create_primary_key(
        "faceImage_pkey",
        "faceImage",
        ["employeeId", "imagePath"],
    )
    op.drop_column("faceImage", "id")
