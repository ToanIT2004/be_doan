"""allow one face image record per employee

Revision ID: d2e5f8a9b3c4
Revises: c9d4e7f8a1b2
"""

from typing import Sequence, Union

from alembic import op


revision: str = "d2e5f8a9b3c4"
down_revision: Union[str, Sequence[str], None] = "c9d4e7f8a1b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        "uq_faceImage_employeeId_imagePath",
        "faceImage",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_faceImage_employeeId",
        "faceImage",
        ["employeeId"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_faceImage_employeeId",
        "faceImage",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_faceImage_employeeId_imagePath",
        "faceImage",
        ["employeeId", "imagePath"],
    )
