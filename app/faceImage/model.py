from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class FaceImage(Base):
    __tablename__ = "faceImage"
    __table_args__ = (
        UniqueConstraint(
            "employeeId",
            name="uq_faceImage_employeeId",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    employeeId: Mapped[int] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"),
        index=True,
    )
    imagePath: Mapped[str] = mapped_column(String(500))
    createdAt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updatedAt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    employee = relationship("Employee", back_populates="faceImages")
