from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class EmployeeFaceEmbedding(Base):
    __tablename__ = "employeeFaceEmbeddings"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    employeeId: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"), index=True)
    imagePath: Mapped[str | None] = mapped_column(String(500), nullable=True)
    embedding: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    modelName: Mapped[str] = mapped_column(String(100), nullable=False)
    createdAt: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updatedAt: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    employee = relationship("Employee", back_populates="faceEmbeddings")
