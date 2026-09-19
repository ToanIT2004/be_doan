from datetime import date

from sqlalchemy import Date, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Attendance(Base):
    __tablename__ = "attendances"
    __table_args__ = (
        UniqueConstraint(
            "employeeId",
            "attendanceDate",
            name="uq_attendances_employee_date",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    employeeId: Mapped[int] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attendanceDate: Mapped[date] = mapped_column(Date, nullable=False)
    checkIn: Mapped[str] = mapped_column(String(40), nullable=False)
    checkOut: Mapped[str | None] = mapped_column(String(40), nullable=True)

    employee = relationship("Employee", back_populates="attendances")
