from datetime import date

from sqlalchemy import CheckConstraint, Date, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class LeaveRequest(Base):
    __tablename__ = "leaveRequests"
    __table_args__ = (
        CheckConstraint(
            '"leaveType" IN (\'annual_leave\', \'sick_leave\', \'unpaid_leave\', \'personal_leave\', \'other\')',
            name="ck_leaveRequests_leaveType",
        ),
        CheckConstraint(
            'status IN (\'pending\', \'approved\', \'rejected\')',
            name="ck_leaveRequests_status",
        ),
        CheckConstraint(
            '"endDate" >= "startDate"',
            name="ck_leaveRequests_date_range",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    leaveType: Mapped[str] = mapped_column(String(30), nullable=False)
    startDate: Mapped[date] = mapped_column(Date, nullable=False)
    endDate: Mapped[date] = mapped_column(Date, nullable=False)
    reason: Mapped[str] = mapped_column(String(1000), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    rejectReason: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    employeeId: Mapped[int] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    employee = relationship("Employee", back_populates="leaveRequests")
