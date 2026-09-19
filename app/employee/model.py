from datetime import date, datetime

from sqlalchemy import Date, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.attendance.model import Attendance
from app.faceImage.model import FaceImage
from app.leaveRequest.model import LeaveRequest
from app.recognition.model import EmployeeFaceEmbedding


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    password: Mapped[str] = mapped_column(String(255))
    fullname: Mapped[str] = mapped_column(String(100), index=True)
    phone: Mapped[str] = mapped_column(String(30))
    dob: Mapped[date | None] = mapped_column(Date, nullable=True)
    avatarUrl: Mapped[str | None] = mapped_column(String(500), nullable=True)
    workDate: Mapped[date | None] = mapped_column(Date, nullable=True)
    departmentName: Mapped[str | None] = mapped_column(String(100), nullable=True)
    role: Mapped[str] = mapped_column(String(100), nullable=False, default="user")
    createdAt: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updatedAt: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    faceImages: Mapped[list["FaceImage"]] = relationship("FaceImage", back_populates="employee", cascade="all, delete-orphan")
    faceEmbeddings: Mapped[list["EmployeeFaceEmbedding"]] = relationship("EmployeeFaceEmbedding", back_populates="employee", cascade="all, delete-orphan")
    leaveRequests: Mapped[list["LeaveRequest"]] = relationship("LeaveRequest", back_populates="employee", cascade="all, delete-orphan")
    attendances: Mapped[list["Attendance"]] = relationship("Attendance", back_populates="employee", cascade="all, delete-orphan")
