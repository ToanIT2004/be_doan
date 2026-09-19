from datetime import date

from pydantic import BaseModel, ConfigDict


class AttendanceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    employeeId: int
    attendanceDate: date
    checkIn: str
    checkOut: str | None
