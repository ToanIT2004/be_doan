from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class EmployeeBase(BaseModel):
    username: str
    fullname: str
    phone: str
    dob: date | None = None
    avatarUrl: str | None = None
    workDate: date | None = None
    departmentName: str | None = None
    role: str = "user"

class EmployeeCreate(EmployeeBase):
    password: str


class EmployeeUpdate(BaseModel):
    password: str | None = None
    fullname: str | None = None
    phone: str | None = None
    dob: date | None = None
    avatarUrl: str | None = None
    workDate: date | None = None
    departmentName: str | None = None
    role: str | None = None


class EmployeeRead(EmployeeBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    createdAt: datetime
    updatedAt: datetime
