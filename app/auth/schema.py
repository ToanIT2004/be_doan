from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class LoginRequest(BaseModel):
    username: str
    password: str


class EmployeeAuthRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    fullname: str
    phone: str
    dob: date | None = None
    avatarUrl: str | None = None
    workDate: date | None = None
    departmentName: str | None = None
    role: str
    createdAt: datetime


class TokenResponse(BaseModel):
    accessToken: str
    tokenType: str = "bearer"
    employee: EmployeeAuthRead


class TokenPayload(BaseModel):
    sub: str
    employeeId: int
    username: str
    role: str
    exp: int


class CurrentEmployee(EmployeeAuthRead):
    model_config = ConfigDict(from_attributes=True)
