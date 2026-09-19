from datetime import date
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


LeaveType = Literal[
    "annual_leave",
    "sick_leave",
    "unpaid_leave",
    "personal_leave",
    "other",
]
LeaveStatus = Literal["pending", "approved", "rejected"]


class LeaveRequestCreate(BaseModel):
    leaveType: LeaveType = Field(
        validation_alias=AliasChoices("leaveType", "leavelType"),
    )
    startDate: date
    endDate: date
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_date_range(self):
        if self.endDate < self.startDate:
            raise ValueError("endDate must be on or after startDate")
        return self


class LeaveRequestUpdate(BaseModel):
    id: int = Field(gt=0)
    leaveType: LeaveType | None = Field(
        default=None,
        validation_alias=AliasChoices("leaveType", "leavelType"),
    )
    startDate: date | None = None
    endDate: date | None = None
    reason: str | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode="after")
    def require_update_field(self):
        fields = self.model_dump(exclude={"id"}, exclude_none=True)
        if not fields:
            raise ValueError("At least one field must be provided for update")
        if self.startDate is not None and self.endDate is not None and self.endDate < self.startDate:
            raise ValueError("endDate must be on or after startDate")
        return self


class LeaveRequestReview(BaseModel):
    id: int = Field(gt=0)
    status: Literal["approved", "rejected"]
    rejectReason: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_reject_reason(self):
        if self.status == "rejected":
            if self.rejectReason is None or not self.rejectReason.strip():
                raise ValueError("rejectReason is required when status is rejected")
            self.rejectReason = self.rejectReason.strip()
        else:
            self.rejectReason = None
        return self


class LeaveRequestRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    leaveType: LeaveType
    startDate: date
    endDate: date
    reason: str
    status: LeaveStatus
    rejectReason: str | None
    employeeId: int
