from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _normalize_upload_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("imagePath must not be empty")

    # Postman clients commonly pass the old upload response (`/face/file.jpg`).
    # Store one canonical public path so it also maps to app/uploads on disk.
    normalized = normalized.lstrip("/")
    if normalized.startswith("app/uploads/"):
        normalized = normalized.removeprefix("app/")
    if not normalized.startswith("uploads/"):
        normalized = f"uploads/{normalized}"
    return f"/{normalized}"


class FaceImageCreate(BaseModel):
    employeeId: int = Field(gt=0)
    imagePath: str = Field(min_length=1, max_length=500)

    @field_validator("imagePath")
    @classmethod
    def normalize_image_path(cls, value: str) -> str:
        return _normalize_upload_path(value)


class FaceImageUpdate(BaseModel):
    employeeId: int | None = Field(default=None, gt=0)
    imagePath: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def require_at_least_one_field(self):
        if not self.model_fields_set:
            raise ValueError("At least one field is required")
        if any(value is None for value in self.model_dump(exclude_unset=True).values()):
            raise ValueError("Updated fields must not be null")
        return self

    @field_validator("imagePath")
    @classmethod
    def normalize_image_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_upload_path(value)


class FaceImageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    employeeId: int
    imagePath: str
    createdAt: datetime
    updatedAt: datetime
