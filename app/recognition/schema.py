from datetime import datetime

from pydantic import BaseModel, Field

from app.attendance.schema import AttendanceRead
from app.antispoofing.schema import LivenessResult
from app.detection.schema import BoundingBox, ImageInfo


class IdentityMatch(BaseModel):
    employeeId: int | None = None
    username: str | None = None
    fullname: str | None = None
    similarity: float = Field(ge=-1, le=1)
    status: str


class RecognizedFace(BaseModel):
    box: BoundingBox
    detectionConfidence: float = Field(ge=0, le=1)
    liveness: LivenessResult
    identity: IdentityMatch
    attendance: AttendanceRead | None = None


class FaceRecognitionResponse(BaseModel):
    image: ImageInfo
    faces: list[RecognizedFace]
    count: int = Field(ge=0)
    inferenceMs: float = Field(ge=0)


class EnrollmentResponse(BaseModel):
    employeeId: int
    embeddingCount: int = Field(ge=0)
    modelName: str
    createdAt: datetime | None = None


class RecognitionHealthResponse(BaseModel):
    status: str
    modelName: str
    modelLoaded: bool
    threshold: float
