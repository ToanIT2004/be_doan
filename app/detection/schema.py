from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)


class FaceDetection(BaseModel):
    box: BoundingBox
    confidence: float = Field(ge=0, le=1)
    classId: int = Field(ge=0)
    label: str = "face"


class ImageInfo(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class FaceDetectionResponse(BaseModel):
    frameId: str | None = None
    image: ImageInfo
    faces: list[FaceDetection]
    count: int = Field(ge=0)
    inferenceMs: float = Field(ge=0)


class DetectionHealthResponse(BaseModel):
    status: str
    modelPath: str
    modelExists: bool
    modelLoaded: bool
    device: str
