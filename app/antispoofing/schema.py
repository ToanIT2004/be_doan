from typing import Literal

from pydantic import BaseModel, Field


LivenessStatus = Literal["live", "spoof", "uncertain"]


class LivenessResult(BaseModel):
    status: LivenessStatus
    liveScore: float = Field(ge=0, le=1)
    spoofScore: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    frameCount: int = Field(ge=1)
    acceptedFrames: int = Field(ge=1)
    bestFrameIndex: int = Field(ge=0)
    inferenceMs: float = Field(ge=0)
    modelName: str


class AntiSpoofHealthResponse(BaseModel):
    status: str
    modelPath: str
    modelExists: bool
    modelLoaded: bool
    modelName: str
    liveThreshold: float = Field(ge=0, le=1)
    spoofThreshold: float = Field(ge=0, le=1)
    maxUncertainty: float = Field(ge=0, le=1)
    inputSize: int = Field(gt=0)
