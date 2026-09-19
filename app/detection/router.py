import base64
import binascii
import json
import os
from typing import Annotated
from uuid import uuid4

from fastapi import (APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status)
from starlette.concurrency import run_in_threadpool

from app.detection.schema import DetectionHealthResponse, FaceDetectionResponse
from app.detection.service import (DetectorUnavailableError, FaceDetector, InvalidImageError, get_face_detector)

router = APIRouter(prefix="/api/v1/detection", tags=["face-detection"])
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
MAX_FRAME_BYTES = int(os.getenv("DETECTION_MAX_FRAME_BYTES", str(8 * 1024 * 1024)))
PREVIEW_IMAGE_SIZE = int(os.getenv("YOLO_PREVIEW_IMAGE_SIZE", "416"))

def _validate_frame(image_bytes: bytes) -> None:
    if not image_bytes:
        raise InvalidImageError("Image is empty")
    if len(image_bytes) > MAX_FRAME_BYTES:
        raise InvalidImageError(f"Image is too large; maximum is {MAX_FRAME_BYTES} bytes")


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, InvalidImageError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))


@router.get("/health", response_model=DetectionHealthResponse)
def detection_health(detector: Annotated[FaceDetector, Depends(get_face_detector)]) -> DetectionHealthResponse:
    exists = detector.model_path.is_file()
    return DetectionHealthResponse(
        status="ready" if exists else "model_missing",
        modelPath=str(detector.model_path),
        modelExists=exists,
        modelLoaded=detector.model_loaded,
        device=detector.device,
    )


@router.post("/faces", response_model=FaceDetectionResponse)
async def detect_faces(image: Annotated[UploadFile, File(description="JPEG, PNG, or WebP frame")], detector: Annotated[FaceDetector, Depends(get_face_detector)]) -> FaceDetectionResponse:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only JPEG, PNG, and WebP images are supported")

    image_bytes = await image.read(MAX_FRAME_BYTES + 1)
    try:
        _validate_frame(image_bytes)
        return await run_in_threadpool(detector.detect, image_bytes)
    except (InvalidImageError, DetectorUnavailableError) as exc:
        raise _http_error(exc) from exc
    finally:
        await image.close()


def _decode_text_frame(message: str) -> tuple[bytes, str]:
    try:
        payload = json.loads(message)
        frame_id = str(payload.get("frameId") or uuid4().hex)
        encoded = payload["image"]
        if not isinstance(encoded, str):
            raise ValueError
        if encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        return base64.b64decode(encoded, validate=True), frame_id
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, binascii.Error) as exc:
        raise InvalidImageError("Text frames must be JSON with frameId and a base64 image") from exc


@router.websocket("/ws/faces")
async def detect_faces_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    detector = get_face_detector()
    sequence = 0

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            try:
                if message.get("bytes") is not None:
                    image_bytes = message["bytes"]
                    frame_id = str(sequence)
                    sequence += 1
                elif message.get("text") is not None:
                    image_bytes, frame_id = _decode_text_frame(message["text"])
                else:
                    raise InvalidImageError("Frame must contain binary or text data")

                _validate_frame(image_bytes)
                result = await run_in_threadpool(
                    detector.detect,
                    image_bytes,
                    frame_id,
                    PREVIEW_IMAGE_SIZE,
                )
                await websocket.send_json(result.model_dump())
            except (InvalidImageError, DetectorUnavailableError) as exc:
                await websocket.send_json(
                    {
                        "error": {
                            "code": "invalid_image"
                            if isinstance(exc, InvalidImageError)
                            else "detector_unavailable",
                            "message": str(exc),
                        }
                    }
                )
    except WebSocketDisconnect:
        pass
