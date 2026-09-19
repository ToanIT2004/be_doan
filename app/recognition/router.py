import os
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.antispoofing.schema import AntiSpoofHealthResponse, LivenessResult
from app.antispoofing.service import (
    AntiSpoofError,
    AntiSpoofUnavailableError,
    MobileNetV3AntiSpoof,
    get_anti_spoof_recognizer,
)
from app.attendance.crud import mark_attendance
from app.auth.security import get_current_employee, require_admin
from app.database import get_db
from app.detection.router import ALLOWED_CONTENT_TYPES, MAX_FRAME_BYTES, _validate_frame
from app.detection.schema import ImageInfo
from app.detection.service import (
    DetectorUnavailableError,
    FaceDetector,
    InvalidImageError,
    get_face_detector,
)
from app.employee.crud import get_employee
from app.recognition import schema
from app.recognition.service import (
    ArcFaceRecognizer,
    RecognitionError,
    RecognitionUnavailableError,
    get_arcface_recognizer,
    now_ms,
)


router = APIRouter(prefix="/api/v1/recognition", tags=["face-recognition"])
MAX_ANTI_SPOOF_FRAMES = int(os.getenv("ANTI_SPOOF_MAX_FRAMES", "16"))
MIN_ANTI_SPOOF_FRAMES = int(os.getenv("ANTI_SPOOF_MIN_FRAMES", "3"))


def _crop_face_with_padding(source, box, padding_ratio: float = 0.30):
    """Keep some surrounding texture without including too much background."""
    height, width = source.shape[:2]
    pad_x = round(box.width * padding_ratio)
    pad_y = round(box.height * padding_ratio)
    x1 = max(0, box.x1 - pad_x)
    y1 = max(0, box.y1 - pad_y)
    x2 = min(width, box.x2 + pad_x)
    y2 = min(height, box.y2 + pad_y)
    return source[y1:y2, x1:x2]


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (InvalidImageError, RecognitionError, AntiSpoofError)):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(exc),
    )


async def _identity_and_attendance(
    crop,
    db: Session,
    recognizer: ArcFaceRecognizer,
) -> tuple[schema.IdentityMatch, object | None]:
    embedding = await run_in_threadpool(
        recognizer.embedding_from_image,
        crop,
        False,
    )
    identity = await run_in_threadpool(recognizer.match_embedding, db, embedding)
    attendance = None
    if identity.status == "recognized" and identity.employeeId is not None:
        attendance = await run_in_threadpool(
            mark_attendance,
            db,
            identity.employeeId,
        )
    return identity, attendance


def _not_evaluated_identity() -> schema.IdentityMatch:
    return schema.IdentityMatch(similarity=0.0, status="not_evaluated")


@router.get("/health", response_model=schema.RecognitionHealthResponse)
def recognition_health(
    recognizer: Annotated[ArcFaceRecognizer, Depends(get_arcface_recognizer)],
) -> schema.RecognitionHealthResponse:
    return schema.RecognitionHealthResponse(
        status="ready" if recognizer.model_loaded else "not_loaded",
        modelName=recognizer.model_name,
        modelLoaded=recognizer.model_loaded,
        threshold=recognizer.threshold,
    )


@router.get("/anti-spoof/health", response_model=AntiSpoofHealthResponse)
def anti_spoof_health(
    anti_spoof: Annotated[
        MobileNetV3AntiSpoof,
        Depends(get_anti_spoof_recognizer),
    ],
) -> AntiSpoofHealthResponse:
    exists = anti_spoof.model_path.is_file()
    return AntiSpoofHealthResponse(
        status="ready" if exists else "model_missing",
        modelPath=str(anti_spoof.model_path),
        modelExists=exists,
        modelLoaded=anti_spoof.model_loaded,
        modelName=anti_spoof.model_name,
        liveThreshold=anti_spoof.live_threshold,
        spoofThreshold=anti_spoof.spoof_threshold,
        maxUncertainty=anti_spoof.max_uncertainty,
        inputSize=anti_spoof.input_size,
    )


@router.post(
    "/admin/employee/{employee_id}/enroll",
    response_model=schema.EnrollmentResponse,
    dependencies=[Depends(require_admin)],
)
async def enroll_employee_faces(
    employee_id: int,
    db: Session = Depends(get_db),
    recognizer: ArcFaceRecognizer = Depends(get_arcface_recognizer),
) -> schema.EnrollmentResponse:
    employee = get_employee(db, employee_id)
    if employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    image_paths = [face_image.imagePath for face_image in employee.faceImages]
    try:
        embeddings = await run_in_threadpool(
            recognizer.enroll_employee,
            db,
            employee_id,
            image_paths,
        )
    except (RecognitionError, RecognitionUnavailableError) as exc:
        raise _http_error(exc) from exc

    return schema.EnrollmentResponse(
        employeeId=employee_id,
        embeddingCount=len(embeddings),
        modelName=recognizer.model_name,
        createdAt=embeddings[0].createdAt if embeddings else None,
    )


@router.post(
    "/attendance",
    response_model=schema.FaceRecognitionResponse,
    dependencies=[Depends(get_current_employee)],
)
async def recognize_attendance_sequence(
    frames: Annotated[
        list[UploadFile],
        File(description="3-16 JPEG, PNG, or WebP frames from one short capture"),
    ],
    db: Session = Depends(get_db),
    detector: FaceDetector = Depends(get_face_detector),
    anti_spoof: MobileNetV3AntiSpoof = Depends(get_anti_spoof_recognizer),
    recognizer: ArcFaceRecognizer = Depends(get_arcface_recognizer),
) -> schema.FaceRecognitionResponse:
    """Recommended quality-weighted multi-frame attendance endpoint."""
    if not MIN_ANTI_SPOOF_FRAMES <= len(frames) <= MAX_ANTI_SPOOF_FRAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Submit between {MIN_ANTI_SPOOF_FRAMES} and "
                f"{MAX_ANTI_SPOOF_FRAMES} frames"
            ),
        )

    started_at = time.perf_counter()
    valid_items = []
    try:
        for upload in frames:
            if upload.content_type not in ALLOWED_CONTENT_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Only JPEG, PNG, and WebP images are supported",
                )
            image_bytes = await upload.read(MAX_FRAME_BYTES + 1)
            _validate_frame(image_bytes)
            detection = await run_in_threadpool(detector.detect, image_bytes)
            if len(detection.faces) != 1:
                continue
            source = detector.decode_image(image_bytes)
            detected_face = detection.faces[0]
            crop = _crop_face_with_padding(source, detected_face.box)
            valid_items.append((detection, detected_face, crop))

        if len(valid_items) < MIN_ANTI_SPOOF_FRAMES:
            raise AntiSpoofError(
                f"At least {MIN_ANTI_SPOOF_FRAMES} frames must contain exactly one face"
            )

        crops = [item[2] for item in valid_items]
        confidences = [item[1].confidence for item in valid_items]
        liveness: LivenessResult = await run_in_threadpool(
            anti_spoof.predict_sequence,
            crops,
            confidences,
        )
        best_detection, best_face, best_crop = valid_items[liveness.bestFrameIndex]

        identity = _not_evaluated_identity()
        attendance = None
        if liveness.status == "live":
            identity, attendance = await _identity_and_attendance(
                best_crop,
                db,
                recognizer,
            )

        face = schema.RecognizedFace(
            box=best_face.box,
            detectionConfidence=best_face.confidence,
            liveness=liveness,
            identity=identity,
            attendance=attendance,
        )
        return schema.FaceRecognitionResponse(
            image=ImageInfo(
                width=best_detection.image.width,
                height=best_detection.image.height,
            ),
            faces=[face],
            count=1,
            inferenceMs=now_ms(started_at),
        )
    except (
        InvalidImageError,
        DetectorUnavailableError,
        AntiSpoofError,
        AntiSpoofUnavailableError,
        RecognitionError,
        RecognitionUnavailableError,
    ) as exc:
        raise _http_error(exc) from exc
    finally:
        for upload in frames:
            await upload.close()
