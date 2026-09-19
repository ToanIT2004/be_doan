import os
import threading
import time
from pathlib import Path
from typing import Any

from app.detection.schema import (BoundingBox, FaceDetection, FaceDetectionResponse, ImageInfo)


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a supported image."""


class DetectorUnavailableError(RuntimeError):
    """Raised when the detector or one of its dependencies is unavailable."""


class FaceDetector:
    def __init__(self) -> None:
        self.model_path = Path(os.getenv("YOLO_FACE_MODEL_PATH", "models/yolov8n-face.pt"))
        self.confidence = float(os.getenv("YOLO_FACE_CONFIDENCE", "0.5"))
        self.image_size = int(os.getenv("YOLO_FACE_IMAGE_SIZE", "640"))
        self.device = os.getenv("YOLO_DEVICE", "cpu")
        self.max_pixels = int(os.getenv("DETECTION_MAX_PIXELS", "16777216"))
        self._model: Any | None = None
        self._model_lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model
            if not self.model_path.is_file():
                raise DetectorUnavailableError(
                    f"YOLO face model not found at '{self.model_path}'. "
                    "Set YOLO_FACE_MODEL_PATH to a face-detection .pt file."
                )
            # Keep runtime settings inside the project for containers and
            # restricted service accounts. An explicit user setting wins.
            os.environ.setdefault(
                "YOLO_CONFIG_DIR", str(Path("Ultralytics").resolve())
            )
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise DetectorUnavailableError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

            try:
                self._model = YOLO(str(self.model_path))
            except Exception as exc:
                raise DetectorUnavailableError(
                    f"Could not load YOLO face model: {exc}"
                ) from exc
            return self._model

    def decode_image(self, image_bytes: bytes) -> Any:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise DetectorUnavailableError("OpenCV and NumPy are required. Run: pip install -r requirements.txt") from exc

        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError("The uploaded data is not a valid image")

        height, width = image.shape[:2]
        if width * height > self.max_pixels:
            raise InvalidImageError(
                f"Image resolution is too large; maximum is {self.max_pixels} pixels"
            )
        return image

    def detect(
        self,
        image_bytes: bytes,
        frame_id: str | None = None,
        image_size: int | None = None,
    ) -> FaceDetectionResponse:
        image = self.decode_image(image_bytes)
        model = self._load_model()
        height, width = image.shape[:2]
        inference_size = self.image_size if image_size is None else image_size

        if inference_size < 32:
            raise InvalidImageError("YOLO image size must be at least 32 pixels")

        started_at = time.perf_counter()
        try:
            # Ultralytics/PyTorch model instances are not assumed to be thread-safe.
            with self._model_lock:
                results = model.predict(
                    source=image,
                    conf=self.confidence,
                    imgsz=inference_size,
                    device=self.device,
                    verbose=False,
                )
        except Exception as exc:
            raise DetectorUnavailableError(f"YOLO inference failed: {exc}") from exc
        inference_ms = round((time.perf_counter() - started_at) * 1000, 2)

        faces: list[FaceDetection] = []
        if results:
            result = results[0]
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            rows = boxes.data.cpu().tolist() if boxes is not None else []
            for x1, y1, x2, y2, confidence, class_id, *_ in rows:
                left = max(0, min(width, round(x1)))
                top = max(0, min(height, round(y1)))
                right = max(left, min(width, round(x2)))
                bottom = max(top, min(height, round(y2)))
                class_number = int(class_id)
                if isinstance(names, dict):
                    label = str(names.get(class_number, "face"))
                elif class_number < len(names):
                    label = str(names[class_number])
                else:
                    label = "face"
                faces.append(
                    FaceDetection(
                        box=BoundingBox(
                            x1=left,
                            y1=top,
                            x2=right,
                            y2=bottom,
                            width=right - left,
                            height=bottom - top,
                        ),
                        confidence=round(float(confidence), 6),
                        classId=class_number,
                        label=label,
                    )
                )

        return FaceDetectionResponse(
            frameId=frame_id,
            image=ImageInfo(width=width, height=height),
            faces=faces,
            count=len(faces),
            inferenceMs=inference_ms,
        )


face_detector = FaceDetector()


def get_face_detector() -> FaceDetector:
    return face_detector
