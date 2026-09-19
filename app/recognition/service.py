import os
import threading
import time
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.employee.model import Employee
from app.file.config import FILE_STORAGE_ROOT
from app.recognition.model import EmployeeFaceEmbedding
from app.recognition.schema import IdentityMatch


class RecognitionUnavailableError(RuntimeError):
    """Raised when ArcFace or one of its runtime dependencies is unavailable."""


class RecognitionError(ValueError):
    """Raised when a face cannot be converted into a usable embedding."""


class ArcFaceRecognizer:
    def __init__(self) -> None:
        self.model_name = os.getenv("ARCFACE_MODEL_NAME", "buffalo_l")
        self.threshold = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.55"))
        self.det_size = int(os.getenv("ARCFACE_DET_SIZE", "640"))
        self.model_root = Path(os.getenv("ARCFACE_MODEL_ROOT", "models/insightface"))
        self.providers = [
            provider.strip()
            for provider in os.getenv("ARCFACE_PROVIDERS", "CPUExecutionProvider").split(",")
            if provider.strip()
        ]
        self.ctx_id = int(os.getenv("ARCFACE_CTX_ID", "-1"))
        self._app: Any | None = None
        self._lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self._app is not None

    def _load_app(self) -> Any:
        if self._app is not None:
            return self._app

        with self._lock:
            if self._app is not None:
                return self._app
            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise RecognitionUnavailableError(
                    "InsightFace is not installed. Run: pip install -r requirements.txt"
                ) from exc

            try:
                self.model_root.mkdir(parents=True, exist_ok=True)
                app = FaceAnalysis(name=self.model_name, root=str(self.model_root), providers=self.providers)
                app.prepare(ctx_id=self.ctx_id, det_size=(self.det_size, self.det_size))
            except Exception as exc:
                raise RecognitionUnavailableError(f"Could not load ArcFace model: {exc}") from exc

            self._app = app
            return self._app

    def read_image_file(self, image_path: str) -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise RecognitionUnavailableError("OpenCV is required. Run: pip install -r requirements.txt") from exc

        relative_path = image_path.strip().replace("\\", "/").lstrip("/")
        if relative_path.startswith("app/uploads/"):
            relative_path = relative_path.removeprefix("app/uploads/")
        elif relative_path.startswith("uploads/"):
            relative_path = relative_path.removeprefix("uploads/")

        storage_root = FILE_STORAGE_ROOT.resolve()
        path = (storage_root / relative_path).resolve()
        if not path.is_relative_to(storage_root):
            raise RecognitionError(f"Invalid face image path: {image_path}")

        image = cv2.imread(str(path))
        if image is None:
            raise RecognitionError(f"Could not read face image: {image_path}")
        return image

    def embedding_from_image(self, image: Any, require_single_face: bool = True) -> list[float]:
        app = self._load_app()
        faces = app.get(image)
        if not faces:
            raise RecognitionError("No face found for ArcFace enrollment")
        if require_single_face and len(faces) != 1:
            raise RecognitionError("Enrollment image must contain exactly one face")

        face = max(
            faces,
            key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
        )
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            raise RecognitionError("ArcFace did not return an embedding")

        return [float(value) for value in embedding.tolist()]

    def enroll_employee(self, db: Session, employee_id: int, image_paths: list[str]) -> list[EmployeeFaceEmbedding]:
        if not image_paths:
            raise RecognitionError("At least one face image is required for enrollment")

        db.query(EmployeeFaceEmbedding).filter(
            EmployeeFaceEmbedding.employeeId == employee_id
        ).delete(synchronize_session=False)

        db_embeddings: list[EmployeeFaceEmbedding] = []
        for image_path in image_paths:
            image = self.read_image_file(image_path)
            embedding = self.embedding_from_image(image, require_single_face=True)
            db_embedding = EmployeeFaceEmbedding(
                employeeId=employee_id,
                imagePath=image_path,
                embedding=embedding,
                modelName=self.model_name,
            )
            db_embeddings.append(db_embedding)

        db.add_all(db_embeddings)
        db.commit()
        for db_embedding in db_embeddings:
            db.refresh(db_embedding)
        return db_embeddings

    def match_embedding(self, db: Session, embedding: list[float]) -> IdentityMatch:
        try:
            import numpy as np
        except ImportError as exc:
            raise RecognitionUnavailableError("NumPy is required. Run: pip install -r requirements.txt") from exc

        rows = (
            db.query(EmployeeFaceEmbedding, Employee)
            .join(Employee, Employee.id == EmployeeFaceEmbedding.employeeId)
            .all()
        )
        if not rows:
            return IdentityMatch(similarity=0.0, status="unknown")

        query = np.array(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            raise RecognitionError("Face embedding is empty")

        best_employee: Employee | None = None
        best_similarity = -1.0
        for db_embedding, employee in rows:
            candidate = np.array(db_embedding.embedding, dtype=np.float32)
            candidate_norm = np.linalg.norm(candidate)
            if candidate_norm == 0:
                continue
            similarity = float(np.dot(query, candidate) / (query_norm * candidate_norm))
            if similarity > best_similarity:
                best_similarity = similarity
                best_employee = employee

        if best_employee is None or best_similarity < self.threshold:
            return IdentityMatch(
                similarity=round(max(best_similarity, 0.0), 6),
                status="unknown",
            )

        return IdentityMatch(
            employeeId=best_employee.id,
            username=best_employee.username,
            fullname=best_employee.fullname,
            similarity=round(best_similarity, 6),
            status="recognized",
        )


arcface_recognizer = ArcFaceRecognizer()


def get_arcface_recognizer() -> ArcFaceRecognizer:
    return arcface_recognizer


def now_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
