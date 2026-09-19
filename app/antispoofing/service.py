import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from app.antispoofing.schema import LivenessResult


class AntiSpoofUnavailableError(RuntimeError):
    """Raised when the anti-spoofing model or runtime is unavailable."""


class AntiSpoofError(ValueError):
    """Raised when frames cannot be evaluated for liveness."""


class MobileNetV3AntiSpoof:
    """ONNX MobileNetV3 PAD inference with quality-weighted frame fusion.

    The exported training model uses class order ``spoof=0, live=1`` and
    returns two logits. A one-output sigmoid model is also accepted.
    """

    def __init__(self) -> None:
        self.model_path = Path(
            os.getenv(
                "ANTI_SPOOF_MODEL_PATH",
                "models/antispoofing/mobilenetv3_pad.onnx",
            )
        )
        self.model_name = os.getenv("ANTI_SPOOF_MODEL_NAME", "mobilenetv3-small-pad")
        self.input_size = int(os.getenv("ANTI_SPOOF_INPUT_SIZE", "160"))
        self.live_threshold = float(os.getenv("ANTI_SPOOF_LIVE_THRESHOLD", "0.75"))
        self.spoof_threshold = float(os.getenv("ANTI_SPOOF_SPOOF_THRESHOLD", "0.30"))
        self.max_uncertainty = float(os.getenv("ANTI_SPOOF_MAX_UNCERTAINTY", "0.18"))
        self.providers = [
            item.strip()
            for item in os.getenv(
                "ANTI_SPOOF_PROVIDERS", "CPUExecutionProvider"
            ).split(",")
            if item.strip()
        ]
        if not 0 <= self.spoof_threshold < self.live_threshold <= 1:
            raise ValueError(
                "Anti-spoof thresholds must satisfy "
                "0 <= spoof threshold < live threshold <= 1"
            )
        if not 0 <= self.max_uncertainty <= 1:
            raise ValueError("ANTI_SPOOF_MAX_UNCERTAINTY must be between 0 and 1")

        self._session: Any | None = None
        self._input_name: str | None = None
        self._lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self._session is not None

    def _load_session(self) -> Any:
        if self._session is not None:
            return self._session

        with self._lock:
            if self._session is not None:
                return self._session
            if not self.model_path.is_file():
                raise AntiSpoofUnavailableError(
                    f"Anti-spoofing model not found at '{self.model_path}'. "
                    "Train/export it with training/train_antispoof.py or set "
                    "ANTI_SPOOF_MODEL_PATH."
                )
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise AntiSpoofUnavailableError(
                    "ONNX Runtime is not installed. Run: pip install -r requirements.txt"
                ) from exc

            try:
                session = ort.InferenceSession(
                    str(self.model_path),
                    providers=self.providers,
                )
                self._input_name = session.get_inputs()[0].name
                self._session = session
            except Exception as exc:
                raise AntiSpoofUnavailableError(
                    f"Could not load anti-spoofing model: {exc}"
                ) from exc
            return self._session

    def _preprocess(self, face_crops: Sequence[Any]) -> Any:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise AntiSpoofUnavailableError(
                "OpenCV and NumPy are required for anti-spoofing"
            ) from exc

        tensors = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for crop in face_crops:
            if crop is None or not hasattr(crop, "size") or crop.size == 0:
                raise AntiSpoofError("Face crop is empty")
            resized = cv2.resize(
                crop,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_AREA,
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
            tensors.append(np.transpose(normalized, (2, 0, 1)))
        if not tensors:
            raise AntiSpoofError("At least one face crop is required")
        return np.stack(tensors).astype(np.float32)

    @staticmethod
    def _live_probabilities(outputs: Any) -> Any:
        import numpy as np

        logits = np.asarray(outputs, dtype=np.float32)
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        if logits.ndim != 2:
            raise AntiSpoofUnavailableError(
                f"Unexpected anti-spoof model output shape: {logits.shape}"
            )
        if logits.shape[1] == 1:
            return 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -30, 30)))
        if logits.shape[1] != 2:
            raise AntiSpoofUnavailableError(
                "Anti-spoof model must output one live logit or two logits "
                "in class order [spoof, live]"
            )
        stable = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(stable)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities[:, 1]

    @staticmethod
    def frame_quality(face_crop: Any, detection_confidence: float = 1.0) -> float:
        """Return a cheap [0, 1] quality score for temporal weighting."""
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise AntiSpoofUnavailableError(
                "OpenCV and NumPy are required for quality scoring"
            ) from exc

        if face_crop is None or face_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 250.0, 1.0)
        brightness = float(gray.mean())
        brightness_score = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
        contrast_score = min(float(gray.std()) / 64.0, 1.0)
        detector_score = float(np.clip(detection_confidence, 0.0, 1.0))
        score = (
            0.40 * sharpness
            + 0.25 * brightness_score
            + 0.15 * contrast_score
            + 0.20 * detector_score
        )
        return float(np.clip(score, 0.05, 1.0))

    def predict_sequence(
        self,
        face_crops: Sequence[Any],
        detection_confidences: Sequence[float] | None = None,
    ) -> LivenessResult:
        try:
            import numpy as np
        except ImportError as exc:
            raise AntiSpoofUnavailableError("NumPy is required") from exc

        if not face_crops:
            raise AntiSpoofError("At least one face crop is required")
        confidences = list(detection_confidences or [1.0] * len(face_crops))
        if len(confidences) != len(face_crops):
            raise AntiSpoofError(
                "Detection confidences must have the same length as face crops"
            )

        batch = self._preprocess(face_crops)
        session = self._load_session()
        started_at = time.perf_counter()
        try:
            with self._lock:
                raw_outputs = session.run(None, {self._input_name: batch})[0]
        except Exception as exc:
            raise AntiSpoofUnavailableError(
                f"Anti-spoofing inference failed: {exc}"
            ) from exc
        inference_ms = round((time.perf_counter() - started_at) * 1000, 2)

        live_scores = self._live_probabilities(raw_outputs)
        if len(live_scores) != len(face_crops):
            raise AntiSpoofUnavailableError(
                "Anti-spoofing output batch size does not match input batch size"
            )
        qualities = np.array(
            [
                self.frame_quality(crop, confidence)
                for crop, confidence in zip(face_crops, confidences)
            ],
            dtype=np.float32,
        )
        weights = qualities / qualities.sum()
        live_score = float(np.sum(weights * live_scores))
        uncertainty = float(
            np.sqrt(np.sum(weights * np.square(live_scores - live_score)))
        )

        if live_score <= self.spoof_threshold:
            decision = "spoof"
        elif live_score >= self.live_threshold and uncertainty <= self.max_uncertainty:
            decision = "live"
        else:
            decision = "uncertain"

        best_index = int(np.argmax(qualities * (0.5 + 0.5 * live_scores)))
        return LivenessResult(
            status=decision,
            liveScore=round(live_score, 6),
            spoofScore=round(1.0 - live_score, 6),
            uncertainty=round(min(uncertainty, 1.0), 6),
            frameCount=len(face_crops),
            acceptedFrames=len(face_crops),
            bestFrameIndex=best_index,
            inferenceMs=inference_ms,
            modelName=self.model_name,
        )


anti_spoof_recognizer = MobileNetV3AntiSpoof()


def get_anti_spoof_recognizer() -> MobileNetV3AntiSpoof:
    return anti_spoof_recognizer
