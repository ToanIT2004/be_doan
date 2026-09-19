import os
import unittest
from unittest.mock import patch

import numpy as np

from app.antispoofing.service import MobileNetV3AntiSpoof


class _FakeSession:
    def __init__(self, logits) -> None:
        self.logits = np.asarray(logits, dtype=np.float32)

    def run(self, *_args, **_kwargs):
        return [self.logits]


class AntiSpoofDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = patch.dict(
            os.environ,
            {
                "ANTI_SPOOF_LIVE_THRESHOLD": "0.75",
                "ANTI_SPOOF_SPOOF_THRESHOLD": "0.30",
                "ANTI_SPOOF_MAX_UNCERTAINTY": "0.18",
            },
        )
        self.env.start()
        self.crop = np.full((160, 160, 3), 127, dtype=np.uint8)

    def tearDown(self) -> None:
        self.env.stop()

    def predict(self, logits):
        recognizer = MobileNetV3AntiSpoof()
        recognizer._input_name = "images"
        recognizer._session = _FakeSession(logits)
        crops = [self.crop.copy() for _ in logits]
        return recognizer.predict_sequence(crops, [0.9] * len(crops))

    def test_live(self) -> None:
        result = self.predict([[-4.0, 4.0], [-3.0, 3.0]])
        self.assertEqual(result.status, "live")

    def test_spoof(self) -> None:
        result = self.predict([[4.0, -4.0], [3.0, -3.0]])
        self.assertEqual(result.status, "spoof")

    def test_uncertain_for_disagreeing_frames(self) -> None:
        result = self.predict([[-4.0, 4.0], [4.0, -4.0]])
        self.assertEqual(result.status, "uncertain")
        self.assertGreater(result.uncertainty, 0.18)


if __name__ == "__main__":
    unittest.main()
