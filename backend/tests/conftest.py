import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pipeline import Item  # noqa: E402,F401


class FakeClassifier:
    """Returns a fixed ranking; records how many crops it was asked to classify."""

    def __init__(self, ranking=None):
        self.ranking = ranking or [("plastic", 0.8), ("glass", 0.15), ("metal", 0.05)]
        self.calls: list[int] = []

    def model(self):
        return None

    def classes(self):
        return [m for m, _ in self.ranking]

    def classify(self, crops):
        self.calls.append(len(crops))
        assert all(c.size > 0 for c in crops)
        return [list(self.ranking) for _ in crops]


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def model(self):
        return None

    def detect(self, image, conf):
        return [b for b in self.boxes if b[1] >= conf]


@pytest.fixture
def image():
    return np.full((400, 600, 3), 200, dtype=np.uint8)


def real_models_available() -> bool:
    return (ROOT / "models" / "waste-cls.pt").is_file() and (ROOT / "models" / "yolov8n.pt").is_file()


os.environ.setdefault("CORS_ORIGINS", "http://localhost:5115")
