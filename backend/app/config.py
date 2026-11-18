from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CLASSIFIER_PATH = Path(os.getenv("CLASSIFIER_PATH", str(BASE_DIR / "models" / "waste-cls.pt")))
DETECTOR_PATH = Path(os.getenv("DETECTOR_PATH", str(BASE_DIR / "models" / "yolov8n.pt")))
METRICS_PATH = Path(os.getenv("METRICS_PATH", str(BASE_DIR / "models" / "metrics.json")))
DEVICE = os.getenv("DEVICE", "cpu")

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "8")) * 1024 * 1024
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2048"))
# Material predictions below this probability are flagged as uncertain in the response and UI.
UNCERTAIN_BELOW = float(os.getenv("UNCERTAIN_BELOW", "0.5"))

SAMPLE_DIR = BASE_DIR / "sample_media"
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:5115,http://127.0.0.1:5115").split(",")
    if o.strip()
]
