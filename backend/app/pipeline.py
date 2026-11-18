"""Two-stage waste pipeline: a COCO detector finds candidate items, a TrashNet classifier names their material.

Shared by the FastAPI app and the Tkinter desktop client.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

MATERIALS = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "organic"]

STREAMS = {
    "cardboard": "Paper & cardboard recycling",
    "paper": "Paper & cardboard recycling",
    "glass": "Glass recycling",
    "metal": "Metal recycling",
    "plastic": "Plastic recycling",
    "trash": "General (non-recyclable) waste",
    "organic": "Compost / wet waste",
}

# BGR colours per material for annotated images; the frontend uses matching hex values.
COLORS = {
    "cardboard": (64, 140, 196),
    "paper": (230, 216, 160),
    "glass": (196, 220, 52),
    "metal": (180, 180, 180),
    "plastic": (240, 160, 60),
    "trash": (110, 110, 240),
    "organic": (90, 200, 90),
}

# COCO classes that can plausibly be discarded items. People, animals, vehicles and furniture are ignored.
ITEM_CLASSES = {
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "book", "vase", "scissors", "toothbrush",
    "cell phone", "remote", "keyboard", "mouse", "laptop", "handbag", "backpack", "umbrella", "sports ball",
    "frisbee", "teddy bear", "hair drier", "clock", "tie", "suitcase", "kite", "baseball bat", "tennis racket",
}
# Food is labelled organic directly from the detector: TrashNet has no organic class to learn it from.
FOOD_CLASSES = {"banana", "apple", "orange", "broccoli", "carrot", "sandwich", "hot dog", "pizza", "donut", "cake"}

MIN_BOX_FRACTION = 0.01  # ignore boxes smaller than 1% of the image area


class ModelUnavailable(RuntimeError):
    pass


class Classifier(Protocol):
    def classify(self, crops: list[np.ndarray]) -> list[list[tuple[str, float]]]: ...


class Detector(Protocol):
    def detect(self, image: np.ndarray, conf: float) -> list[tuple[str, float, list[float]]]: ...


@dataclass
class Item:
    material: str
    confidence: float
    stream: str
    uncertain: bool
    source: str  # "classifier" or "detector"
    box: list[float]
    object: str | None = None
    object_confidence: float | None = None
    top3: list[dict[str, float | str]] = field(default_factory=list)


@dataclass
class Analysis:
    items: list[Item]
    counts: dict[str, int]
    whole_image: bool
    inference_ms: float
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total"] = len(self.items)
        return d


def pad_box(box: list[float], width: int, height: int, pad: float = 0.1) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return (max(0, int(x1 - dx)), max(0, int(y1 - dy)), min(width, int(x2 + dx)), min(height, int(y2 + dy)))


def analyze(image: np.ndarray, classifier: Classifier, detector: Detector | None, conf: float = 0.25,
            uncertain_below: float = 0.5) -> Analysis:
    h, w = image.shape[:2]
    start = time.perf_counter()
    candidates = []
    if detector is not None:
        for label, score, box in detector.detect(image, conf):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if (label in ITEM_CLASSES or label in FOOD_CLASSES) and area >= MIN_BOX_FRACTION * w * h:
                candidates.append((label, score, box))

    items: list[Item] = []
    to_classify = [c for c in candidates if c[0] not in FOOD_CLASSES]
    whole = not candidates
    if whole:
        to_classify = [(None, None, [0.0, 0.0, float(w), float(h)])]

    crops = []
    for _, _, box in to_classify:
        x1, y1, x2, y2 = pad_box(box, w, h)
        crops.append(image[y1:y2, x1:x2])
    for (label, score, box), probs in zip(to_classify, classifier.classify(crops) if crops else []):
        material, p = probs[0]
        items.append(Item(material=material, confidence=round(p, 4), stream=STREAMS[material],
                          uncertain=p < uncertain_below, source="classifier", box=[round(v, 1) for v in box],
                          object=label, object_confidence=None if score is None else round(score, 4),
                          top3=[{"material": m, "probability": round(q, 4)} for m, q in probs[:3]]))
    for label, score, box in candidates:
        if label in FOOD_CLASSES:
            items.append(Item(material="organic", confidence=round(score, 4), stream=STREAMS["organic"],
                              uncertain=score < uncertain_below, source="detector", box=[round(v, 1) for v in box],
                              object=label, object_confidence=round(score, 4)))

    items.sort(key=lambda i: i.confidence, reverse=True)
    counts = dict(Counter(i.material for i in items).most_common())
    return Analysis(items, counts, whole, round((time.perf_counter() - start) * 1000, 1), w, h)


def annotate(image: np.ndarray, analysis: Analysis) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    thickness = max(2, round(min(h, w) / 300))
    scale = max(0.5, min(h, w) / 900)
    for it in analysis.items:
        color = COLORS[it.material]
        x1, y1, x2, y2 = (int(v) for v in it.box)
        if not analysis.whole_image:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = f"{it.material} {it.confidence:.0%}" + ("?" if it.uncertain else "")
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        tx, ty = max(0, min(x1, w - tw - 6)), max(y1, th + base + 4)
        cv2.rectangle(out, (tx, ty - th - base - 4), (tx + tw + 6, ty), color, -1)
        cv2.putText(out, text, (tx + 3, ty - base - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (17, 21, 27), 1, cv2.LINE_AA)
    return out


def decode_image(data: bytes, max_side: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The file is not a valid image (supported: JPEG, PNG, BMP, WebP).")
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        f = max_side / max(h, w)
        image = cv2.resize(image, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return image


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not encode image")
    return buf.tobytes()


class _Lazy:
    """Loads ultralytics weights on first use; predictors are not thread-safe, so calls are serialised."""

    def __init__(self, path: Path, device: str, hint: str) -> None:
        self.path, self.device, self.hint = path, device, hint
        self._model = None
        self.error: str | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if not self.path.is_file():
                        self.error = f"Weights not found at '{self.path}'. {self.hint}"
                        raise ModelUnavailable(self.error)
                    try:
                        from ultralytics import YOLO

                        self._model = YOLO(str(self.path))
                        self.error = None
                    except Exception as exc:  # corrupted or incompatible weights
                        self.error = f"Could not load '{self.path.name}': {exc}"
                        raise ModelUnavailable(self.error) from exc
        return self._model


class YoloClassifier(_Lazy):
    def __init__(self, path: Path, device: str = "cpu") -> None:
        super().__init__(path, device, "Train it with 'python scripts/train_classifier.py --data <TrashNet folder>'.")

    def classes(self) -> list[str]:
        names = self.model().names
        return [names[i] for i in sorted(names)]

    def classify(self, crops: list[np.ndarray]) -> list[list[tuple[str, float]]]:
        model = self.model()
        with self._lock:
            results = model.predict(crops, device=self.device, verbose=False)
        out = []
        for r in results:
            probs = r.probs.data.cpu().numpy()
            order = np.argsort(probs)[::-1]
            out.append([(str(model.names[int(i)]), float(probs[i])) for i in order])
        return out


class YoloDetector(_Lazy):
    def __init__(self, path: Path, device: str = "cpu") -> None:
        super().__init__(path, device, "Download it with 'python scripts/download_detector.py'.")

    def detect(self, image: np.ndarray, conf: float) -> list[tuple[str, float, list[float]]]:
        model = self.model()
        with self._lock:
            r = model.predict(image, conf=conf, iou=0.45, device=self.device, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            return []
        return [(str(model.names[int(c)]), float(s), [float(v) for v in b])
                for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())]
