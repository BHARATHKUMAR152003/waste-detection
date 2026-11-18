from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import config
from .pipeline import STREAMS, ModelUnavailable, YoloClassifier, YoloDetector, analyze, annotate, decode_image, encode_jpeg

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = FastAPI(title="Waste Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
classifier = YoloClassifier(config.CLASSIFIER_PATH, config.DEVICE)
detector = YoloDetector(config.DETECTOR_PATH, config.DEVICE)


@app.on_event("startup")
def warm_up() -> None:
    """Loads both models in the background so the first request doesn't pay the PyTorch import cost."""

    def _load() -> None:
        for m in (classifier, detector):
            try:
                m.model()
            except ModelUnavailable:
                pass  # reported via /api/health and 503 responses

    threading.Thread(target=_load, daemon=True).start()


@app.exception_handler(ModelUnavailable)
async def model_unavailable(_: Request, exc: ModelUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc), "code": "model_unavailable"})


def _conf(value: float) -> float:
    if not 0.05 <= value <= 0.95:
        raise HTTPException(422, "Confidence must be between 0.05 and 0.95.")
    return value


async def _read_limited(file: UploadFile) -> bytes:
    data = await file.read(config.MAX_IMAGE_BYTES + 1)
    if len(data) > config.MAX_IMAGE_BYTES:
        raise HTTPException(413, f"File is larger than {config.MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    if not data:
        raise HTTPException(422, "The uploaded file is empty.")
    return data


def _run(data: bytes, conf: float, use_detector: bool, include_image: bool) -> dict[str, Any]:
    try:
        image = decode_image(data, config.MAX_IMAGE_SIDE)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    result = analyze(image, classifier, detector if use_detector else None, conf, config.UNCERTAIN_BELOW)
    body = result.to_dict()
    if include_image:
        body["annotated_image"] = "data:image/jpeg;base64," + base64.b64encode(encode_jpeg(annotate(image, result))).decode()
    return body


def _metrics() -> dict[str, Any] | None:
    try:
        return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "classifier": {"present": config.CLASSIFIER_PATH.is_file(), "loaded": classifier.loaded, "error": classifier.error},
        "detector": {"present": config.DETECTOR_PATH.is_file(), "loaded": detector.loaded, "error": detector.error},
    }


@app.get("/api/model")
def model_info() -> dict[str, Any]:
    m = _metrics()
    return {
        "classes": classifier.classes(),
        "streams": STREAMS,
        "detector": config.DETECTOR_PATH.name if config.DETECTOR_PATH.is_file() else None,
        "uncertain_below": config.UNCERTAIN_BELOW,
        "metrics": None if m is None else {k: m.get(k) for k in (
            "dataset", "base_model", "epochs", "imgsz", "test_accuracy", "per_class", "confusion", "split_sizes", "trained_at")},
    }


@app.get("/api/samples")
def samples() -> list[dict[str, str]]:
    if not config.SAMPLE_DIR.is_dir():
        return []
    return [{"name": p.name, "url": f"/api/samples/{p.name}"}
            for p in sorted(config.SAMPLE_DIR.iterdir()) if p.suffix.lower() in IMAGE_EXT]


@app.get("/api/samples/{name}")
def sample(name: str) -> FileResponse:
    path = (config.SAMPLE_DIR / name).resolve()
    if path.parent != config.SAMPLE_DIR.resolve() or not path.is_file() or path.suffix.lower() not in IMAGE_EXT:
        raise HTTPException(404, "Sample not found.")
    return FileResponse(path)


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...), conf: float = Form(0.25), detect: bool = Form(True)) -> dict[str, Any]:
    if Path(file.filename or "").suffix.lower() not in IMAGE_EXT:
        raise HTTPException(422, "Unsupported image type. Use JPEG, PNG, BMP or WebP.")
    data = await _read_limited(file)
    return await run_in_threadpool(_run, data, _conf(conf), detect, True)


@app.post("/api/analyze/frame")
async def analyze_frame(file: UploadFile = File(...), conf: float = Form(0.25), detect: bool = Form(True)) -> dict[str, Any]:
    """Webcam frame: processed in memory and discarded. Returns boxes and labels only."""
    data = await _read_limited(file)
    return await run_in_threadpool(_run, data, _conf(conf), detect, False)
