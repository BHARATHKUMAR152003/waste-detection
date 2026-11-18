import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.pipeline import encode_jpeg

from .conftest import FakeClassifier, FakeDetector, real_models_available


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "classifier", FakeClassifier())
    monkeypatch.setattr(main, "detector", FakeDetector([("bottle", 0.9, [10, 10, 200, 300])]))
    with TestClient(main.app) as c:
        yield c


def jpeg(h=240, w=320):
    return encode_jpeg(np.full((h, w, 3), 180, dtype=np.uint8))


def test_analyze_image_returns_items_and_annotated_image(client):
    res = client.post("/api/analyze/image", files={"file": ("x.jpg", jpeg(), "image/jpeg")}, data={"conf": "0.3"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["total"] == 1
    assert body["items"][0]["material"] == "plastic"
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")


def test_frame_endpoint_skips_the_image_and_can_disable_detection(client):
    body = client.post("/api/analyze/frame", files={"file": ("f.jpg", jpeg(), "image/jpeg")}, data={"detect": "false"}).json()
    assert "annotated_image" not in body
    assert body["whole_image"] is True


@pytest.mark.parametrize("name, data, conf, status", [
    ("x.gif", b"GIF89a", "0.25", 422),
    ("x.jpg", b"not an image", "0.25", 422),
    ("x.jpg", b"", "0.25", 422),
    ("x.jpg", None, "0.99", 422),
])
def test_validation(client, name, data, conf, status):
    payload = jpeg() if data is None else data
    res = client.post("/api/analyze/image", files={"file": (name, payload, "image/jpeg")}, data={"conf": conf})
    assert res.status_code == status


def test_size_limit(client, monkeypatch):
    monkeypatch.setattr(main.config, "MAX_IMAGE_BYTES", 1000)
    res = client.post("/api/analyze/image", files={"file": ("x.jpg", jpeg(600, 800), "image/jpeg")})
    assert res.status_code == 413


def test_sample_path_traversal_is_blocked(client):
    assert client.get("/api/samples/..%2Fapp%2Fmain.py").status_code == 404
    assert client.get("/api/samples/missing.jpg").status_code == 404


def test_missing_weights_return_503(monkeypatch, tmp_path):
    from app.pipeline import YoloClassifier

    monkeypatch.setattr(main, "classifier", YoloClassifier(tmp_path / "nope.pt"))
    with TestClient(main.app) as c:
        res = c.post("/api/analyze/image", files={"file": ("x.jpg", jpeg(), "image/jpeg")}, data={"detect": "false"})
    assert res.status_code == 503
    assert res.json()["code"] == "model_unavailable"


@pytest.mark.skipif(not real_models_available(), reason="trained weights not present")
def test_real_models_classify_held_out_samples():
    """End-to-end with the real weights on the bundled held-out TrashNet samples."""
    with TestClient(main.app) as c:
        info = c.get("/api/model").json()
        assert set(info["classes"]) == {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
        assert info["metrics"]["test_accuracy"] > 0.7
        samples = c.get("/api/samples").json()
        assert samples
        correct = 0
        for s in samples:
            raw = c.get(s["url"]).content
            body = c.post("/api/analyze/image", files={"file": (s["name"], raw, "image/jpeg")}, data={"detect": "false"}).json()
            correct += body["items"][0]["material"] == s["name"].split("-")[0]
        assert correct >= len(samples) - 2, json.dumps(samples)
