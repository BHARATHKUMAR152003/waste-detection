import numpy as np
import pytest

from app.pipeline import analyze, annotate, decode_image, encode_jpeg, pad_box

from .conftest import FakeClassifier, FakeDetector


def test_whole_image_is_classified_when_nothing_is_detected(image):
    clf = FakeClassifier()
    result = analyze(image, clf, FakeDetector([]))
    assert result.whole_image is True
    assert clf.calls == [1]
    assert result.items[0].material == "plastic"
    assert result.items[0].stream == "Plastic recycling"
    assert result.items[0].box == [0.0, 0.0, 600.0, 400.0]


def test_detected_items_are_cropped_and_classified(image):
    clf = FakeClassifier()
    det = FakeDetector([("bottle", 0.9, [10, 10, 200, 300]), ("cup", 0.6, [300, 50, 450, 250])])
    result = analyze(image, clf, det)
    assert result.whole_image is False
    assert clf.calls == [2]
    assert [i.object for i in result.items] == ["bottle", "cup"]
    assert result.counts == {"plastic": 2}
    assert result.items[0].top3[0] == {"material": "plastic", "probability": 0.8}


def test_people_tiny_boxes_and_low_confidence_are_ignored(image):
    clf = FakeClassifier()
    det = FakeDetector([
        ("person", 0.95, [0, 0, 300, 400]),
        ("bottle", 0.9, [10, 10, 15, 15]),  # far below 1% of the image area
        ("cup", 0.1, [100, 100, 300, 300]),
    ])
    result = analyze(image, clf, det, conf=0.25)
    assert result.whole_image is True
    assert clf.calls == [1]


def test_food_is_organic_without_the_classifier(image):
    clf = FakeClassifier()
    result = analyze(image, clf, FakeDetector([("banana", 0.7, [50, 50, 250, 200])]))
    assert clf.calls == []
    assert result.items[0].material == "organic"
    assert result.items[0].source == "detector"
    assert result.items[0].stream == "Compost / wet waste"


def test_low_probability_is_flagged_uncertain(image):
    clf = FakeClassifier([("trash", 0.34), ("paper", 0.33), ("plastic", 0.33)])
    result = analyze(image, clf, None, uncertain_below=0.5)
    assert result.items[0].uncertain is True


def test_pad_box_is_clamped_to_the_image():
    assert pad_box([0, 0, 100, 100], 120, 90) == (0, 0, 110, 90)


def test_decode_rejects_non_images_and_downscales():
    with pytest.raises(ValueError, match="not a valid image"):
        decode_image(b"not an image", 1024)
    big = np.zeros((3000, 1500, 3), dtype=np.uint8)
    out = decode_image(encode_jpeg(big), 1000)
    assert max(out.shape[:2]) == 1000


def test_annotate_keeps_image_size(image):
    result = analyze(image, FakeClassifier(), FakeDetector([("bottle", 0.9, [10, 10, 200, 300])]))
    assert annotate(image, result).shape == image.shape
