"""Fine-tunes a YOLOv8n classifier on TrashNet and reports accuracy on a held-out test split.

TrashNet (Gary Thung & Mindy Yang, MIT licence): 2,527 photos in six classes
(cardboard, glass, metal, paper, plastic, trash). Download `dataset-resized.zip` from
https://github.com/garythung/trashnet and unzip it, then:

    python scripts/train_classifier.py --data path/to/dataset-resized --epochs 15

Writes models/waste-cls.pt and models/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

try:  # use the OS certificate store (needed behind TLS-inspecting corporate proxies)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


def split_dataset(src: Path, dst: Path, seed: int) -> dict[str, dict[str, int]]:
    """Stratified 70/15/15 split, copied into the folder layout ultralytics expects."""
    if dst.exists():
        shutil.rmtree(dst)
    rng = random.Random(seed)
    sizes: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for cls in CLASSES:
        files = sorted(p for p in (src / cls).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise SystemExit(f"No images found for class '{cls}' in {src}")
        rng.shuffle(files)
        n_val = n_test = max(1, round(len(files) * 0.15))
        parts = {"test": files[:n_test], "val": files[n_test:n_test + n_val], "train": files[n_test + n_val:]}
        for split, items in parts.items():
            out = dst / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, out / f.name)
            sizes[split][cls] = len(items)
    return sizes


def evaluate(model, test_dir: Path) -> dict:
    names = model.names
    confusion = {a: Counter() for a in CLASSES}
    for cls in CLASSES:
        images = sorted((test_dir / cls).iterdir())
        for i in range(0, len(images), 32):
            for r in model.predict([str(p) for p in images[i:i + 32]], verbose=False, device="cpu"):
                confusion[cls][names[int(r.probs.top1)]] += 1
    total = sum(sum(c.values()) for c in confusion.values())
    correct = sum(confusion[c][c] for c in CLASSES)
    per_class = {}
    for c in CLASSES:
        tp = confusion[c][c]
        predicted = sum(confusion[a][c] for a in CLASSES)
        actual = sum(confusion[c].values())
        per_class[c] = {
            "precision": round(tp / predicted, 4) if predicted else 0.0,
            "recall": round(tp / actual, 4) if actual else 0.0,
            "support": actual,
        }
    return {
        "test_accuracy": round(correct / total, 4),
        "per_class": per_class,
        "confusion": {a: {b: confusion[a][b] for b in CLASSES} for a in CLASSES},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="TrashNet dataset-resized folder")
    ap.add_argument("--workdir", type=Path, default=Path.home() / "waste-train", help="scratch folder (keep it outside synced folders)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    split_dir = args.workdir / "split"
    sizes = split_dataset(args.data, split_dir, args.seed)
    print("split sizes:", json.dumps(sizes))

    model = YOLO("yolov8n-cls.pt")
    model.train(data=str(split_dir), epochs=args.epochs, imgsz=args.imgsz, seed=args.seed, deterministic=True,
                workers=0, device="cpu", batch=32, project=str(args.workdir / "runs"), name="waste", exist_ok=True,
                plots=False, verbose=False)

    best = args.workdir / "runs" / "waste" / "weights" / "best.pt"
    trained = YOLO(str(best))
    report = evaluate(trained, split_dir / "test")
    print("test accuracy:", report["test_accuracy"])

    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    shutil.copy2(best, models / "waste-cls.pt")
    meta = {
        "dataset": "TrashNet (dataset-resized), stratified 70/15/15 split",
        "base_model": "yolov8n-cls (ImageNet-pretrained)",
        "classes": CLASSES,
        "split_sizes": sizes,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **report,
    }
    (models / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("saved", models / "waste-cls.pt")


if __name__ == "__main__":
    main()
