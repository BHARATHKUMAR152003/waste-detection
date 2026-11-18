"""Downloads the COCO-pretrained YOLOv8n detector (Ultralytics, AGPL-3.0) used to locate items before classification.

Usage:  python scripts/download_detector.py
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
TARGET = ROOT / "models" / "yolov8n.pt"

try:  # use the OS certificate store (needed behind TLS-inspecting corporate proxies)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

if __name__ == "__main__":
    if TARGET.is_file() and TARGET.stat().st_size > 1_000_000:
        print(f"Detector already present: {TARGET}")
        sys.exit(0)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    tmp = TARGET.with_suffix(".part")
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    if tmp.stat().st_size < 1_000_000:
        tmp.unlink()
        sys.exit("Download looks incomplete (file too small).")
    tmp.replace(TARGET)
    print(f"Saved {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")
