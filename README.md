# Waste Detection Using Computer Vision

Classify waste items by **material** (cardboard, glass, metal, paper, plastic, trash) with a YOLOv8 classifier
fine-tuned on the TrashNet dataset, and get the matching disposal stream. Works on photos and a live webcam in the
browser, and in a **Tkinter + OpenCV desktop app**. Rebuilt from the project on my resume.

| Part | Port | Tech |
| --- | --- | --- |
| Frontend | 5115 | React 18, TypeScript, Vite, Tailwind CSS |
| Backend | 8115 | FastAPI, Ultralytics YOLOv8, OpenCV, PyTorch (CPU) |
| Desktop | — | Tkinter + OpenCV (`backend/desktop/waste_desktop.py`) |

## How it works

1. **Locate** (optional, on by default) — the COCO-pretrained YOLOv8n detector finds candidate objects. Only object
   classes that can be discarded items are kept (bottle, cup, can-like items, books, utensils, …); people, vehicles and
   furniture are ignored, as are boxes smaller than 1 % of the image.
2. **Classify** — each crop (padded by 10 %) is classified by the TrashNet-trained YOLOv8n-cls model. If nothing was
   located, the whole image is classified.
3. **Organic** — TrashNet has no food class, so food detected by the COCO model (banana, apple, sandwich, …) is labelled
   *organic → compost* directly by the detector.
4. Each result has a disposal stream, the top-3 material probabilities, and an **uncertain** flag when the top
   probability is below 50 % (`UNCERTAIN_BELOW`).

Camera frames and uploads are processed in memory and never stored.

## Model and evaluation

`scripts/train_classifier.py` fine-tunes `yolov8n-cls` (ImageNet-pretrained) on TrashNet with a seeded, stratified
70 / 15 / 15 train / validation / test split, then evaluates the best checkpoint on the **held-out test split** and
writes `models/waste-cls.pt` plus `models/metrics.json` (accuracy, per-class precision/recall, confusion matrix). The
web UI shows these numbers. The bundled `sample_media/` photos come from the test split, so the model never saw them.

TrashNet photos show a single item on a plain background; real, cluttered scenes are harder, so accuracy there will
be lower than the test figure. The *trash* class is also the smallest (137 images).

Dataset: [TrashNet](https://github.com/garythung/trashnet) by Gary Thung and Mindy Yang (MIT licence). Detector
weights: Ultralytics YOLOv8n (AGPL-3.0).

## Run locally

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_detector.py        # COCO YOLOv8n weights (~6 MB)
uvicorn app.main:app --port 8115
# desktop client instead of / in addition to the web app:
python desktop/waste_desktop.py
```

```bash
cd frontend
npm install
npm run dev        # http://localhost:5115
```

The trained classifier (`models/waste-cls.pt`, ~3 MB) and its `metrics.json` are committed. To retrain, download
`dataset-resized.zip` from the TrashNet repository, unzip it and run
`python scripts/train_classifier.py --data path/to/dataset-resized --epochs 15` (about an hour on a laptop CPU).

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/health` | Model files present / loaded / errors |
| GET | `/api/model` | Classes, disposal streams, evaluation metrics |
| GET | `/api/samples`, `/api/samples/{name}` | Held-out sample photos |
| POST | `/api/analyze/image` | multipart `file`, `conf` (0.05–0.95), `detect` (bool) → items + annotated image |
| POST | `/api/analyze/frame` | Same, without the annotated image (webcam) |

Missing weights return `503 {"code": "model_unavailable"}`; invalid or oversized images return 422 / 413.

## Tests

```bash
cd backend && pip install -r requirements-dev.txt && pytest
cd frontend && npx vitest run && npm run build
```

Pipeline and API tests use fake models (crop selection, filtering, organic handling, uncertainty, validation, size
limit, path traversal, missing weights). One end-to-end test runs the real weights on the held-out samples and is
skipped when the weights are absent.

## Deployment

- **Backend**: a container or app host with ~1.5 GB RAM (PyTorch + two YOLO models), e.g. Render, Railway or Azure
  App Service. Start with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, set `CORS_ORIGINS` to the frontend URL
  and run `scripts/download_detector.py` during the build.
- **Frontend**: static site — build with `VITE_API_URL=<backend URL> npm run build` and publish `frontend/dist`.
  The webcam needs HTTPS.

## Limitations

- Six TrashNet materials plus detector-based organic; no e-waste, textiles or hazardous classes.
- Disposal streams are generic; local recycling rules differ.
- CPU inference: roughly 100–300 ms per frame on a laptop, so the webcam view updates a few times per second.
