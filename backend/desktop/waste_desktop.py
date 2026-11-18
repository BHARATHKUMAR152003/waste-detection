"""Tkinter desktop client: live webcam (OpenCV) or image file -> waste material + disposal stream.

Run from the backend folder:  python desktop/waste_desktop.py
Uses the same models and pipeline as the web API (app/pipeline.py).
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config  # noqa: E402
from app.pipeline import ModelUnavailable, YoloClassifier, YoloDetector, analyze, annotate  # noqa: E402

DISPLAY_WIDTH = 720


class WasteApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.classifier = YoloClassifier(config.CLASSIFIER_PATH, config.DEVICE)
        self.detector = YoloDetector(config.DETECTOR_PATH, config.DEVICE)
        self.capture: cv2.VideoCapture | None = None
        self.photo: ImageTk.PhotoImage | None = None

        root.title("Waste Detection — desktop")
        root.minsize(980, 560)
        controls = ttk.Frame(root, padding=8)
        controls.pack(side=tk.TOP, fill=tk.X)
        self.cam_button = ttk.Button(controls, text="Start camera", command=self.toggle_camera)
        self.cam_button.pack(side=tk.LEFT)
        ttk.Button(controls, text="Open image…", command=self.open_image).pack(side=tk.LEFT, padx=6)
        ttk.Label(controls, text="Detector confidence").pack(side=tk.LEFT, padx=(16, 4))
        self.conf = tk.DoubleVar(value=0.25)
        ttk.Scale(controls, from_=0.05, to=0.9, variable=self.conf, length=160).pack(side=tk.LEFT)
        self.use_detector = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Locate items first", variable=self.use_detector).pack(side=tk.LEFT, padx=12)

        body = ttk.Frame(root, padding=8)
        body.pack(fill=tk.BOTH, expand=True)
        self.view = ttk.Label(body, text="Start the camera or open an image.", anchor=tk.CENTER)
        self.view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.results = tk.Text(body, width=40, state=tk.DISABLED, wrap=tk.WORD)
        self.results.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        root.protocol("WM_DELETE_WINDOW", self.close)

    def show(self, frame) -> None:
        try:
            result = analyze(frame, self.classifier, self.detector if self.use_detector.get() else None,
                             self.conf.get(), config.UNCERTAIN_BELOW)
        except ModelUnavailable as exc:
            self.stop_camera()
            messagebox.showerror("Model unavailable", str(exc))
            return
        shown = annotate(frame, result)
        h, w = shown.shape[:2]
        if w > DISPLAY_WIDTH:
            shown = cv2.resize(shown, (DISPLAY_WIDTH, int(h * DISPLAY_WIDTH / w)))
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)))
        self.view.configure(image=self.photo, text="")

        lines = [f"{len(result.items)} item(s) · {result.inference_ms:.0f} ms", ""]
        if result.whole_image:
            lines.append("No separate items found — classified the whole image.\n")
        for it in result.items:
            obj = f" ({it.object})" if it.object else ""
            flag = "  [uncertain]" if it.uncertain else ""
            lines.append(f"• {it.material}{obj} {it.confidence:.0%}{flag}\n  → {it.stream}")
        self.results.configure(state=tk.NORMAL)
        self.results.delete("1.0", tk.END)
        self.results.insert(tk.END, "\n".join(lines))
        self.results.configure(state=tk.DISABLED)

    def open_image(self) -> None:
        self.stop_camera()
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            frame = cv2.imread(path)
            if frame is None:
                messagebox.showerror("Unreadable file", "That file is not a supported image.")
            else:
                self.show(frame)

    def toggle_camera(self) -> None:
        if self.capture is not None:
            self.stop_camera()
            return
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.capture = None
            messagebox.showerror("Camera", "No camera could be opened.")
            return
        self.cam_button.configure(text="Stop camera")
        self.tick()

    def tick(self) -> None:
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if ok:
            self.show(frame)
        self.root.after(30, self.tick)

    def stop_camera(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.cam_button.configure(text="Start camera")

    def close(self) -> None:
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    WasteApp(window)
    window.mainloop()
