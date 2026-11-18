import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, CameraOff } from 'lucide-react';
import { api, MATERIAL_COLORS, type Analysis } from '../api';
import ResultList from './ResultList';

interface Props {
  conf: number;
  detect: boolean;
  onError: (msg: string | null) => void;
}

const MAX_FRAME_WIDTH = 640;

function cameraError(e: unknown): string {
  const name = (e as { name?: string })?.name;
  if (name === 'NotAllowedError') return 'Camera permission was denied. Allow camera access in the browser and try again.';
  if (name === 'NotFoundError') return 'No camera was found on this device.';
  if (name === 'NotReadableError') return 'The camera is in use by another application.';
  return 'Could not start the camera.';
}

export default function WebcamMode({ conf, detect, onError }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const running = useRef(false);
  const settings = useRef({ conf, detect });
  const [active, setActive] = useState(false);
  const [result, setResult] = useState<Analysis | null>(null);

  settings.current = { conf, detect };

  const draw = (r: Analysis) => {
    const c = overlay.current;
    const v = video.current;
    if (!c || !v) return;
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);
    const s = v.videoWidth / r.width;
    ctx.lineWidth = Math.max(2, c.width / 320);
    ctx.font = `${Math.max(14, c.width / 40)}px Inter, Segoe UI, sans-serif`;
    for (const it of r.items) {
      const [x1, y1, x2, y2] = it.box.map((n) => n * s);
      const color = MATERIAL_COLORS[it.material] ?? '#cfd6de';
      if (!r.whole_image) {
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
      const text = `${it.material} ${(it.confidence * 100).toFixed(0)}%${it.uncertain ? '?' : ''}`;
      const th = parseInt(ctx.font, 10) + 6;
      ctx.fillStyle = color;
      ctx.fillRect(x1, Math.max(0, y1 - (r.whole_image ? 0 : th)), ctx.measureText(text).width + 8, th);
      ctx.fillStyle = '#11151b';
      ctx.fillText(text, x1 + 4, Math.max(th - 6, y1 - (r.whole_image ? -th + 6 : 6)));
    }
  };

  const stop = useCallback(() => {
    running.current = false;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    setActive(false);
  }, []);

  const loop = useCallback(async () => {
    const v = video.current;
    const grab = document.createElement('canvas');
    while (running.current && v) {
      if (v.readyState < 2) {
        await new Promise((r) => setTimeout(r, 100));
        continue;
      }
      const scale = Math.min(1, MAX_FRAME_WIDTH / v.videoWidth);
      grab.width = Math.round(v.videoWidth * scale);
      grab.height = Math.round(v.videoHeight * scale);
      grab.getContext('2d')?.drawImage(v, 0, 0, grab.width, grab.height);
      const blob = await new Promise<Blob | null>((r) => grab.toBlob(r, 'image/jpeg', 0.8));
      if (!blob || !running.current) break;
      try {
        const r = await api.analyzeFrame(blob, settings.current.conf, settings.current.detect);
        if (!running.current) break;
        setResult(r);
        draw(r);
      } catch (e) {
        onError((e as Error).message);
        stop();
        break;
      }
    }
  }, [onError, stop]);

  const start = async () => {
    onError(null);
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      onError('Camera access needs a secure context (HTTPS or localhost) and a browser that supports getUserMedia.');
      return;
    }
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      stream.current = s;
      if (video.current) {
        video.current.srcObject = s;
        await video.current.play();
      }
      running.current = true;
      setActive(true);
      void loop();
    } catch (e) {
      stop();
      onError(cameraError(e));
    }
  };

  useEffect(() => stop, [stop]);

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <div className="panel min-w-0">
        {active ? (
          <button type="button" className="btn btn-danger" onClick={stop}>
            <CameraOff size={16} aria-hidden="true" /> Stop camera
          </button>
        ) : (
          <button type="button" className="btn btn-primary" onClick={() => void start()}>
            <Camera size={16} aria-hidden="true" /> Start camera
          </button>
        )}
        <div className="relative mt-4 overflow-hidden rounded-xl border border-ink-700 bg-ink-950">
          <video ref={video} playsInline muted className={`block h-auto w-full ${active ? '' : 'hidden'}`} aria-label="Camera preview" />
          <canvas ref={overlay} className={`pointer-events-none absolute inset-0 h-full w-full ${active ? '' : 'hidden'}`} aria-hidden="true" />
          {!active && (
            <p className="px-6 py-20 text-center text-sm text-fog-400">
              Hold an item up to the camera. Frames are analysed in memory and never stored.
            </p>
          )}
        </div>
      </div>
      {result && active && <ResultList result={result} live />}
    </div>
  );
}
