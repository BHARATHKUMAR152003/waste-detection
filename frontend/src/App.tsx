import { useEffect, useState } from 'react';
import { Camera, Image as ImageIcon, Recycle } from 'lucide-react';
import { api, type ModelInfo } from './api';
import ImageMode from './components/ImageMode';
import ModelCard from './components/ModelCard';
import WebcamMode from './components/WebcamMode';

type Mode = 'image' | 'webcam';

export default function App() {
  const [mode, setMode] = useState<Mode>('image');
  const [conf, setConf] = useState(0.25);
  const [detect, setDetect] = useState(true);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.model().then(setInfo, (e) => setError(e.message));
  }, []);

  return (
    <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
      <header className="mb-5 flex items-center gap-3">
        <span className="grid h-10 w-10 place-items-center rounded-xl bg-accent-400/15 text-accent-300">
          <Recycle size={22} aria-hidden="true" />
        </span>
        <div>
          <h1 className="text-xl font-semibold text-fog-100">Waste Detection Using Computer Vision</h1>
          <p className="text-xs text-fog-400">YOLOv8 · OpenCV · FastAPI — material classification with disposal guidance</p>
        </div>
      </header>

      {info && (
        <div className="mb-4">
          <ModelCard info={info} />
        </div>
      )}

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div role="tablist" aria-label="Input" className="flex gap-2">
          {(
            [
              ['image', 'Photo', ImageIcon],
              ['webcam', 'Webcam', Camera],
            ] as const
          ).map(([key, label, Icon]) => (
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={mode === key}
              className={`btn ${mode === key ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setMode(key)}
            >
              <Icon size={16} aria-hidden="true" /> {label}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={detect} onChange={(e) => setDetect(e.target.checked)} className="h-4 w-4 accent-teal-400" />
          Locate separate items first
        </label>
        <label className="flex items-center gap-2 text-sm">
          Detector confidence
          <input type="range" min={0.05} max={0.9} step={0.05} value={conf} onChange={(e) => setConf(Number(e.target.value))} disabled={!detect} />
          <output className="w-10 tabular-nums">{Math.round(conf * 100)}%</output>
        </label>
      </div>

      {error && (
        <p role="alert" className="mb-4 rounded-xl border border-red-400/40 bg-red-400/10 px-4 py-3 text-sm text-red-200">
          {error}
        </p>
      )}

      <main>
        {mode === 'image' ? <ImageMode conf={conf} detect={detect} onError={setError} /> : <WebcamMode conf={conf} detect={detect} onError={setError} />}
      </main>
    </div>
  );
}
