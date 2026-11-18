import { useEffect, useRef, useState } from 'react';
import { ImageUp, Loader2 } from 'lucide-react';
import { api, apiUrl, type Analysis, type Sample } from '../api';
import ResultList from './ResultList';

interface Props {
  conf: number;
  detect: boolean;
  onError: (msg: string | null) => void;
}

export default function ImageMode({ conf, detect, onError }: Props) {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [result, setResult] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.samples().then(setSamples, () => setSamples([]));
  }, []);

  const run = async (blob: Blob, name: string) => {
    setBusy(true);
    onError(null);
    try {
      setResult(await api.analyzeImage(blob, name, conf, detect));
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const runSample = async (s: Sample) => {
    const res = await fetch(apiUrl(s.url));
    if (!res.ok) {
      onError('Could not load the sample image.');
      return;
    }
    await run(new Blob([await res.arrayBuffer()], { type: res.headers.get('Content-Type') ?? 'image/jpeg' }), s.name);
  };

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <div className="panel min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" className="btn btn-primary" onClick={() => input.current?.click()} disabled={busy}>
            {busy ? <Loader2 size={16} className="animate-spin" aria-hidden="true" /> : <ImageUp size={16} aria-hidden="true" />}
            Upload photo
          </button>
          <input
            ref={input}
            type="file"
            accept="image/jpeg,image/png,image/bmp,image/webp"
            className="sr-only"
            aria-label="Upload photo"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void run(f, f.name);
              e.target.value = '';
            }}
          />
          {samples.map((s) => (
            <button key={s.name} type="button" className="btn btn-secondary" disabled={busy} onClick={() => void runSample(s)}>
              Sample: {s.name.split('-')[0]}
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs text-fog-400">JPEG, PNG, BMP or WebP up to 8 MB. Samples are held-out TrashNet test photos.</p>
        <div className="mt-4 overflow-hidden rounded-xl border border-ink-700 bg-ink-950">
          {result?.annotated_image ? (
            <img src={result.annotated_image} alt="Photo with waste labels" className="block h-auto w-full" />
          ) : (
            <p className="px-6 py-20 text-center text-sm text-fog-400">Upload a photo of an item or try a sample.</p>
          )}
        </div>
      </div>
      {result && <ResultList result={result} />}
    </div>
  );
}
