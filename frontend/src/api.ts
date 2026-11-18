export interface WasteItem {
  material: string;
  confidence: number;
  stream: string;
  uncertain: boolean;
  source: 'classifier' | 'detector';
  box: [number, number, number, number];
  object: string | null;
  object_confidence: number | null;
  top3: { material: string; probability: number }[];
}

export interface Analysis {
  items: WasteItem[];
  counts: Record<string, number>;
  whole_image: boolean;
  inference_ms: number;
  width: number;
  height: number;
  total: number;
  annotated_image?: string;
}

export interface ModelInfo {
  classes: string[];
  streams: Record<string, string>;
  detector: string | null;
  uncertain_below: number;
  metrics: null | {
    dataset: string;
    base_model: string;
    epochs: number;
    imgsz: number;
    test_accuracy: number;
    per_class: Record<string, { precision: number; recall: number; support: number }>;
    confusion: Record<string, Record<string, number>>;
    split_sizes: Record<string, Record<string, number>>;
    trained_at: string;
  };
}

export interface Sample {
  name: string;
  url: string;
}

const BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '');

export const apiUrl = (path: string) => `${BASE}${path}`;

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), init);
  } catch {
    throw new Error('Cannot reach the API. Is the backend running on port 8115?');
  }
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === 'string') msg = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(msg);
  }
  return (await res.json()) as T;
}

const form = (file: Blob, name: string, conf: number, detect: boolean) => {
  const f = new FormData();
  f.append('file', file, name);
  f.append('conf', String(conf));
  f.append('detect', String(detect));
  return f;
};

export const api = {
  model: () => call<ModelInfo>('/api/model'),
  samples: () => call<Sample[]>('/api/samples'),
  analyzeImage: (file: Blob, name: string, conf: number, detect: boolean) =>
    call<Analysis>('/api/analyze/image', { method: 'POST', body: form(file, name, conf, detect) }),
  analyzeFrame: (file: Blob, conf: number, detect: boolean) =>
    call<Analysis>('/api/analyze/frame', { method: 'POST', body: form(file, 'frame.jpg', conf, detect) }),
};

export const MATERIAL_COLORS: Record<string, string> = {
  cardboard: '#c48c40',
  paper: '#a0d8e6',
  glass: '#34dcc4',
  metal: '#b4b4b4',
  plastic: '#3ca0f0',
  trash: '#f06e6e',
  organic: '#5ac85a',
};
