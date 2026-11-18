import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../App';

const model = {
  classes: ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
  streams: {},
  detector: 'yolov8n.pt',
  uncertain_below: 0.5,
  metrics: {
    dataset: 'TrashNet',
    base_model: 'yolov8n-cls',
    epochs: 15,
    imgsz: 224,
    test_accuracy: 0.875,
    per_class: Object.fromEntries(['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'].map((c) => [c, { precision: 0.9, recall: 0.85, support: 10 }])),
    confusion: Object.fromEntries(
      ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'].map((r) => [r, Object.fromEntries(['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'].map((c) => [c, r === c ? 9 : 0]))]),
    ),
    split_sizes: {},
    trained_at: '',
  },
};

const analysis = {
  items: [
    {
      material: 'plastic',
      confidence: 0.91,
      stream: 'Plastic recycling',
      uncertain: false,
      source: 'classifier',
      box: [0, 0, 10, 10],
      object: 'bottle',
      object_confidence: 0.8,
      top3: [
        { material: 'plastic', probability: 0.91 },
        { material: 'glass', probability: 0.06 },
      ],
    },
    { material: 'trash', confidence: 0.35, stream: 'General (non-recyclable) waste', uncertain: true, source: 'classifier', box: [0, 0, 5, 5], object: 'cup', object_confidence: 0.5, top3: [] },
  ],
  counts: { plastic: 1, trash: 1 },
  whole_image: false,
  inference_ms: 120,
  width: 10,
  height: 10,
  total: 2,
  annotated_image: 'data:image/jpeg;base64,AAAA',
};

let calls: { url: string; init?: RequestInit }[];

beforeEach(() => {
  calls = [];
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    const respond = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
    if (url === '/api/model') return respond(model);
    if (url === '/api/samples') return respond([{ name: 'plastic-1.jpg', url: '/api/samples/plastic-1.jpg' }]);
    if (url === '/api/samples/plastic-1.jpg') return new Response(new Blob(['x'], { type: 'image/jpeg' }));
    if (url === '/api/analyze/image') return respond(analysis);
    return respond({ detail: 'not found' }, 404);
  }));
});

afterEach(() => vi.unstubAllGlobals());

it('shows model accuracy, analyses a sample and lists materials with disposal streams', async () => {
  const user = userEvent.setup();
  render(<App />);
  expect(await screen.findByText(/87\.5% accuracy on held-out test photos/)).toBeInTheDocument();

  await user.click(await screen.findByRole('button', { name: 'Sample: plastic' }));
  const results = await screen.findByRole('region', { name: 'Results' });
  expect(within(results).getByText('2 items · 120 ms')).toBeInTheDocument();
  expect(within(results).getByText('→ Plastic recycling')).toBeInTheDocument();
  expect(within(results).getByText(/Low confidence/)).toBeInTheDocument();
  expect(within(results).getByText(/Also possible: glass 6%/)).toBeInTheDocument();
  expect(screen.getByAltText('Photo with waste labels')).toBeInTheDocument();

  const post = calls.find((c) => c.url === '/api/analyze/image')!;
  const body = post.init!.body as FormData;
  expect(body.get('detect')).toBe('true');
  expect(body.get('conf')).toBe('0.25');
});

it('sends detect=false when item location is turned off', async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.click(screen.getByRole('checkbox', { name: /locate separate items/i }));
  await user.click(await screen.findByRole('button', { name: 'Sample: plastic' }));
  await screen.findByRole('region', { name: 'Results' });
  expect((calls.find((c) => c.url === '/api/analyze/image')!.init!.body as FormData).get('detect')).toBe('false');
});
