import { MATERIAL_COLORS, type Analysis } from '../api';

export default function ResultList({ result, live = false }: { result: Analysis; live?: boolean }) {
  return (
    <section className="panel" aria-label="Results" aria-live={live ? 'off' : 'polite'}>
      <h2 className="panel-title">
        {result.total} item{result.total === 1 ? '' : 's'} · {result.inference_ms.toFixed(0)} ms
      </h2>
      {result.whole_image && (
        <p className="mt-1 text-xs text-fog-400">No separate items were located, so the whole image was classified.</p>
      )}
      <ul className="mt-3 space-y-2">
        {result.items.map((it, i) => (
          <li key={i} className="rounded-xl border border-ink-700 p-3 text-sm">
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-2 font-medium capitalize text-fog-100">
                <span aria-hidden="true" className="h-2.5 w-2.5 rounded-full" style={{ background: MATERIAL_COLORS[it.material] }} />
                {it.material}
                {it.object && <span className="text-xs font-normal normal-case text-fog-400">({it.object})</span>}
              </span>
              <span className="tabular-nums">{(it.confidence * 100).toFixed(0)}%</span>
            </div>
            <p className="mt-1 text-xs text-fog-300">→ {it.stream}</p>
            {it.uncertain && <p className="mt-1 text-xs text-amber-200">Low confidence — check before sorting.</p>}
            {it.top3.length > 1 && (
              <p className="mt-1 text-xs text-fog-400">
                Also possible: {it.top3.slice(1).map((t) => `${t.material} ${(t.probability * 100).toFixed(0)}%`).join(', ')}
              </p>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
