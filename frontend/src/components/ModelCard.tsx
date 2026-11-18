import type { ModelInfo } from '../api';

export default function ModelCard({ info }: { info: ModelInfo }) {
  const m = info.metrics;
  return (
    <details className="panel">
      <summary className="cursor-pointer text-sm font-semibold text-fog-100">
        Model: YOLOv8n classifier fine-tuned on TrashNet
        {m && <span className="ml-2 font-normal text-fog-300">— {(m.test_accuracy * 100).toFixed(1)}% accuracy on held-out test photos</span>}
      </summary>
      {m ? (
        <div className="mt-3 space-y-3 text-sm">
          <p className="text-fog-300">
            {m.dataset}; base {m.base_model}; {m.epochs} epochs at {m.imgsz}px. Items are first located with {info.detector ?? 'a detector'} (COCO),
            then each crop is classified. Food detected by the COCO model is labelled organic directly.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] text-left text-xs">
              <caption className="mb-1 text-left text-fog-400">Confusion matrix (rows = true class, columns = predicted)</caption>
              <thead>
                <tr>
                  <th className="py-1 pr-2" />
                  {info.classes.map((c) => (
                    <th key={c} className="py-1 pr-2 font-medium capitalize">
                      {c}
                    </th>
                  ))}
                  <th className="py-1 pr-2 font-medium">Recall</th>
                  <th className="py-1 font-medium">Precision</th>
                </tr>
              </thead>
              <tbody>
                {info.classes.map((r) => (
                  <tr key={r} className="border-t border-ink-700">
                    <th className="py-1 pr-2 font-medium capitalize">{r}</th>
                    {info.classes.map((c) => (
                      <td key={c} className={`py-1 pr-2 tabular-nums ${r === c ? 'text-accent-300' : m.confusion[r][c] ? 'text-amber-200' : 'text-fog-400'}`}>
                        {m.confusion[r][c]}
                      </td>
                    ))}
                    <td className="py-1 pr-2 tabular-nums">{(m.per_class[r].recall * 100).toFixed(0)}%</td>
                    <td className="py-1 tabular-nums">{(m.per_class[r].precision * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-fog-400">
            TrashNet photos show one item on a plain background. Cluttered real-world scenes are harder, so treat low-confidence results as
            suggestions.
          </p>
        </div>
      ) : (
        <p className="mt-2 text-sm text-fog-400">No evaluation report found (models/metrics.json).</p>
      )}
    </details>
  );
}
