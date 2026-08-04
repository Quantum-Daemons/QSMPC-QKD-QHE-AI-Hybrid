// The models behind a use case, and what they actually score.
//
// Everything here comes from the model registry (src/qsh/models/registry.py -> config.defaults.json).
// No architecture name, parameter count, metric, baseline, dataset licence or Hub URL is written in
// this file, so the page cannot drift from what the training runs produced.
//
// Two display rules that are the entire point of the panel:
//
//   * A model that has never been trained shows "not trained" — never a baseline standing in for a
//     measurement. The published baseline is what we are aiming at, not what we achieved.
//   * The honest metric is shown next to the flattering one. In four of these five industries the
//     aggregate number looks excellent while the class that matters is broken, so showing only the
//     aggregate would be true and misleading at the same time.
import { modelsFor, type ModelSpec } from "../lib/config";

const TIER_LABEL: Record<string, string> = {
  t1_crypto: "encrypted (FHE/MPC-feasible)",
  t2_industry: "industry model (plaintext)",
  t3_large: "large model (remote / WebGPU)",
};

/** Metrics that flatter and must never appear without their honest counterpart beside them. */
// Exact names were not enough: the set matched only whole keys, so quantum_qnn_accuracy,
// quantum_svc_accuracy and quantum_*_binary_f1 escaped the companion label and were rendered in the
// headline block — an aggregate presented as the result, which is the thing this project reports
// per-class metrics to avoid. Matched by suffix now, so a prefixed aggregate cannot slip through.
const COMPANION_METRICS = new Set([
  "accuracy", "aggregate_accuracy", "auroc", "point_adjusted_f1", "auprc_baseline",
]);
const COMPANION_SUFFIXES = ["_accuracy", "_binary_f1", "_auroc", "_point_adjusted_f1"];
const isCompanion = (k: string) =>
  COMPANION_METRICS.has(k) || COMPANION_SUFFIXES.some((sfx) => k.endsWith(sfx));

/** Run metadata that lives in the same record but is not a model metric. */
const NOT_A_METRIC = new Set([
  "wall_clock_s", "resolution", "n_eval", "n_train", "n_test", "positive_rate",
  "decision_threshold", "best_val_auc",
]);

function fmt(v: unknown): string {
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(4);
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v);
}

function MetricRows({ spec }: { spec: ModelSpec }) {
  const achieved = spec.achieved ?? null;
  if (!achieved) {
    return (
      <p className="muted small">
        Not trained in this build — no measured metrics. The published baseline below is the target,
        not a result.
      </p>
    );
  }
  const numeric = Object.entries(achieved).filter(
    ([k, v]) => typeof v === "number" && Number.isFinite(v as number) && !NOT_A_METRIC.has(k));
  const headline = numeric.filter(([k]) => !isCompanion(k));
  const companion = numeric.filter(([k]) => isCompanion(k));

  return (
    <table className="mini-table">
      <tbody>
        {headline.map(([k, v]) => (
          <tr key={k} className={k === spec.metric.name ? "metric-headline" : undefined}>
            <td><code>{k}</code></td>
            <td className="num">{fmt(v)}</td>
            <td className="muted small">
              {k === spec.metric.name && spec.metric.baseline != null
                ? `baseline ${spec.metric.baseline}${spec.metric.ceiling != null
                    ? ` · ceiling ${spec.metric.ceiling}` : ""}`
                : ""}
            </td>
          </tr>
        ))}
        {companion.map(([k, v]) => (
          <tr key={k} className="metric-companion">
            <td><code>{k}</code></td>
            <td className="num">{fmt(v)}</td>
            <td className="muted small">companion — shown so the gap is visible</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ModelPanel({ usecase }: { usecase: string }) {
  const specs = modelsFor(usecase);
  if (specs.length === 0) return null;

  return (
    <section className="card">
      <h3>Models</h3>
      <p className="muted small">
        The industry model runs in plaintext; the encrypted model is what homomorphic evaluation can
        actually carry. The gap between them is measured, not asserted.
      </p>

      {specs.map((spec) => (
        <div key={spec.key} className="model-block">
          <div className="model-head">
            <strong>{spec.label}</strong>
            <span className="tag">{TIER_LABEL[spec.tier] ?? spec.tier}</span>
            {spec.runs_encrypted ? <span className="tag">runs under CKKS</span> : null}
          </div>
          <p className="muted small">{spec.architecture}</p>

          <MetricRows spec={spec} />

          <p className="muted small">
            <strong>Data:</strong> {spec.dataset.name} — {spec.dataset.licence}{" "}
            <a href={spec.dataset.url} target="_blank" rel="noreferrer noopener">source</a>
            {!spec.dataset.publishable && " · weights not published (licence)"}
          </p>
          {spec.metric.baseline_source && (
            <p className="muted small"><strong>Baseline:</strong> {spec.metric.baseline_source}</p>
          )}
          {typeof spec.achieved?.baseline_gap_note === "string" && (
            <p className="muted small"><strong>Gap to baseline:</strong> {spec.achieved.baseline_gap_note}</p>
          )}
          {spec.notes && <p className="muted small">{spec.notes}</p>}
          {spec.hf_repo_id && (
            <p className="muted small">
              <strong>Weights:</strong>{" "}
              <a href={`https://huggingface.co/${spec.hf_repo_id}`} target="_blank" rel="noreferrer noopener">
                {spec.hf_repo_id}
              </a>
              {spec.hf_revision
                ? <> · pinned to <code>{spec.hf_revision.slice(0, 12)}</code></>
                : <> · <em>no pinned revision; the browser will not fetch an unpinned model</em></>}
            </p>
          )}
        </div>
      ))}
    </section>
  );
}
