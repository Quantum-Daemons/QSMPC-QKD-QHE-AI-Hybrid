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
import { useState } from "react";
import { browserModels, evaluatePublishedModel, type ModelEvalResult } from "../lib/onnxInference";

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
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toLocaleString();
    // Don't let a near-miss print as a clean 1.0000. The satellite student's teacher-agreement is
    // 0.999987, which toFixed(4) rounds to "1.0000" — read as exact agreement when it is not.
    // toPrecision(3) is no better here ("1.00"), so widen the decimals until the value stops
    // rendering as an exact 1 or 0.
    for (let d = 4; d <= 8; d++) {
      const r = v.toFixed(d);
      if (Number(r) !== 1 && Number(r) !== 0) return r;
    }
    return v.toExponential(2);
  }
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
  // Flatten one level of nesting before filtering. genomics.prs keeps every measured value under
  // `measured` (n_samples, global_high_risk_cut, by_super_population...) and its only numeric
  // top-level key is wall_clock_s, which NOT_A_METRIC excludes — so the filter returned nothing and
  // the component rendered an empty <table>, hiding the ancestry stratification that the model's own
  // registry note calls "the finding, not a footnote".
  const flat: Array<[string, number]> = [];
  // `not_measured` holds published baselines and nulls for things this build did NOT measure --
  // walking into it would present someone else's 0.628 as this run's result. `licence_audit` is
  // provenance, not a metric.
  const NOT_A_SUBTREE = new Set(["not_measured", "licence_audit"]);
  const walk = (obj: Record<string, unknown>, prefix: string, depth: number) => {
    for (const [k, v] of Object.entries(obj)) {
      if (NOT_A_SUBTREE.has(k)) continue;
      const key = prefix ? `${prefix}.${k}` : k;
      if (typeof v === "number" && Number.isFinite(v)) {
        if (!NOT_A_METRIC.has(k) && !NOT_A_METRIC.has(key)) flat.push([key, v]);
      } else if (v && typeof v === "object" && !Array.isArray(v) && depth < 3) {
        walk(v as Record<string, unknown>, key, depth + 1);
      }
    }
  };
  walk(achieved as Record<string, unknown>, "", 0);
  // Strip the redundant `measured.` prefix now that nothing unmeasured can reach here.
  const numeric: Array<[string, number]> = flat.map(
    ([k, v]) => [k.startsWith("measured.") ? k.slice("measured.".length) : k, v]);
  // A record that measured nothing numeric must say why rather than render an empty table.
  const notMeasured = (achieved as Record<string, unknown>).not_measured as
    | { reason?: string } | undefined;
  if (numeric.length === 0) {
    return (
      <p className="muted small">
        No numeric metric recorded for this model.
        {notMeasured?.reason ? ` ${notMeasured.reason}` : ""}
      </p>
    );
  }
  const bothZero = Boolean(
    (achieved as Record<string, unknown>).both_scored_zero
    ?? ((achieved as Record<string, number>).teacher_metric === 0
        && (achieved as Record<string, number>).student_metric === 0
        && "teacher_metric" in (achieved as object)));
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
        {bothZero && (
          <tr className="metric-companion">
            <td colSpan={3} className="muted small">
              teacher and student both score 0 on the fidelity slice, so <code>metric_delta</code> is 0 by
              cancellation — not because encryption was free. That slice is a later window than the one the
              teacher's headline metric is computed over, so the two are not comparable.
            </td>
          </tr>
        )}
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


/** Run one published model in the browser, or say precisely why it cannot be run.
 *
 * Every reason below is a different fact about the model and they must not be collapsed into a
 * single "unavailable": a model nobody trained, a model whose weights are licence-blocked, and a
 * model that is published but ships no held-out sample are three separate states, and flattening
 * them would hide which one applies.
 */
function ModelRunner({ usecase, spec }: { usecase: string; spec: ModelSpec }) {
  const [res, setRes] = useState<ModelEvalResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [ran, setRan] = useState(false);

  const fetchable = browserModels(usecase).some((m) => m.spec.key === spec.key);
  const blocked =
    !spec.achieved ? "never trained in this build, so there is nothing to run - the baseline above is the target, not a result"
    : !spec.hf_repo_id ? "deliberately unpublished: the source data's licence does not clearly permit redistributing derived weights"
    : !spec.hf_revision ? "published but not pinned to a commit, and the browser will not fetch an unpinned model"
    : !fetchable ? "published, but it ships no browser-runnable ONNX graph at that revision"
    : null;

  const run = async () => {
    setBusy(true);
    setRes(await evaluatePublishedModel(usecase, undefined, spec.key));
    setRan(true);
    setBusy(false);
  };

  if (blocked) return <p className="muted small"><em>Cannot be run here: {blocked}.</em></p>;

  const isForecast = res?.task === "forecast";
  const fmt = (v: number | null) =>
    v == null ? "\u2014" : isForecast ? v.toFixed(4) : `${(v * 100).toFixed(1)}%`;

  return (
    <div style={{ marginTop: 6 }}>
      <button className="btn ghost" onClick={run} disabled={busy} style={{ fontSize: 12 }}>
        {busy ? "Running\u2026" : "\u25b6 Run this model in your browser"}
      </button>
      {ran && res === null && (
        <p className="muted small">
          Published and pinned, but it ships no <code>eval_sample.json</code> at that revision, so
          there is no held-out sample to score it on. Reported rather than passed off as a success.
        </p>
      )}
      {res?.error && (
        <p className="small" style={{ color: "var(--warn)" }}>{res.error}</p>
      )}
      {res?.ran && (
        <p className="muted small">
          {isForecast ? "forecast MAE" : "accuracy"} measured here <b>{fmt(res.score)}</b> vs expected{" "}
          <b>{fmt(res.expected)}</b> on {res.n} held-out rows ({res.backend}
          {res.bytes == null ? ", cached" : `, ${(res.bytes / 1e6).toFixed(1)} MB`}).{" "}
          {res.matchesExpected
            ? "Reproduced - this is the published model running here."
            : "MISMATCH - the figure is withheld rather than shown."}
        </p>
      )}
    </div>
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
          <ModelRunner usecase={usecase} spec={spec} />
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
