// Runs the PUBLISHED Hugging Face model in the browser, on a held-out sample from the corpus it was
// actually trained on, and checks the answer against a figure measured offline on those same rows.
//
// This panel exists because the browser tier never worked. A 401 on the private HF repos masked a
// deeper problem: each page bundled a sample from a *different* dataset than its published model was
// trained on -- PaySim 8 features against a 32-feature AMLworld model, Wisconsin WDBC tabular rows
// against a BreastMNIST image model. So the fetch could not have succeeded even public.
//
// The check is the point. plan18's audit rule is that "a result labelled MEASURED that produced no
// network or compute activity is the most serious defect class in this project". A number the page
// can verify against an independently-measured expectation is the opposite of that.
import { useEffect, useState } from "react";
import { browserModel, evaluatePublishedModel, type ModelEvalResult } from "../lib/onnxInference";

export default function PublishedModelPanel({ usecase }: { usecase: string }) {
  const [res, setRes] = useState<ModelEvalResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [attempted, setAttempted] = useState(false);

  // Whether anything is fetchable is known from the registry before any click, so say it up front
  // rather than offering a button whose only possible outcome is "there was nothing to verify".
  // Genomics is the standing case: `genomics.prs` is deliberately unpublished pending the per-score
  // licence audit, so its button could never do anything.
  const target = browserModel(usecase);

  useEffect(() => { setRes(null); setAttempted(false); }, [usecase]);

  const run = async () => {
    setBusy(true);
    setRes(await evaluatePublishedModel(usecase));
    setAttempted(true);
    setBusy(false);
  };

  const isForecast = res?.task === "forecast";
  // A classifier is read as a percentage and higher is better; a forecaster is an absolute error
  // and lower is better. Showing both the same way would invite exactly the wrong reading.
  const fmt = (v: number | null) =>
    v == null ? "—" : isForecast ? v.toFixed(4) : `${(v * 100).toFixed(1)}%`;
  const scoreLabel = isForecast ? "forecast MAE measured here" : "accuracy measured here";
  const baselineLabel = isForecast
    ? "persistence baseline (predict no change)"
    : "majority-class baseline";

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h3>Published model · verified in your browser</h3>
      <p className="muted small" style={{ marginTop: 0 }}>
        Downloads the model published to Hugging Face at a pinned commit and evaluates it here, on
        held-out rows from the corpus it was trained on. The accuracy is then compared against a
        figure measured offline with onnxruntime on <b>those same rows</b> — if they disagree, this
        panel says so instead of showing the number.
      </p>

      {target ? (
        <div className="btn-row" style={{ marginTop: 10 }}>
          <button className="btn" onClick={run} disabled={busy}>
            {busy ? "Running…" : "▶ Run published model"}
          </button>
        </div>
      ) : (
        <p className="muted small" style={{ marginTop: 10 }}>
          <b>Nothing to verify on this page.</b> No model for this use case is published at a pinned
          revision, so there is no artefact to fetch and no offline figure to check it against. That
          is a deliberate state, not a gap waiting to be filled: publication is withheld where the
          source data's licence does not clearly permit redistributing derived weights. The pipeline
          above still runs its own bundled model, and its metrics come from that.
        </p>
      )}

      {attempted && res === null && (
        <p className="muted small" style={{ marginTop: 10 }}>
          The model is published, but it ships no <code>eval_sample.json</code> at that revision, so
          there is no held-out sample to score it on. Reported rather than passed off as a success.
        </p>
      )}

      {res && res.error && (
        <div className="note" style={{ marginTop: 10, borderColor: "var(--warn)", color: "var(--warn)" }}>
          ⚠ The published model did not run: <code>{res.error}</code>
          <div style={{ marginTop: 6 }}>
            Reported rather than silently replaced. The pipeline's own bundled model is unaffected.
          </div>
        </div>
      )}

      {res?.ran && (
        <>
          <table className="mini-table" style={{ marginTop: 12 }}>
            <tbody>
              <tr className={res.matchesExpected ? "metric-headline" : undefined}>
                <td>{scoreLabel}</td>
                <td className="num">{fmt(res.score)}</td>
                <td className="muted small">{res.n} held-out {isForecast ? "windows" : "rows"}, in this tab</td>
              </tr>
              <tr>
                <td>expected (offline, same {isForecast ? "windows" : "rows"})</td>
                <td className="num">{fmt(res.expected)}</td>
                <td className="muted small">onnxruntime, measured by scripts/prep_model_inputs.py</td>
              </tr>
              <tr className="metric-companion">
                <td>{baselineLabel}</td>
                <td className="num">{fmt(res.baseline)}</td>
                <td className="muted small">
                  {isForecast ? "error from simply repeating the last observed step"
                              : "what guessing the commonest label would score"}
                </td>
              </tr>
              <tr className="metric-companion">
                <td>execution provider</td>
                <td className="num">{res.backend}</td>
                <td className="muted small">
                  {res.bytes == null ? "served from cache" : `${(res.bytes / 1e6).toFixed(1)} MB transferred`}
                </td>
              </tr>
            </tbody>
          </table>

          {!res.beatsBaseline && (
            <div className="note" style={{ marginTop: 10, borderColor: "var(--warn)", color: "var(--warn)" }}>
              ⚠ <b>The model does not beat the trivial baseline</b> on these {isForecast ? "windows" : "rows"}
              {isForecast && res.score != null && res.baseline != null
                ? ` — its forecast error (${res.score.toFixed(4)}) is about `
                  + `${(res.score / res.baseline).toFixed(0)}× that of simply repeating the last `
                  + "observed value. Telemanom derives its anomaly signal from forecast residuals, so "
                  + "residuals this large are dominated by model error rather than by anomalies. The "
                  + "published event-wise F0.5 should be read with that in mind."
                : "."} Shown because it is measurable here, not because it flatters the model.
            </div>
          )}
          {res.matchesExpected ? (
            <div className="note" style={{ marginTop: 10 }}>
              ✓ The browser reproduced the offline figure on the same rows, so this really is the
              published model running here — not a bundled stand-in.
            </div>
          ) : (
            <div className="note" style={{ marginTop: 10, borderColor: "var(--warn)", color: "var(--warn)" }}>
              ⚠ The browser got {fmt(res.score)} where {fmt(res.expected)} was expected on these
              {isForecast ? " windows" : " rows"}. Something differs — preprocessing, quantisation, or
              the execution provider — so treat this number as unverified rather than as a result.
            </div>
          )}

          <p className="muted" style={{ fontSize: 11, marginTop: 10 }}>{res.provenance}</p>
        </>
      )}
    </div>
  );
}
