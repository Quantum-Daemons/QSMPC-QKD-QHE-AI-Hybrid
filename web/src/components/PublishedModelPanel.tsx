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
import { evaluatePublishedModel, type ModelEvalResult } from "../lib/onnxInference";

export default function PublishedModelPanel({ usecase }: { usecase: string }) {
  const [res, setRes] = useState<ModelEvalResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [attempted, setAttempted] = useState(false);

  useEffect(() => { setRes(null); setAttempted(false); }, [usecase]);

  const run = async () => {
    setBusy(true);
    setRes(await evaluatePublishedModel(usecase));
    setAttempted(true);
    setBusy(false);
  };

  const pct = (v: number | null) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h3>Published model · verified in your browser</h3>
      <p className="muted small" style={{ marginTop: 0 }}>
        Downloads the model published to Hugging Face at a pinned commit and evaluates it here, on
        held-out rows from the corpus it was trained on. The accuracy is then compared against a
        figure measured offline with onnxruntime on <b>those same rows</b> — if they disagree, this
        panel says so instead of showing the number.
      </p>

      <div className="btn-row" style={{ marginTop: 10 }}>
        <button className="btn" onClick={run} disabled={busy}>
          {busy ? "Running…" : "▶ Run published model"}
        </button>
      </div>

      {attempted && res === null && (
        <p className="muted small" style={{ marginTop: 10 }}>
          No model-native sample is published for this use case yet, so there is nothing to verify
          here. The pipeline above still runs its own bundled distilled model.
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
                <td>accuracy measured here</td>
                <td className="num">{pct(res.accuracy)}</td>
                <td className="muted small">{res.n} held-out rows, in this tab</td>
              </tr>
              <tr>
                <td>expected (offline, same rows)</td>
                <td className="num">{pct(res.expected)}</td>
                <td className="muted small">onnxruntime, measured by scripts/prep_model_inputs.py</td>
              </tr>
              <tr className="metric-companion">
                <td>majority-class baseline</td>
                <td className="num">{pct(res.majorityBaseline)}</td>
                <td className="muted small">what guessing the commonest label would score</td>
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

          {res.matchesExpected ? (
            <div className="note" style={{ marginTop: 10 }}>
              ✓ The browser reproduced the offline figure on the same rows, so this really is the
              published model running here — not a bundled stand-in.
            </div>
          ) : (
            <div className="note" style={{ marginTop: 10, borderColor: "var(--warn)", color: "var(--warn)" }}>
              ⚠ The browser got {pct(res.accuracy)} where {pct(res.expected)} was expected on these
              rows. Something differs — preprocessing, quantisation, or the execution provider — so
              treat this number as unverified rather than as a result.
            </div>
          )}

          <p className="muted" style={{ fontSize: 11, marginTop: 10 }}>{res.provenance}</p>
        </>
      )}
    </div>
  );
}
