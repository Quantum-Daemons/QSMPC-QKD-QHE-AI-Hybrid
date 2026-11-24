// The satellite QCNNs on real telemetry, and their tree network under CKKS (Round 86; the owner's choices of 2026-11-24:
// both circuits on this page, OPS-SAT first, and the encrypted tree network run here). Figures come from the records
// through web/public/data/satellite/satellite_qcnn_<corpus>.json (scripts/export_satellite_qcnn.py); the clear readouts
// and the CKKS run are computed in this tab.
import { useEffect, useMemo, useState } from "react";
import StageBadge from "./StageBadge";
import { clearReadout, treeReadoutCkksOffThread, type TreeCkksRun, type TreeNetwork } from "../lib/clientQcnnTreeCkks";
import { qcnnInferCkks, type QcnnCkksResult } from "../lib/clientQcnnCkks";
import { qcnnInferTrained, type QcnnWeights } from "../lib/quantumSim";

type Three = { average_precision: number; minority_f1: number; balanced_accuracy: number };
type Spread = Record<keyof Three, { mean: number; sd: number; min: number; max: number }>;
interface CorpusFile {
  corpus: "opssat" | "esa"; title: string; features: string[]; weights: QcnnWeights & { scale: number; bias: number;
    feature_mean: number[]; feature_std: number[]; encoding_scale?: number; n_qubits: number };
  tree: { maps: number[][][][]; readout: number[] };
  rows_about: string; rows: number[][]; labels: number[];
  figures: { qcnn_seed0: Three; qcnn_spread: Spread; logistic_regression: Three; classical_surrogate: Three;
    reference_rows?: Record<string, { auc_pr: number | null }> | null; note?: string | null; test_rows: number; test_positive: number;
    trainer_rule_minority_f1: number };
  ckks_record: { rows: number; blocks: number; per_row: { tree_ms: number; tree_bytes_up: number; pauli_ms: number; pauli_bytes_up: number };
    tree_max_abs_readout_error: number; tree_decisions_differing: number; pauli_rows: number };
  onnx_record: { bytes: number; max_abs_readout_difference: number; decisions_differing: number; test_rows: number };
  licence: string; licence_url: string; attribution: string; modifications: string;
}

const f3 = (v: number) => v.toFixed(3);
const kb = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(1)} kB`);

export default function SatelliteQcnnCard() {
  const [corpus, setCorpus] = useState<"opssat" | "esa">("opssat");
  const [file, setFile] = useState<CorpusFile | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [run, setRun] = useState<{ run: TreeCkksRun; ranIn: string; ms: number } | null>(null);
  const [one, setOne] = useState<QcnnCkksResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    setFile(null); setRun(null); setOne(null); setError(null);
    fetch(`${import.meta.env.BASE_URL}data/satellite/satellite_qcnn_${corpus}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j: CorpusFile) => { if (live) setFile(j); })
      .catch((e: unknown) => { if (live) setFailed(String(e)); });
    return () => { live = false; };
  }, [corpus]);
  const net: TreeNetwork | null = useMemo(() => (file ? {
    maps: file.tree.maps, readout: file.tree.readout, scale: file.weights.scale, bias: file.weights.bias,
    feature_mean: file.weights.feature_mean, feature_std: file.weights.feature_std,
    encoding_scale: file.weights.encoding_scale ?? 1, n_qubits: file.weights.n_qubits,
  } : null), [file]);
  const clear = useMemo(() => (file && net ? clearReadout(net, file.rows) : null), [file, net]);
  const counts = useMemo(() => {
    if (!file || !net || !clear) return null;
    let tp = 0, fp = 0, fn = 0, tn = 0;
    clear.forEach((r, i) => {
      const pos = r * net.scale + net.bias > 0, y = file.labels[i] === 1;
      if (pos && y) tp++; else if (pos) fp++; else if (y) fn++; else tn++;
    });
    return { tp, fp, fn, tn };
  }, [file, net, clear]);
  const stateVectorCheck = useMemo(() => {
    if (!file || !clear) return null;
    const sv = qcnnInferTrained(file.rows[0], file.weights);
    return Math.abs(sv.expectation - clear[0]);
  }, [file, clear]);

  if (failed) return <div className="card" style={{ marginTop: 14 }}><h3>A QCNN on real telemetry</h3><p className="small" style={{ color: "var(--danger)" }}>Its data did not load: {failed}.</p></div>;
  if (!file || !net || !clear || !counts) return null;
  const fig = file.figures;
  const worst = run ? Math.max(...Array.from(run.run.readouts, (v, i) => Math.abs(v - clear[i]))) : null;
  const flips = run ? Array.from(run.run.readouts).filter((v, i) => (v * net.scale + net.bias > 0) !== (clear[i] * net.scale + net.bias > 0)).length : null;

  const runTree = async () => {
    setBusy("tree"); setError(null);
    try {
      const t0 = performance.now();
      const out = await treeReadoutCkksOffThread(net, file.rows);
      setRun({ ...out, ms: performance.now() - t0 });
    } catch (e) { setError(`The encrypted run failed ${String((e as Error).message ?? e)}`); } finally { setBusy(null); }
  };
  const runOne = async () => {
    setBusy("one"); setError(null);
    try { setOne(await qcnnInferCkks(file.weights, file.rows[0])); } catch (e) { setError(String(e)); } finally { setBusy(null); }
  };

  return (
    <div className="card" id="sat-qcnn" style={{ marginTop: 14 }}>
      <h3>A QCNN on real telemetry, and its tree network under CKKS <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        An 8-wire tree QCNN, trained on real satellite telemetry as a pre-registration fixed before any run
        (<code>references/satellite-qcnn-tn-preregistration.md</code>). Its circuit is a tree tensor network, simulated
        here classically and exactly; it is not quantum hardware, and nothing here is a claim of quantum advantage.
      </p>
      <label className="small">
        Corpus:{" "}
        <select value={corpus} onChange={(e) => setCorpus(e.target.value as "opssat" | "esa")}>
          <option value="opssat">OPS-SAT-AD (segments)</option>
          <option value="esa">ESA-ADB Mission1 (32-minute windows)</option>
        </select>
      </label>
      <div style={{ overflowX: "auto" }}>
        <table className="table small" style={{ marginTop: 8 }}>
          <thead><tr><th>{file.title}: {fig.test_rows.toLocaleString("en-US")} test rows, {fig.test_positive.toLocaleString("en-US")} anomalous</th>
            <th>Average precision</th><th>Minority F1 at the validation threshold</th><th>Balanced accuracy</th></tr></thead>
          <tbody>
            <tr><td>This QCNN (seed 0)</td><td>{f3(fig.qcnn_seed0.average_precision)}</td><td>{f3(fig.qcnn_seed0.minority_f1)}</td><td>{f3(fig.qcnn_seed0.balanced_accuracy)}</td></tr>
            <tr><td>QCNN, five seeds (mean and sd)</td>
              {(["average_precision", "minority_f1", "balanced_accuracy"] as const).map((k) => (
                <td key={k}>{f3(fig.qcnn_spread[k].mean)} &plusmn; {f3(fig.qcnn_spread[k].sd)}</td>))}</tr>
            <tr><td>Logistic regression, the same eight inputs</td><td>{f3(fig.logistic_regression.average_precision)}</td><td>{f3(fig.logistic_regression.minority_f1)}</td><td>{f3(fig.logistic_regression.balanced_accuracy)}</td></tr>
            <tr><td><b>Classical surrogate</b>: logistic regression on the circuit&apos;s own features</td><td><b>{f3(fig.classical_surrogate.average_precision)}</b></td><td>{f3(fig.classical_surrogate.minority_f1)}</td><td>{f3(fig.classical_surrogate.balanced_accuracy)}</td></tr>
          </tbody>
        </table>
      </div>
      <p className="small" style={{ margin: "4px 0 0" }}>
        <b>The classical surrogate leads</b> on average precision: a classical model with the circuit&apos;s own
        hypothesis class beats the trained circuit, which is ahead of plain logistic regression on the same inputs.
        {fig.reference_rows ? <> The shipped OPS-SAT models read all nineteen features (average precision {Object.values(fig.reference_rows).map((r) => (r.auc_pr ?? 0).toFixed(3)).join(" and ")}): not like for like.</> : null}
        {fig.note ? <> {fig.note} At its own rule, a positive logit, seed 0 scores a minority F1 of {f3(fig.trainer_rule_minority_f1)}, reported, not chosen.</> : null}
      </p>
      <p className="small" style={{ margin: "8px 0 0" }}>
        In this tab, on {file.rows_about}: the tree contraction gives {counts.tp} true and {counts.fp} false alarms, {counts.fn} missed
        and {counts.tn} correctly quiet, at the circuit&apos;s own threshold. On the first row it equals the state vector
        to {stateVectorCheck != null ? stateVectorCheck.toExponential(1) : "-"}.
      </p>
      <div className="btn-row" style={{ marginTop: 10 }}>
        <button className="btn" disabled={busy !== null} onClick={runTree}>
          {busy === "tree" ? "Running under CKKS..." : `Run all ${file.rows.length.toLocaleString("en-US")} rows under CKKS, one row a slot`}
        </button>
        <button className="btn ghost" disabled={busy !== null} onClick={runOne}>
          {busy === "one" ? "Running..." : "Run the first row the per-row way"}
        </button>
      </div>
      {error ? <p className="small" style={{ color: "var(--danger)" }}>{error}</p> : null}
      {run && worst != null ? (
        <p className="small" style={{ margin: "8px 0 0" }}>
          {run.run.rows.toLocaleString("en-US")} rows in {run.run.blocks} block(s), in {run.ranIn}: the largest difference between a
          decrypted readout and the clear one is {worst.toExponential(2)}, and {flips} decision(s) differ. Sent{" "}
          {kb(run.run.upBytes)} ({run.run.ciphertextsUp} ciphertexts), received {kb(run.run.downBytes)}, keys {kb(run.run.keyBytes)};
          keys {Math.round(run.run.keyMs)} ms, encryption {Math.round(run.run.encryptMs)} ms, evaluation {Math.round(run.run.evaluateMs)} ms,
          decryption {Math.round(run.run.decryptMs)} ms.
        </p>
      ) : null}
      {one ? (
        <p className="small" style={{ margin: "6px 0 0" }}>
          The first row the per-row way (one inner product over {one.nTerms.toLocaleString("en-US")} products): error {one.maxError.toExponential(2)},
          {one.matchesClear ? " the same class" : " a different class"}; sent {kb(one.upBytes)} for this one row, in {Math.round(one.totalMs)} ms.
        </p>
      ) : null}
      <p className="muted small" style={{ margin: "8px 0 0" }}>
        Recorded on node over {file.ckks_record.rows.toLocaleString("en-US")} test rows: largest error {file.ckks_record.tree_max_abs_readout_error.toExponential(1)},
        {" "}{file.ckks_record.tree_decisions_differing} decisions changed, {file.ckks_record.per_row.tree_ms.toFixed(2)} ms and {kb(file.ckks_record.per_row.tree_bytes_up)} a row
        one row a slot, against {file.ckks_record.per_row.pauli_ms.toFixed(0)} ms and {kb(file.ckks_record.per_row.pauli_bytes_up)} a row the per-row way
        ({file.ckks_record.pauli_rows} rows). CKKS with N = 16384 and 360 modulus bits, inside the HomomorphicEncryption.org
        standard&apos;s post-quantum bound; the weights are public and the key stays in this tab. The same contraction as an ONNX
        graph ({kb(file.onnx_record.bytes)}) equals the state vector on all {file.onnx_record.test_rows.toLocaleString("en-US")} test rows to{" "}
        {file.onnx_record.max_abs_readout_difference.toExponential(1)}. Data: {file.attribution}; licence{" "}
        <a href={file.licence_url} target="_blank" rel="noreferrer">{file.licence}</a>; {file.modifications}.
      </p>
    </div>
  );
}
