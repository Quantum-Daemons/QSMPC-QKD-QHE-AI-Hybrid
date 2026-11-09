// The trained QCNN's class with its input under CKKS (Round 86). Every number comes from the run or from the weights file, and the caveats are part of the result.
import type { QcnnCkksReport } from "../lib/clientCompute";
import StageBadge from "./StageBadge";

const mb = (b: number) => `${(b / 1e6).toFixed(2)} MB`;
const ms = (v: number) => `${v.toFixed(0)} ms`;
const f = (v: number | null) => (v == null ? "n/a" : v.toFixed(4));

export default function QcnnCkksPanel({ report }: { report: QcnnCkksReport }) {
  const r = report.result;
  const name = (c: number) => report.classNames?.[c] ?? String(c);
  const { label, qcnn, classical, surrogate, surrogateFeatures, baseline } = report.headline;
  const verdict = qcnn == null || classical == null ? "" : qcnn > classical ? "The circuit is ahead on this split." : qcnn === classical ? "The two are level on this split." : "The classical model is ahead on this split.";
  const rows: [string, string][] = [
    ["Features the client encrypts", `${r.nTerms.toLocaleString()} products of sines and cosines of its ${r.nFeatures} encoded angles (3^${r.nFeatures}), in ${r.nCiphertexts} ciphertext${r.nCiphertexts > 1 ? "s" : ""}`],
    ["Evaluation under CKKS", `${r.nOutputs} inner product${r.nOutputs > 1 ? "s" : ""} with the circuit's coefficients: ${r.rotations} rotations, one multiplicative level, no relinearisation key`],
    ["Ciphertexts and keys", `${mb(r.upBytes)} up, ${mb(r.downBytes)} down, ${mb(r.keyBytes)} of public and Galois keys`],
    ["Wall-clock", `${ms(r.totalMs)}: building the form from the circuit ${ms(r.formMs)}, keys ${ms(r.keyMs)}, encrypting ${ms(r.encryptMs)}, evaluating ${ms(r.evaluateMs)}, decrypting ${ms(r.decryptMs)}`],
    ["Largest difference from the same circuit in the clear", `${r.maxError.toExponential(1)} on the decrypted readout${r.nOutputs > 1 ? "s" : ""}; the form itself, in the clear, ${r.maxIdentityError.toExponential(1)}`],
    [`${label} recorded on the held-out split`, `the trained QCNN ${f(qcnn)}; logistic regression on the same inputs and split ${f(classical)}; ${surrogate == null ? "" : `the same estimator on the same ${surrogateFeatures?.toLocaleString()} products ${f(surrogate)}; `}the constant-predictor baseline ${f(baseline)}. ${verdict}`],
  ];
  return (
    <div className="card" style={{ marginTop: 14 }} id="qcnn-ckks">
      <h3>The trained QCNN under CKKS <StageBadge kind="notional" /></h3>
      <p style={{ marginTop: 4 }}>
        The {report.qubits}-wire circuit&apos;s readout is exactly a linear form in the products of sines and cosines of its encoded angles, because the encoding is a product of real states. So the client computes
        those products from its own features and encrypts them, and the server, which holds the circuit, returns one encrypted inner product per readout. The server handles only ciphertexts.
      </p>
      <p style={{ margin: "6px 0 10px" }}>
        Class from the decrypted readout: <b>{name(r.cls)}</b>.{" "}
        <span className={`badge-state ${r.matchesClear ? "done" : "error"}`}>{r.matchesClear ? "matches the circuit in the clear" : `DISAGREES with the circuit in the clear (${name(r.clearClass)})`}</span>
      </p>
      <table className="table">
        <tbody>
          {rows.map(([k, v]) => <tr key={k}><th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{k}</th><td>{v}</td></tr>)}
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 10 }}>
        This is classical computation under RLWE, not quantum homomorphic encryption. The circuit is simulated classically, as it is when it runs in the clear, and nothing here is faster or more accurate than the
        circuit in the clear: the figures above are that circuit&apos;s. The evaluator knows the circuit, and the {r.nTerms.toLocaleString()} coefficients can be recovered from that many chosen scores, which matters only
        for a private model. Both parties run in this tab, so the separation is notional. The training-time figures come from the weights file and are not re-scored here.
      </p>
    </div>
  );
}
