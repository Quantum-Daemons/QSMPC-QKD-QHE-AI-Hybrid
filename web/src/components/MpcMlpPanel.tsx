// A published finance MLP under MPC, on one held-out row (Round 86). Every number comes from the run; the caveats are part of the result.
import { modelsFor } from "../lib/config";
import type { MpcMlpReport } from "../lib/clientCompute";
import StageBadge from "./StageBadge";

const mb = (b: number) => `${(b / 1e6).toFixed(2)} MB`;
const s = (ms: number) => `${(ms / 1000).toFixed(1)} s`;

export default function MpcMlpPanel({ report }: { report: MpcMlpReport }) {
  const r = report.result;
  const name = (c: number) => report.classes[String(c)] ?? String(c);
  const models = modelsFor("finance_aml");
  const f1 = (key: string) => {
    const a = models.find((m) => m.key === key)?.achieved as Record<string, unknown> | null | undefined;
    const v = a?.minority_class_f1 ?? a?.student_metric;
    return typeof v === "number" ? v.toFixed(3) : "-";
  };
  const f1b = (key: string) => {
    const a = models.find((m) => m.key === key)?.achieved as { xgboost_same_features?: { minority_class_f1?: number } } | null | undefined;
    const v = a?.xgboost_same_features?.minority_class_f1;
    return typeof v === "number" ? v.toFixed(3) : "-";
  };
  const rows: [string, string][] = [
    ["ReLUs in garbled circuits", `${r.reluCount}, in ${r.batches} batches, ${r.andGates.toLocaleString()} AND gates with the argmax`],
    ["Garbled-circuit traffic", `${mb(r.gcBytes)} in ${r.gcMessages.toLocaleString()} messages between the two notional parties`],
    ["CKKS", `${r.rotations} rotations and ${r.conjugations} conjugations; ${mb(r.ckksBytes)} of ciphertexts, ${mb(r.keyBytes)} of keys`],
    ["Wall-clock", `${s(r.totalMs)} in ${report.ranIn}: garbled circuits ${s(r.gcMs)} (compiling them ${s(r.gcCompileMs)}), CKKS ${s(r.ckksMs)}`],
    ["Masks", `reals uniform on +-2^${r.maskBits.dense} on the hidden layers, +-2^${r.maskBits.logit} on the logits`],
    ["Statistical distance of everything masked, this run", `at most 2^${r.statisticalDistanceLog2.toFixed(1)}`],
    ["Largest error against the clear network's logits", r.logitMaxAbsError.toExponential(2)],
    ["Logits the trainer recorded for this row", `${report.expectedLogits.map((v) => v.toFixed(3)).join(", ")}; reconstructed here ${r.logits.map((v) => v.toFixed(3)).join(", ")}`],
    ["Minority-class F1 over the whole held-out window", `this model ${f1(report.key)}; the same features through XGBoost ${f1b(report.key)}; on the 32 per-account features, the CKKS linear student ${f1("finance_aml.student")} and the gradient-boosting teacher ${f1("finance_aml.gbm")}`],
  ];
  return (
    <div className="card" style={{ marginTop: 14 }} id="mpc-mlp">
      <h3>The published MLP under MPC <StageBadge kind="notional" /></h3>
      <p style={{ marginTop: 4 }}>
        <code>{report.key}</code> on one held-out row of AMLworld&apos;s test window, index {report.rowIndex} of {report.rowCount}, labelled <b>{name(report.rowLabel)}</b>, with no
        activation in either party&apos;s hands. The client scales the row in the clear and encrypts it. CKKS evaluates each dense layer and the server masks every slot before the client
        decrypts; garbled circuits apply the ReLUs and the argmax on additive shares. The row&apos;s {r.reluCount ? report.nFeatures : ""} features include per-transaction windows that see transactions up to the end of its batch of 2,048, as the Graph Feature Preprocessor paper processes them.
      </p>
      <p style={{ margin: "6px 0 10px" }}>
        Class from the 2PC argmax: <b>{name(r.prediction)}</b>.{" "}
        <span className={`badge-state ${r.matchesClear ? "done" : "error"}`}>{r.matchesClear ? "matches the clear network" : "DISAGREES with the clear network"}</span>
      </p>
      <table className="table">
        <tbody>
          {rows.map(([k, v]) => <tr key={k}><th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{k}</th><td>{v}</td></tr>)}
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 10 }}>
        Both parties run in this tab, so the separation is notional, and only that lets the page compare the result with the clear network. The protocol is semi-honest, and the masks are
        bounded reals, which hide a value only statistically; the bound above is this project&apos;s own, for this run. AMLworld is synthetic by IBM&apos;s own statement. The published figures on this corpus reach higher (63.23 for GFP + XGBoost, 68.16 for Multi-PNA+EU, 63.72 for a GNN also evaluated under CKKS); this model has no cycle enumeration or message passing.
      </p>
    </div>
  );
}
