// The medical CNN student under MPC, on one held-out image (Round 86, the owner's request of 2026-11-07).
//
// Every number here comes from the run. The self-test is only possible because both parties are in this tab, and the panel says so; the caveats are part of
// the result, not a footnote to it, so they are not folded away.
import { CnnOcclusion } from "./ModelExplanations";
import StageBadge from "./StageBadge";
import type { MpcCnnReport } from "../lib/clientCompute";

const mb = (b: number) => `${(b / 1e6).toFixed(1)} MB`;
const s = (ms: number) => `${(ms / 1000).toFixed(1)} s`;

export default function MpcCnnPanel({ report }: { report: MpcCnnReport }) {
  const r = report.result;
  const name = (c: number) => report.classes?.[String(c)] ?? String(c);
  const rows: [string, string][] = [
    ["ReLUs in garbled circuits", `${r.reluCount} (the convolution's and the first dense layer's), in ${r.batches} batches, ${r.andGates.toLocaleString()} AND gates with the argmax`],
    ["Garbled-circuit traffic", `${mb(r.gcBytes)} in ${r.gcMessages.toLocaleString()} messages between the two notional parties`],
    ["CKKS", `${r.rotations} rotations and ${r.conjugations} conjugations; ${mb(r.ckksBytes)} of ciphertexts, ${mb(r.keyBytes)} of keys`],
    ["Wall-clock", `${s(r.totalMs)} in ${report.ranIn}: garbled circuits ${s(r.gcMs)} (compiling them ${s(r.gcCompileMs)}), CKKS ${s(r.ckksMs)}`],
    ["Masks", `reals uniform on +-2^${r.maskBits.conv} on the convolution and first dense layer, +-2^${r.maskBits.logit} on the logits`],
    ["Statistical distance of everything masked, this run", `at most 2^${r.statisticalDistanceLog2.toFixed(1)}`],
    ["Largest logit error against the clear network", r.logitMaxAbsError.toExponential(2)],
  ];
  return (
    <div className="card" style={{ marginTop: 14 }} id="mpc-cnn">
      <h3>The CNN student under MPC <StageBadge kind="notional" /></h3>
      <p style={{ marginTop: 4 }}>
        One held-out BreastMNIST test image, index {report.imageIndex} of {report.imageCount}, labelled <b>{name(report.imageLabel)}</b>, through the published student
        with no activation in either party&apos;s hands. CKKS evaluates the convolution and both dense layers, and the server masks every slot before the client decrypts; garbled
        circuits apply the ReLUs and the argmax on additive shares; average pooling runs on each party&apos;s own share.
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
      {report.explanation && <CnnOcclusion e={report.explanation} className={name(report.explanation.predicted)} />}
      <p className="muted small" style={{ marginTop: 10 }}>
        Both parties run in this tab, so the separation is notional, and only that lets the page compare the result with the clear network. The protocol is semi-honest: the garbled circuits are
        designed against malicious parties, but the CKKS half and the shares are not authenticated, and a malicious client can extract weights from protocols of this shape (MUSE, USENIX Security 2021). The
        masks are bounded reals, which hide a value only statistically: Gazelle and the 2025-26 CKKS hybrids (BLB, EncFormer) mask over the plaintext ring instead, and their security theorems do not cover this. The bound above is
        this project&apos;s own, for this run. The class is the student&apos;s, on an image MedMNIST states is not for clinical use: it is not a diagnosis.
      </p>
    </div>
  );
}
