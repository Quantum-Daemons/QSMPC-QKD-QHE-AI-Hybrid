// The tensor-network checks of the medical, telecom and satellite QCNNs (Round 86, batch 6; the owner's choice of
// 2026-11-26), pre-registered in references/qcnn-tensor-network-checks-preregistration.md: how much entanglement each
// trained circuit builds, against the bound its width sets, and a matrix-product-state classifier on the same inputs.
// Every figure is read from web/public/data/qcnn_tn_checks.json, which scripts/train/qcnn_tn_checks.py writes.
import { useEffect, useState } from "react";

export interface TnArtifact {
  circuit: string; wires: number; largest_schmidt_rank: number; bound: number; schmidt_rows: number;
  headline: "test_macro_f1" | "test_minority_f1"; qcnn: number; logistic_regression: number; mps: number;
  mps_bond_dimension: number; mps_average_precision: number | null;
  mps_minority_f1_at_validation_best_threshold: number | null; mps_balanced_accuracy_at_validation_best_threshold: number | null;
}
export interface TnChecks { preregistration: string; record: string; statement: string; artifacts: Record<string, TnArtifact> }

let cached: Promise<TnChecks | null> | null = null;
/** The shipped checks, fetched once a page load. */
export function loadTnChecks(): Promise<TnChecks | null> {
  cached ??= fetch(`${import.meta.env.BASE_URL}data/qcnn_tn_checks.json`)
    .then((r) => ((r.headers.get("content-type") ?? "").includes("application/json") ? r.json() : null))
    .catch(() => null);
  return cached;
}

export function useTnChecks(): TnChecks | null {
  const [d, setD] = useState<TnChecks | null>(null);
  useEffect(() => { let on = true; loadTnChecks().then((v) => { if (on) setD(v); }); return () => { on = false; }; }, []);
  return d;
}

/** The artifact a page's QCNN card shows: the weights file `loadQcnnWeights` picks for that page and width. */
export function tnKeyFor(usecase: string, qubits: number): string | null {
  if (usecase === "medical_fl") return "medical";
  if (usecase === "telecom_ids") return qubits === 7 ? "telecom_q7" : "telecom";
  return null;
}

/** The Schmidt-rank sentence, shared by the circuit card and the satellite card. */
export function schmidtSentence(a: TnArtifact): string {
  const full = a.largest_schmidt_rank === a.bound;
  return `After the encoding and after every layer, over ${a.schmidt_rows.toLocaleString("en-US")} test rows and every cut of the wire order, `
    + `the state's largest Schmidt rank is ${a.largest_schmidt_rank} of the ${a.bound} any ${a.wires}-wire state can reach`
    + (full ? ": the trained circuit builds as much entanglement as its width allows, and is still simulated exactly here." : ".");
}

export default function QcnnTnCheck({ artifactKey }: { artifactKey: string }) {
  const d = useTnChecks();
  const a = d?.artifacts[artifactKey];
  if (!d || !a) return null;
  const metric = a.headline === "test_macro_f1" ? "macro F1" : "minority-class F1";
  const rows: [string, number][] = [["this QCNN", a.qcnn], ["logistic regression", a.logistic_regression],
    [`an MPS classifier (bond dimension ${a.mps_bond_dimension})`, a.mps]];
  // Ties are named as ties: on the medical page all three score the same minority-class F1 to six places.
  const best = Math.max(...rows.map((r) => r[1]));
  const leaders = rows.filter((r) => Math.abs(r[1] - best) < 5e-7).map((r) => r[0]);
  const verdict = leaders.length === rows.length ? "All three score the same."
    : leaders.length > 1 ? `${leaders.join(" and ")} tie for the highest.`
    : leaders[0] === "this QCNN" ? "The QCNN scores highest." : `The highest is ${leaders[0]}, a classical model.`;
  return (
    <div id="qcnn-tn-check">
      <h3 style={{ marginTop: 12 }}>Classically simulable, and checked</h3>
      <p>{d.statement} {schmidtSentence(a)}</p>
      <p>
        On the same inputs and split, test {metric}: {rows.map(([name, v], i) => (
          <span key={name}>{i ? (i === rows.length - 1 ? " and " : ", ") : ""}{name} <b>{v.toFixed(4)}</b></span>))}.
        {" "}{verdict} The MPS is a tensor
        network on each input&apos;s rank among the training rows, its bond dimension chosen on the validation rows.
      </p>
      <p className="small muted">Pre-registration: <code>{d.preregistration}</code>; record: <code>{d.record}</code>.</p>
    </div>
  );
}
