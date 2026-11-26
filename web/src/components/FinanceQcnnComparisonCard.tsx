// The finance QCNN on AMLworld against quantum kernels and classical comparators (Round 86, batch 6; the owner's choices
// of 2026-11-26), pre-registered in references/finance-qcnn-amlworld-comparison-preregistration.md. Its criterion 2: the
// page states every arm's figure with its interval and the prevalence, says the circuit is classically simulable with no
// quantum advantage claimed, and says which comparators score above it, whatever ships. Every figure is read from
// web/public/data/finance/qcnn_comparison.json, which scripts/train/finance_qcnn_comparison.py --record writes.
import { useEffect, useState } from "react";
import StageBadge from "./StageBadge";

interface Row { key: string; label: string; kind: string; test_ap: number; ap_ci95: [number, number]; shipped_minus_this_ci95: [number, number] | null }
interface Comparison {
  preregistration: string; record: string; test_rows: number; test_positive: number; prevalence: number; twice_prevalence: number;
  rows: Row[]; spreads: Record<string, { mean_test_ap: number; sd_test_ap: number; seeds: number }>;
  shipped_seed: { seed: number; test_ap: number; minority_f1: number }; criterion_1_signal: boolean;
  comparators_above_the_shipped_seed: string[]; schmidt: { largest: number; bound: number; rows: number } | null;
  statement: string; licence: string; licence_url: string; attribution: string; modifications: string;
}

const f4 = (v: number) => v.toFixed(4);

export default function FinanceQcnnComparisonCard() {
  const [d, setD] = useState<Comparison | null>(null);
  useEffect(() => {
    let on = true;
    fetch(`${import.meta.env.BASE_URL}data/finance/qcnn_comparison.json`)
      .then((r) => ((r.headers.get("content-type") ?? "").includes("application/json") ? r.json() : null))
      .then((j: Comparison | null) => { if (on) setD(j); })
      .catch(() => { if (on) setD(null); });
    return () => { on = false; };
  }, []);
  if (!d) return null;
  const rows = [...d.rows].sort((a, b) => b.test_ap - a.test_ap);
  return (
    <div className="card" id="finance-qcnn-comparison" style={{ marginTop: 14 }}>
      <h3>The QCNN on AMLworld, against quantum kernels and classical models <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        A pre-registered comparison (<code>{d.preregistration}</code>) on the same eight AMLworld inputs and the same fit
        sample, scored on the {d.test_rows.toLocaleString("en-US")} transactions of the test window, {d.test_positive.toLocaleString("en-US")} of
        them laundering: a prevalence of {(d.prevalence * 100).toFixed(3)} percent, so an average precision of {f4(d.prevalence)} is
        what a random ranking scores. {d.statement}
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table small" style={{ marginTop: 8 }}>
          <thead><tr><th>Model</th><th>Kind</th><th>Test average precision (95% interval)</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key}><td>{r.key === "qcnn1_shipped" ? <b>{r.label}</b> : r.label}</td><td>{r.kind}</td>
                <td>{f4(r.test_ap)} ({f4(r.ap_ci95[0])} to {f4(r.ap_ci95[1])})</td></tr>))}
          </tbody>
        </table>
      </div>
      <p className="small" style={{ margin: "6px 0 0" }}>
        Across seeds: {Object.entries(d.spreads).map(([k, s], i) => (
          <span key={k}>{i ? "; " : ""}{k === "qcnn1" ? "QCNN-1" : k === "qcnn2" ? "QCNN-2" : "without entanglement"} {f4(s.mean_test_ap)} &plusmn; {f4(s.sd_test_ap)} over {s.seeds}</span>))}.
        The seed best on validation, seed {d.shipped_seed.seed}, scores {f4(d.shipped_seed.test_ap)}: {d.criterion_1_signal ? "at least" : "under"} twice
        the prevalence ({f4(d.twice_prevalence)}).{" "}
        {d.comparators_above_the_shipped_seed.length
          ? <>Above it: {d.comparators_above_the_shipped_seed.join(", ")}.</>
          : <>No comparator scores above it.</>}
        {d.schmidt ? <> Its largest Schmidt rank over {d.schmidt.rows.toLocaleString("en-US")} test rows, every cut and every layer, traced in double
          precision, is {d.schmidt.largest} of the {d.schmidt.bound} an eight-wire state can reach.</> : null}
      </p>
      <p className="muted small" style={{ margin: "8px 0 0" }}>
        Record: <code>{d.record}</code>. Data: {d.attribution} Licence <a href={d.licence_url} target="_blank" rel="noreferrer">{d.licence}</a>; {d.modifications}
      </p>
    </div>
  );
}
