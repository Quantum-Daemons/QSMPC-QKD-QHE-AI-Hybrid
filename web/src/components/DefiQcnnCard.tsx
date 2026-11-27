// The DeFi page's QCNN on Aave v3 liquidations, trained and not encrypted (Round 86, batch 8; checklist QC54, the owner's
// choices of 2026-11-26 and 2026-11-27). Pre-registered in references/defi-qcnn-aave-v3-preregistration.md: when both of
// its criteria hold, the page shows the circuit beside its comparators, with every arm's figure, its interval and the
// prevalence, the statement that it is classically simulable with no quantum advantage claimed, and which comparators
// score above it. The figures come from web/public/data/defi/qcnn_comparison.json (scripts/train/defi_qcnn_comparison.py
// record), the circuit and its sample from defi_qcnn.json (scripts/export_defi_qcnn.py); the sample's re-score is
// computed in this tab by the page's own tree contraction.
//
// The page's private check computes Aave V3's own rule, a health factor below 1; this circuit is a separate, learned
// classifier, and nothing here runs under encryption.
import { useEffect, useState } from "react";
import StageBadge from "./StageBadge";
import { clearReadout, type TreeNetwork } from "../lib/clientQcnnTreeCkks";
import { averagePrecision } from "../lib/averagePrecision";

interface Row { key: string; label: string; kind: string; validation_ap: number; test_ap: number; ap_ci95: [number, number];
  shipped_minus_this_ci95: [number, number] | null }
interface Comparison {
  preregistration: string; record: string; task: string; features: string[]; test_rows: number; test_positive: number;
  prevalence: number; twice_prevalence: number; validation_rows: number; validation_positive: number;
  validation_prevalence: number; rows: Row[];
  spreads: Record<string, { mean_test_ap: number; sd_test_ap: number; seeds: number }>;
  shipped_seed: { seed: number; test_ap: number; test_auc: number; minority_f1: number }; criterion_1_signal: boolean;
  criteria_met: boolean; comparators_above_the_shipped_seed: string[]; statement: string;
  licence: string; licence_url: string; attribution: { citation: string }; modifications: string;
}
interface Circuit {
  seed: number; rows_about: string; rows: number[][]; labels: number[];
  weights: { scale: number; bias: number; feature_mean: number[]; feature_std: number[]; encoding_scale?: number; n_qubits: number };
  tree: { maps: number[][][][]; readout: number[] };
  sample: { rows: number; positive: number; average_precision: number };
  export_check: { test_rows: number; page_vs_state_vector: { max_abs_readout_difference: number; decisions_differing: number };
                  criterion_met: boolean };
}

const f4 = (v: number) => v.toFixed(4);
const pct = (v: number) => `${(v * 100).toFixed(2)} percent`;
const ci = (r: Row) => `${f4(r.ap_ci95[0])} to ${f4(r.ap_ci95[1])}`;

export default function DefiQcnnCard() {
  const [d, setD] = useState<Comparison | null>(null);
  const [c, setC] = useState<Circuit | null>(null);
  const [rescore, setRescore] = useState<{ ap: number; ms: number } | null>(null);
  useEffect(() => {
    let on = true;
    const json = (path: string) => fetch(`${import.meta.env.BASE_URL}data/defi/${path}`)
      .then((r) => ((r.headers.get("content-type") ?? "").includes("application/json") ? r.json() : null))
      .catch(() => null);
    Promise.all([json("qcnn_comparison.json"), json("defi_qcnn.json")]).then(([a, b]) => {
      if (on) { setD(a as Comparison | null); setC(b as Circuit | null); }
    });
    return () => { on = false; };
  }, []);
  if (!d || !c || !d.criteria_met) return null;
  const rows = [...d.rows].sort((a, b) => b.test_ap - a.test_ap);
  const by = (k: string) => d.rows.find((r) => r.key === k);
  const ship = by("qcnn8_shipped"), ens = by("qcnn8_ensemble"), noent = by("noent_ensemble"), four = by("qcnn4_ensemble");
  const spread = d.spreads.qcnn8;
  const lr = by("logistic_regression");
  const fell = d.rows.filter((r) => r.test_ap < r.validation_ap);
  const rose = d.rows.filter((r) => r.test_ap >= r.validation_ap);
  const circuitsAbove = d.rows.filter((r) => r.key !== "qcnn8_shipped" && r.kind.includes("circuit")
    && r.test_ap > d.shipped_seed.test_ap);
  const run = () => {
    const net: TreeNetwork = { maps: c.tree.maps, readout: c.tree.readout, scale: c.weights.scale, bias: c.weights.bias,
      feature_mean: c.weights.feature_mean, feature_std: c.weights.feature_std,
      encoding_scale: c.weights.encoding_scale ?? 1, n_qubits: c.weights.n_qubits };
    const t0 = performance.now();
    const readout = clearReadout(net, c.rows);
    const logits = Array.from(readout, (r) => r * net.scale + net.bias);
    setRescore({ ap: averagePrecision(logits, c.labels), ms: performance.now() - t0 });
  };
  const ex = c.export_check.page_vs_state_vector;
  return (
    <div className="card" id="defi-qcnn" style={{ marginTop: 14 }}>
      <h3>A QCNN on Aave v3 liquidations, trained and not encrypted <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        This page&apos;s private check computes Aave V3&apos;s own rule, a health factor below 1, so it has no learned classifier
        for a QCNN to reproduce. This circuit is a separate classifier, pre-registered beside it
        (<code>{d.preregistration}</code>); its positive class is {d.task}. It reads eight of the borrow&apos;s numeric inputs, chosen on the fit window alone ({d.features.join(", ")}), and
        is scored on the {d.test_rows.toLocaleString("en-US")} borrows of the test window, {d.test_positive.toLocaleString("en-US")}{" "}
        of them liquidated: at that prevalence, {(d.prevalence * 100).toFixed(2)} percent, a random ranking scores an average
        precision of {f4(d.prevalence)}.
      </p>
      <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
        {ship && (
          <li>
            The seed the rule ships, seed {d.shipped_seed.seed}, the best of {spread?.seeds ?? "the"} on validation, scores{" "}
            {f4(ship.test_ap)} ({ci(ship)}), above the {f4(d.twice_prevalence)} the pre-registration asked.{" "}
            {d.comparators_above_the_shipped_seed.length
              ? <>Comparators above it: {d.comparators_above_the_shipped_seed.join("; ")}.</>
              : <>No comparator scores above it.</>}
            {circuitsAbove.length ? <> Of the circuit&apos;s other arms, above it: {circuitsAbove.map((r) => r.label).join("; ")}.</> : null}
          </li>
        )}
        {ship && (
          <li>
            The figures do not carry from one window to the next: {fell.length} of the {d.rows.length} rows below score lower on
            the later test window than on validation ({pct(d.validation_prevalence)} liquidated on validation,{" "}
            {pct(d.prevalence)} on test), the shipped seed from {f4(ship.validation_ap)} to {f4(ship.test_ap)}
            {lr ? <> and logistic regression from {f4(lr.validation_ap)} to {f4(lr.test_ap)}</> : null}
            {rose.length ? <>; only {rose.map((r) => `${r.label}, ${f4(r.validation_ap)} to ${f4(r.test_ap)}`).join("; ")}, scores higher</> : null}.
            The test column ranks these models under that shift; it does not rank the methods.
          </li>
        )}
        {spread && ens && (
          <li>
            The seed is not the method: over {spread.seeds} seeds the circuit scores {f4(spread.mean_test_ap)} &plusmn;{" "}
            {f4(spread.sd_test_ap)}, and their mean probability {f4(ens.test_ap)} ({ci(ens)}).
          </li>
        )}
        {noent && ens && (
          <li>
            Without its entangling gates the circuit scores {f4(noent.test_ap)} ({ci(noent)}), {noent.ap_ci95[1] < ens.ap_ci95[0]
              ? "below" : noent.ap_ci95[0] > ens.ap_ci95[1] ? "above" : "within the interval of"} the entangled circuit&apos;s{" "}
            {f4(ens.test_ap)}{four ? <>; on the first four inputs alone, four wires, {f4(four.test_ap)} ({ci(four)})</> : null}.
          </li>
        )}
        <li>{d.statement}</li>
      </ul>
      <div style={{ overflowX: "auto" }}>
        <table className="table small" style={{ marginTop: 8 }}>
          <thead><tr><th>Model</th><th>Kind</th><th>Validation average precision</th>
            <th>Test average precision (95% interval)</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key}><td>{r.key === "qcnn8_shipped" ? <b>{r.label}</b> : r.label}</td><td>{r.kind}</td>
                <td>{f4(r.validation_ap)}</td><td>{f4(r.test_ap)} ({ci(r)})</td></tr>))}
          </tbody>
        </table>
      </div>
      <p className="small" style={{ marginTop: 6 }}>
        The page&apos;s tree contraction of this seed equals the trainer&apos;s state vector on all{" "}
        {c.export_check.test_rows.toLocaleString("en-US")} test borrows within {ex.max_abs_readout_difference.toExponential(1)},
        no decision differing (criterion 2). This tab can re-score {c.rows_about}.{" "}
        <button type="button" className="btn ghost" onClick={run}>Re-score the sample here</button>
      </p>
      {rescore ? (
        <p className="small" style={{ marginTop: 4 }}>
          Re-scored in this tab: an average precision of {f4(rescore.ap)} over the sample&apos;s {c.sample.rows.toLocaleString("en-US")}{" "}
          borrows ({c.sample.positive} liquidated), in {rescore.ms.toFixed(0)} ms;{" "}
          {Math.abs(rescore.ap - c.sample.average_precision) <= 1e-9
            ? <>equal, within 1e-9, to the value recorded for these rows.</>
            : <>it DIFFERS from the {f4(c.sample.average_precision)} recorded for these rows.</>}
        </p>
      ) : null}
      <p className="muted small" style={{ margin: "8px 0 0" }}>
        Record: <code>{d.record}</code>. Data: {d.attribution.citation} Licence{" "}
        <a href={d.licence_url} target="_blank" rel="noreferrer">{d.licence}</a>; {d.modifications}
      </p>
    </div>
  );
}
