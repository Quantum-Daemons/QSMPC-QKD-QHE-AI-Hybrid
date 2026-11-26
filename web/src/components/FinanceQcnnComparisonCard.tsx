// The finance QCNN on AMLworld against quantum kernels and classical comparators (Round 86, batch 6; the owner's choices
// of 2026-11-26), pre-registered in references/finance-qcnn-amlworld-comparison-preregistration.md. Its criterion 2: the
// page states every arm's figure with its interval and the prevalence, says the circuit is classically simulable with no
// quantum advantage claimed, and says which comparators score above it, whatever ships. Every figure is read from
// web/public/data/finance/qcnn_comparison.json, which scripts/train/finance_qcnn_comparison.py --record writes.
//
// ONE ROW IS NOT AN ARM (the owner's choice of 2026-11-26, made after the record was first written): logistic regression on
// the circuit's own tanh-encoded inputs, the trainer's baseline as the served weights file records it. The arms' logistic
// regression reads the standardised inputs and scores far lower, so without this row the card read as if every classical
// model trailed the circuit, when this one leads the circuit's seed average. It is labelled in the table and in the summary.
// The summary's verdicts are computed from the rows, as every verdict on these pages is, so a rerun that reorders them
// cannot leave a sentence asserting the old order.
import { useEffect, useState } from "react";
import StageBadge from "./StageBadge";

interface Row {
  key: string; label: string; kind: string; test_ap: number; ap_ci95: [number, number];
  shipped_minus_this_ci95: [number, number] | null; not_an_arm?: boolean;
}
interface Comparison {
  preregistration: string; record: string; test_rows: number; test_positive: number; prevalence: number; twice_prevalence: number;
  rows: Row[]; spreads: Record<string, { mean_test_ap: number; sd_test_ap: number; seeds: number }>;
  shipped_seed: { seed: number; test_ap: number; minority_f1: number }; criterion_1_signal: boolean;
  comparators_above_the_shipped_seed: string[]; schmidt: { largest: number; bound: number; rows: number } | null;
  statement: string; licence: string; licence_url: string; attribution: string; modifications: string;
}

const f4 = (v: number) => v.toFixed(4);
const ci = (r: Row) => `${f4(r.ap_ci95[0])} to ${f4(r.ap_ci95[1])}`;
const KERNELS = ["product", "zz", "projected"];

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
  const by = (k: string) => d.rows.find((r) => r.key === k);
  const ship = by("qcnn1_shipped"), ens = by("qcnn1_ensemble"), noent = by("noent_ensemble"), lrEnc = d.rows.find((r) => r.not_an_arm);
  const kernels = d.rows.filter((r) => KERNELS.includes(r.key)).map((r) => r.test_ap);
  const spread = d.spreads.qcnn1;
  // The shipped seed against a row, from the interval of their paired difference: ahead, behind, or level within it.
  const versusShip = (r: Row) => {
    const p = r.shipped_minus_this_ci95;
    if (!p) return "";
    const where = p[0] > 0 ? "ahead of" : p[1] < 0 ? "behind" : "level, within the interval, with";
    return `${where} the row that is not an arm (difference ${f4(p[0])} to ${f4(p[1])})`;
  };
  return (
    <div className="card" id="finance-qcnn-comparison" style={{ marginTop: 14 }}>
      <h3>The QCNN on AMLworld, against quantum kernels and classical models <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        Pre-registered (<code>{d.preregistration}</code>): the same eight AMLworld inputs and fit sample for every model, scored on
        the {d.test_rows.toLocaleString("en-US")} transactions of the test window, {d.test_positive.toLocaleString("en-US")} of them
        laundering. At that prevalence, {(d.prevalence * 100).toFixed(3)} percent, a random ranking scores an average precision
        of {f4(d.prevalence)}.
      </p>
      <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
        {ship && (
          <li>
            The seed the rule ships, seed {d.shipped_seed.seed}, the best of {spread?.seeds ?? "the"} on validation, scores {f4(ship.test_ap)} ({ci(ship)}).{" "}
            {d.comparators_above_the_shipped_seed.length
              ? <>Arms above it: {d.comparators_above_the_shipped_seed.join(", ")}.</>
              : <>No arm scores above it{lrEnc ? <>, and it is {versusShip(lrEnc)}</> : null}.</>}
          </li>
        )}
        {spread && ens && (
          <li>
            The seed is not the method: over {spread.seeds} seeds the circuit scores {f4(spread.mean_test_ap)} &plusmn; {f4(spread.sd_test_ap)},
            and their mean probability {f4(ens.test_ap)} ({ci(ens)})
            {lrEnc ? <>, {ens.ap_ci95[1] < lrEnc.ap_ci95[0] ? "below" : ens.ap_ci95[0] > lrEnc.ap_ci95[1] ? "above" : "overlapping"} logistic
              regression on the circuit&apos;s own tanh-encoded inputs, {f4(lrEnc.test_ap)} ({ci(lrEnc)}). That row is not an arm: it is the
              trainer&apos;s baseline as the served weights file records it, added after the record was written; the arms&apos; logistic regression
              reads the standardised inputs instead.</> : "."}
          </li>
        )}
        {noent && ens && kernels.length > 0 && (
          <li>
            Without its entangling gates the circuit scores {f4(noent.test_ap)} (the mean probability of its {d.spreads.noent?.seeds ?? "few"} seeds),
            {noent.ap_ci95[1] < ens.ap_ci95[0] ? " below " : " not below "}the entangled circuit&apos;s {f4(ens.test_ap)}; the three quantum kernels
            score {f4(Math.min(...kernels))} to {f4(Math.max(...kernels))}.
          </li>
        )}
        <li>
          The comparison runs the checks of Bowles, Ahmed and Schuld&apos;s benchmark study of quantum classifiers (arXiv:2403.07059, 2024):
          classical baselines, many seeds and the circuit without entanglement. They found classical models ahead overall and
          entanglement often unnecessary; here a classical model leads the circuit&apos;s seed average, and removing the entanglement
          {noent && ens && noent.ap_ci95[1] < ens.ap_ci95[0] ? " lowers" : " does not lower"} the circuit&apos;s figure.
        </li>
        <li>
          {d.statement}{d.schmidt ? <> Its largest Schmidt rank over {d.schmidt.rows.toLocaleString("en-US")} test rows, every cut and every
            layer, traced in double precision, is {d.schmidt.largest} of the {d.schmidt.bound} an eight-wire state can reach.</> : null}
        </li>
      </ul>
      <div style={{ overflowX: "auto" }}>
        <table className="table small" style={{ marginTop: 8 }}>
          <thead><tr><th>Model</th><th>Kind</th><th>Test average precision (95% interval)</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key}>
                <td>{r.key === "qcnn1_shipped" ? <b>{r.label}</b> : r.not_an_arm ? <i>{r.label}</i> : r.label}</td>
                <td>{r.kind}</td>
                <td>{f4(r.test_ap)} ({ci(r)})</td>
              </tr>))}
          </tbody>
        </table>
      </div>
      <p className="muted small" style={{ margin: "8px 0 0" }}>
        Record: <code>{d.record}</code>. Data: {d.attribution} Licence <a href={d.licence_url} target="_blank" rel="noreferrer">{d.licence}</a>; {d.modifications}
      </p>
    </div>
  );
}
