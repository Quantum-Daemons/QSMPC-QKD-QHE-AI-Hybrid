// This tab's differential-privacy releases about one population, composed, and what their epsilon means as odds and as
// a test (Round 86, the owner's choices of 2026-11-22: a ledger of the releases reruns repeat, and DP explained as
// odds). The ledger is this tab's memory only (lib/dpLedger.ts); this panel reads it and holds nothing itself.
import { useSyncExternalStore } from "react";
import { formatDelta } from "../lib/clientDp";
import {
  bestTestPower, ledgerSummary, ledgerVersion, oddsFactor, POPULATIONS, type Population, posteriorAtMost,
  subscribeLedger,
} from "../lib/dpLedger";

/** The false-positive rate the table's test column is read at, and the prior its sentence starts from. */
export const LEDGER_FPR = 0.05;
export const LEDGER_PRIOR = 0.1;

const f2 = (x: number) => x.toFixed(2);
const factor = (x: number) => (x >= 100 ? Math.round(x).toLocaleString("en-US") : x.toFixed(2));
const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
/** A posterior so close to 1 that one decimal would round it to certainty, which it is not. */
const posterior = (x: number) => (x >= 0.9995 ? "over 99.9%" : pct(x));

export default function DpLedgerPanel({ population, thisRun, thisRunLabel }: {
  population: Population;
  /** The epsilon the card states for this run's release (or releases), at the population's delta. */
  thisRun: { epsilon: number; delta: number };
  thisRunLabel: string;
}) {
  useSyncExternalStore(subscribeLedger, ledgerVersion, ledgerVersion);
  const s = ledgerSummary(population);
  // The tab's total gets its own row only once it is more than this run's: one run of the medical page is four
  // releases, and its total is the run's own figure until a second run or the comparison adds to it.
  const more = s !== null && s.epsilon > thisRun.epsilon + 1e-9;
  const rows = [{ label: thisRunLabel, epsilon: thisRun.epsilon }];
  if (more) rows.push({ label: `All ${s.releases} releases this tab has made`, epsilon: s.epsilon });
  const widest = rows[rows.length - 1].epsilon;
  const { who, unit, aUnit, units } = POPULATIONS[population];
  return (
    <div style={{ marginTop: 12 }} id={`dp-ledger-${population}`}>
      <h4>This tab&apos;s releases, and what epsilon means</h4>
      <p className="small" style={{ marginTop: 4 }}>
        Each run draws fresh noise and releases again about {who}, so runs add up: concentrated differential privacy
        adds rho across releases about the same {units} (Bun and Steinke, TCC 2016), and the total converts to epsilon
        once. {s === null
          ? "This tab has made no release about them yet."
          : `This tab has made ${s.releases} ${s.releases === 1 ? "release" : "releases"} about them, rho ${s.rho.toPrecision(3)} in all.`}
      </p>
      <table className="table">
        <thead>
          <tr>
            <th scope="col" style={{ textAlign: "left" }}>Released</th>
            <th scope="col">Epsilon</th>
            <th scope="col">Odds multiplied by at most</th>
            <th scope="col">A test wrong for 1 in 20 finds at most</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{r.label}</th>
              <td>{f2(r.epsilon)}</td>
              <td>{factor(oddsFactor(r.epsilon))}</td>
              <td>{pct(bestTestPower(r.epsilon, thisRun.delta, LEDGER_FPR))}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 8 }}>
        In words: someone who put the chance that one given {unit} is in the data at 1 in 10 can, after
        {more ? " all of this tab's releases" : ` ${thisRunLabel.charAt(0).toLowerCase()}${thisRunLabel.slice(1)}`}, put it at
        most at {posterior(posteriorAtMost(widest, LEDGER_PRIOR))},
        whatever else they knew: epsilon bounds how far any result moves anyone&apos;s odds. A survey of 963 people found
        odds-based explanations of epsilon more effective than ones that show outputs or leave epsilon out (Nanayakkara,
        Smart, Cummings, Kaptchuk and Redmiles, USENIX Security 2023). That reading is exact when
        delta is 0. With delta {formatDelta(thisRun.delta)} the exact statement is the last column: a test of
        whether {aUnit} is in the data that wrongly flags 1 in 20 of those that are not finds at most that share of those that are
        (Kairouz, Oh and Viswanath, ICML 2015). The ledger lives in this tab: it cannot see other tabs, earlier visits or
        anyone else&apos;s runs, and a reload empties it, though what was released stays released.
      </p>
    </div>
  );
}
