// 5G-NIDD's two base stations, federated (Round 86, batch 6; the owner's choice of 2026-11-26).
// A pre-registered comparison (references/telecom-nidd-federated-preregistration.md), run offline by
// scripts/train/telecom_nidd_federated.py: each station alone, FedAvg under the page's pairwise masks, and pooling (the
// page's CKKS student), each scored on each station's test flows. Every offline figure is read from
// web/public/data/telecom_nidd/federated.json; the run's figures (the replayed last aggregation and the global model under
// CKKS) come from `scoreNiddFederated` in clientCompute.ts as `artifacts.nidd_federated`.
import { useEffect, useId, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartReadout, HINT_POINTER, ReadoutTooltip } from "./ChartReadout";
import { MarksHidden } from "./VerificationChart";
import StageBadge from "./StageBadge";

type ByStation = { station_1: number; station_2: number };
interface Arm { test: ByStation; flamby_metric: number }
interface Federated {
  stations: string[];
  comparison: { pooled: Arm; alone: Arm[]; federated: Arm };
  per_round: ({ round: number } & ByStation)[];
  criteria: { "1_federated_above_each_station_alone": boolean; "2_within_0_01_of_pooled": boolean;
              federated_minus_pooled: number; federated_minus_best_alone: number };
  federated: { rounds: number; local_iterations: number; l2: number; temperature: number; weights: number[] };
  preregistration: { file: string; sha256: string }; record: string;
  attribution: { dataset: string; licence: string; licence_url: string; citation: string; download: string };
}
export interface NiddFederatedLive {
  replay: { coordinates: number; bitExact: number; ms: number; publicKeys: string[]; firstWords: string[][]; weights: number[] };
  encrypted: { rows: number; ciphertexts: number; ms: number; maxAbsError: number; decidedFlips: number; criterion3: boolean;
               perStation: { station: string; rows: number; encryptedAccuracy: number | null; plaintextAccuracy: number | null;
                             decisionsThatDiffer: number }[] };
}

const tick = { fill: "var(--muted)", fontSize: 11 };
const f3 = (v: number) => v.toFixed(3);
const pct = (v: number | null) => (v == null ? "n/a" : `${(v * 100).toFixed(2)}%`);

export default function NiddFederatedCard({ live }: { live: NiddFederatedLive | null }) {
  const captionId = useId();
  const [d, setD] = useState<Federated | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    let on = true;
    fetch(`${import.meta.env.BASE_URL}data/telecom_nidd/federated.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j: Federated) => { if (on) setD(j); })
      .catch((e: unknown) => { if (on) setFailed(String(e)); });
    return () => { on = false; };
  }, []);
  const title = "5G-NIDD's two base stations, federated";
  if (failed) {
    return <div className="card" id="nidd-federated" style={{ marginTop: 14 }}><h3>{title}</h3>
      <p className="small" style={{ color: "var(--danger)" }}>The federation&apos;s figures could not be read ({failed}).</p></div>;
  }
  if (!d) return null;
  const c = d.comparison, cr = d.criteria, fed = d.federated;
  // Short names in the table, the chart and the sentences; the file's long names say which CSV file each is.
  const s1 = "station 1", s2 = "station 2";
  const bestAlone = Math.max(...c.alone.map((a) => a.flamby_metric));
  const rows: { label: string; arm: Arm; strong?: boolean }[] = [
    { label: "Station 1 alone", arm: c.alone[0] },
    { label: "Station 2 alone", arm: c.alone[1] },
    { label: "Federated: FedAvg under the page's pairwise masks", arm: c.federated, strong: true },
    { label: "Pooled: every flow in one place (the page's CKKS student)", arm: c.pooled },
  ];
  // Why the comparison came out as it did, in the record's own figures, and only where they say it: when one station's
  // own model beats the federation by the mean, it is because that model carries to the other station's flows while
  // the other's does not carry back, and a model trained on both scores between them on the first station's flows.
  const key = ["station_1", "station_2"] as const;
  const b = c.alone[0].flamby_metric >= c.alone[1].flamby_metric ? 0 : 1, o = 1 - b;
  const [sb, so] = [b === 0 ? s1 : s2, o === 0 ? s1 : s2];
  const lowB = Math.min(c.alone[o].test[key[b]], c.alone[b].test[key[b]]), highB = Math.max(c.alone[o].test[key[b]], c.alone[b].test[key[b]]);
  const between = [c.federated.test[key[b]], c.pooled.test[key[b]]].every((v) => v > lowB && v < highB);
  const why = !cr["1_federated_above_each_station_alone"] && c.alone[b].flamby_metric > c.federated.flamby_metric
    ? `The best by the mean is ${sb}'s own model: on ${so}'s flows it scores ${f3(c.alone[b].test[key[o]])}, `
      + (f3(c.alone[b].test[key[o]]) === f3(c.alone[o].test[key[o]]) ? `as ${so}'s own does to three places`
        : `against ${f3(c.alone[o].test[key[o]])} for ${so}'s own`)
      + `, while ${so}'s model scores ${f3(c.alone[o].test[key[b]])} on ${sb}'s.` + (between ? ` Trained on both, federated or pooled, one linear model scores between the two on ${sb}'s `
      + `flows (${f3(c.federated.test[key[b]])} and ${f3(c.pooled.test[key[b]])}).` : "")
    : null;
  const ys = d.per_round.flatMap((p) => [p.station_1, p.station_2]).concat([c.pooled.flamby_metric, bestAlone]);
  // The axis in steps of 0.05, so its ticks are even; the first round sits far below the rest and stays on it.
  const lo = Math.max(0, Math.floor((Math.min(...ys) - 0.005) * 20) / 20), hi = Math.min(1, Math.ceil((Math.max(...ys) + 0.002) * 20) / 20);
  const yTicks = Array.from({ length: Math.round((hi - lo) * 20) + 1 }, (_, i) => Math.round((lo + i * 0.05) * 100) / 100);
  return (
    <div className="card" id="nidd-federated" style={{ marginTop: 14 }}>
      <h3>{title} <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        A pre-registered comparison, run offline on every 5G-NIDD flow: the two base stations train one detector without
        pooling their flows, by FedAvg for {fed.rounds} rounds of {fed.local_iterations} local L-BFGS iterations, each
        round&apos;s two models weighted by their stations&apos; fit rows ({fed.weights.map((w) => w.toFixed(3)).join(" and ")})
        and summed under the page&apos;s pairwise masks; the penalty ({fed.l2.toExponential(0)}) and the temperature
        ({fed.temperature.toFixed(3)}) were chosen from masked sums of the stations&apos; validation losses. Each model is
        scored on each station&apos;s test flows; the mean is FLamby&apos;s metric, as the medical page averages its
        hospitals. Station 1 is {d.stations[0] ?? "the first base station"}, station 2 {d.stations[1] ?? "the second"}.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="small" style={{ width: "100%", borderCollapse: "collapse" }}>
          <caption className="muted" style={{ textAlign: "left" }}>Macro-F1 over the nine classes, on each station&apos;s test flows</caption>
          <thead><tr><th scope="col" style={{ textAlign: "left" }}>Model</th><th scope="col">{s1}</th><th scope="col">{s2}</th>
            <th scope="col">mean</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label} style={r.strong ? { fontWeight: 600 } : undefined}>
                <td>{r.label}</td><td style={{ textAlign: "center" }}>{f3(r.arm.test.station_1)}</td>
                <td style={{ textAlign: "center" }}>{f3(r.arm.test.station_2)}</td>
                <td style={{ textAlign: "center" }}>{f3(r.arm.flamby_metric)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small" style={{ marginTop: 6 }}>
        {cr["1_federated_above_each_station_alone"]
          ? <>By the mean, the federated model is above each station alone, by {f3(cr.federated_minus_best_alone)} over the better one.</>
          : <>By the mean, the federated model is not above each station alone: {f3(cr.federated_minus_best_alone)} against the better one.</>}
        {" "}{cr["2_within_0_01_of_pooled"]
          ? <>It comes within 0.01 of pooling ({cr.federated_minus_pooled >= 0 ? "+" : ""}{f3(cr.federated_minus_pooled)}).</>
          : <>It is {f3(-cr.federated_minus_pooled)} below pooling.</>}
        {why ? <>{" "}{why}</> : null}
      </p>
      <figure className="chart-figure" style={{ margin: "6px 0 0" }}>
        <figcaption id={captionId} className="small muted">
          <b>The federated model after each round, on each station&apos;s test flows</b>. The dashed line is pooling&apos;s
          mean, {f3(c.pooled.flamby_metric)}; the dotted line the better station alone&apos;s mean, {f3(bestAlone)}. The rounds
          were fixed before the run, and nothing was chosen from these curves.
        </figcaption>
        <ChartReadout hint={HINT_POINTER}>
          <MarksHidden>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={d.per_round} role="img" title="Macro-F1 of the federated model by round, by base station"
                aria-describedby={captionId} margin={{ top: 10, right: 20, bottom: 8, left: 4 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" />
                <XAxis dataKey="round" type="number" domain={[1, fed.rounds]} tick={tick} allowDecimals={false}
                  ticks={[1, ...Array.from({ length: Math.floor(fed.rounds / 10) }, (_, i) => (i + 1) * 10)]}
                  label={{ value: "round", position: "insideBottom", offset: -6, fill: "var(--muted)", fontSize: 11 }} />
                <YAxis type="number" domain={[lo, hi]} ticks={yTicks} tick={tick} width={48} tickFormatter={(v: number) => v.toFixed(2)} />
                <Tooltip content={<ReadoutTooltip labelName="round" labelFormatter={(l) => `round ${l}`}
                  formatter={(v) => Number(v).toFixed(4)} />} />
                <Legend verticalAlign="bottom" wrapperStyle={{ fontSize: 11, paddingTop: 12 }} />
                <ReferenceLine y={c.pooled.flamby_metric} stroke="var(--text)" strokeDasharray="6 4" />
                <ReferenceLine y={bestAlone} stroke="var(--muted)" strokeDasharray="1 4" />
                <Line dataKey="station_1" name={s1} stroke="var(--accent)" strokeWidth={1.8} dot={false} isAnimationActive={false} />
                <Line dataKey="station_2" name={s2} stroke="var(--accent-2)" strokeWidth={1.8} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </MarksHidden>
        </ChartReadout>
      </figure>
      {live ? (
        <div style={{ marginTop: 8 }}>
          <h4 style={{ margin: "6px 0 2px" }}>This run, in this tab</h4>
          <p className="small" style={{ margin: 0 }}>
            The last round&apos;s aggregation, replayed: two notional stations drew fresh X25519 keys
            ({live.replay.publicKeys.join(", ")}), masked their last local models, and the aggregator summed the masked
            shares (first words {live.replay.firstWords.map((w) => w.join(" ")).join(" | ")}):{" "}
            <b style={{ color: live.replay.bitExact === live.replay.coordinates ? undefined : "var(--danger)" }}>
              {live.replay.bitExact} of {live.replay.coordinates} coordinates
            </b>{" "}equal the shipped global model bit for bit, in {live.replay.ms.toFixed(1)} ms.
            {" "}The global model under CKKS over the student&apos;s {live.encrypted.rows.toLocaleString("en-US")} held-out flows,
            {" "}{live.encrypted.ciphertexts} ciphertexts, {live.encrypted.ms.toFixed(0)} ms: largest deviation
            {" "}{live.encrypted.maxAbsError.toExponential(2)}, {live.encrypted.decidedFlips} decision(s) flipped beyond twice
            that; criterion 3 <b>{live.encrypted.criterion3 ? "holds" : "fails"}</b>.
          </p>
          <div style={{ overflowX: "auto" }}>
            <table className="small" style={{ width: "100%", borderCollapse: "collapse", marginTop: 4 }}>
              <thead><tr><th scope="col" style={{ textAlign: "left" }}>Held-out flows from</th><th scope="col">rows</th>
                <th scope="col">accuracy, encrypted</th><th scope="col">accuracy, plaintext</th><th scope="col">decisions that differ</th></tr></thead>
              <tbody>
                {live.encrypted.perStation.map((p) => (
                  <tr key={p.station}><td>{p.station}</td><td style={{ textAlign: "center" }}>{p.rows}</td>
                    <td style={{ textAlign: "center" }}>{pct(p.encryptedAccuracy)}</td>
                    <td style={{ textAlign: "center" }}>{pct(p.plaintextAccuracy)}</td>
                    <td style={{ textAlign: "center" }}>{p.decisionsThatDiffer}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <p className="small muted" style={{ marginTop: 8 }}>
          Run the pipeline to replay the last aggregation in this tab and score the federated model under CKKS.
        </p>
      )}
      <p className="small muted" style={{ marginTop: 6 }}>
        What it is not: with two stations, each can compute the other&apos;s update from the global model it receives and its
        own, so the masks hide the updates from the aggregator only. The stations exchanged their row counts and each
        feature&apos;s mean and deviation in the clear, to standardise alike. The mask seeds come from X25519 alone, which is
        not post-quantum, and the masked updates are not sealed. One process played both stations and the aggregator,
        offline and once; no differential privacy is applied. The split is the student&apos;s random one, so a
        station&apos;s test flows share attack sessions with its training flows; a station&apos;s model on the other
        station&apos;s flows is the cross-station reading. Pre-registration: <code>{d.preregistration.file}</code>; record:{" "}
        <code>{d.record}</code>. Data: {d.attribution.dataset},{" "}
        <a href={d.attribution.licence_url} target="_blank" rel="noreferrer">{d.attribution.licence}</a>; {d.attribution.citation}
      </p>
    </div>
  );
}
