// VUS-PR beside the satellite forecaster's event-wise F0.5 (Round 86, the owner's choice of 2026-11-14). The F0.5 reads
// Telemanom's flags at its thresholds; VUS-PR reads the continuous scores with no threshold. Computed offline by
// scripts/measure_satellite_vus_pr.py with TSB-AD's own code on the published forecaster's scores; every figure is read
// from web/public/data/satellite/vus_pr.json.
import { useEffect, useState } from "react";
import CardExport from "./CardExport";
import KeyPoints from "./KeyPoints";
import StageBadge from "./StageBadge";
import { modelsFor } from "../lib/config";

interface Spread { median: number; min: number; max: number; draws: number }
interface VusPr {
  model: string; hf_revision: string; tsb_ad: string; thresholds: number; version: string;
  series: { grid: string; test_range: [string, string]; steps: number; anomalous_steps: number; anomalous_share: number; events: number; warm_up_rows: number; warm_up_rule: string };
  score: string; window: { samples: number; minutes_each_side: number; rule: string };
  vus_pr: number; vus_roc: number; point_wise_average_precision: number;
  sensitivity: Record<string, number>; no_skill: { permuted: Spread; random_walk: Spread };
  event_wise_f0_5_corrected: number | null; comparator: string;
}

const f3 = (v: number) => v.toFixed(3);
const f4 = (v: number) => v.toFixed(4);

/** "window_250_thresholds_1000" -> "window 250, 1,000 thresholds"; "window_0" -> "window 0 (no buffer)". */
function sensLabel(k: string): string {
  const m = /^window_(\d+)(?:_thresholds_(\d+))?$/.exec(k);
  if (!m) return k;
  const w = Number(m[1]);
  return `window ${w}${w === 0 ? " (no buffer)" : ""}${m[2] ? `, ${Number(m[2]).toLocaleString("en-US")} thresholds` : ""}`;
}

export default function SatVusPrCard() {
  const [v, setV] = useState<VusPr | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/satellite/vus_pr.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j: VusPr) => { if (live) setV(j); })
      .catch((e: unknown) => { if (live) setFailed(String(e)); });
    return () => { live = false; };
  }, []);
  if (failed) {
    return <div className="card" id="sat-vus-pr" style={{ marginTop: 14 }}><h3>Threshold-free ranking: VUS-PR</h3><p className="small" style={{ color: "var(--danger)" }}>Its recorded figures did not load: {failed}.</p></div>;
  }
  if (!v) return null;
  const ds = modelsFor("satellite_comms").find((m) => m.key === v.model)?.dataset;
  const s = v.series, nsk = v.no_skill;
  const sens = Object.entries(v.sensitivity);
  const timesPerm = nsk.permuted.median > 0 ? v.vus_pr / nsk.permuted.median : null;
  // The same window at 1,000 thresholds, from the sensitivity row: the figure's dependence on TSB-AD's grid.
  const fineVus = sens.find(([k]) => k === `window_${v.window.samples}_thresholds_1000`)?.[1] ?? null;
  const noBuffer = v.sensitivity.window_0;
  // How far apart TSB-AD's thresholds sit in the ranking: `np.linspace(0, len(score) - 1, thre)` over the sorted scores.
  const gridStep = Math.round((s.steps - 1) / (v.thresholds - 1) / 100) * 100;
  return (
    <div className="card" id="sat-vus-pr" style={{ marginTop: 14 }}>
      <h3>Threshold-free ranking of the forecaster&apos;s scores: VUS-PR <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        The event-wise F0.5 on this page reads Telemanom&apos;s flags at its thresholds, a figure for the operating point. VUS-PR, which
        TSB-AD (Liu and Paparrizos, NeurIPS 2024) recommends for anomaly detection in time series, reads the continuous scores with no
        threshold: how well they rank the anomalous steps above the rest, with soft credit for scores just before or after an event
        (Paparrizos et al., PVLDB 2022). Here it is computed with TSB-AD {v.tsb_ad}&apos;s own code ({v.thresholds} thresholds, version
        &apos;{v.version}&apos;) on the published forecaster&apos;s scores over the test half, {s.test_range[0].slice(0, 10)} to{" "}
        {s.test_range[1].slice(0, 10)} ({s.steps.toLocaleString("en-US")} steps of {s.grid}; {s.events} anomalies over{" "}
        {s.anomalous_steps.toLocaleString("en-US")} steps, {(100 * s.anomalous_share).toFixed(3)}% of them). Computed offline and
        recorded; nothing here ran in your tab.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table small">
          <thead><tr><th>Figure</th><th>Value</th><th>What it reads</th></tr></thead>
          <tbody>
            <tr><td>VUS-PR</td><td><b>{f3(v.vus_pr)}</b></td><td>the ranking, buffer up to {v.window.samples} steps ({v.window.minutes_each_side} minutes each side of an event)</td></tr>
            <tr><td>Point-wise average precision</td><td>{f3(v.point_wise_average_precision)}</td><td>the same ranking with no buffer and no existence credit</td></tr>
            <tr><td>No skill: the score shuffled</td><td>{f4(nsk.permuted.median)}</td><td>median of {nsk.permuted.draws} draws ({f4(nsk.permuted.min)} to {f4(nsk.permuted.max)})</td></tr>
            <tr><td>No skill: a random walk</td><td>{f4(nsk.random_walk.median)}</td><td>median of {nsk.random_walk.draws} draws ({f4(nsk.random_walk.min)} to {f4(nsk.random_walk.max)})</td></tr>
            {v.event_wise_f0_5_corrected != null && (
              <tr><td>Event-wise F0.5, corrected</td><td>{f3(v.event_wise_f0_5_corrected)}</td><td>Telemanom&apos;s flags at its thresholds, ESA-ADB&apos;s scoring: the operating point</td></tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="muted small" style={{ margin: "6px 0 10px" }}>
        The score: {v.score}. The window: {v.window.rule}. Sensitivity: {sens.map(([k, x]) => `${sensLabel(k)}: ${f3(x)}`).join("; ")}.
        Warm-up: {s.warm_up_rule}.
      </p>
      <KeyPoints title="What it shows" items={[
        <><b>The scores rank the anomalies far above chance.</b> VUS-PR {f3(v.vus_pr)} against {f4(nsk.permuted.median)} for the same
          scores shuffled{timesPerm ? `, about ${Math.round(timesPerm).toLocaleString("en-US")} times as high` : ""}, and{" "}
          {f4(nsk.random_walk.median)} for a random walk, the no-skill score a 2026 preprint found hardest for VUS-PR to tell from
          a detector (Lyu, arXiv:2607.11969).</>,
        <><b>Events count, not steps.</b> The point-wise average precision, {f3(v.point_wise_average_precision)}, counts steps, so a long
          event weighs by its length. VUS-PR multiplies recall by the share of the {s.events} events the scores have reached at each
          threshold (the existence reward), so each event counts{noBuffer != null ? `; with no buffer it is ${f3(noBuffer)}` : ""}.
          {v.event_wise_f0_5_corrected != null && <> The event-wise F0.5, {f3(v.event_wise_f0_5_corrected)}, reads Telemanom&apos;s flags at
          its thresholds, event by event.</>}</>,
      ]} />
      <div className="note" style={{ marginTop: 10, fontSize: 12 }}>
        <KeyPoints title="What it does not show" items={[
          <><b>A comparison.</b> {v.comparator[0].toUpperCase() + v.comparator.slice(1)}. The figure is not comparable with TSB-AD&apos;s
            leaderboard (other data) or with TimeEval&apos;s RangePrVUS (linear slopes, a larger buffer).</>,
          <><b>A stable figure for one event.</b> With {s.events} anomalies, one event moves the existence credit by about{" "}
            {(100 / s.events).toFixed(0)} points of recall; the sensitivity row shows how much the buffer moves the figure.</>,
          <><b>A figure free of its grid.</b> TSB-AD places its {v.thresholds} thresholds evenly through the ranking of the scores, about{" "}
            {gridStep.toLocaleString("en-US")} steps apart here, coarse beside {s.anomalous_steps.toLocaleString("en-US")} anomalous steps: with 1,000 the same scores give{" "}
            {fineVus != null ? f3(fineVus) : "a different figure"}. The card reports the reference code&apos;s default.</>,
        ]} />
      </div>
      <p className="muted small" style={{ marginTop: 6 }}>
        Data: {ds ? <>{ds.name}, {ds.attribution}</> : "ESA-ADB, Mission1 lightweight channels 41-46"} Model: the published forecaster at
        revision {v.hf_revision.slice(0, 7)}. Code: TSB-AD {v.tsb_ad} (Apache-2.0), run offline.
      </p>
      <CardExport name="satellite_comms-vus-pr" label="the VUS-PR figures" rows={[
        { figure: "VUS-PR", value: v.vus_pr }, { figure: "VUS-ROC", value: v.vus_roc },
        { figure: "point-wise average precision", value: v.point_wise_average_precision },
        { figure: "no skill, shuffled (median)", value: nsk.permuted.median }, { figure: "no skill, random walk (median)", value: nsk.random_walk.median },
        ...sens.map(([k, x]) => ({ figure: `VUS-PR, ${sensLabel(k)}`, value: x })),
      ]} json={v} />
    </div>
  );
}
