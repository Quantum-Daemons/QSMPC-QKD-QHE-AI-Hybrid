// The heart-sound federation (Round 86): the PhysioNet/CinC Challenge 2016 recordings, each source held out in turn,
// trained offline on this project's GPU and recorded. Every figure is read from web/public/data/medical/pcg_fl.json
// except the Challenge's own baseline, which carries its source; nothing here runs in the visitor's tab.
import { useEffect, useState } from "react";
import CardExport from "./CardExport";
import KeyPoints from "./KeyPoints";
import StageBadge from "./StageBadge";
import type { Stat } from "../lib/cxrFl";
import { PCG_RANDOM, PCG_SOURCES, pcgFlRows, type PcgFl } from "../lib/pcgFl";

const f3 = (v: number) => v.toFixed(3);
const pct = (v: number) => `${(100 * v).toFixed(0)}%`;
const stat = (s?: Stat) => (!s ? "-" : s.sd == null ? f3(s.mean) : `${f3(s.mean)} ± ${f3(s.sd)}`);

export default function PcgFlCard() {
  const [c, setC] = useState<PcgFl | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/medical/pcg_fl.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j: PcgFl) => { if (live) setC(j); })
      .catch((e: unknown) => { if (live) setFailed(String(e)); });
    return () => { live = false; };
  }, []);
  if (failed) {
    return <div className="card" id="pcg-fl" style={{ marginTop: 14 }}><h3>Heart-sound federation</h3><p className="small" style={{ color: "var(--danger)" }}>Its recorded figures did not load: {failed}.</p></div>;
  }
  if (!c) return null;
  const held = PCG_SOURCES.filter((s) => c.summary[s]);
  const random = c.summary[PCG_RANDOM]?.pooled;
  const mean = (f: (s: string) => number | undefined) => {
    const v = held.map(f).filter((x): x is number => x != null);
    return v.reduce((a, b) => a + b, 0) / v.length;
  };
  const pooledMean = mean((s) => c.summary[s].pooled?.auroc.mean);
  const fedMean = mean((s) => c.summary[s].fedavg?.auroc.mean);
  const shares = held.map((s) => c.sources[s].abnormal / c.sources[s].recordings);
  const n = c.training_recording_counts;
  const below = held.filter((s) => (c.summary[s].pooled?.auroc.mean ?? 1) < 0.5 || (c.summary[s].fedavg?.auroc.mean ?? 1) < 0.5);
  const worst = held.reduce((w, s) => ((c.summary[s].pooled?.auroc.mean ?? 1) < (c.summary[w].pooled?.auroc.mean ?? 1) ? s : w), held[0]);
  return (
    <div className="card" id="pcg-fl" style={{ marginTop: 14 }}>
      <h3>Heart-sound federation, each source held out <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        Normal against abnormal heart sounds on the PhysioNet/CinC Challenge 2016 recordings: six databases collected by different teams, with different
        stethoscopes and populations, and abnormal recordings from {pct(Math.min(...shares))} to {pct(Math.max(...shares))} of each. Each database is held out once; a small
        CNN ({c.recipe.params.toLocaleString("en-US")} parameters) on {c.recipe.window_s}-second log-mel windows is trained on the other five, pooled and under FedAvg with
        each database as a site, and scored on the held-out one, recording by recording. {c.recipe.epochs} epochs or rounds, {c.recipe.seeds.length} seeds.
        Trained on this project&apos;s GPU and recorded: nothing here ran in your tab, and the weights are not published.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table small">
          <thead>
            <tr><th rowSpan={2}>Held out</th><th rowSpan={2}>Recordings</th><th rowSpan={2}>Abnormal</th><th colSpan={2}>AUROC on the held-out source</th><th colSpan={2}>Mean of sensitivity and specificity</th></tr>
            <tr><th>pooled</th><th>FedAvg</th><th>pooled</th><th>FedAvg</th></tr>
          </thead>
          <tbody>{held.map((s) => {
            const r = c.summary[s], src = c.sources[s];
            return (
              <tr key={s}>
                <td>{s}</td><td>{src.recordings.toLocaleString("en-US")}</td><td>{pct(src.abnormal / src.recordings)}</td>
                <td>{stat(r.pooled?.auroc)}</td><td>{stat(r.fedavg?.auroc)}</td><td>{f3(r.pooled?.macc.mean ?? NaN)}</td><td>{f3(r.fedavg?.macc.mean ?? NaN)}</td>
              </tr>
            );
          })}</tbody>
        </table>
      </div>
      <p className="muted small" style={{ margin: "6px 0 10px" }}>
        Mean over the seeds, ± their standard deviation. A recording&apos;s score is the mean of its windows&apos; logits; the threshold is 0, with a loss that weights the
        classes evenly. The data name no patients, so a source&apos;s recordings are not independent where a patient has several.
      </p>
      <KeyPoints title="What it shows" items={[
        <><b>A new source is hard.</b> Held out, the sources average {f3(pooledMean)} pooled and {f3(fedMean)} federated. With the recordings split at random instead, so that one patient&apos;s recordings can fall on both sides, the same network reaches {f3(random?.auroc.mean ?? NaN)}: most of that gap is the change of source, not the model.</>,
        <><b>On some sources the score turns round.</b> Below 0.5, a held-out source&apos;s abnormal recordings score lower than its normal ones: so it went on {below.join(", ")}, in every seed. The lowest pooled figure is {worst}, {f3(c.summary[worst].pooled?.auroc.mean ?? NaN)}, on {c.sources[worst].recordings.toLocaleString("en-US")} recordings with {pct(c.sources[worst].abnormal / c.sources[worst].recordings)} abnormal: what the network took from the other sources, whose abnormal shares differ, does not carry over and here points the other way.</>,
        <><b>The Challenge saw the same.</b> Its example algorithm, scored leaving one database out, ranged from 0.47 to 0.86 in this mean of sensitivity and specificity, 0.59 ± 0.15 on average (Clifford et al., Computing in Cardiology 2016).</>,
        <><b>Federation is not a cure for a source shift.</b> FedAvg was below pooled training on {held.filter((s) => (c.summary[s].fedavg?.auroc.mean ?? 0) < (c.summary[s].pooled?.auroc.mean ?? 0)).length} of {held.length} held-out sources. It weights each site by its size, so the largest database dominates the average, and a site whose abnormal share differs from the rest pulls the model its own way.</>,
      ]} />
      <div className="note" style={{ marginTop: 10, fontSize: 12 }}>
        <KeyPoints title="What it does not show" items={[
          <><b>Labels from the contributors.</b> Normal recordings came from healthy subjects and abnormal ones mostly from patients with valve defects or coronary artery disease; no finer diagnosis is given.</>,
          <><b>Three counts of one training set.</b> The released files list {n.released_zip.toLocaleString("en-US")} recordings, all used here; the PhysioNet page says {n.physionet_page_introduction.toLocaleString("en-US")} and {n.physionet_page_announcement.toLocaleString("en-US")}, and the Challenge paper&apos;s table {n.cinc_2016_paper_table_1.toLocaleString("en-US")}.</>,
          <><b>Recorded, not run here, and not a device.</b> The training ran offline ({c.environment.gpu}, driver {c.environment.driver}, PyTorch {c.environment.torch}). Nothing on this page is a diagnosis.</>,
        ]} />
      </div>
      <p className="muted small" style={{ marginTop: 6 }}>
        Data: {c.attribution.provider}, <a href={c.attribution.download} target="_blank" rel="noreferrer">{c.attribution.dataset}</a>, under the <a href={c.attribution.licence_url} target="_blank" rel="noreferrer">{c.attribution.licence}</a>; {c.attribution.citation}; {c.attribution.platform_citation}.
      </p>
      <CardExport name="medical_fl-heart-sound-federation" label="the heart-sound federation table" rows={pcgFlRows(c)} json={c} />
    </div>
  );
}
