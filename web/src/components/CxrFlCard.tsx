// The chest X-ray federation (Round 86, the owner's item (3)): NIH ChestX-ray14, Effusion, trained offline on this
// project's GPU and recorded. Every figure below is read from web/public/data/medical/cxr_fl.json except the published
// ones, which carry their sources; nothing here runs in the visitor's tab.
import { useEffect, useState } from "react";
import CardExport from "./CardExport";
import Details from "./Details";
import KeyPoints from "./KeyPoints";
import StageBadge from "./StageBadge";
import { CXR_ARMS, cxrFlRows, type CxrFl, type Stat } from "../lib/cxrFl";

const f3 = (v: number) => v.toFixed(3);
const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
const stat = (s: Stat) => (s.sd == null ? f3(s.mean) : `${f3(s.mean)} ± ${f3(s.sd)}`);
const range = (vals: number[]) => {
  const lo = Math.min(...vals), hi = Math.max(...vals);
  return lo === hi ? f3(lo) : `${f3(lo)} to ${f3(hi)}`;
};

/** Effusion AUROC on the official test split as each paper prints it: context for the table, not a reproduction. */
const PUBLISHED: [string, string, string][] = [
  ["Wang et al., CVPR 2017 (arXiv:1705.02315v5, Table 17)", "ResNet-50, ImageNet-pretrained, 1024 px", "0.759"],
  ["Tang et al., 2018 (arXiv:1807.07532, Table 1)", "baseline and attention-guided curriculum", "0.790, 0.819"],
  ["Baltruschat et al., Sci Rep 2019 (Table 5)", "ResNet-38, -50, -101 from scratch at 448 px, with age, sex and view", "0.822, 0.818, 0.818"],
  ["Tayebi Arasteh et al., Sci Rep 2023 (Table 2)", "ResNet-50, ImageNet-21K, five datasets as sites: alone, FedAvg", "0.815, 0.826"],
];

export default function CxrFlCard() {
  const [c, setC] = useState<CxrFl | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/medical/cxr_fl.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j: CxrFl) => { if (live) setC(j); })
      .catch((e: unknown) => { if (live) setFailed(String(e)); });
    return () => { live = false; };
  }, []);
  if (failed) {
    return <div className="card" id="cxr-fl" style={{ marginTop: 14 }}><h3>Chest X-ray federation</h3><p className="small" style={{ color: "var(--danger)" }}>Its recorded figures did not load: {failed}.</p></div>;
  }
  if (!c) return null;
  const s = c.summary, d = c.data;
  const central = s["centralised ResNet-18"], fed = s["FedAvg ResNet-18"], paOnly = s["PA images only, ResNet-18"], cnx = s["FedAvg ConvNeXt-atto"];
  const alone = [1, 2, 3, 4].map((h) => s[`hospital ${h} alone, ResNet-18`]).filter(Boolean);
  const hospAp = d.hospitals.map((h) => h.ap_share);
  const untrained = c.untrained_view_probe_auroc;
  const negView = [central, fed].map((a) => a.score_view_auroc_negatives.mean);
  const posView = [central, fed].map((a) => a.score_view_auroc_positives.mean);
  const camOf = (a: typeof central) => a.cam?.all;
  return (
    <div className="card" id="cxr-fl" style={{ marginTop: 14 }}>
      <h3>Chest X-ray federation, four hospitals by view <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        Effusion on NIH ChestX-ray14, a ResNet-18 trained from scratch at 224 px. Four simulated hospitals take the official training patients, every patient in one
        hospital, with AP shares from {pct(Math.min(...hospAp))} to {pct(Math.max(...hospAp))}, an outpatient clinic to an intensive-care unit; every model is tested on the
        official test split, {d.test.images.toLocaleString("en-US")} images of {d.test.patients.toLocaleString("en-US")} patients once {d.dropped_duplicates} pixel-identical
        duplicates are dropped. {c.recipe.epochs} epochs, or {c.recipe.epochs} FedAvg rounds of one local epoch, with {c.recipe.seeds.length} training seeds; ConvNeXt-atto once.
        Trained on this project&apos;s GPU and recorded: nothing here ran in your tab, and the weights are not published.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table small">
          <thead><tr><th>Hospital</th><th>Patients</th><th>Training images</th><th>AP share</th><th>Effusion</th></tr></thead>
          <tbody>{d.hospitals.map((h) => (
            <tr key={h.hospital}><td>{h.hospital}</td><td>{h.patients.toLocaleString("en-US")}</td><td>{h.train_images.toLocaleString("en-US")}</td><td>{pct(h.ap_share)}</td><td>{pct(h.effusion_prevalence)}</td></tr>
          ))}</tbody>
        </table>
      </div>
      <div style={{ overflowX: "auto", marginTop: 8 }}>
        <table className="table small">
          <thead>
            <tr><th rowSpan={2}>Trained</th><th rowSpan={2}>Seeds</th><th colSpan={3}>AUROC on the official test split</th><th colSpan={2}>Specificity at sensitivity 0.80</th><th rowSpan={2}>AP against PA among the negatives</th></tr>
            <tr><th>all</th><th>PA</th><th>AP</th><th>PA</th><th>AP</th></tr>
          </thead>
          <tbody>{CXR_ARMS.filter((a) => s[a]).map((a) => (
            <tr key={a}>
              <td>{a}</td><td>{s[a].seeds}</td><td>{stat(s[a].auroc_overall)}</td><td>{f3(s[a].auroc_PA.mean)}</td><td>{f3(s[a].auroc_AP.mean)}</td>
              <td>{f3(s[a].specificity_PA.mean)}</td><td>{f3(s[a].specificity_AP.mean)}</td><td>{f3(s[a].score_view_auroc_negatives.mean)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <p className="muted small" style={{ margin: "6px 0 10px" }}>
        Mean over the seeds, ± their standard deviation where there are several; the first seed&apos;s 95% interval, from resampling patients, is {f3(central.interval_overall_first_seed[0])} to {f3(central.interval_overall_first_seed[1])} pooled.
        The threshold is each model&apos;s own, chosen on its validation patients for sensitivity 0.80. The last column is the AUROC with which the Effusion score alone tells AP from PA among the true negatives: 0.5 would mean the score carries no view.
      </p>
      <KeyPoints title="What it shows" items={[
        <><b>Federation.</b> FedAvg reaches {f3(fed.auroc_overall.mean)} against {f3(central.auroc_overall.mean)} pooled, above every hospital alone ({range(alone.map((a) => a.auroc_overall.mean))}).</>,
        <><b>The view is in the score.</b> The view alone predicts Effusion with AUROC {f3(d.view_only_auroc)} on the test split, a weak shortcut; but among the true negatives the score tells AP from PA with AUROC {range(negView)}, so one threshold passes {pct(central.specificity_PA.mean)} of PA negatives and {pct(central.specificity_AP.mean)} of AP ones. Among the positives it is {range(posView)}: AP effusions score lower. AP is the harder view, {f3(central.auroc_AP.mean)} against {f3(central.auroc_PA.mean)}.</>,
        <><b>PA only.</b> Trained on PA images alone, the model matches the pooled one on PA ({f3(paOnly.auroc_PA.mean)}) and loses {f3(central.auroc_AP.mean - paOnly.auroc_AP.mean)} on AP ({f3(paOnly.auroc_AP.mean)}).</>,
        <><b>A probe sees the view, trained or not.</b> A linear probe reads AP against PA from the features at AUROC {f3(central.view_probe_auroc?.mean ?? NaN)}, and at {f3(untrained.view_probe_auroc.mean)} from an untrained network&apos;s, as Baltruschat et al. found 0.998 for a from-scratch ResNet-50: it says the view is visible, not that the classifier relies on it.</>,
        ...(cnx ? [<><b>ConvNeXt-atto, once.</b> Under FedAvg with the ResNet&apos;s {c.recipe.epochs}-epoch recipe, {cnx.params?.toLocaleString("en-US")} parameters: {f3(cnx.auroc_overall.mean)}. ConvNeXt&apos;s own recipes train for hundreds of epochs with heavier augmentation; this compares the two under one short recipe, not each at its best.</>] : []),
        <><b>Where the maps point.</b> On NIH&apos;s {d.boxes.effusion_boxes} Effusion boxes, all in the test split, the class activation map&apos;s maximum lands inside the box {pct(camOf(central)?.pointing_hit_rate.mean ?? NaN)} of the time pooled and {pct(camOf(fed)?.pointing_hit_rate.mean ?? NaN)} federated ({pct(camOf(fed)?.pointing_hit_rate_15px.mean ?? NaN)} with Zhang et al.&apos;s 15-pixel tolerance), against {pct(camOf(central)?.box_area_share.mean ?? NaN)} for a uniform map, {pct(d.boxes.centre_baseline.all.pointing_hit_rate)} for the image&apos;s centre and {pct(untrained.cam.all.pointing_hit_rate.mean)} for an untrained network. Saporta et al. report a Grad-CAM hit rate of 0.507 for effusion on CheXpert&apos;s segmentations, from an ensemble of DenseNet-121s scored on its true positives only: these maps, 7 by 7 before upsampling, are weak evidence of where the model looks.</>,
      ]} />
      <Details summary="Published figures on the same test split, for context">
        <table className="table small">
          <thead><tr><th>Source</th><th>Model</th><th>Effusion AUROC</th></tr></thead>
          <tbody>{PUBLISHED.map(([src, model, v]) => <tr key={src}><td>{src}</td><td>{model}</td><td>{v}</td></tr>)}</tbody>
        </table>
        <p className="muted small">Different models, inputs and training from this card&apos;s, so none is a figure to reproduce here. Tayebi Arasteh et al. treat whole datasets as sites; no paper found splits sites by view.</p>
      </Details>
      <div className="note" style={{ marginTop: 10, fontSize: 12 }}>
        <KeyPoints title="What it does not show" items={[
          <><b>Noisy labels.</b> The labels were mined from the reports. Re-read on the images by a radiologist, 70 to 74% of a sample of Effusion labels held (Oakden-Rayner, Academic Radiology 2020), and an uncertain mention is labelled No Finding.</>,
          <><b>Simulated hospitals.</b> One dataset split by view: real hospitals also differ in scanners, populations and labelling, which this split does not model.</>,
          <><b>Recorded, not run here, and not a device.</b> The training ran offline ({c.environment.gpu}, driver {c.environment.driver}, PyTorch {c.environment.torch}). Nothing on this page is a diagnosis.</>,
        ]} />
      </div>
      <p className="muted small" style={{ marginTop: 6 }}>
        Data: {c.attribution.provider}, <a href={c.attribution.download} target="_blank" rel="noreferrer">{c.attribution.dataset}</a>; {c.attribution.citation}
      </p>
      <CardExport name="medical_fl-cxr-federation" label="the chest X-ray federation table" rows={cxrFlRows(c)} json={c} />
    </div>
  );
}
