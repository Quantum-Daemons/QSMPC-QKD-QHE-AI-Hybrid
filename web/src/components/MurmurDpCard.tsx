// Murmur detection under differential privacy (Round 86, the owner's choices of 2026-11-14): CirCor DigiScope, the
// person as the unit, trained offline on this project's GPU and recorded. Every figure is read from
// web/public/data/medical/murmur_dp.json; nothing here runs in the visitor's tab.
import { useEffect, useState } from "react";
import CardExport from "./CardExport";
import KeyPoints from "./KeyPoints";
import StageBadge from "./StageBadge";
import { MURMUR_ARMS, MURMUR_EPS, MURMUR_ROUTES, murmurDpRows, pick, type MurmurDp, type Stat2 } from "../lib/murmurDp";

const f3 = (v: number) => v.toFixed(3);
const f2 = (v: number) => v.toFixed(2);
const pct = (v: number) => `${(100 * v).toFixed(0)}%`;
const stat = (s?: Stat2) => (!s ? "-" : s.sd == null ? f3(s.mean) : `${f3(s.mean)} ± ${f3(s.sd)}`);
const HEAD = "head-panns";

export default function MurmurDpCard() {
  const [c, setC] = useState<MurmurDp | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/medical/murmur_dp.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j: MurmurDp) => { if (live) setC(j); })
      .catch((e: unknown) => { if (live) setFailed(String(e)); });
    return () => { live = false; };
  }, []);
  if (failed) {
    return <div className="card" id="murmur-dp" style={{ marginTop: 14 }}><h3>Murmur detection under differential privacy</h3><p className="small" style={{ color: "var(--danger)" }}>Its recorded figures did not load: {failed}.</p></div>;
  }
  if (!c) return null;
  const s = c.summary, d = c.data, pub = s.public;
  const cen = (eps: number | null, kind = "DP", route = "public", model = HEAD) => pick(s[route], model, "central", kind, eps);
  const bal = (eps: number) => pick(pub, HEAD, "central", "DP", eps)?.at_thresholds;
  const full8 = pick(pub, HEAD, "central-full", "DP", 8), fed8 = pick(pub, HEAD, "federated", "DP", 8);
  const val8 = cen(8, "DP", "circor"), val1 = cen(1, "DP", "circor");
  const naive8 = pick(s.naive, "head-public", "central", "DP", 8);
  const search8 = pick(s.search, HEAD, "central", "DP", 8), search1 = pick(s.search, HEAD, "central", "DP", 1);
  const op8 = pick(s.objpert, HEAD, "central", "DP", 8), op1 = pick(s.objpert, HEAD, "central", "DP", 1);
  const routeCell = (route: string, model: string, eps: number) => stat(pick(s[route], model, "central", "DP", eps)?.auroc);
  return (
    <div className="card" id="murmur-dp" style={{ marginTop: 14 }}>
      <h3>Murmur detection under differential privacy, the person as the unit <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        Present against absent on the CirCor DigiScope recordings ({d.people.toLocaleString("en-US")} people, {d.records_labelled.toLocaleString("en-US")} labelled
        records), scored on {d.test_people} held-out people ({d.test_records} records, {d.test_records_present} with a murmur). {d.people_with_two_records} people
        came to both screening campaigns under two patient IDs, so the person, not the ID, is the unit of the split and of privacy. A logistic head is trained
        privately on frozen features of an audio network trained on AudioSet (PANNs CNN14), the recipe Abadi et al. used for CIFAR-10: features learned on public
        data, only the last layer private. Learning rates and decision thresholds are chosen on a stand-in of the same size from the PhysioNet/CinC 2016 recordings,
        so epsilon covers everything that touched CirCor. Delta {c.recipe.delta.toExponential(0)}. Trained on this project&apos;s GPU and recorded: nothing here ran in
        your tab, and no weights are published.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table small">
          <thead>
            <tr><th>AUROC on the held-out people</th><th>No noise</th><th>Clip only</th>{MURMUR_EPS.map((e) => <th key={e}>ε {e}</th>)}</tr>
          </thead>
          <tbody>{MURMUR_ARMS.map(([arm, label]) => (
            <tr key={arm}>
              <td>{label}</td>
              <td>{stat(pick(pub, HEAD, arm, "non-private", null)?.auroc)}</td>
              <td>{stat(pick(pub, HEAD, arm, "clip only", null)?.auroc)}</td>
              {MURMUR_EPS.map((e) => <td key={e}>{stat(pick(pub, HEAD, arm, "DP", e)?.auroc)}</td>)}
            </tr>
          ))}</tbody>
        </table>
      </div>
      <p className="muted small" style={{ margin: "6px 0 10px" }}>
        Mean over five seeds, ± their standard deviation. A record&apos;s score is the mean of its windows&apos; logits. Central DP-SGD samples about{" "}
        {c.recipe.expected_batch} people a step and clips each person&apos;s gradient to {c.recipe.clip}; the four sites each add a share of discrete Gaussian
        noise inside the secure sum, a guarantee that holds while all four add theirs.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table small">
          <thead>
            <tr><th>Every route, central arm</th><th>PANNs ε 8</th><th>PANNs ε 1</th><th>PhysioNet CNN ε 8</th><th>PhysioNet CNN ε 1</th></tr>
          </thead>
          <tbody>{MURMUR_ROUTES.filter(([route]) => s[route]).map(([route, label]) => (
            <tr key={route}>
              <td>{label}</td>
              <td>{routeCell(route, HEAD, 8)}</td><td>{routeCell(route, HEAD, 1)}</td>
              <td>{routeCell(route, "head-public", 8)}</td><td>{routeCell(route, "head-public", 1)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <KeyPoints title="What it shows" items={[
        <><b>Public features keep most of the accuracy.</b> The head reaches {stat(cen(null, "non-private")?.auroc)} without noise and {stat(cen(8)?.auroc)} at ε 8,{" "}
          {stat(cen(1)?.auroc)} at ε 1, with {c.recipe.params[HEAD]} weights to learn under noise. With a few hundred people, private learning needs features learned on
          public data or far more data (Tramèr and Boneh, ICLR 2021); the same head on the heart-sound card&apos;s CNN features, which come from a single database,
          keeps less ({stat(cen(8, "DP", "public", "head-public")?.auroc)} at ε 8).</>,
        <><b>Tuning on public data cost almost nothing here.</b> Choosing the rates on CirCor&apos;s own validation people gives {stat(val8?.auroc)} at ε 8 and{" "}
          {stat(val1?.auroc)} at ε 1, but that epsilon leaves the choice out (Papernot and Steinke, ICLR 2022). The stand-in has to be new to the feature network: drawn
          from the databases the heart-sound card&apos;s CNN was trained on, its validation was too easy and the rates it chose gave {stat(naive8?.auroc)} at ε 8. An
          accounted search over three public-chosen rates gives {stat(search8?.auroc)} and {stat(search1?.auroc)}; objective perturbation (Redberg, Koskela and
          Wang, NeurIPS 2023), which needs no rate at all, {stat(op8?.auroc)} and {stat(op1?.auroc)}.</>,
        <><b>The distributed noise costs nothing extra.</b> At ε 8 the four sites&apos; discrete Gaussian gives {stat(fed8?.auroc)}, the central full-batch Gaussian
          {" "}{stat(full8?.auroc)}: the same noise once quantised to 2^14 a clip, as Kairouz, Liu and Steinke (ICML 2021) report.</>,
        <><b>The ranking survives, the threshold does not.</b> At logit 0 the private head calls {pct(bal(1)?.zero?.share_predicted_positive.mean ?? NaN)} of records
          positive at ε 1, against {pct(d.test_records_present / d.test_records)} that are, so its balanced accuracy is {f2(bal(1)?.zero?.balanced_accuracy.mean ?? NaN)}.
          A threshold chosen on the stand-in raises it to {f2(bal(8)?.balanced?.balanced_accuracy.mean ?? NaN)} at ε 8 and{" "}
          {f2(bal(1)?.balanced?.balanced_accuracy.mean ?? NaN)} at ε 1, and still calls too few positive. AUROC alone hides this (Wen et al., 2026).</>,
      ]} />
      <div className="note" style={{ marginTop: 10, fontSize: 12 }}>
        <KeyPoints title="What it does not show" items={[
          <><b>A large test.</b> {d.test_people} people and {d.test_records_present} records with a murmur: differences under about 0.03 are within the seeds&apos; spread.</>,
          <><b>Choices made before the runs.</b> Which features, arms and routes to compare was decided after earlier runs on CirCor; epsilon covers each run, not that
            choice.</>,
          <><b>The page&apos;s own sampler.</b> The study draws its noise in floating point; the medical table above samples it exactly, in integers.</>,
          <><b>A device.</b> CirCor&apos;s public training set is children and young people in north-eastern Brazil; nothing here is a diagnosis.</>,
        ]} />
      </div>
      <p className="muted small" style={{ marginTop: 6 }}>
        Data: {c.attribution.circor.provider}, <a href={c.attribution.circor.download} target="_blank" rel="noreferrer">{c.attribution.circor.dataset}</a>, under the{" "}
        <a href={c.attribution.circor.licence_url} target="_blank" rel="noreferrer">{c.attribution.circor.licence}</a>; {c.attribution.circor.citation}; {c.attribution.platform_citation}.
        Stand-in: {c.attribution.physionet_2016.dataset}, {c.attribution.physionet_2016.licence}. Features: {c.attribution.panns.model},{" "}
        <a href={c.attribution.panns.licence_url} target="_blank" rel="noreferrer">{c.attribution.panns.licence}</a>, {c.attribution.panns.use}; {c.attribution.panns.citation}.
      </p>
      <CardExport name="medical_fl-murmur-dp" label="the murmur DP figures" rows={murmurDpRows(c)} json={c} />
    </div>
  );
}
