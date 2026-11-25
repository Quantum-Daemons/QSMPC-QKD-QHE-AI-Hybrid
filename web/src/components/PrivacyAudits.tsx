// What the models' outputs leak, as recorded (the owner's choice of 2026-11-24: the third layer of the PETs evaluation,
// references/pets-evaluation-2026-11.md, shown on the page). Every figure is read from web/src/lib/privacyAudits.json,
// which scripts/export_privacy_audits.py copies from the records in docs/measurements; nothing is recomputed in the tab,
// and no reconstructed record or image is shown, because the records hold none.
import { useId, useState } from "react";
import {
  Bar, CartesianGrid, Cell, ComposedChart, Legend, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip,
  XAxis, YAxis,
} from "recharts";
import Details from "./Details";
import KeyPoints from "./KeyPoints";
import { ChartReadout, HINT_POINTER, ReadoutTooltip } from "./ChartReadout";
import { MarksHidden } from "./VerificationChart";
import audits from "../lib/privacyAudits.json";

type Study = (typeof audits.membership)[number];
type Model = Study["models"][number];
type Arm = (typeof audits.dp_one_run.arms)[number];

const tick = { fill: "var(--muted)", fontSize: 11 };
const FPRS = audits.fprs;
const X_DOMAIN: [number, number] = [5e-4, 0.2];
const Y_DOMAIN: [number, number] = [1e-4, 1];
/** A colour and a shape for each model, so two lines differ by more than colour (WCAG 2.2 SC 1.4.1). */
const SERIES_COLOURS = ["var(--danger)", "var(--accent)", "var(--accent-2)", "var(--warn)", "var(--quantum)"] as const;
const MARKS = [
  { shape: "diamond", dash: "" },
  { shape: "circle", dash: "6 3" },
  { shape: "square", dash: "2 3" },
  { shape: "triangle", dash: "8 3 2 3" },
  { shape: "cross", dash: "1 3" },
] as const;

const pct = (v: number | null) => (v == null ? "-" : `${(100 * v).toFixed(2)}%`);
const fprName = (f: number) => `${(100 * f).toString()}%`;
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

function MembershipChart({ study, captionId }: { study: Study; captionId: string }) {
  const series = study.models.map((m, i) => ({
    m, mark: MARKS[i % MARKS.length], colour: SERIES_COLOURS[i % SERIES_COLOURS.length],
    points: m.lira.tpr.map((tpr, k) => ({ fpr: FPRS[k], tpr, model: m.label }))
      .filter((p): p is { fpr: number; tpr: number; model: string } => p.tpr != null && p.tpr > 0),
  }));
  return (
    <ChartReadout hint={HINT_POINTER}>
      <MarksHidden>
      <ResponsiveContainer width="100%" height={260}>
        <ScatterChart role="img" aria-describedby={captionId}
          title={`LiRA's true-positive rate against the false-positive rate, log-log, for ${study.title}`}
          margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis type="number" dataKey="fpr" name="false-positive rate" scale="log" domain={X_DOMAIN} ticks={FPRS}
            allowDataOverflow tick={tick} tickFormatter={fprName}
            label={{ value: "false-positive rate (log)", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} />
          <YAxis type="number" dataKey="tpr" name="true-positive rate" scale="log" domain={Y_DOMAIN}
            ticks={[1e-4, 1e-3, 1e-2, 1e-1, 1]} allowDataOverflow tick={tick} width={52}
            tickFormatter={(v: number) => (v >= 0.01 ? `${100 * v}%` : `${(100 * v).toPrecision(1)}%`)} />
          <Tooltip content={<ReadoutTooltip />}
            formatter={(v: number, name: string, entry: { payload?: { model: string } }, i: number) =>
              [pct(v), i === 0 ? `${entry.payload?.model ?? ""}: ${name}` : name]} />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          <ReferenceLine segment={[{ x: X_DOMAIN[0], y: X_DOMAIN[0] }, { x: X_DOMAIN[1], y: X_DOMAIN[1] }]}
            stroke="var(--muted)" strokeDasharray="4 4" />
          {series.map(({ m, mark, colour, points }, i) => (
            <Scatter key={m.key} name={`${m.label} (${mark.shape})`} data={points} fill={SERIES_COLOURS[i % SERIES_COLOURS.length]}
              shape={mark.shape} line={{ stroke: colour, strokeDasharray: mark.dash }} lineType="joint"
              isAnimationActive={false} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
      </MarksHidden>
    </ChartReadout>
  );
}

function rmiaCell(m: Model, k: number) {
  const v = m.rmia.tpr[k];
  if (v != null) return pct(v);
  const why = m.rmia.not_resolvable.find((s) => s.startsWith(`fpr ${FPRS[k]}`));
  return <span title={why ?? undefined}>not resolvable</span>;
}

function DpAuditChart({ study, captionId }: { study: string; captionId: string }) {
  const rows = audits.dp_one_run.arms.filter((a: Arm) => a.study === study).map((a: Arm) => ({ ...a, name: a.arm }));
  return (
    <ChartReadout hint={HINT_POINTER}>
      <ResponsiveContainer width="100%" height={34 * rows.length + 64}>
        <ComposedChart layout="vertical" data={rows} role="img" aria-describedby={captionId}
          title={`${study}: for each arm, the epsilon its accountant states beside the one-run audit's lower bound`}
          margin={{ top: 8, right: 24, bottom: 18, left: 8 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" tick={tick} domain={[0, "auto"]}
            label={{ value: "epsilon", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} />
          <YAxis type="category" dataKey="name" tick={{ ...tick, width: 140 }} width={150} interval={0} />
          <Tooltip content={<ReadoutTooltip />} formatter={(v: number, name: string) => [v.toFixed(2), name]} />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="accounted" name="the accountant's epsilon (grey)" fill="var(--muted)" isAnimationActive={false} barSize={10} />
          <Bar dataKey="lower_bound" name="the audit's lower bound (green; red where a bug was seeded)" fill="var(--accent-2)"
            isAnimationActive={false} barSize={10}>
            {rows.map((r) => <Cell key={r.name} fill={r.seeded_bug ? "var(--danger)" : "var(--accent-2)"} />)}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
    </ChartReadout>
  );
}

export default function PrivacyAudits() {
  const [pick, setPick] = useState(0);
  const study = audits.membership[pick];
  const miaCaption = useId(), dpCaption = useId();
  const pipeline = audits.membership.find((s) => s.file.startsWith("mia-breastmnist-pipeline"));
  const teacher = pipeline?.models.find((m) => m.key === "teacher");
  const honest = audits.dp_one_run.arms.filter((a) => !a.seeded_bug);
  const seeded = audits.dp_one_run.arms.filter((a) => a.seeded_bug);
  const files = Object.keys(audits.sources);
  return (
    <div className="card">
      <p className="small" style={{ marginTop: 0 }}>
        What this project&apos;s own models give away through their outputs, measured offline with published attacks and
        recorded in <code>docs/measurements</code>: {files.length} records, read here, not run in your tab. An attack&apos;s
        figure is a lower bound on what leaks, never proof that nothing does, and only metrics are published: no
        reconstructed record or image.
      </p>
      <KeyPoints items={[
        <>Membership inference: shadow-model attacks (LiRA and RMIA, 64 shadow models each) on {audits.membership.length} studies
          in {new Set(audits.membership.map((s) => s.use_case)).size} use cases. The measure is the true-positive rate at a low
          false-positive rate, per record, because an average hides the records most exposed.</>,
        teacher ? <>In BreastMNIST&apos;s whole pipeline the teacher leaks most: {pct(teacher.lira.tpr[0])} of its members found at a
          false-positive rate of {fprName(FPRS[0])}, against {pipeline!.models.filter((m) => m.key !== "teacher").map((m) => pct(m.lira.tpr[0])).join(" and ")} through
          the students.</> : null,
        <>A one-run DP audit: {honest.filter((a) => a.lower_bound <= a.accounted).length} of the {honest.length} arms measure under
          the epsilon their accountant states, and {seeded.filter((a) => a.lower_bound > a.accounted).length} of the {seeded.length}{" "}
          arms with a bug seeded on purpose measure over it. Under the accountant is consistent with the accounting and proves
          nothing more.</>,
      ].filter(Boolean)} />

      <h3 style={{ marginBottom: 4 }}>Membership inference</h3>
      <label className="small">
        Study:{" "}
        <select value={pick} onChange={(e) => setPick(Number(e.target.value))}>
          {audits.membership.map((s, i) => <option key={s.file} value={i}>{s.use_case}: {s.title}</option>)}
        </select>
      </label>
      <figure className="chart-figure" style={{ margin: "8px 0" }}>
        <figcaption id={miaCaption} className="small muted">
          <b>{study.title}.</b> LiRA (Carlini et al., IEEE S&amp;P 2022), online, with {study.shadow_models} shadow models
          ({study.membership}): the share of members it finds (true-positive rate) at each false-positive rate, both on log
          scales. One record is {study.unit}; {study.audit_records} records were audited. The dashed diagonal is a guess
          (TPR = FPR): a model whose line lies on it tells the attack nothing more.
        </figcaption>
        <MembershipChart study={study} captionId={miaCaption} />
      </figure>
      <div style={{ overflowX: "auto" }}>
        <table className="table small">
          <thead>
            <tr><th>Model</th><th>AUC, LiRA / RMIA</th>
              {FPRS.map((f) => <th key={f}>TPR at {fprName(f)} FPR, LiRA / RMIA</th>)}
              <th>Records found in most targets at {fprName(FPRS[0])}</th></tr>
          </thead>
          <tbody>
            {study.models.map((m) => (
              <tr key={m.key}>
                <td title={m.description}>{m.label}</td>
                <td>{m.lira.auc.toFixed(3)} / {m.rmia.auc.toFixed(3)}</td>
                {FPRS.map((f, k) => <td key={f}>{pct(m.lira.tpr[k])} / {rmiaCell(m, k)}</td>)}
                <td>{m.exposed} of {m.audit_records}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small" style={{ margin: "4px 0 0" }}>
        RMIA is Zarifzadeh, Liu and Shokri (ICML 2024); &quot;not resolvable&quot; means its top score ties across more
        non-members than that false-positive rate allows. Data: {study.data}. Record: <code>docs/measurements/{study.file}</code>.
      </p>

      <h3 style={{ marginTop: 22, marginBottom: 4 }}>Differential privacy, audited in one run</h3>
      <figure className="chart-figure" style={{ margin: "8px 0" }}>
        <figcaption id={dpCaption} className="small muted">
          <b>The epsilon each accountant states, and the lower bound an audit measures.</b> {audits.dp_one_run.method}, at
          {" "}{Math.round(100 * audits.dp_one_run.confidence)}% confidence, with canaries an adversary sees in every release.
          Two arms carry a bug seeded on purpose; the audit should put them over their accountant, and does.
        </figcaption>
        {[...new Set(audits.dp_one_run.arms.map((a) => a.study))].map((study) => (
          <div key={study}>
            <div className="small" style={{ margin: "8px 0 0" }}><b>{study}</b></div>
            <DpAuditChart study={study} captionId={dpCaption} />
          </div>
        ))}
      </figure>
      <p className="muted small" style={{ margin: "4px 0 0" }}>
        Canaries: {sum(audits.dp_one_run.arms.map((a) => a.canaries)).toLocaleString("en-US")} over the {audits.dp_one_run.arms.length} arms.
        Record: <code>docs/measurements/{audits.dp_one_run.file}</code>.
      </p>

      <h3 style={{ marginTop: 22, marginBottom: 4 }}>A guessing game in one run</h3>
      <p className="small muted" style={{ marginTop: 0 }}>
        {audits.dp_kcandidate.method}. Each slot holds one of k canaries; the adversary names it or abstains. The lower bound
        on mu is the largest mu-GDP claim the run rejects at 95% confidence; beside it, the accuracy of naming every slot
        against {audits.dp_kcandidate.bound} (riskcal {audits.dp_kcandidate.riskcal}). The murmur study is not Gaussian
        DP, so its rows compare epsilon, the lower bound read inside the Gaussian family as the paper does.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table small">
          <thead><tr><th>Study and arm</th><th>Game</th><th>Lower bound / accountant</th><th>Naming every slot (95% interval)</th>
            <th>Bound 1 - f(1/k)</th><th>Holds</th></tr></thead>
          <tbody>
            {audits.dp_kcandidate.rows.map((r) => (
              <tr key={`${r.study}-${r.arm}-${r.game}`} style={r.seeded_bug ? { color: "var(--danger)" } : undefined}>
                <td>{r.study}: {r.arm}</td>
                <td>{r.game}</td>
                <td>{r.mu_accounted != null
                  ? <>mu {r.mu_lower_bound.toFixed(3)} / {r.mu_accounted.toFixed(3)}</>
                  : <>epsilon {r.epsilon_lower_bound.toFixed(2)} / {Number(r.epsilon_accounted).toFixed(2)}</>}</td>
                <td>{pct(r.accuracy)} ({pct(r.interval_95[0])} to {pct(r.interval_95[1])}); a guess {pct(r.baseline)}</td>
                <td>{pct(r.reconstruction_bound)}</td>
                <td>{r.under_the_accountant && r.within_the_bound ? "both" : r.seeded_bug ? "neither: the seeded bug is caught" : "no"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small" style={{ margin: "4px 0 14px" }}>
        Red rows carry a bug seeded on purpose. Record: <code>docs/measurements/{audits.dp_kcandidate.file}</code>.
      </p>

      <Details summary="Reconstruction, attribute inference and gradient inversion">
        {audits.reconstruction.map((r) => (
          <div key={r.file}>
            <p className="small" style={{ marginBottom: 4 }}>
              <b>{r.use_case}: {r.title}, an informed adversary</b> (Balle, Cherubin and Hayes, IEEE S&amp;P 2022), who knows
              every training record but one: the share of the {r.records.toLocaleString("en-US")} records rebuilt within a
              relative error of 0.1%, 1% and 10%.
            </p>
            <table className="table small">
              <thead><tr><th>Route</th><th>Median relative error</th><th>Within 0.1%</th><th>Within 1%</th><th>Within 10%</th></tr></thead>
              <tbody>
                {r.routes.map((x) => (
                  <tr key={x.label}><td>{x.label}</td><td>{x.median_relative_error.toExponential(2)}</td>
                    {x.share_within.map((s, k) => <td key={k}>{pct(s)}</td>)}</tr>
                ))}
              </tbody>
            </table>
            <p className="muted small" style={{ margin: "4px 0 10px" }}>Record: <code>docs/measurements/{r.file}</code>.</p>
          </div>
        ))}
        <p className="small" style={{ marginBottom: 4 }}>
          <b>Medical: attribute inference on the heart models</b>, asked {audits.attribute_inference.attribute}; accuracy on
          the training rows (members) and on held-out rows, beside imputation that uses no model (Jayaraman and Evans, CCS 2022).
        </p>
        <table className="table small">
          <thead><tr><th>Model</th><th>Attack</th><th>Members</th><th>Non-members</th></tr></thead>
          <tbody>
            {audits.attribute_inference.rows.map((x) => (
              <tr key={`${x.model}-${x.attack}`}><td>{x.model}</td><td>{x.attack}</td><td>{pct(x.members)}</td><td>{pct(x.non_members)}</td></tr>
            ))}
          </tbody>
        </table>
        <p className="muted small" style={{ margin: "4px 0 10px" }}>Record: <code>docs/measurements/{audits.attribute_inference.file}</code>.</p>
        <p className="small" style={{ marginBottom: 4 }}>
          <b>Medical: gradient inversion on the chest X-ray federation</b> (Geiping et al., NeurIPS 2020), one gradient step at
          the published weights with the labels known: the median structural similarity (SSIM) of each reconstruction to its
          film, beside a flat grey image and the most similar other film.
        </p>
        <table className="table small">
          <thead><tr><th>Setting</th><th>SSIM, reconstruction</th><th>SSIM, flat grey</th><th>SSIM, nearest other film</th><th>Films closer to themselves</th></tr></thead>
          <tbody>
            {audits.gradient_inversion.rows.map((x) => (
              <tr key={x.setting}><td>{x.setting}</td><td>{x.ssim_median.toFixed(3)}</td><td>{x.flat_grey_ssim_median.toFixed(3)}</td>
                <td>{x.nearest_other_film_ssim_median.toFixed(3)}</td><td>{x.films_closer_to_themselves}</td></tr>
            ))}
          </tbody>
        </table>
        <p className="muted small" style={{ margin: "4px 0 0" }}>
          This bounds that attack at that budget; it is not evidence that a hospital&apos;s gradient is safe to show the
          aggregator. Record: <code>docs/measurements/{audits.gradient_inversion.file}</code>.
        </p>
      </Details>
    </div>
  );
}
