// The 3D nodule student reported at each of its steps, as Lee et al. report theirs (Round 86, batch 8; the owner's
// choice of 2026-11-27). Pre-registered in references/medical-nodule3d-reporting-preregistration.md, computed offline by
// scripts/train/medical_nodule3d_report.py and exported by scripts/export_nodule3d_reporting.py to
// web/public/data/medical/nodule3d_reporting.json, which this reads; nothing renders until that file exists.
import { useEffect, useState } from "react";

type Fig = { auc: number; auc_ci95: [number, number]; auprc: number; auprc_ci95: [number, number] };
type Mean = { auc: number; auc_sd: number; auprc: number; auprc_sd: number };
interface Reporting {
  preregistration: string; record: string; device: string; resamples: number; seeds: number;
  test_part: { crops: number; patients: number; malignant: number; malignant_share: number };
  teacher: Fig & { what: string };
  students: { square: Mean; control: Mean };
  shipped_seed: Fig & { seed: number };
  encrypted: Record<string, Fig & { plaintext_same_run: Fig; max_abs_logit_error: number; crops: number }>;
}

const f3 = (v: number) => v.toFixed(3);
const ci = (f: Fig, k: "auc" | "auprc") => `${f3(f[k])} (${f3(f[`${k}_ci95`][0])} to ${f3(f[`${k}_ci95`][1])})`;
const mean = (m: Mean, k: "auc" | "auprc") => `${f3(m[k])} (sd ${f3(m[`${k}_sd`])})`;

export default function Nodule3dReportingSection() {
  const [d, setD] = useState<Reporting | null>(null);
  useEffect(() => {
    let on = true;
    fetch(`${import.meta.env.BASE_URL}data/medical/nodule3d_reporting.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j: Reporting | null) => { if (on && j?.teacher) setD(j); })
      .catch(() => undefined);
    return () => { on = false; };
  }, []);
  if (!d) return null;
  const sets = Object.entries(d.encrypted).sort(([a], [b]) => a.localeCompare(b));
  return (
    <div style={{ marginTop: 10 }}>
      <h4 style={{ margin: "6px 0 2px" }}>At each step, as Lee et al. report theirs</h4>
      <p className="small" style={{ margin: 0 }}>
        AUC and AUPRC for malignant against benign at each step, on the same {d.test_part.crops} test crops of{" "}
        {d.test_part.patients} patients, as Lee et al. (Radiology: Artificial Intelligence 7(6):e240798, 2025) report their
        reference, approximated and encrypted networks. Brackets: 95% percentile intervals over {d.resamples.toLocaleString("en-US")}{" "}
        resamples of the test patients, or the standard deviation over the {d.seeds} seeds. A model that ranks at random has an
        AUPRC of about {f3(d.test_part.malignant_share)}, the malignant share. Pre-registered; no figure here decides anything.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="small" style={{ width: "100%", borderCollapse: "collapse", marginTop: 4 }}>
          <thead><tr><th scope="col" style={{ textAlign: "left" }}>Step</th><th scope="col">AUC</th><th scope="col">AUPRC</th></tr></thead>
          <tbody>
            <tr><td>The teacher: the mean of the 3D ResNet-18&apos;s five fold networks</td>
              <td style={{ textAlign: "center" }}>{ci(d.teacher, "auc")}</td><td style={{ textAlign: "center" }}>{ci(d.teacher, "auprc")}</td></tr>
            <tr><td>The ReLU control, {d.seeds} seeds, in the clear</td>
              <td style={{ textAlign: "center" }}>{mean(d.students.control, "auc")}</td>
              <td style={{ textAlign: "center" }}>{mean(d.students.control, "auprc")}</td></tr>
            <tr><td>The square student, {d.seeds} seeds, in the clear</td>
              <td style={{ textAlign: "center" }}>{mean(d.students.square, "auc")}</td>
              <td style={{ textAlign: "center" }}>{mean(d.students.square, "auprc")}</td></tr>
            <tr><td>The shipped seed ({d.shipped_seed.seed}), in the clear</td>
              <td style={{ textAlign: "center" }}>{ci(d.shipped_seed, "auc")}</td><td style={{ textAlign: "center" }}>{ci(d.shipped_seed, "auprc")}</td></tr>
            {sets.map(([name, e]) => (
              <tr key={name}><td>The shipped seed under CKKS {name}: its decrypted logits (largest error {e.max_abs_logit_error.toExponential(1)})</td>
                <td style={{ textAlign: "center" }}>{ci(e, "auc")}</td><td style={{ textAlign: "center" }}>{ci(e, "auprc")}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted" style={{ marginTop: 4 }}>
        {d.teacher.what}. Each CKKS row is one run, since the scheme&apos;s noise differs between runs. Computed on the{" "}
        {d.device === "cpu" ? "CPU" : d.device}. Pre-registration: <code>{d.preregistration}</code>; record: <code>{d.record}</code>.
      </p>
    </div>
  );
}
