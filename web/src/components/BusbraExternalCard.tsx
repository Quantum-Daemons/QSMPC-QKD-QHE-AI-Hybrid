// The published BreastMNIST models on BUS-BRA, an external site (Round 86, batches 12 and 13; the owner's choice of
// 2026-11-28 to show it). Pre-registered in references/medical-busbra-external-test-preregistration.md and run offline by
// scripts/check_busbra_external.py; scripts/export_outside_data_cards.py writes the file this reads, and every figure
// below is read from it. Nothing renders until it loads.
import { useEffect, useState } from "react";
import CardExport from "./CardExport";
import StageBadge from "./StageBadge";

interface Model {
  key: string; label: string; breastmnist_auc: number; busbra_auc: number; busbra_auc_ci95: [number, number];
  busbra_auprc: number; busbra_accuracy: number; malignant_detected: number; per_device_auc: Record<string, number>;
}
interface Busbra {
  preregistration: string; record: string; images: number; patients: number; malignant: number;
  devices: Record<string, number>; models: Model[];
  ckks: { student_largest_logit_error: number; student_decisions_changed: number;
    square_student: Record<string, { largest_logit_error: number; decisions_changed: number }> };
  attribution: string;
}

const f2 = (v: number) => v.toFixed(2);
const f3 = (v: number) => v.toFixed(3);

export default function BusbraExternalCard() {
  const [d, setD] = useState<Busbra | null>(null);
  useEffect(() => {
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/medical/busbra_external.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j: Busbra | null) => { if (live && j?.models) setD(j); })
      .catch(() => undefined);
    return () => { live = false; };
  }, []);
  if (!d) return null;
  const aucs = d.models.map((m) => m.busbra_auc), inside = d.models.map((m) => m.breastmnist_auc);
  const devices = Object.keys(d.devices).length;
  const sq = d.ckks.square_student;
  return (
    <div className="card" id="medical-busbra" style={{ marginTop: 14 }}>
      <h3>The BreastMNIST models on another site: BUS-BRA <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        Every BreastMNIST figure on this page is measured on BreastMNIST&apos;s own test split, which comes from one site. The
        published models were scored, unchanged, on BUS-BRA: {d.images.toLocaleString("en-US")} breast ultrasound images of{" "}
        {d.patients.toLocaleString("en-US")} patients from Brazil, biopsy-proven, from {devices} devices, each image resized
        whole to 28 x 28 by cubic spline as MedMNIST v2 states its method. Their malignant-class AUC falls from{" "}
        {f2(Math.min(...inside))}-{f2(Math.max(...inside))} to {f2(Math.min(...aucs))}-{f2(Math.max(...aucs))}.
        BUS-BRA&apos;s images are framed differently from BUSI&apos;s, so the fall is not the site&apos;s alone. No model was retrained,
        recalibrated or thresholded on it.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table small">
          <thead><tr><th>Model</th><th>AUC, BreastMNIST test</th><th>AUC, BUS-BRA (95% by patient)</th><th>Accuracy, BUS-BRA</th>
            <th>Malignant found</th></tr></thead>
          <tbody>
            {d.models.map((m) => (
              <tr key={m.key}>
                <td>{m.label}</td><td>{f3(m.breastmnist_auc)}</td>
                <td><b>{f3(m.busbra_auc)}</b> ({f3(m.busbra_auc_ci95[0])}-{f3(m.busbra_auc_ci95[1])})</td>
                <td>{f3(m.busbra_accuracy)}</td><td>{m.malignant_detected} of {d.malignant}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small" style={{ marginTop: 6 }}>
        Under CKKS the students still decide every BUS-BRA image as their plaintext networks do: the CKKS student&apos;s largest
        logit error is {d.ckks.student_largest_logit_error.toExponential(1)} with {d.ckks.student_decisions_changed} decisions
        changed, and the square student&apos;s is{" "}
        {Object.entries(sq).map(([s, v], i) => <span key={s}>{i ? " and " : ""}{v.largest_logit_error.toExponential(1)} under {s}</span>)}{" "}
        with no decision changed. Pre-registered (<code>{d.preregistration}</code>); record <code>{d.record}</code>.
      </p>
      <p className="muted small" style={{ marginTop: 6 }}>Data: {d.attribution}.</p>
      <CardExport name="medical_fl-busbra-external" label="the BUS-BRA figures" rows={d.models.map((m) => ({
        model: m.label, auc_breastmnist_test: m.breastmnist_auc, auc_busbra: m.busbra_auc,
        auc_busbra_low: m.busbra_auc_ci95[0], auc_busbra_high: m.busbra_auc_ci95[1], accuracy_busbra: m.busbra_accuracy,
        malignant_found: m.malignant_detected,
      }))} json={d} />
    </div>
  );
}
