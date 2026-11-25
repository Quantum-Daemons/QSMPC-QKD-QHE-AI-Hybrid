// The satellite forecaster's threshold, against two calibrated ones (Round 86, batch 6; the owner's choice of 2026-11-26).
// A pre-registered comparison (references/satellite-threshold-calibration-preregistration.md): each threshold fitted on
// ESA-ADB's validation months alone and scored once on the test period's 29 anomalies, by
// scripts/measure_satellite_thresholds.py. The page's rule does not change; every figure is read from
// web/public/data/satellite/thresholds.json, which scripts/export_satellite_thresholds_card.py copies from the record.
import { useEffect, useState } from "react";
import StageBadge from "./StageBadge";

interface Arm { key: string; label: string; f0_5: number; tp: number; fp: number; fn: number; share_flagged: number }
interface Thresholds {
  arms: Arm[]; events: string; metric: string; vus_pr_of_the_shared_score: number; ndt_matches_the_students_record: boolean;
  preregistration: { file: string; sha256: string }; record: string;
  attribution: { dataset: string; licence: string; licence_url: string; citation: string; download: string };
}

export default function SatThresholdsCard() {
  const [t, setT] = useState<Thresholds | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/satellite/thresholds.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j: Thresholds) => { if (live) setT(j); })
      .catch((e: unknown) => { if (live) setFailed(String(e)); });
    return () => { live = false; };
  }, []);
  if (failed) {
    return <div className="card" id="sat-thresholds" style={{ marginTop: 14 }}><h3>The forecaster&apos;s threshold, against calibrated ones</h3>
      <p className="small" style={{ color: "var(--danger)" }}>The comparison&apos;s figures could not be read ({failed}).</p></div>;
  }
  if (!t) return null;
  const ndt = t.arms.find((a) => a.key === "ndt");
  return (
    <div className="card" id="sat-thresholds" style={{ marginTop: 14 }}>
      <h3>The forecaster&apos;s threshold, against calibrated ones <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        A pre-registered comparison, run offline: the CKKS student&apos;s forecasts, their errors smoothed as Telemanom smooths
        them, and three rules that turn them into alarms, each fitted on the validation months alone and scored once on
        {" "}{t.events}, by {t.metric}. Only the threshold differs; the encrypted forecasts are the same. The page keeps its rule,
        Telemanom&apos;s non-parametric dynamic threshold{ndt ? <>, which scores {ndt.f0_5.toFixed(3)}</> : null}: a threshold fixed on
        three months does not carry across seven years of telemetry whose nominal level drifts.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="small" style={{ width: "100%", borderCollapse: "collapse" }}>
          <caption className="muted" style={{ textAlign: "left" }}>Corrected event-wise F0.5 on the test period&apos;s anomalies, by rule</caption>
          <thead><tr><th scope="col" style={{ textAlign: "left" }}>Rule</th><th scope="col">F0.5</th><th scope="col">tp / fp / fn</th><th scope="col">test steps flagged</th></tr></thead>
          <tbody>
            {t.arms.map((a) => (
              <tr key={a.key} style={a.key === "ndt" ? { fontWeight: 600 } : undefined}>
                <td>{a.label}</td><td style={{ textAlign: "center" }}>{a.f0_5.toFixed(3)}</td>
                <td style={{ textAlign: "center" }}>{a.tp} / {a.fp} / {a.fn}</td>
                <td style={{ textAlign: "center" }}>{(a.share_flagged * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted" style={{ marginTop: 6 }}>
        VUS-PR of the score the rules share, which no threshold changes: {t.vus_pr_of_the_shared_score.toFixed(3)}. The NDT row
        {t.ndt_matches_the_students_record ? " reproduces the student's own record" : " does not reproduce the student's record"}.
        Pre-registration: <code>{t.preregistration.file}</code>; record: <code>{t.record}</code>. Data: {t.attribution.dataset},{" "}
        <a href={t.attribution.licence_url} target="_blank" rel="noreferrer">{t.attribution.licence}</a>; {t.attribution.citation}
      </p>
    </div>
  );
}
