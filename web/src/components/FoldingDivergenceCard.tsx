// The genomics page's encrypted predictor of 3D folding divergence (Round 86, batch 6; the owner's option D1, 2026-11-26),
// pre-registered in references/genomics-3d-divergence-preregistration.md. A small linear model per window, taught by
// Akita's predictions from Gilbertson et al.'s data, runs under CKKS (node-seal) on one person's encrypted carrier
// indicators; the card sets the decrypted outputs beside the plaintext ones and beside the teacher. Every figure is read
// from web/public/data/genomics/folding_divergence.json, which scripts/train/genomics_3d_divergence.py writes only when
// the pre-registered criteria hold.
import { useEffect, useState } from "react";
import StageBadge from "./StageBadge";
import { encryptedDense } from "../lib/clientFHE";

interface Window { chr: string; start: number; test_r2: number; test_spearman: number; super_population_r2: number; test_rows: number }
interface Payload {
  what: string; preregistration: string; record: string; windows: Window[];
  variants: { rsid: string; chr: string; pos: number; ref: string; alt: string; window: number }[];
  W: number[][]; b: number[]; sample_ids: string[]; super_population: string[]; carriers: number[][];
  teacher: (number | null)[][]; genotype_caveat: string; licence: string; licence_url: string; attribution: string;
  modifications: string;
}
interface Run { person: number; outputs: number[]; plain: number[]; maxAbsError: number; ms: number; ctBytes: number | null }

const mb = (w: Window) => `${w.chr}:${(w.start / 1e6).toFixed(2)} Mb`;
const fmt = (v: number) => v.toExponential(3);

export default function FoldingDivergenceCard() {
  const [d, setD] = useState<Payload | null>(null);
  const [person, setPerson] = useState(0);
  const [run, setRun] = useState<Run | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let on = true;
    fetch(`${import.meta.env.BASE_URL}data/genomics/folding_divergence.json`)
      .then((r) => ((r.headers.get("content-type") ?? "").includes("application/json") ? r.json() : null))
      .then((j: Payload | null) => { if (on) setD(j); })
      .catch(() => { if (on) setD(null); });
    return () => { on = false; };
  }, []);
  if (!d) return null;
  const go = async () => {
    setBusy(true); setError(null);
    try {
      const r = await encryptedDense(d.W, d.b, d.carriers[person]);
      setRun({ person, outputs: r.logits, plain: r.plainLogits, maxAbsError: r.maxAbsError, ms: r.fheMs, ctBytes: r.ctBytes });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const shown = run && run.person === person ? run : null;
  return (
    <div className="card" id="folding-divergence" style={{ marginTop: 14 }}>
      <h3>Predicted 3D genome folding, computed under encryption <StageBadge kind="real" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        {d.what} A ridge regression per window on whether the person carries each of its {Math.round(d.variants.length / d.windows.length)} chosen
        variants, taught by Akita&apos;s predictions for 2,457 individuals and tested on individuals it was not fitted on
        (pre-registered: <code>{d.preregistration}</code>). Not a disease risk, and not a measurement of anyone&apos;s chromatin.
      </p>
      <label className="small">
        Person:{" "}
        <select value={person} onChange={(e) => setPerson(Number(e.target.value))}>
          {d.sample_ids.map((s, i) => <option key={s} value={i}>{s} ({d.super_population[i]})</option>)}
        </select>
      </label>{" "}
      <button className="btn" type="button" disabled={busy} onClick={go}>
        {busy ? "Predicting under CKKS..." : `Predict the ${d.windows.length} windows under CKKS`}
      </button>
      {error ? <p className="small" style={{ color: "var(--danger)" }}>{error}</p> : null}
      <div style={{ overflowX: "auto" }}>
        <table className="table small" style={{ marginTop: 8 }}>
          <thead><tr><th>Window</th><th>Test R&sup2; (super-population only)</th><th>Akita&apos;s divergence for this person</th>
            <th>Prediction, decrypted</th><th>|decrypted - plaintext|</th></tr></thead>
          <tbody>
            {d.windows.map((w, k) => (
              <tr key={k}>
                <td>{mb(w)}</td>
                <td>{w.test_r2.toFixed(3)} ({w.super_population_r2.toFixed(3)})</td>
                <td>{d.teacher[person][k] == null ? "no value in the table" : fmt(d.teacher[person][k] as number)}</td>
                <td>{shown ? fmt(shown.outputs[k]) : "-"}</td>
                <td>{shown ? Math.abs(shown.outputs[k] - shown.plain[k]).toExponential(1) : "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {shown ? (
        <p className="small" style={{ margin: "6px 0 0" }}>
          The person&apos;s {d.carriers[person].length} carrier indicators went into one ciphertext; the {d.windows.length} outputs came
          back in one, and only they were decrypted. Largest difference from the plaintext model {shown.maxAbsError.toExponential(1)},
          in {Math.round(shown.ms)} ms{shown.ctBytes != null ? `, a ${(shown.ctBytes / 1e3).toFixed(0)} kB ciphertext` : ""}.
        </p>
      ) : null}
      <p className="muted small" style={{ margin: "8px 0 0" }}>
        {d.genotype_caveat} Record: <code>{d.record}</code>. {d.attribution} Licence:{" "}
        <a href={d.licence_url} target="_blank" rel="noreferrer">{d.licence}</a>; {d.modifications}
      </p>
    </div>
  );
}
