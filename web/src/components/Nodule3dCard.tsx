// The 3D nodule student on the medical page (Round 86, batch 4; references/medical-nodule3d-square-student-
// preregistration.md and its addendum). Twelve LIDC-IDRI test crops, fixed by the addendum's section 4 before any model
// was trained, are drawn as three planes with the exact contribution map (Nodule3dViewer) and run end to end under CKKS
// in a Web Worker (lib/clientFHESquare3d.ts). The figures for the whole test part come from the record through
// web/public/data/medical/nodule3d_square_student.json (scripts/export_nodule3d_student.py); the run is computed here.
import { useEffect, useMemo, useState } from "react";
import Nodule3dViewer from "./Nodule3dViewer";
import StageBadge from "./StageBadge";
import { contributionMap, memoryNote, plainSquare3dLogits, runSquare3dOffThread, type Square3dRun,
  type Square3dWeights, VOXELS } from "../lib/clientFHESquare3d";

const LIMIT = 1e-3;
const CLASS_NAMES = ["benign", "malignant"];

interface CropEntry { label: number; readers: number; median_rating: number; sha256: string;
  plain_logits: [number, number]; u8_base64: string }
interface CkksFigure { N: number; coeff_mod_bits: number[]; scale_bits: number; modulus_bits: number; crops: number;
  max_abs_logit_error: number; decisions_differ: number; median_ms: number; max_ms: number;
  ciphertexts: { input: number; output: number }; bytes: { input: number; output: number }; criterion_3: boolean }
type Part = { patients: number; nodules: number; malignant: number; benign: number };
interface NoduleFile {
  weights: Square3dWeights & { decision: { platt: [number, number]; threshold: number; rule: string }; export_seed: number };
  crops_about: string;
  crops: CropEntry[];
  figures: {
    export_seed: number; teacher: string; teachers_oof_auc: Record<string, number>;
    mean_test_auc: { control: number; square: number }; sd_test_auc: { control: number; square: number };
    square_test_aucs: number[];
    export: { test_auc: number; val_auc: number; test_balanced_accuracy: number;
              bootstrap: { auc: number; ci95: [number, number]; resamples: number } };
    criteria: { "1_accuracy": boolean; "2_no_collapse": boolean; "3_fidelity": Record<string, boolean> };
    ckks: Record<string, CkksFigure>;
    split_counts: { train: Part; val: Part; test: Part };
    malignant_share: { train: number; val: number; test: number };
    majority_class_rate_test: number; context_not_compared: string;
  };
  licence: string; licence_url: string; attribution: string; modifications: string; data_usage_policy: string;
}

const f3 = (v: number) => v.toFixed(3);
const kb = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(2)} MB` : `${(b / 1e3).toFixed(0)} kB`);
const TEACHER_NAMES: Record<string, string> = {
  resnet3d18: "3D ResNet-18 from scratch", ctfm_probe: "CT-FM linear probe", totalseg_size: "TotalSegmentator nodule size",
};

/** One request per page load: the page mounts this card twice in quick succession (both fetches were seen in the
 *  network log, 18 ms apart), and the file is 0.8 MB. A failed request is forgotten, so a later mount can retry. */
let loading: Promise<NoduleFile> | null = null;
function loadFile(): Promise<NoduleFile> {
  loading ??= fetch(`${import.meta.env.BASE_URL}data/medical/nodule3d_square_student.json`)
    .then((r) => (r.ok ? r.json() as Promise<NoduleFile> : Promise.reject(new Error(`HTTP ${r.status}`))))
    .catch((e) => { loading = null; throw e; });
  return loading;
}

/** The voxels of one shipped crop: its base64 bytes, read as the network reads them, (u / 255 - 0.5) / 0.5. */
function decode(c: CropEntry): { u8: Uint8Array; v: Float64Array } {
  const bin = atob(c.u8_base64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  if (u8.length !== VOXELS) throw new Error(`a crop holds ${u8.length} voxels, not ${VOXELS}`);
  const v = new Float64Array(VOXELS);
  for (let i = 0; i < VOXELS; i++) v[i] = (u8[i] / 255 - 0.5) / 0.5;
  return { u8, v };
}

/** The validation-fitted decision: Platt scaling of the logit difference, then the threshold chosen on validation. */
function malignantAt(logits: [number, number], d: NoduleFile["weights"]["decision"]): boolean {
  const [a, b] = d.platt;
  return 1 / (1 + Math.exp(-(a * (logits[1] - logits[0]) + b))) >= d.threshold;
}

/** Where the shipped seed stands on test among the five square seeds; it was chosen on validation, before any test
 *  figure was read, by the pre-registration's rule, so the card says where that left it. */
function seedPosition(fig: NoduleFile["figures"]): string {
  const all = fig.square_test_aucs;
  const rank = all.filter((v) => v > fig.export.bootstrap.auc + 1e-9).length + 1;
  return `On test it ranks ${rank} of ${all.length} square seeds (${f3(Math.min(...all))} to ${f3(Math.max(...all))}); ` +
    "the pre-registration chooses by validation, so no seed was picked on test figures.";
}

function setLabel(name: string, p: { N: number; coeff_mod_bits: number[]; scale_bits: number }): string {
  const total = p.coeff_mod_bits.reduce((a, b) => a + b, 0);
  return `${name}: N = ${p.N.toLocaleString("en-US")}, ${p.coeff_mod_bits.join("+")} bits (${total}), scale 2^${p.scale_bits}`;
}

export default function Nodule3dCard() {
  const [file, setFile] = useState<NoduleFile | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [pick, setPick] = useState(0);
  const [set, setSet] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<{ crop: number; run: Square3dRun; ranIn: string; wallMs: number } | null>(null);

  useEffect(() => {
    let live = true;
    loadFile()
      .then((d) => { if (live) { setFile(d); setSet(Object.keys(d.weights.ckks).sort()[0] ?? ""); } })
      .catch((e) => { if (live) setFailed((e as Error)?.message ?? String(e)); });
    return () => { live = false; };
  }, []);

  const crop = useMemo(() => (file ? decode(file.crops[pick]) : null), [file, pick]);
  const contribution = useMemo(() => (file && crop ? contributionMap(crop.v, file.weights) : null), [file, crop]);
  const plain = useMemo(() => (file && crop ? plainSquare3dLogits(crop.v, file.weights) : null), [file, crop]);

  if (failed) return <div className="card" style={{ marginTop: 14 }}><p className="small">The 3D nodule student&apos;s file did not load: {failed}</p></div>;
  if (!file || !crop || !plain || !contribution) return null;
  const fig = file.figures;
  const c = file.crops[pick];
  const note = memoryNote();
  const o = outcome && outcome.crop === pick ? outcome : null;

  const run = async () => {
    setOutcome(null); setFailed(null);
    const t0 = performance.now();
    try {
      setBusy(`Building the keys for ${set}, then encrypting the crop's 27 taps...`);
      const r = await runSquare3dOffThread(crop.v, file.weights, set);
      setOutcome({ crop: pick, run: r.run, ranIn: r.ranIn, wallMs: performance.now() - t0 });
    } catch (e) { setFailed((e as Error)?.message ?? String(e)); }
    finally { setBusy(null); }
  };

  return (
    <div className="card" id="nodule3d-student" style={{ marginTop: 14 }}>
      <h3>A 3D nodule, end to end under CKKS: the 3D square student <StageBadge kind="notional" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        The same design as the square student above, in three dimensions: one 3 x 3 x 3 convolution with four filters, a
        square, a 2 x 2 x 2 average pool and a dense layer, trained on LIDC-IDRI chest CT nodules (28 mm cubes at 1 mm) with
        soft labels from the best of three teacher candidates. The client encrypts a crop&apos;s 27 convolution taps; the
        evaluator convolves, squares and applies the pooled dense layer on the ciphertexts; the client decrypts two logits.
        Both parties run in this tab, so the separation is notional. Pre-registered before any image was fetched, with an
        addendum before any model was trained.
      </p>

      <table className="facts-table small">
        <tbody>
          <tr><th scope="row">Test AUC, mean of five seeds</th>
            <td>square student {f3(fig.mean_test_auc.square)} (sd {f3(fig.sd_test_auc.square)}); control with ReLU {f3(fig.mean_test_auc.control)} (sd {f3(fig.sd_test_auc.control)})</td></tr>
          <tr><th scope="row">The shipped seed ({fig.export_seed}, best on validation)</th>
            <td>test AUC {f3(fig.export.bootstrap.auc)}, 95% interval {f3(fig.export.bootstrap.ci95[0])} to {f3(fig.export.bootstrap.ci95[1])} over test patients ({fig.export.bootstrap.resamples.toLocaleString("en-US")} resamples); balanced accuracy {f3(fig.export.test_balanced_accuracy)} at the validation threshold. {seedPosition(fig)}</td></tr>
          <tr><th scope="row">Teacher candidates, out of fold</th>
            <td>{Object.entries(fig.teachers_oof_auc).map(([k, v]) => `${TEACHER_NAMES[k] ?? k} ${f3(v)}`).join("; ")}; the teacher: {TEACHER_NAMES[fig.teacher] ?? fig.teacher}</td></tr>
          <tr><th scope="row">Test part</th>
            <td>{fig.split_counts.test.nodules} nodules of {fig.split_counts.test.patients} patients, {fig.split_counts.test.malignant} malignant; always answering benign scores {f3(fig.majority_class_rate_test)}</td></tr>
          {Object.entries(fig.ckks).sort(([a], [b]) => a.localeCompare(b)).map(([name, k]) => (
            <tr key={name}><th scope="row">Under {name}, every test crop</th>
              <td>largest |decrypted - plaintext| logit {k.max_abs_logit_error.toExponential(2)} against {LIMIT.toExponential(0)}, {k.decisions_differ} decisions changed of {k.crops}; median {(k.median_ms / 1000).toFixed(1)} s a crop; {kb(k.bytes.input)} up, {kb(k.bytes.output)} back{k.criterion_3 ? "" : "; did not meet the limit, so not offered"}</td></tr>
          ))}
        </tbody>
      </table>

      {/* A fieldset's min-width defaults to its content's, so without minWidth 0 the select's max-width: 100% would
          resolve against a box as wide as its longest option and push the card sideways on a phone. */}
      <fieldset className="small" style={{ border: 0, padding: 0, margin: "10px 0 6px", minWidth: 0 }}>
        <legend className="muted">Test crop (the addendum&apos;s twelve: the first six of each label)</legend>
        <select value={pick} disabled={busy != null} onChange={(e) => { setPick(Number(e.target.value)); setOutcome(null); }}
          aria-label="Test crop" style={{ maxWidth: "100%" }}>
          {file.crops.map((k, i) => (
            <option key={k.sha256} value={i}>
              {`Crop ${i + 1}: ${CLASS_NAMES[k.label]} by the readers (median rating ${k.median_rating}, ${k.readers} reader${k.readers === 1 ? "" : "s"})`}
            </option>
          ))}
        </select>
      </fieldset>
      <Nodule3dViewer volume={crop.u8} contribution={contribution}
        label={`Crop ${pick + 1}: ${CLASS_NAMES[c.label]} by the readers`} />
      <p className="small">
        In the clear: logits {plain[0].toFixed(3)} (benign) and {plain[1].toFixed(3)} (malignant); at the validation
        threshold the student says <b>{malignantAt(plain, file.weights.decision) ? "malignant" : "benign"}</b>, the readers
        say {CLASS_NAMES[c.label]}.
      </p>

      <fieldset className="small" style={{ border: 0, padding: 0, margin: "4px 0 8px" }}>
        <legend className="muted">Parameter set</legend>
        {Object.entries(file.weights.ckks).sort(([a], [b]) => a.localeCompare(b)).map(([name, p]) => (
          <label key={name} style={{ display: "block", minHeight: 24 }}>
            <input type="radio" name="nodule3d-set" checked={set === name} disabled={busy != null}
              onChange={() => { setSet(name); setOutcome(null); }} /> {setLabel(name, p)}
          </label>
        ))}
      </fieldset>
      {note ? <p className="small" style={{ color: "var(--warning, var(--danger))" }}>{note}</p> : null}
      <button className="btn" onClick={run} disabled={busy != null || !set}>{busy ?? `Run crop ${pick + 1} under ${set}`}</button>
      {failed ? <p className="small" style={{ color: "var(--danger)" }}>It did not run: {failed}</p> : null}
      {o ? (
        <table className="facts-table small" style={{ marginTop: 8 }}>
          <tbody>
            <tr><th scope="row">Decrypted logits</th><td>{o.run.logits[0].toFixed(6)} and {o.run.logits[1].toFixed(6)}; in the clear {o.run.plain[0].toFixed(6)} and {o.run.plain[1].toFixed(6)}</td></tr>
            <tr><th scope="row">Largest |decrypted - plaintext| logit</th><td>{o.run.maxAbsError.toExponential(2)} against the pre-registered {LIMIT.toExponential(0)}: <b>{o.run.maxAbsError <= LIMIT ? "within" : "OVER"}</b></td></tr>
            <tr><th scope="row">Decision at the validation threshold</th><td>{malignantAt(o.run.logits, file.weights.decision) ? "malignant" : "benign"} decrypted, {malignantAt(o.run.plain, file.weights.decision) ? "malignant" : "benign"} in the clear: {malignantAt(o.run.logits, file.weights.decision) === malignantAt(o.run.plain, file.weights.decision) ? "the same" : "DIFFERENT"}</td></tr>
            <tr><th scope="row">Time</th><td>keys {o.run.ms.keys.toFixed(0)} ms, encrypt {o.run.ms.encrypt.toFixed(0)} ms, evaluate {o.run.ms.evaluate.toFixed(0)} ms, decrypt {o.run.ms.decrypt.toFixed(1)} ms; {(o.wallMs / 1000).toFixed(1)} s in all, in {o.ranIn}</td></tr>
            <tr><th scope="row">Crossed, encrypted</th><td>{kb(o.run.bytes.input)} up ({o.run.ciphertexts.input} ciphertexts), {kb(o.run.bytes.output)} back ({o.run.ciphertexts.output})</td></tr>
          </tbody>
        </table>
      ) : null}

      <p className="muted small" style={{ marginTop: 8 }}>
        The figures above are the record&apos;s, over the whole test part (<code>docs/measurements/medical-nodule3d-square-student.json</code>);
        the twelve crops are not a sample of the student&apos;s best. {fig.context_not_compared}, so its figures are cited,
        not compared. Labels are the readers&apos; median malignancy rating (above 3 malignant, below 3 benign, exactly 3
        left out), not a diagnosis. For research and demonstration only, not for clinical use.
        Data: LIDC-IDRI (<a href={file.licence_url} target="_blank" rel="noreferrer">CC BY 3.0</a>), {file.attribution}{" "}
        Changes: {file.modifications}. Use is subject to <a href={file.data_usage_policy} target="_blank" rel="noreferrer">TCIA&apos;s
        Data Usage Policies and Restrictions</a>, which forbid attempts to identify participants.
      </p>
    </div>
  );
}
