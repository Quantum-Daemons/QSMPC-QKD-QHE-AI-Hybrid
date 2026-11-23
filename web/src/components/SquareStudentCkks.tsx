// A fully encrypted pass on the medical page (Round 86, the owner's choice of 2026-11-19). The square-activation student
// runs end to end under CKKS in this tab, with no layer decrypted in between, in its own CKKS context for each parameter
// set its published weights list (lib/clientFHESquare.ts, in a Web Worker). The weights and the 156 official test images
// are the model's pinned Hub files, each checked against the SHA-256 recorded for its revision.
import { useState } from "react";
import { type SquareRun, type SquareWeights, runSquareStudentOffThread } from "../lib/clientFHESquare";
import BfvBudgetMeter from "./BfvBudgetMeter";
import CkksLadder from "./CkksLadder";
import { modelsFor } from "../lib/config";
import { type ModelInputFile, publishedSampleOf, publishedWeightsOf } from "../lib/onnxInference";
import SquareStudentSetsChart, { type SquareTabPoint } from "./SquareStudentSetsChart";
import StageBadge from "./StageBadge";
import { EncryptedAgreementChart, type EncryptedAgreementPlot } from "./VerificationChart";

const KEY = "medical_fl.square_student";
/** The pre-registered limit on a decrypted logit's distance from the plaintext one. */
const LIMIT = 1e-3;
const MALIGNANT = 0;

interface Outcome { set: string; rows: number[]; runs: SquareRun[]; ranIn: string; wallMs: number }

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : NaN;
};

/** Eight images, four of each class, the first of each in the split's order; or every image. */
function rowsFor(labels: number[], count: number): number[] {
  if (count >= labels.length) return labels.map((_, i) => i);
  const half = Math.floor(count / 2);
  const pick = (cls: number) => labels.map((l, i) => (l === cls ? i : -1)).filter((i) => i >= 0).slice(0, half);
  return [...pick(MALIGNANT), ...pick(1 - MALIGNANT)].sort((a, b) => a - b);
}

function setLabel(name: string, p: { N: number; coeff_mod_bits: number[]; scale_bits: number }): string {
  const total = p.coeff_mod_bits.reduce((a, b) => a + b, 0);
  return `${name}: N = ${p.N.toLocaleString("en-US")}, ${p.coeff_mod_bits.join("+")} bits (${total}), scale 2^${p.scale_bits}`;
}

export default function SquareStudentCkks() {
  const spec = modelsFor("medical_fl").find((m) => m.key === KEY);
  const [file, setFile] = useState<ModelInputFile | null>(null);
  const [weights, setWeights] = useState<SquareWeights | null>(null);
  const [set, setSet] = useState("A2");
  const [count, setCount] = useState(8);
  const [busy, setBusy] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  if (!spec?.hf_revision) return null;
  const labels = (file?.labels ?? []) as number[];

  const load = async () => {
    setBusy("Fetching the 156 test images and the weights..."); setFailed(null);
    try {
      const [got, bytes] = await Promise.all([publishedSampleOf(spec), publishedWeightsOf(spec)]);
      if (got.kind !== "ok") throw new Error(got.kind === "absent" ? "the model's repository has no evaluation sample at its pinned revision" : got.reason);
      const w = JSON.parse(new TextDecoder().decode(bytes)) as SquareWeights;
      if (!w.ckks || !Object.keys(w.ckks).length) throw new Error("the published weights list no CKKS parameter set");
      setFile(got.file); setWeights(w); setSet(Object.keys(w.ckks).sort()[0]);
    } catch (e) { setFailed((e as Error)?.message ?? String(e)); }
    finally { setBusy(null); }
  };

  const run = async () => {
    if (!file || !weights) return;
    setFailed(null); setOutcome(null);
    const rows = rowsFor(labels, count);
    const t0 = performance.now();
    try {
      setBusy(`Building the keys for ${set}, then image 1 of ${rows.length}...`);
      const batch = await runSquareStudentOffThread(rows.map((i) => file.samples[i] as number[]), weights, set,
        (done) => setBusy(done < rows.length ? `Image ${done + 1} of ${rows.length} under CKKS...` : "Finishing..."));
      setOutcome({ set, rows, runs: batch.runs, ranIn: batch.ranIn, wallMs: performance.now() - t0 });
    } catch (e) { setFailed((e as Error)?.message ?? String(e)); }
    finally { setBusy(null); }
  };

  const o = outcome;
  const worst = o ? Math.max(...o.runs.map((r) => r.maxAbsError)) : NaN;
  const unchanged = o ? o.runs.filter((r) => r.decisionAgrees).length : 0;
  const acc = (pick: (r: SquareRun) => number) =>
    o ? o.runs.filter((r, k) => pick(r) === labels[o.rows[k]]).length / o.runs.length : NaN;
  const plainCls = (r: SquareRun) => (r.plain[1] > r.plain[0] ? 1 : 0);
  // The sample's own split and licence fields, which every published sample carries (prep_model_inputs.py) and the
  // shared type leaves undeclared.
  const meta = (file ?? {}) as { split?: string; licence?: string; licence_url?: string; attribution?: string; modifications?: string };
  // This run on the sets' plot: its largest error and its median time per image, encrypt to decrypt, keys excluded, as
  // the record times a set.
  const tabPoint: SquareTabPoint | null = o ? {
    set: o.set, worst, images: o.runs.length, ranIn: o.ranIn,
    medianMs: median(o.runs.map((r) => r.ms.encrypt + r.ms.evaluate + r.ms.decrypt)),
  } : null;
  const plot: EncryptedAgreementPlot | null = o && file ? {
    kind: "encrypted-agreement",
    points: o.runs.map((r, k) => ({
      enc: r.logits[1] - r.logits[0], plain: r.plain[1] - r.plain[0], agree: r.decisionAgrees,
      label: labels[o.rows[k]], encCls: r.predicted, plainCls: plainCls(r),
    })),
    rows: o.runs.length, classes: 2, maxAbsError: worst,
    heldOut: true, split: meta.split, corpus: "MedMNIST v2 BreastMNIST",
    modelKey: KEY, modelLabel: spec.label,
    licence: { name: meta.licence ?? "", url: meta.licence_url ?? null,
               attribution: meta.attribution ?? "", modifications: meta.modifications ?? "" },
    fheMs: o.runs.reduce((s, r) => s + r.ms.evaluate, 0),
    workNote: `Each image travelled as nine ciphertexts and came back as two; ${o.runs.reduce((s, r) => s + r.ms.evaluate, 0).toFixed(0)} ms `
      + `of encrypted evaluation for the ${o.runs.length} images in all.`,
    arithmeticNote: "This model is a convolution, a square and a dense layer: eleven plaintext multiplications (the nine "
      + "kernel taps, then each class's dense weights), one ciphertext multiplication with relinearisation, three levels of "
      + "rescaling and twenty rotations, with no bootstrapping and the square as the activation itself rather than an "
      + "approximation of one, so it does more homomorphic arithmetic than this page's linear students and far less than "
      + "that ResNet.",
    scoreKind: "logit", classNames: weights?.classes,
    accuracy: { encrypted: acc((r) => r.predicted), plaintext: acc(plainCls),
                majority: Math.max(labels.filter((l) => l === 0).length, labels.filter((l) => l === 1).length) / labels.length },
  } : null;

  return (
    <div className="card" id="square-student" style={{ marginTop: 14 }}>
      <h3>A fully encrypted pass: the square student <StageBadge kind="notional" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        The CKKS student above decrypts each layer&apos;s output and runs its ReLU and pooling in the clear. This student
        ({KEY}) replaces the ReLU with a square, so nothing in between needs decrypting: the client encrypts a 14 x 14 test
        image&apos;s nine convolution taps, the evaluator convolves, squares and applies the pooled dense layer on the
        ciphertexts, and the client decrypts two logits. That is three multiplications deep, one more than this page&apos;s
        CKKS context allows, so it runs in its own context, under either parameter set that met the pre-registered limit
        (every decrypted logit within {LIMIT.toExponential(0)} of the plaintext one, no decision changed). Both parties
        run in this tab, so the separation is notional.
      </p>
      <SquareStudentSetsChart run={tabPoint} />
      {!file || !weights ? (
        <button className="btn" onClick={load} disabled={busy != null}>
          {busy ?? "Load the 156 test images and the weights (0.4 MB from the model's Hub repository)"}
        </button>
      ) : (
        <>
          <fieldset className="small" style={{ border: 0, padding: 0, margin: "4px 0 8px" }}>
            <legend className="muted">Parameter set</legend>
            {Object.entries(weights.ckks).sort(([a], [b]) => a.localeCompare(b)).map(([name, p]) => (
              <label key={name} style={{ display: "block", minHeight: 24 }}>
                <input type="radio" name="square-set" checked={set === name} disabled={busy != null}
                  onChange={() => { setSet(name); setOutcome(null); }} /> {setLabel(name, p)}
              </label>
            ))}
          </fieldset>
          <fieldset className="small" style={{ border: 0, padding: 0, margin: "0 0 8px" }}>
            <legend className="muted">Images</legend>
            {[8, labels.length].map((n) => (
              <label key={n} style={{ display: "inline-block", minHeight: 24, marginRight: 14 }}>
                <input type="radio" name="square-count" checked={count === n} disabled={busy != null}
                  onChange={() => { setCount(n); setOutcome(null); }} /> {n === labels.length ? `all ${n}` : `${n}, four of each class`}
              </label>
            ))}
          </fieldset>
          <button className="btn" onClick={run} disabled={busy != null}>{busy ?? `Run ${count === labels.length ? `all ${count} images` : `${count} images`} under ${set}`}</button>
          {failed ? <p className="small" style={{ color: "var(--danger)" }}>It did not run: {failed}</p> : null}
          {o ? (
            <>
              <table className="facts-table small" style={{ marginTop: 8 }}>
                <tbody>
                  <tr><th scope="row">Parameter set</th><td>{setLabel(o.set, weights.ckks[o.set])}</td></tr>
                  <tr><th scope="row">Images</th><td>{o.runs.length} of the official test split ({o.rows.filter((i) => labels[i] === MALIGNANT).length} malignant, {o.rows.filter((i) => labels[i] !== MALIGNANT).length} normal or benign)</td></tr>
                  <tr><th scope="row">Largest |decrypted - plaintext| logit</th><td>{worst.toExponential(2)} against the pre-registered {LIMIT.toExponential(0)}: <b>{worst <= LIMIT ? "within" : "OVER"}</b></td></tr>
                  <tr><th scope="row">Decisions unchanged by encryption</th><td>{unchanged} of {o.runs.length}</td></tr>
                  <tr><th scope="row">Accuracy on these images</th><td>{acc((r) => r.predicted).toFixed(4)} decrypted, {acc(plainCls).toFixed(4)} in the clear (the model&apos;s argmax, its pre-registered operating point)</td></tr>
                  <tr><th scope="row">Median time per image</th><td>encrypt {median(o.runs.map((r) => r.ms.encrypt)).toFixed(0)} ms, evaluate {median(o.runs.map((r) => r.ms.evaluate)).toFixed(0)} ms, decrypt {median(o.runs.map((r) => r.ms.decrypt)).toFixed(1)} ms; {(o.wallMs / 1000).toFixed(1)} s in all with the keys, in {o.ranIn}</td></tr>
                  <tr><th scope="row">Crossed, encrypted</th><td>{(o.runs[0].bytes.input / 1e6).toFixed(2)} MB per image (nine ciphertexts), {(o.runs[0].bytes.output / 1e3).toFixed(0)} kB back (two)</td></tr>
                </tbody>
              </table>
              {plot ? <EncryptedAgreementChart plot={plot} /> : null}
              <CkksLadder ladder={o.runs[0].ladder} setName={o.set} />
            </>
          ) : null}
          <p className="muted small" style={{ marginTop: 8 }}>
            Every slot of the decrypted ciphertexts holds the logit: each vector is tiled with period 1,024, so the
            rotate-and-add sums one whole period into every slot and no partial sum, which would expose a product of a
            weight and a pixel, is decrypted. At its argmax this student flags more images malignant than the CKKS student
            above (on the whole split, recall 0.738 against 0.571, specificity 0.693 against 0.842), so its accuracy, 0.705,
            is below the 0.731 of always answering benign. Pre-registration: <code>references/medical-square-student-preregistration.md</code>.
            MedMNIST v2 BreastMNIST (Yang et al., Scientific Data 2023); for research and demonstration only, not for
            clinical use.
          </p>
          <BfvBudgetMeter />
        </>
      )}
    </div>
  );
}
