#!/usr/bin/env node
/**
 * A trained QCNN's prediction under CKKS, end to end, against the circuit in the clear (Round 86).
 *
 * This runs the module the page runs, on the real node-seal, on the shipped weights and on rows the page can receive, and asserts what the panel reports: the form over the Pauli
 * products equals the circuit's readout to rounding, the decrypted readout is within the bound of the clear one, the class is the same, and the counts are the design's.
 *
 * Run: npm run check:qcnn-ckks
 */
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPO = resolve(WEB, "..");
const { build } = createRequire(join(WEB, "package.json"))("esbuild");
const sealWasm = join(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
const tmp = mkdtempSync(join(tmpdir(), "qsh-qcnn-ckks-"));
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };
try {
  await build({
    stdin: { contents: `export { qcnnInferCkks } from ${JSON.stringify(join(WEB, "src/lib/clientQcnnCkks"))};
export { pauliForm, pauliFeatures, evaluateForm } from ${JSON.stringify(join(WEB, "src/lib/qcnnPauli"))};
export { qcnnInferTrained, stateVectorOps } from ${JSON.stringify(join(WEB, "src/lib/quantumSim"))};
export { recordQcnn, gateCounts } from ${JSON.stringify(join(WEB, "src/lib/circuitDiagram"))};`, resolveDir: WEB, loader: "ts" },
    bundle: true, format: "esm", platform: "node", outfile: join(tmp, "b.mjs"), logLevel: "error",
    define: { "import.meta.env": '{"BASE_URL":"/"}' },
    plugins: [{ name: "wasm-url", setup(b) {
      b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
      b.onLoad({ filter: /.*/, namespace: "wasmurl" }, () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    } }],
  });
  const m = await import(pathToFileURL(join(tmp, "b.mjs")).href);
  const pub = (p) => JSON.parse(readFileSync(join(WEB, "public/data", p), "utf8"));
  const cases = [
    ["finance", "models/qcnn_finance/weights.json", pub("finance/paysim_sample.json").samples_by_space.raw ?? pub("finance/paysim_sample.json").samples],
    ["medical", "models/qcnn_medical/weights.json", pub("medical/breast_cancer_sample.json").samples_by_space.raw ?? pub("medical/breast_cancer_sample.json").samples],
    ["telecom 7 wires", "models/qcnn_telecom_q7/weights.json", pub("telecom/student_heldout.json").samples],
    ["telecom 8 wires", "models/qcnn_telecom/weights.json", pub("telecom/student_heldout.json").samples],
  ];
  for (const [name, rel, samples] of cases) {
    const w = JSON.parse(readFileSync(join(REPO, rel), "utf8"));
    const nf = w.n_features ?? w.n_qubits, nTerms = 3 ** nf, outputs = w.readout?.readout_qubits ? 1 << w.readout.readout_qubits : 1;
    // Every row through the form, in the clear: the identity holds for every input, not only the ones CKKS happens to be run on.
    const form = m.pauliForm(w);
    let idMax = 0;
    for (const row of samples.slice(0, 200)) {
      const x = row.slice(0, nf), phi = m.pauliFeatures(w, x), clear = m.qcnnInferTrained(x, w);
      const values = w.readout?.readout_qubits ? clear.probs : [clear.expectation];
      form.coefficients.forEach((c, i) => { idMax = Math.max(idMax, Math.abs(m.evaluateForm(c, phi) - values[i])); });
    }
    check(idMax < 1e-12, `${name}: the linear form over ${nTerms} Pauli products equals the circuit's readout on ${Math.min(200, samples.length)} rows (max difference ${idMax.toExponential(2)})`);
    check(form.nTerms === nTerms && form.coefficients.length === outputs, `${name}: ${form.nTerms} terms, ${form.coefficients.length} readout(s)`);
    // The page's diagram is drawn from a recording of the circuit function. Replay that recording on a state vector and it must reproduce the circuit's readout.
    const recorded = m.recordQcnn(w);
    let replayMax = 0;
    for (const row of samples.slice(0, 8)) {
      const x = row.slice(0, nf), N = 1 << w.n_qubits;
      const re = new Float64Array(N), im = new Float64Array(N);
      re[0] = 1;
      const ops = m.stateVectorOps(re, im, N);
      for (const [q, value] of x.entries()) {
        const z = w.input_space === "raw" ? (value - w.feature_mean[q]) / (w.feature_std[q] || 1) : value;
        ops.RY(q, (w.encoding_scale ?? 1) * Math.tanh(z) * Math.PI);
      }
      for (const block of recorded.blocks) {
        if (block.kind === "encode") continue;
        for (const g of block.gates) {
          if (g.name === "CNOT") ops.CNOT(g.wires[0], g.wires[1]);
          else if (g.name === "X") ops.X(g.wires[0]);
          else if (g.wires.length === 1) ops[g.name](g.wires[0], g.angle);
          else ops[g.name](g.wires[0], g.wires[1], g.angle);
        }
      }
      const clear = m.qcnnInferTrained(x, w);
      if (w.readout?.readout_qubits) {
        const k = 1 << w.readout.readout_qubits, p = new Float64Array(k);
        for (let i = 0; i < N; i++) { let c = 0; recorded.readout.wires.forEach((wire, j) => { c |= ((i >>> wire) & 1) << j; }); p[c] += re[i] * re[i] + im[i] * im[i]; }
        p.forEach((v, c) => { replayMax = Math.max(replayMax, Math.abs(v - clear.probs[c])); });
      } else {
        let z = 0;
        for (let i = 0; i < N; i++) z += (i & 1 ? -1 : 1) * (re[i] * re[i] + im[i] * im[i]);
        replayMax = Math.max(replayMax, Math.abs(z - clear.expectation));
      }
    }
    const total = Object.values(m.gateCounts(recorded)).reduce((a, b) => a + b, 0);
    check(replayMax < 1e-12, `${name}: the gates the page's diagram draws reproduce the circuit (${total} gates, ${recorded.blocks.length} blocks, max difference ${replayMax.toExponential(2)})`);
    // Sixteen rows under CKKS, spread over the sample.
    const step = Math.max(1, Math.floor(samples.length / 16));
    let maxErr = 0, agree = 0, ran = 0, last = null;
    for (let i = 0; i < samples.length && ran < 16; i += step, ran++) {
      const r = await m.qcnnInferCkks(w, samples[i].slice(0, nf));
      maxErr = Math.max(maxErr, r.maxError); agree += r.matchesClear ? 1 : 0; last = r;
    }
    check(agree === ran, `${name}: the class from the decrypted readout is the clear circuit's on all ${ran} rows (${agree})`);
    check(maxErr < 1e-6, `${name}: the decrypted readout is within 1e-6 of the clear one (max ${maxErr.toExponential(2)})`);
    const nCt = Math.ceil(nTerms / 4096);
    check(last.nCiphertexts === nCt && last.rotations === outputs * 12, `${name}: ${last.nCiphertexts} input ciphertext(s), ${last.rotations} rotations, the design's 12 per readout`);
    console.log(`--- ${name}: ${(last.totalMs / 1000).toFixed(2)} s, up ${(last.upBytes / 1e6).toFixed(2)} MB, down ${(last.downBytes / 1e6).toFixed(2)} MB, keys ${(last.keyBytes / 1e6).toFixed(1)} MB, form ${last.formMs.toFixed(0)} ms`);
  }
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
if (failed) { console.error(`check:qcnn-ckks FAILED: ${failed}`); process.exit(1); }
console.log("check:qcnn-ckks OK");
process.exit(0);
