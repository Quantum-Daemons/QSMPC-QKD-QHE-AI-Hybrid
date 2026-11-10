#!/usr/bin/env node
/**
 * A published tabular MLP under MPC, end to end, against the logits its trainer recorded (Round 86).
 *
 * This runs the module the page runs, on the real node-seal and the real Polytune, on a held-out row of each tier, and asserts what the panel reports:
 * the class from the 2PC argmax is the clear network's, the logits reconstructed from the two shares are those of the graph on the Hub, what the client
 * decoded is masked, and the circuits and the ciphertexts are the size the design says.
 *
 * Run: npm run check:hybrid-mlp
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
const mpcWasm = join(WEB, "src/wasm/qsh_mpc/qsh_mpc_bg.wasm");
const tmp = mkdtempSync(join(tmpdir(), "qsh-hybrid-mlp-"));
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };

try {
  await build({
    stdin: { contents: `export { hybridMlpInference, scaleRow } from ${JSON.stringify(join(WEB, "src/lib/clientHybridMlp"))};`, resolveDir: WEB, loader: "ts" },
    bundle: true, format: "esm", platform: "node", outfile: join(tmp, "b.mjs"), logLevel: "error",
    define: { "import.meta.env": '{"BASE_URL":"/"}' },
    plugins: [{ name: "wasm-url", setup(b) {
      b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
      b.onLoad({ filter: /.*/, namespace: "wasmurl" }, () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    } }],
  });
  globalThis.fetch = async (url) => {
    if (String(url).endsWith("qsh_mpc_bg.wasm")) return new Response(readFileSync(mpcWasm), { headers: { "Content-Type": "application/wasm" } });
    throw new Error(`unexpected fetch of ${url}`);
  };
  const m = await import(pathToFileURL(join(tmp, "b.mjs")).href);
  const cfg = JSON.parse(readFileSync(join(WEB, "src/lib/config.defaults.json"), "utf8")).hybrid_cnn;
  const andPerRelu = (cfg.relu_ring_bits - 1) * 2 + cfg.frac_bits + cfg.dense_mask_bits;
  for (const [dir, rowIndex] of [["finance_aml_mlp_large", 0], ["finance_aml_mlp_large", 12]]) {
    const w = JSON.parse(readFileSync(join(REPO, "models", dir, "weights.json"), "utf8"));
    const row = w.fixtures.rows[rowIndex], expected = w.fixtures.logits[rowIndex];
    const started = Date.now();
    const r = await m.hybridMlpInference(w, row);
    console.log(`--- ${w.key}, row ${rowIndex}, label ${w.fixtures.labels[rowIndex]}: ${((Date.now() - started) / 1000).toFixed(1)} s`);
    const hidden = w.layers.filter((l) => l.activation === "relu");
    const relus = hidden.reduce((a, l) => a + l.out, 0);
    const batches = hidden.reduce((a, l) => a + Math.ceil(l.out / cfg.conv_relu_batch), 0);
    const wantAnd = relus * andPerRelu + 367;
    const block = (n) => 1 << Math.ceil(Math.log2(Math.max(n, 2)));
    const ciphertexts = w.layers.reduce((a, l) => a + Math.ceil(l.out / Math.floor(4096 / block(l.in))), 0);
    const expectedClass = expected[1] > expected[0] ? 1 : 0;
    check(r.matchesClear && r.prediction === expectedClass, `the 2PC class (${r.prediction}) is the clear network's (${r.clearPrediction}) and the graph's (${expectedClass})`);
    const err = Math.max(...r.logits.map((v, i) => Math.abs(v - expected[i])));
    check(err < 5e-3, `the logits from the shares are within 5e-3 of the recorded graph's (${err.toExponential(2)})`);
    check(r.clientViewMedianAbs > 2 ** Math.min(cfg.dense_mask_bits, cfg.logit_mask_bits) / 8, `what the client decoded is masked (median |value| ${r.clientViewMedianAbs.toExponential(2)})`);
    check(r.reluCount === relus && r.batches === batches, `${r.reluCount} ReLUs in ${r.batches} batches`);
    check(r.andGates === wantAnd, `the circuits are the size the design says (${r.andGates} AND gates, expected ${wantAnd})`);
    check(r.statisticalDistanceLog2 < -20, `the run's bound on the masks is below 2^-20 (2^${r.statisticalDistanceLog2.toFixed(1)})`);
    check(r.conjugations === ciphertexts, `every ciphertext the client decrypts had its imaginary half cancelled (${r.conjugations} of ${ciphertexts})`);
    check(r.maxLayerError < 5e-3, `each layer's reconstruction is within 5e-3 of the clear layer (${r.maxLayerError.toExponential(2)})`);
  }
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
if (failed) { console.error(`check:hybrid-mlp FAILED: ${failed}`); process.exit(1); }
console.log("check:hybrid-mlp OK");
process.exit(0);
