#!/usr/bin/env node
/**
 * The compact transaction GNN under MPC, end to end, against the logits its trainer recorded (Round 86).
 *
 * This runs the module the page runs, on the real node-seal and the real Polytune, on held-out transactions, and asserts what the panel reports: the class from the 2PC argmax is the clear network's, the
 * logits reconstructed from the two shares are those the trainer recorded, what the client decoded is masked, and the circuits and the ciphertexts are the size the design says. It also asserts the two
 * properties that make this a GNN on a fixed tree: swapping two slots of one direction changes nothing, and an empty slot contributes nothing.
 *
 * Run: npm run check:hybrid-gnn
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
const tmp = mkdtempSync(join(tmpdir(), "qsh-hybrid-gnn-"));
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };

try {
  await build({
    stdin: { contents: `export { hybridGnnInference, gnnRow, gnnForward, secondLayer } from ${JSON.stringify(join(WEB, "src/lib/clientHybridGnn"))};`, resolveDir: WEB, loader: "ts" },
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
  const w = JSON.parse(readFileSync(join(REPO, "models/finance_aml_gnn_compact/weights.json"), "utf8"));
  const D = w.slots.per_direction;

  // ---- the network in the clear, on every fixture row
  let worst = 0;
  for (let i = 0; i < w.fixtures.labels.length; i++) {
    const clear = m.gnnForward(w, m.gnnRow(w, i));
    worst = Math.max(worst, ...clear.map((v, k) => Math.abs(v - w.fixtures.logits[i][k])));
  }
  check(worst < 2e-3, `the clear forward pass gives the trainer's logits on all ${w.fixtures.labels.length} rows (worst ${worst.toExponential(2)})`);

  // ---- what makes it a GNN on a fixed tree
  const row = m.gnnRow(w, 0);
  const base = m.gnnForward(w, row);
  const swap = structuredClone(row);
  [swap.slots[0], swap.slots[D - 1]] = [swap.slots[D - 1], swap.slots[0]];
  const swapped = m.gnnForward(w, swap);
  check(Math.max(...swapped.map((v, k) => Math.abs(v - base[k]))) < 1e-9, "swapping two slots of one direction leaves the logits unchanged");
  const firstEmpty = w.fixtures.slots.map((r, i) => [i, r.findIndex((s) => s.every((v) => v === 0))]).find(([, s]) => s >= 0);
  check(firstEmpty !== undefined, "some fixture row has an empty slot, so the padding is exercised");
  const emptyRow = m.gnnRow(w, firstEmpty[0]);
  check(emptyRow.slots[firstEmpty[1]].every((v) => v === 0), "an empty slot is still exactly zero after the client's scaling");
  const moved = structuredClone(emptyRow);
  const noBiasA = w.layers.a_slot.every((r) => r.length === w.slots.features.length);
  check(noBiasA && m.gnnForward(w, moved).every((v, k) => v === m.gnnForward(w, emptyRow)[k]), "the per-slot map has no bias, as `a_slot` is one matrix over the slot's features");

  // ---- the protocol, on one laundering row and one clean row
  for (const rowIndex of [0, 12]) {
    const expected = w.fixtures.logits[rowIndex];
    const started = Date.now();
    const r = await m.hybridGnnInference(w, m.gnnRow(w, rowIndex));
    console.log(`--- ${w.key}, row ${rowIndex}, label ${w.fixtures.labels[rowIndex]}: ${((Date.now() - started) / 1000).toFixed(1)} s`);
    const expectedClass = expected[1] > expected[0] ? 1 : 0;
    check(r.matchesClear && r.prediction === expectedClass, `the 2PC class (${r.prediction}) is the clear network's (${r.clearPrediction}) and the trainer's (${expectedClass})`);
    const err = Math.max(...r.logits.map((v, i) => Math.abs(v - expected[i])));
    check(err < 5e-3, `the logits from the shares are within 5e-3 of the recorded ones (${err.toExponential(2)})`);
    check(r.clientViewMedianAbs > 2 ** Math.min(cfg.dense_mask_bits, cfg.logit_mask_bits) / 8, `what the client decoded is masked (median |value| ${r.clientViewMedianAbs.toExponential(2)})`);
    const hidden = [w.layers.q1.weight.length + (2 + 4 * D) * w.layers.a_slot.length, w.layers.q2.weight.length + 2 * w.layers.second[0].bias.length, w.layers.c.weight.length];
    const relus = hidden.reduce((a, b) => a + b, 0), batches = hidden.reduce((a, b) => a + Math.ceil(b / cfg.conv_relu_batch), 0);
    check(r.reluCount === relus && relus === w.relu_count && r.batches === batches, `${r.reluCount} ReLUs in ${r.batches} batches, as the file says (${w.relu_count})`);
    check(r.andGates === relus * andPerRelu + 367, `the circuits are the size the design says (${r.andGates} AND gates, expected ${relus * andPerRelu + 367})`);
    check(r.statisticalDistanceLog2 < -19, `the run's bound on the masks is below 2^-19 (2^${r.statisticalDistanceLog2.toFixed(1)})`);
    const block = (n) => 1 << Math.ceil(Math.log2(Math.max(n, 2)));
    const second = m.secondLayer(w);
    const perCt = (nIn) => Math.floor(4096 / block(nIn));
    const ciphertexts = Math.ceil(w.layers.q1.weight.length / perCt(w.layers.q1.weight[0].length)) + w.layers.a_slot.length + Math.ceil(second.out / perCt(second.in)) + Math.ceil(w.layers.c.weight.length / perCt(second.out)) + Math.ceil(w.layers.d.weight.length / perCt(w.layers.c.weight.length));
    check(r.conjugations === ciphertexts, `one evaluation per output dimension serves every account at once (${r.conjugations} ciphertexts, ${ciphertexts} expected)`);
    check(r.maxLayerError < 5e-3, `each layer's reconstruction is within 5e-3 of the clear layer (${r.maxLayerError.toExponential(2)})`);
  }
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
if (failed) { console.error(`check:hybrid-gnn FAILED: ${failed}`); process.exit(1); }
console.log("check:hybrid-gnn OK");
