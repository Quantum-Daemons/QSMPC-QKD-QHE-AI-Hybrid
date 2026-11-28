#!/usr/bin/env node
/**
 * The medical CNN student under MPC, end to end, against the clear network.
 *
 * WHAT THIS IS FOR. The page's card says the published student ran with no activation in either party's hands. Nothing else can check that sentence. The
 * crate's own test covers one ReLU circuit and the CKKS module has no ReLU. This runs one held-out image through the module the page runs, the real node-seal and
 * the real Polytune, and asserts what the card reports:
 *
 *   - the class from the 2PC argmax is the clear network's, and the logits reconstructed from the two shares are close to it;
 *   - what the client decoded is masked: the median magnitude it saw is about half the mask's width, and not the activations' few units;
 *   - the circuit is the size the design says, so a cheaper or a different circuit cannot pass unnoticed;
 *   - the run's own bound on the masks is what it should be;
 *   - the class that holds the weights was never handed a decryptor.
 *
 * Run: npm run check:hybrid-cnn (about 20 s)
 *
 * `--all [--out FILE]` (Round 86, batch 12) runs every image of the published sample, the official test split's 156, and
 * writes the record the Models panel's "its class matched the clear network on all 156 test images" and the models doc's
 * table rest on (`docs/measurements/medical-cnn-mpc-<date>.json`, `tests/test_r86_cnn_mpc_record.py` holds both to it);
 * every image is held to the same checks as the one. About 45 minutes; its timings want a quiet machine.
 */
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPO = resolve(WEB, "..");
const MODEL = join(REPO, "models", "medical_breastmnist_student");
const { build } = createRequire(join(WEB, "package.json"))("esbuild");
const sealWasm = join(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
const mpcWasm = join(WEB, "src/wasm/qsh_mpc/qsh_mpc_bg.wasm");

const tmp = mkdtempSync(join(tmpdir(), "qsh-hybrid-cnn-"));
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };

try {
  await build({
    stdin: { contents: `export { hybridStudentInference } from ${JSON.stringify(join(WEB, "src/lib/clientHybridCnn"))};`, resolveDir: WEB, loader: "ts" },
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
  const weights = JSON.parse(readFileSync(join(MODEL, "weights.json"), "utf8"));
  const sample = JSON.parse(readFileSync(join(MODEL, "eval_sample.json"), "utf8"));
  const all = process.argv.includes("--all");
  const toImage = (flat) => {
    const size = Math.sqrt(flat.length);
    return Array.from({ length: size }, (_, y) => flat.slice(y * size, (y + 1) * size));
  };
  const runs = [];
  for (const k of all ? sample.samples.map((_, i) => i) : [1]) {
    const one = await m.hybridStudentInference(weights, toImage(sample.samples[k]));
    runs.push({ k, r: one });
    if (all && (k + 1) % 13 === 0) console.log(`  ${k + 1} of ${sample.samples.length} images`);
  }
  const r = runs[0].r;

  const cfg = JSON.parse(readFileSync(join(WEB, "src/lib/config.defaults.json"), "utf8")).hybrid_cnn;
  const andPerRelu = (bits) => (cfg.relu_ring_bits - 1) * 2 + cfg.frac_bits + bits;
  const dims = (d) => d.out;
  const nConv = weights.conv.out_ch * 14 * 14, nDense = weights.dense.slice(0, -1).reduce((a, d) => a + dims(d), 0);
  const wantAnd = nConv * andPerRelu(cfg.conv_mask_bits) + nDense * andPerRelu(cfg.dense_mask_bits) + 367;

  check(r.matchesClear, `the 2PC class (${r.prediction}) is the clear network's (${r.clearPrediction})`);
  check(r.logitMaxAbsError < 2e-3, `logits reconstructed from the shares are within 2e-3 of the clear network's (${r.logitMaxAbsError.toExponential(2)})`);
  check(r.clientViewMedianAbs > 2 ** Math.min(cfg.conv_mask_bits, cfg.dense_mask_bits) / 8, `what the client decoded is masked (median |value| ${r.clientViewMedianAbs.toExponential(2)})`);
  check(r.reluCount === nConv + nDense && r.batches === Math.ceil(nConv / cfg.conv_relu_batch) + 1, `${r.reluCount} ReLUs in ${r.batches} batches`);
  check(r.andGates === wantAnd, `the circuits are the size the design says (${r.andGates} AND gates, expected ${wantAnd})`);
  check(r.statisticalDistanceLog2 < -20, `the run's bound on the masks is below 2^-20 (2^${r.statisticalDistanceLog2.toFixed(1)})`);
  check(r.conjugations === weights.conv.out_ch + weights.dense.length, `every ciphertext the client decrypts had its imaginary half cancelled (${r.conjugations})`);

  if (all) {
    const labels = sample.labels.map((y) => (Array.isArray(y) ? y[0] : y));
    const med = (xs) => { const s = [...xs].sort((p, q) => p - q); const n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
    for (const { k, r: one } of runs) {
      if (!one.matchesClear || !(one.logitMaxAbsError < 2e-3) || one.andGates !== wantAnd || !(one.statisticalDistanceLog2 < -20)) {
        check(false, `image ${k}: class ${one.prediction} against ${one.clearPrediction}, error ${one.logitMaxAbsError.toExponential(2)}, ${one.andGates} AND gates`);
      }
    }
    const gaps = runs.map(({ r: one }) => Math.abs(one.logits[0] - one.logits[1]));
    const malignant = runs.filter(({ k }) => labels[k] === 0);
    const record = {
      what: "the medical CNN student under MPC (clientHybridCnn.ts's hybridStudentInference) on every image of its published sample, in node with the real node-seal and Polytune",
      sample: "models/medical_breastmnist_student/eval_sample.json: the official BreastMNIST test split, all 156 images",
      date: new Date().toISOString().slice(0, 10), node: process.version,
      images: runs.length,
      class_matches_the_clear_network: runs.filter(({ r: one }) => one.matchesClear).length,
      accuracy: runs.filter(({ k, r: one }) => one.prediction === labels[k]).length / runs.length,
      malignant_images: malignant.length,
      malignant_detected: malignant.filter(({ r: one }) => one.prediction === 0).length,
      largest_logit_error: Math.max(...runs.map(({ r: one }) => one.logitMaxAbsError)),
      smallest_logit_gap: Math.min(...gaps),
      statistical_distance_log2: { min: Math.min(...runs.map(({ r: one }) => one.statisticalDistanceLog2)),
                                   max: Math.max(...runs.map(({ r: one }) => one.statisticalDistanceLog2)) },
      and_gates_per_image: [...new Set(runs.map(({ r: one }) => one.andGates))],
      relus_per_image: [...new Set(runs.map(({ r: one }) => one.reluCount))],
      batches_per_image: [...new Set(runs.map(({ r: one }) => one.batches))],
      gc_megabytes_per_image: med(runs.map(({ r: one }) => one.gcBytes / 1e6)),
      ckks_rotations_per_image: [...new Set(runs.map(({ r: one }) => one.rotations))],
      conjugations_per_image: [...new Set(runs.map(({ r: one }) => one.conjugations))],
      ckks_megabytes_per_image: med(runs.map(({ r: one }) => one.ckksBytes / 1e6)),
      key_megabytes: med(runs.map(({ r: one }) => one.keyBytes / 1e6)),
      wall_clock_s_median: med(runs.map(({ r: one }) => one.totalMs / 1000)),
      gc_s_median: med(runs.map(({ r: one }) => one.gcMs / 1000)),
      ckks_s_median: med(runs.map(({ r: one }) => one.ckksMs / 1000)),
      every_image_passed_the_checks: failed === 0,
    };
    const i = process.argv.indexOf("--out");
    const outFile = i > 0 ? process.argv[i + 1] : join(REPO, "docs", "measurements", `medical-cnn-mpc-${record.date}.json`);
    writeFileSync(outFile, JSON.stringify(record, null, 1) + "\n");
    console.log(`wrote ${outFile}: ${record.class_matches_the_clear_network} of ${record.images} match, accuracy ${record.accuracy.toFixed(4)}`);
  }

  const src = readFileSync(join(WEB, "src/lib/clientHybridCnn.ts"), "utf8");
  const server = src.slice(src.indexOf("class Server {"), src.indexOf("/** The garbled circuits' running totals. */"));
  check(server.length > 500 && !/Decryptor|secretKey|decrypt\(/.test(server), "the class holding the weights has no decryptor and no secret key");
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
if (failed) { console.error(`${failed} check(s) failed`); process.exit(1); }
