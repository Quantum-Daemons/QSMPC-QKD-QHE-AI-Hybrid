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
 */
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
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
  const flat = sample.samples[1];
  const size = Math.sqrt(flat.length);
  const image = Array.from({ length: size }, (_, y) => flat.slice(y * size, (y + 1) * size));
  const r = await m.hybridStudentInference(weights, image);

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

  const src = readFileSync(join(WEB, "src/lib/clientHybridCnn.ts"), "utf8");
  const server = src.slice(src.indexOf("class Server {"), src.indexOf("/** The garbled circuits' running totals. */"));
  check(server.length > 500 && !/Decryptor|secretKey|decrypt\(/.test(server), "the class holding the weights has no decryptor and no secret key");
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
if (failed) { console.error(`${failed} check(s) failed`); process.exit(1); }
