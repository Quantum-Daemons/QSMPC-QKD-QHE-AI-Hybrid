#!/usr/bin/env node
/**
 * The four hospitals' aggregation, five ways, end to end: the module the medical page's worker runs, the real node-seal
 * and the real Lattigo wasm (scripts/build_wasm.sh exports it), on the page's own four-hospital sample.
 *
 * WHAT THIS IS FOR. The comparison's table says three things nothing else can check: that the masks row IS the page's
 * federation, that threshold BGV trains the masks' model to the bit, and how far each CKKS design's aggregate is from
 * the exact FedAvg. It asserts them, and the statistical security each threshold row prints:
 *
 *   - the masks row's accuracies are those of fedHeart's federated arm for the same seed, exactly;
 *   - threshold BGV's every round's aggregate equals the masks' run's, so its model and accuracies are the same;
 *   - each design's largest distance from the exact float64 FedAvg is within its bound (the masks' is their 2^20
 *     fixed point; the CKKS designs' their encoding and smudging);
 *   - the threshold rows state 16, 40 and 128 bits, the masks and the shared key none;
 *   - the bytes a hospital sends are the masks' 56 and, for every encrypted design, a ciphertext of at least 100 kB.
 *
 * Run: npm run check:fed-agg (about 20 s; needs scripts/build_wasm.sh first)
 */
import { createRequire } from "node:module";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { webcrypto } from "node:crypto";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { build } = createRequire(join(WEB, "package.json"))("esbuild");
const sealWasm = join(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
const fedDir = join(WEB, "public/wasm/qsh_fedthresh");
const fedWasm = join(fedDir, "qsh_fedthresh.wasm");

if (!existsSync(fedWasm)) {
  console.error(`FAIL ${fedWasm} is missing: run scripts/build_wasm.sh first`);
  process.exit(1);
}
globalThis.crypto ??= webcrypto;
const tmp = mkdtempSync(join(tmpdir(), "qsh-fed-agg-"));
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };

try {
  await build({
    stdin: { contents: `export { runAggregationComparison, loadFedThresh } from ${JSON.stringify(join(WEB, "src/lib/clientFedAgg"))};
export { fedHeart } from ${JSON.stringify(join(WEB, "src/lib/clientFedHeart"))};`, resolveDir: WEB, loader: "ts" },
    bundle: true, format: "esm", platform: "node", outfile: join(tmp, "b.mjs"), logLevel: "error",
    define: { "import.meta.env": '{"BASE_URL":"/"}' },
    plugins: [{ name: "wasm-url", setup(b) {
      b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
      b.onLoad({ filter: /.*/, namespace: "wasmurl" }, () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    } }],
  });
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    if (String(url).endsWith("qsh_fedthresh.wasm")) return new Response(readFileSync(fedWasm), { headers: { "Content-Type": "application/wasm" } });
    return realFetch(url);
  };
  const m = await import(pathToFileURL(join(tmp, "b.mjs")).href);
  const sites = JSON.parse(readFileSync(join(WEB, "public/data/medical/heart_disease_sites.json"), "utf8"));
  const api = await m.loadFedThresh(pathToFileURL(fedDir).href + "/");
  const seed = 1;   // the page's default
  const r = await m.runAggregationComparison(sites, seed, api);
  const by = Object.fromEntries(r.designs.map((d) => [d.kind, d]));
  const heart = await m.fedHeart(sites, seed);

  for (const d of r.designs) {
    console.log(`     ${d.kind.padEnd(16)} mean ${d.mean.toFixed(4)} max|agg - exact| ${d.maxDiffFromExact.toExponential(2)}`
      + ` bytes ${d.bytesPerHospitalPerRound} ms/round ${d.msPerRound.toFixed(1)} key setup ${d.keySetupMs.toFixed(0)} ms`
      + ` stat. sec ${d.statSecBits ?? "-"} same as masks ${d.identicalToMasks ?? "-"}`);
  }
  check(r.designs.map((d) => d.kind).join() === "masks,ckks-shared-key,ckks-demo,ckks-l40,bgv-128", "five designs, in the table's order");
  check(JSON.stringify(by.masks.federated) === JSON.stringify(heart.federated),
    `the masks row is the page's federation: ${by.masks.federated.map((x) => x.toFixed(3)).join(", ")}`);
  check(by["bgv-128"].identicalToMasks === true, "threshold BGV's every aggregate equals the masks', to the bit");
  check(JSON.stringify(by["bgv-128"].federated) === JSON.stringify(by.masks.federated), "and so its model's accuracies");
  const bounds = { masks: 4 * 0.5 / 2 ** 20 + 1e-12, "ckks-shared-key": 1e-6, "ckks-demo": 5e-6, "ckks-l40": 1e-4, "bgv-128": 4 * 0.5 / 2 ** 20 + 1e-12 };
  for (const d of r.designs) {
    check(d.maxDiffFromExact <= bounds[d.kind], `${d.kind}: within ${bounds[d.kind].toExponential(1)} of the exact FedAvg (${d.maxDiffFromExact.toExponential(2)})`);
    check(Math.abs(d.mean - by.masks.mean) <= 0.02, `${d.kind}: accuracy within 0.02 of the masks' (${d.mean.toFixed(4)})`);
  }
  check(by.masks.statSecBits === null && by["ckks-shared-key"].statSecBits === null
    && by["ckks-demo"].statSecBits === 16 && by["ckks-l40"].statSecBits === 40 && by["bgv-128"].statSecBits === 128,
    "statistical security stated: none, none, 16, 40, 128 bits");
  check(by.masks.bytesPerHospitalPerRound === 56, `a masked share is 56 bytes (${by.masks.bytesPerHospitalPerRound})`);
  check(r.designs.filter((d) => d.kind !== "masks").every((d) => d.bytesPerHospitalPerRound >= 100_000),
    "every encrypted design sends at least a 100 kB ciphertext per hospital per round");
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
console.log(failed ? `${failed} check(s) FAILED` : "all checks passed");
process.exit(failed ? 1 : 0);
