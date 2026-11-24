#!/usr/bin/env node
/**
 * The tree QCNN under CKKS, one row a slot, on the real node-seal (Round 86; references/satellite-qcnn-tn-preregistration.md, D).
 *
 * Usage: node web/scripts/check_qcnn_tree_ckks.mjs IN.json OUT.json
 *   IN: { cases: [{ name, network, rows, expected, weights_path, pauli_rows }] }, written by scripts/measure_qcnn_tree_ckks.py:
 *   the network (qsh.qhe.qcnn_tree), the test rows in raw units, the clear readouts (equal to the state vector), the
 *   artifact for the Pauli-form path (`clientQcnnCkks.ts`), and how many of the rows to run that path on.
 *   OUT: per case, the decrypted readouts' largest difference from the clear ones, the decisions that differ, the
 *   times and the bytes, for both paths.
 */
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { build } = createRequire(join(WEB, "package.json"))("esbuild");
const sealWasm = join(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
const [inPath, outPath] = process.argv.slice(2);
if (!inPath || !outPath) {
  console.error("usage: check_qcnn_tree_ckks.mjs IN.json OUT.json");
  process.exit(2);
}
const tmp = mkdtempSync(join(tmpdir(), "qsh-qcnn-tree-ckks-"));
try {
  await build({
    stdin: { contents: `export { treeReadoutCkks, clearReadout, TREE_CKKS } from ${JSON.stringify(join(WEB, "src/lib/clientQcnnTreeCkks"))};
export { qcnnInferCkks } from ${JSON.stringify(join(WEB, "src/lib/clientQcnnCkks"))};`, resolveDir: WEB, loader: "ts" },
    bundle: true, format: "esm", platform: "node", outfile: join(tmp, "b.mjs"), logLevel: "error",
    define: { "import.meta.env": '{"BASE_URL":"/"}' },
    plugins: [{ name: "wasm-url", setup(b) {
      b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
      b.onLoad({ filter: /.*/, namespace: "wasmurl" }, () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    } }],
  });
  const m = await import(pathToFileURL(join(tmp, "b.mjs")).href);
  const input = JSON.parse(readFileSync(inPath, "utf8"));
  const out = { params: m.TREE_CKKS, cases: [] };
  for (const c of input.cases) {
    const net = c.network;
    const clear = m.clearReadout(net, c.rows);
    let clearMax = 0;
    for (let i = 0; i < clear.length; i++) clearMax = Math.max(clearMax, Math.abs(clear[i] - c.expected[i]));
    const t0 = performance.now();
    const run = await m.treeReadoutCkks(net, c.rows);
    const totalMs = performance.now() - t0;
    let worst = 0, flipped = 0;
    for (let i = 0; i < run.rows; i++) {
      worst = Math.max(worst, Math.abs(run.readouts[i] - c.expected[i]));
      const a = run.readouts[i] * net.scale + net.bias, b = c.expected[i] * net.scale + net.bias;
      if ((a > 0) !== (b > 0)) flipped++;
    }
    // The Pauli-form path, one row at a time, on the first rows.
    const w = JSON.parse(readFileSync(c.weights_path, "utf8"));
    const pauli = { rows: 0, maxError: 0, flipped: 0, totalMs: 0, upBytes: 0, downBytes: 0, keyBytes: 0 };
    for (let i = 0; i < Math.min(c.pauli_rows, c.rows.length); i++) {
      const r = await m.qcnnInferCkks(w, c.rows[i]);
      pauli.rows++;
      pauli.maxError = Math.max(pauli.maxError, Math.abs(r.readouts[0] - c.expected[i]));
      if (!r.matchesClear) pauli.flipped++;
      pauli.totalMs += r.totalMs;
      pauli.upBytes += r.upBytes;
      pauli.downBytes += r.downBytes;
      pauli.keyBytes = r.keyBytes;
    }
    out.cases.push({
      name: c.name, rows: run.rows, blocks: run.blocks, clear_twin_max_difference: clearMax,
      tree: { max_abs_readout_error: worst, decisions_differing: flipped, ciphertexts_up: run.ciphertextsUp,
              up_bytes: run.upBytes, down_bytes: run.downBytes, key_bytes: run.keyBytes, key_ms: run.keyMs,
              encrypt_ms: run.encryptMs, evaluate_ms: run.evaluateMs, decrypt_ms: run.decryptMs, total_ms: totalMs },
      pauli,
    });
    console.log(`${c.name}: ${run.rows} rows in ${run.blocks} block(s); tree max error ${worst.toExponential(2)}, ${flipped} decisions differ, `
      + `${(totalMs / run.rows).toFixed(3)} ms and ${(run.upBytes / run.rows).toFixed(0)} B up a row; Pauli path on ${pauli.rows} rows: `
      + `max error ${pauli.maxError.toExponential(2)}, ${(pauli.totalMs / Math.max(1, pauli.rows)).toFixed(1)} ms and `
      + `${(pauli.upBytes / Math.max(1, pauli.rows)).toFixed(0)} B up a row`);
  }
  out.node = process.version;
  writeFileSync(outPath, JSON.stringify(out, null, 1) + "\n");
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
