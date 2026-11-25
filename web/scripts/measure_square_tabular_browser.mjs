#!/usr/bin/env node
/**
 * F7b's criterion 4 (references/finance-square-student-page-preregistration.md): the finance square network, by the
 * page's own module (src/lib/clientFHESquareTabular.ts), in a Web Worker in headless Chrome, on one block of the rows the
 * finance page ships. Each run is a fresh worker, as the page would start one: it scales the rows, generates the keys,
 * encrypts, evaluates and decrypts, and reports the time from the start of key generation to the last decryption, the
 * size of its WebAssembly memory after the run (a wasm memory only grows, so that is its peak), and the decrypted
 * decisions against the plaintext ones, which the same module computes in float64.
 *
 * node-seal does not export its heap, so the worker wraps WebAssembly.instantiate and instantiateStreaming before the
 * module is loaded and keeps the Memory the instance exports.
 *
 * Usage: node web/scripts/measure_square_tabular_browser.mjs WEIGHTS.json ROWS.json OUT.json [RUNS]
 *   WEIGHTS: the exported seed's weights.json, as scripts/train/finance_square_student_page.py writes it
 *   ROWS: web/public/data/finance/student_heldout.json (raw inputs in `samples`, their names in `feature_names`)
 *   OUT: per set, each run's figures, the browser's version and the load average before each run
 */
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { loadavg, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { build } = createRequire(join(WEB, "package.json"))("esbuild");
const CHROME = process.env.CHROME ?? "google-chrome";
const CDP_PORT = Number(process.env.CDP_PORT ?? 9471);   // none of the gate's ports
const [weightsPath, rowsPath, outPath, runsArg] = process.argv.slice(2);
if (!weightsPath || !rowsPath || !outPath) {
  console.error("usage: measure_square_tabular_browser.mjs WEIGHTS.json ROWS.json OUT.json [RUNS]");
  process.exit(2);
}
const RUNS = Number(runsArg ?? 3);
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));

const WORKER = `
import { runSquareTabular, scaleRow } from ${JSON.stringify(join(WEB, "src/lib/clientFHESquareTabular"))};
let memory = null;
const keep = (r) => {
  const instance = r && r.instance ? r.instance : r;
  for (const v of Object.values(instance.exports)) if (v instanceof WebAssembly.Memory) memory = v;
  return r;
};
const streaming = WebAssembly.instantiateStreaming, plain = WebAssembly.instantiate;
if (streaming) WebAssembly.instantiateStreaming = (...a) => streaming.apply(WebAssembly, a).then(keep);
WebAssembly.instantiate = (...a) => plain.apply(WebAssembly, a).then(keep);
self.onmessage = async (e) => {
  try {
    const { w, samples, set, withPlain } = e.data;
    const ts = performance.now();
    const z = new Float64Array(samples.length * w.n_in);
    for (let i = 0; i < samples.length; i++) z.set(scaleRow(w, samples[i]), i * w.n_in);
    const scaleMs = performance.now() - ts;
    const t0 = performance.now();
    const r = await runSquareTabular(w, z, samples.length, set, () => {});
    const totalMs = performance.now() - t0;
    let flagged = 0;
    for (let i = 0; i < r.rows; i++) if (r.logits[2 * i + 1] > r.logits[2 * i]) flagged++;
    self.postMessage({ ok: true, set, rows: r.rows, blocks: r.blocks, total_ms: totalMs, scale_ms: scaleMs,
      key_ms: r.keyMs, key_bytes: r.keyBytes, per_block: r.per_block, max_abs_error: r.maxAbsError,
      decisions_differ: r.decisionsDiffer, flagged_decrypted: flagged,
      wasm_memory_bytes: memory ? memory.buffer.byteLength : null,
      plain: withPlain ? Array.from(r.plain) : null });
  } catch (err) {
    self.postMessage({ ok: false, error: String(err && err.stack || err) });
  }
};
`;

const PAGE = `<!doctype html><meta charset="utf-8"><title>F7b criterion 4</title>
<script type="module">
const w = await (await fetch("/weights.json")).json();
const rows = await (await fetch("/rows.json")).json();
window.__run = (set, n, withPlain) => new Promise((ok, bad) => {
  const worker = new Worker("/worker.js", { type: "module" });
  worker.onmessage = (e) => { worker.terminate(); e.data.ok ? ok(e.data) : bad(e.data.error); };
  worker.onerror = (e) => { worker.terminate(); bad(String(e.message || e)); };
  worker.postMessage({ w, samples: rows.samples.slice(0, n), set, withPlain });
});
window.__ready = { n: rows.samples.length, names: rows.feature_names.length };
</script>`;

const tmp = mkdtempSync(join(tmpdir(), "qsh-f7b-browser-"));
let server = null, chrome = null;
try {
  const wasmPath = join(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
  const bundled = await build({
    stdin: { contents: WORKER, resolveDir: WEB, loader: "ts" },
    bundle: true, format: "esm", platform: "browser", write: false, logLevel: "error",
    // node-seal's Node-only branch imports these; in a browser that branch never runs, as in the site's own build.
    external: ["module", "fs", "path", "url", "crypto", "worker_threads", "node:*"],
    define: { "import.meta.env": '{"BASE_URL":"/"}' },
    plugins: [{ name: "wasm-url", setup(b) {
      b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
      b.onLoad({ filter: /.*/, namespace: "wasmurl" }, () => ({ contents: 'export default "/seal.wasm";', loader: "js" }));
    } }],
  });
  const files = {
    "/": ["text/html", Buffer.from(PAGE)],
    "/worker.js": ["text/javascript", Buffer.from(bundled.outputFiles[0].contents)],
    "/seal.wasm": ["application/wasm", readFileSync(wasmPath)],
    "/weights.json": ["application/json", readFileSync(weightsPath)],
    "/rows.json": ["application/json", readFileSync(rowsPath)],
  };
  server = createServer((req, res) => {
    const f = files[(req.url ?? "/").split("?")[0]];
    if (!f) { res.writeHead(404); res.end(); return; }
    // As the deployed site: cross-origin isolated, same-origin resources only.
    res.writeHead(200, { "Content-Type": f[0], "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp", "Cross-Origin-Resource-Policy": "same-origin",
      "Cache-Control": "no-store" });
    res.end(f[1]);
  });
  await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
  const base = `http://127.0.0.1:${server.address().port}`;

  chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${CDP_PORT}`, `--user-data-dir=${join(tmp, "profile")}`,
    "--no-first-run", "--no-default-browser-check", "about:blank"], { stdio: "ignore" });
  let target = null;
  for (let k = 0; k < 100 && !target; k++) {
    try { target = (await (await fetch(`http://127.0.0.1:${CDP_PORT}/json`)).json()).find((t) => t.type === "page"); }
    catch { await sleep(100); }
  }
  if (!target) throw new Error("headless Chrome did not answer on its DevTools port");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((ok, bad) => { ws.onopen = ok; ws.onerror = bad; });
  let id = 0;
  const pending = new Map();
  ws.onmessage = (m) => { const msg = JSON.parse(m.data); if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); } };
  const send = (method, params = {}) => new Promise((ok) => { const n = ++id; pending.set(n, ok); ws.send(JSON.stringify({ id: n, method, params })); });
  const evaluate = async (expression) => {
    const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 2000));
    return r.result?.result?.value;
  };
  const version = (await send("Browser.getVersion")).result;
  await send("Page.navigate", { url: `${base}/` });
  let ready = null;
  for (let k = 0; k < 300 && !ready; k++) { ready = await evaluate("window.__ready ?? null"); if (!ready) await sleep(100); }
  if (!ready) throw new Error("the harness page did not load its files");
  const w = JSON.parse(readFileSync(weightsPath, "utf8"));
  const out = { browser: version.product, user_agent: version.userAgent, rows_shipped: ready.n, runs_per_set: RUNS, sets: {} };
  for (const set of Object.keys(w.ckks)) {
    const n = Math.min(ready.n, w.ckks[set].N / 2);
    out.sets[set] = [];
    for (let k = 0; k < RUNS; k++) {
      const before = loadavg().map((x) => Math.round(x * 100) / 100);
      const r = await evaluate(`window.__run(${JSON.stringify(set)}, ${n}, ${k === 0})`);
      out.sets[set].push({ ...r, load_average_before: before });
      console.log(`${set} run ${k + 1}: ${r.rows} rows, ${(r.total_ms / 1000).toFixed(2)} s keys to last decryption `
        + `(keys ${r.key_ms.toFixed(0)} ms), wasm memory ${(r.wasm_memory_bytes / 1e9).toFixed(3)} GB, `
        + `${r.decisions_differ} decisions differ, max |err| ${r.max_abs_error.toExponential(2)}; load ${before.join(" ")}`);
    }
  }
  writeFileSync(outPath, JSON.stringify(out) + "\n");
  ws.close();
} finally {
  if (chrome) chrome.kill();
  if (server) server.close();
  await sleep(300);
  rmSync(tmp, { recursive: true, force: true });
}
