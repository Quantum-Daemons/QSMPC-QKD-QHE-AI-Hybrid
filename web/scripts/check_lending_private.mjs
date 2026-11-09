#!/usr/bin/env node
/**
 * The lending page's private liquidation check, against the same check in the clear.
 *
 * WHAT THIS IS FOR. The page says Bob's balances stayed under his key, the pool's thresholds stayed with the pool, and the only thing the circuit produced was the
 * sign of the margin. This runs the module the page runs, on the real node-seal and the real Polytune, and asserts what the card reports:
 *
 *   - every checkpoint of a default run gets the clear answer, and the margin added back from the two shares is close to the clear margin;
 *   - positions built to sit on either side of zero by one stable unit get the right answer;
 *   - what Bob decoded is masked: the median magnitude across all slots is about half the mask's width, far above any margin;
 *   - the circuit is the size the existing argmax has, and every check cancelled the imaginary half and used the two rotations the design says;
 *   - the sealing is real: a changed byte, a different key or a signature from another key is refused;
 *   - the class holding the weights has no decryptor, which the medical check asserts of the same class.
 *
 * Run: npm run check:lending-private (about 10 s)
 */
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { build } = createRequire(join(WEB, "package.json"))("esbuild");
const sealWasm = join(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
const mpcWasm = join(WEB, "src/wasm/qsh_mpc/qsh_mpc_bg.wasm");

const tmp = mkdtempSync(join(tmpdir(), "qsh-lending-private-"));
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };
const refused = async (run) => { try { await run(); return false; } catch { return true; } };

try {
  const entry = `
    export * from ${JSON.stringify(join(WEB, "src/lib/lending/privateCheck"))};
    export { simulate } from ${JSON.stringify(join(WEB, "src/lib/lending/model"))};
    export { sweepLevel, executeTask, closeTaskSession } from ${JSON.stringify(join(WEB, "src/lib/lending/tasks"))};
    export { poolSize, runLendingTasks } from ${JSON.stringify(join(WEB, "src/lib/lending/pool"))};
    export { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";
    export { lendingCfg } from ${JSON.stringify(join(WEB, "src/lib/config"))};`;
  await build({
    stdin: { contents: entry, resolveDir: WEB, loader: "ts" },
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
  const cfg = m.lendingCfg();
  const sessionKey = crypto.getRandomValues(new Uint8Array(32));
  const session = await m.openCheckSession({ sessionKey, sig: "ml-dsa-65", cfg });

  // ---- a default run's checkpoints
  const run = m.simulate(cfg, 1);
  const outcomes = [];
  for (const c of run.checkpoints) outcomes.push(await session.check(c));
  const bound = outcomes.reduce((s, o) => s + o.statisticalDistance, 0);
  const worst = Math.max(...outcomes.map((o) => o.absError));
  const liquidatable = outcomes.filter((o) => o.flag).length;
  check(outcomes.length >= 2 && outcomes.length <= cfg.check_points + 1 && run.checkpoints.every((c) => c.debtValue > 0),
    `${outcomes.length} checkpoints ran, all on an open position (at most the configured ${cfg.check_points}, plus the first hour Bob can be liquidated)`);
  check(outcomes.every((o) => o.agrees), "every checkpoint's bit is the clear answer");
  check(liquidatable >= 1 && liquidatable < outcomes.length, `the run has both answers: ${liquidatable} of ${outcomes.length} liquidatable`);
  check(worst < 1e-3, `the margin from the shares is within 1 stable unit of the clear margin (worst ${(worst * 1000).toExponential(2)} stable units)`);
  const mask = 2 ** cfg.margin_mask_bits;
  check(outcomes.every((o) => o.bobViewMedianAbs > mask / 8), `what Bob decoded is masked (smallest median |value| ${Math.min(...outcomes.map((o) => o.bobViewMedianAbs)).toExponential(2)}, against margins of about ${Math.max(...outcomes.map((o) => Math.abs(o.clearMargin))).toFixed(0)})`);
  check(outcomes.every((o) => o.andGates === 367), `the circuit is the existing two-value argmax (${outcomes[0].andGates} AND gates)`);
  check(outcomes.every((o) => o.rotations === 2 && o.conjugations === 1), "each check made two rotations and one conjugation");
  check(Math.log2(bound) < -20, `the run's bound on the masks is below 2^-20 (2^${Math.log2(bound).toFixed(1)})`);

  // ---- positions built on either side of zero, by one stable unit
  for (const margin of [1, -1, 25, -25]) {
    const prices = cfg.initial_prices, thresholds = cfg.liquidation_thresholds, amounts = [10, 0.5];
    const collateral = amounts[0] * prices[0] * thresholds[0] + amounts[1] * prices[1] * thresholds[1];
    const debtValue = collateral - margin;
    const o = await session.check({ hour: 100000 + margin, amounts, scaledDebt: debtValue, prices, thresholds, index: 1, debtValue, margin, healthFactor: collateral / debtValue });
    check(o.agrees && o.flag === (margin < 0), `a margin of ${margin} stable units gets the bit ${margin < 0 ? "liquidatable" : "safe"} (reconstructed ${(o.reconstructedMargin * 1000).toFixed(3)})`);
  }

  // ---- the pool's parts: a changed byte, another key and another signer are all refused
  const { submission: key } = m.checkKeys(sessionKey);
  const signer = m.ml_dsa65.keygen(crypto.getRandomValues(new Uint8Array(32)));
  const other = m.ml_dsa65.keygen(crypto.getRandomValues(new Uint8Array(32)));
  const ct = crypto.getRandomValues(new Uint8Array(64));
  const signature = m.ml_dsa65.sign(m.submissionMessage(7, ct), signer.secretKey);
  const verifyWith = (pub) => (sig, msg) => m.ml_dsa65.verify(sig, msg, pub);
  const record = await m.sealSubmission(key, 7, signature, ct);
  const fine = await m.openSubmission(key, record, 7, signature.length, verifyWith(signer.publicKey));
  check(fine.ciphertext.length === ct.length && fine.ciphertext.every((v, i) => v === ct[i]), "an untouched submission opens and verifies");
  const flipped = record.slice(); flipped[40] ^= 1;
  check(await refused(() => m.openSubmission(key, flipped, 7, signature.length, verifyWith(signer.publicKey))), "a changed byte is refused");
  check(await refused(() => m.openSubmission(crypto.getRandomValues(new Uint8Array(32)), record, 7, signature.length, verifyWith(signer.publicKey))), "another session's key is refused");
  check(await refused(() => m.openSubmission(key, record, 8, signature.length, verifyWith(signer.publicKey))), "a record replayed for another hour is refused");
  check(await refused(() => m.openSubmission(key, record, 7, signature.length, verifyWith(other.publicKey))), "a signature from another key is refused");
  session.close();

  // ---- the pool's size, and the task path the workers take
  check(m.poolSize(4, 9, 20) === 4 && m.poolSize(4, 2, 20) === 2 && m.poolSize(4, 9, 4) === 2 && m.poolSize(4, 9, 1) === 1, "the pool is capped by the configuration, the tasks and half the logical cores");
  const levels = cfg.sweep_multipliers.map((multiplier) => ({ kind: "sweep", cfg, multiplier, seeds: [1, 2, 3, 4, 5, 6, 7, 8] }));
  const pooled = await m.runLendingTasks(levels, {}, cfg.max_workers);
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const series = pooled.results.map((r) => mean(r.outcome.liquidations));
  check(series.at(-1) > series[0], `more volatility, more liquidations (mean ${series[0].toFixed(1)} at ${cfg.sweep_multipliers[0]}x, ${series.at(-1).toFixed(1)} at ${cfg.sweep_multipliers.at(-1)}x)`);
  check(pooled.workers === 1 && pooled.ranIn === "the calling thread", "without a Worker the same tasks run on the calling thread, and the run says so");

  // The workers' own path, with a Worker that answers in this thread: finishing is the pool's job, and it hung once when no progress callback was given.
  class FakeWorker {
    static made = 0;
    constructor() { FakeWorker.made++; }
    terminate() {}
    postMessage(message) {
      if (message.id === undefined) return;
      setTimeout(async () => {
        const result = await m.executeTask(message.task, {});
        this.onmessage({ data: { id: message.id, ok: true, result, busyMs: 2 } });
      }, 0);
    }
  }
  globalThis.window = {};
  globalThis.Worker = FakeWorker;
  const timeout = (ms) => new Promise((_, no) => setTimeout(() => no(new Error("the pool did not finish")), ms));
  try {
    const silent = await Promise.race([m.runLendingTasks(levels, {}, 3), timeout(20000)]);
    check(silent.results.length === levels.length && silent.results.every(Boolean), `the pool finishes with no progress callback (${silent.results.length} levels in ${silent.ranIn})`);
    check(FakeWorker.made === silent.workers && silent.workers === m.poolSize(3, levels.length, navigator.hardwareConcurrency), `it made the workers it reports (${FakeWorker.made})`);
    check(silent.busyMs.every((b) => b === 2) && silent.utilisation > 0 && Math.abs(silent.speedUp - silent.utilisation * silent.workers) < 1e-9, "its busy time, speed-up and utilisation are the workers' own figures");
    const seen = [];
    await Promise.race([m.runLendingTasks(levels, {}, 3, (done, total) => seen.push(`${done}/${total}`)), timeout(20000)]);
    check(seen.length === levels.length && seen.at(-1) === `${levels.length}/${levels.length}`, "and it reports every task's progress when asked");
  } finally {
    delete globalThis.window;
    delete globalThis.Worker;
  }
  const again = m.sweepLevel(cfg, 2, [1, 2, 3]), twice = m.sweepLevel(cfg, 2, [1, 2, 3]);
  check(JSON.stringify(again) === JSON.stringify(twice), "a sweep level is reproducible from its seeds");
  const pair = await m.runLendingTasks(run.checkpoints.slice(0, 2).map((input) => ({ kind: "check", input })), { sessionKey, sig: "ml-dsa-87", cfg }, 1);
  check(pair.results.every((r) => r.kind === "check" && r.outcome.agrees && r.signatureBytes === 4627), `the task path runs a check, with the selected signature (${pair.results[0].signatureBytes} B)`);
  m.closeTaskSession();

  const src = readFileSync(join(WEB, "src/lib/lending/privateCheck.ts"), "utf8");
  check(/M < 0/.test(src) && /public, and then Bob can compute M himself/.test(src), "the module states why the thresholds are the pool's secret");
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
if (failed) { console.error(`${failed} check(s) failed`); process.exit(1); }
console.log("private lending check: all checks passed");
