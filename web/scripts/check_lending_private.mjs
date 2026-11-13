#!/usr/bin/env node
/**
 * The lending page's private liquidation check, against the same check in the clear.
 *
 * WHAT THIS IS FOR. The page says Bob's balances stayed under his key, the pool's thresholds stayed with the pool, and the only thing the circuit produced was the
 * sign of the margin. This runs the module the page runs, on the real node-seal and the real Polytune, and asserts what the card reports:
 *
 *   - every checkpoint of a default run gets the clear answer, and the margin added back from the two shares is close to the clear margin;
 *   - positions built to sit on either side of zero by one stable unit get the right answer;
 *   - what the key holder decoded is masked: the median magnitude across all slots is about half the mask's width, far above any margin;
 *   - the circuit is the size the existing argmax has, and every check cancelled the imaginary half and used the two rotations the design says;
 *   - under Carol's custody Bob sends nothing during a check, only masked margins reach her, each settled liquidation is applied to his ciphertext, and her signature binds the hour, the ciphertext and the bit;
 *   - the sealing is real: a changed byte, a different key or a signature from another key is refused;
 *   - the class holding the weights has no decryptor, which the medical check asserts of the same class;
 *   - under a 2-of-3 committee on threshold BGV no one holds the key: two distinct members re-encrypt each masked margin to Carol, all three pairs take a turn, only
 *     slot 0 carries anything, the mask is uniform modulo t, the bits match the clear answer at one stable unit either side of zero, and Carol signs every bit.
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
const fedDir = join(WEB, "public/wasm/qsh_fedthresh");

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
    if (String(url).endsWith("qsh_fedthresh.wasm")) return new Response(readFileSync(join(fedDir, "qsh_fedthresh.wasm")), { headers: { "Content-Type": "application/wasm" } });
    throw new Error(`unexpected fetch of ${url}`);
  };
  const m = await import(pathToFileURL(join(tmp, "b.mjs")).href);
  const cfg = m.lendingCfg();
  const sessionKey = crypto.getRandomValues(new Uint8Array(32));
  const mask = 2 ** cfg.margin_mask_bits;
  const run = m.simulate(cfg, 1);

  async function holds(name, session, checkpoints) {
    const outcomes = [];
    for (const c of checkpoints) outcomes.push(await session.check(c));
    const bound = outcomes.reduce((s, o) => s + o.statisticalDistance, 0);
    const worst = Math.max(...outcomes.map((o) => o.absError));
    check(outcomes.every((o) => o.agrees), `${name}: every checkpoint's bit is the clear answer`);
    check(worst < 1e-3, `${name}: the margin from the shares is within 1 stable unit of the clear margin (worst ${(worst * 1000).toExponential(2)} stable units)`);
    check(outcomes.every((o) => o.keyHolderViewMedianAbs > mask / 8), `${name}: what the key holder decoded is masked (smallest median |value| ${Math.min(...outcomes.map((o) => o.keyHolderViewMedianAbs)).toExponential(2)}, against margins of about ${Math.max(...outcomes.map((o) => Math.abs(o.clearMargin))).toFixed(0)})`);
    check(outcomes.every((o) => o.andGates === 367), `${name}: the circuit is the existing two-value argmax (${outcomes[0].andGates} AND gates)`);
    check(outcomes.every((o) => o.rotations === 2 && o.conjugations === 1), `${name}: each check made two rotations and one conjugation`);
    check(Math.log2(bound) < -20, `${name}: the run's bound on the masks is below 2^-20 (2^${Math.log2(bound).toFixed(1)})`);
    return outcomes;
  }

  // ---- Bob holds the key (the bilateral mode)
  const session = await m.openCheckSession({ sessionKey, sig: "ml-dsa-65", cfg, custody: "borrower" });
  check(run.checkpoints.length >= 2 && run.checkpoints.length <= cfg.check_points + 1 && run.checkpoints.every((c) => c.debtValue > 0),
    `${run.checkpoints.length} checkpoints, all on an open position (at most the configured ${cfg.check_points}, plus the first hour Bob can be liquidated)`);
  const bilateral = await holds("bilateral", session, run.checkpoints);
  const liquidatable = bilateral.filter((o) => o.flag).length;
  check(liquidatable >= 1 && liquidatable < bilateral.length, `the run has both answers: ${liquidatable} of ${bilateral.length} liquidatable`);
  check(bilateral.every((o) => o.messages.filter((x) => x.from === "Bob").length === 1 && o.bitSignatureVerified === null), "bilateral: Bob sends at every check, and there is no signed bit");

  // ---- positions built on either side of zero, by one stable unit
  for (const margin of [1, -1, 25, -25]) {
    const prices = cfg.initial_prices, thresholds = cfg.liquidation_thresholds, amounts = [10, 0.5];
    const collateral = amounts[0] * prices[0] * thresholds[0] + amounts[1] * prices[1] * thresholds[1];
    const debtValue = collateral - margin;
    const o = await session.check({ hour: 100000 + margin, amounts, scaledDebt: debtValue, settled: { seized: [0, 0], scaledRepaid: 0 }, prices, thresholds, index: 1, debtValue, margin, healthFactor: collateral / debtValue });
    check(o.agrees && o.flag === (margin < 0), `a margin of ${margin} stable units gets the bit ${margin < 0 ? "liquidatable" : "safe"} (reconstructed ${(o.reconstructedMargin * 1000).toFixed(3)})`);
  }

  // ---- Carol holds the key: Bob sends once, and a settled liquidation is applied by the pool. Seed 3 has Bob partly liquidated twice.
  const partial = m.simulate(cfg, 3);
  check(partial.checkpoints.some((c) => c.settled.scaledRepaid > 0), "the seed-3 run has checkpoints after a partial liquidation of Bob");
  const setup = await m.setupKeyholder({ sessionKey, sig: "ml-dsa-65", kem: "ml-kem-768+x25519" }, partial.checkpoints[0]);
  check(setup.linkAgreed, "the pool and Carol's own hybrid handshake agrees");
  const carolSession = await m.openCheckSession({ sessionKey, sig: "ml-dsa-65", cfg, custody: "keyholder", keyholder: setup });
  const custody = await holds("Carol's custody", carolSession, partial.checkpoints);
  check(custody.every((o) => o.messages.every((x) => x.from !== "Bob")), "Carol's custody: Bob sends nothing during any check");
  check(custody.every((o) => o.messages.filter((x) => x.to === "Carol").every((x) => x.what === "the masked margin, sealed")), "Carol's custody: only the masked margin reaches Carol");
  check(custody.every((o) => o.bitSignatureVerified === true), "Carol's custody: every signed bit verifies against her public key");
  check(custody.some((o) => o.hour > 0 && Math.abs(o.clearMargin) > 0) && custody.every((o) => o.encryptMs === 0), "Carol's custody: no check encrypts anything new");
  const moved = custody.filter((o) => partial.checkpoints.find((c) => c.hour === o.hour).settled.scaledRepaid > 0);
  check(moved.length >= 1 && moved.every((o) => o.agrees && o.absError < 1e-3), `after a settlement the pool's updated ciphertext still gives the clear margin (${moved.length} checks)`);
  carolSession.close();

  // ---- what Carol's signature binds
  const carolKey = m.ml_dsa65.keygen(crypto.getRandomValues(new Uint8Array(32)));
  const digestOf = crypto.getRandomValues(new Uint8Array(64));
  const signed = m.ml_dsa65.sign(m.bitMessage(360, digestOf, true), carolKey.secretKey);
  const bitOk = (hour, ct, flag) => m.ml_dsa65.verify(signed, m.bitMessage(hour, ct, flag), carolKey.publicKey);
  check(bitOk(360, digestOf, true), "a signed bit verifies");
  check(!bitOk(360, digestOf, false), "the other bit does not");
  check(!bitOk(361, digestOf, true), "the same bit for another hour does not");
  const otherCt = digestOf.slice(); otherCt[0] ^= 1;
  check(!bitOk(360, otherCt, true), "the same bit for another ciphertext does not");

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
  const pair = await m.runLendingTasks(run.checkpoints.slice(0, 2).map((input) => ({ kind: "check", input })), { sessionKey, sig: "ml-dsa-87", cfg, custody: "borrower" }, 1);
  check(pair.results.every((r) => r.kind === "check" && r.outcome.agrees && r.signatureBytes === 4627), `the task path runs a check, with the selected signature (${pair.results[0].signatureBytes} B)`);
  m.closeTaskSession();

  // ---- the same through the task path a worker takes, for Carol's custody
  const init = { sessionKey, sig: "ml-dsa-65", cfg, custody: "keyholder", kem: "ml-kem-768+x25519" };
  const first = await m.runLendingTasks([{ kind: "setup", input: run.checkpoints[0] }], init, 1);
  const viaTasks = await m.runLendingTasks(run.checkpoints.map((input) => ({ kind: "check", input })), { ...init, keyholder: first.results[0].setup }, cfg.max_workers);
  check(viaTasks.results.every((r) => r.outcome.agrees && r.outcome.bitSignatureVerified === true), `the task path runs Carol's custody (${viaTasks.results.length} checks)`);
  m.closeTaskSession();

  // ---- a 2-of-3 committee: no one holds the key, and two members re-encrypt each masked margin to Carol on threshold BGV. Seed 3 again, for the settlements.
  const fedThreshBase = pathToFileURL(fedDir).href + "/";
  const cInit = { sessionKey, sig: "ml-dsa-65", cfg, custody: "committee", kem: "ml-kem-768+x25519", first: partial.checkpoints[0], fedThreshBase };
  const cSession = await m.openCheckSession(cInit);
  const cs = cSession.committee;
  check(cs?.members === 3 && cs.threshold === 2 && cs.statSecBits === 128 && cs.logN === 14 && Math.abs(cs.logQ - 260) < 1,
    `the committee: 3 members, any 2 re-encrypt, N = 2^${cs?.logN}, a ${cs?.logQ.toFixed(1)}-bit Q, t ${cs?.plainModulus}`);
  check(cs.floodBits === 189 && Math.abs(cs.noiseBoundLog2 - 47.85) < 0.01 && cs.plainModulus < 2 ** 53 && cs.plainModulus % 32768 === 1,
    `each share is flooded with ${cs.floodBits}-bit uniform noise against a worst case of 2^${cs.noiseBoundLog2.toFixed(2)}; t is 1 mod 2N and below 2^53`);
  check(cs.linkAgreed && cs.submissionBytes > 3e6, `Bob's one submission reached the pool sealed (${(cs.submissionBytes / 1e6).toFixed(2)} MB), and the pool and Carol's handshake agrees`);
  const cOut = [];
  for (const c of partial.checkpoints) cOut.push(await cSession.check(c));
  const cWorst = Math.max(...cOut.map((o) => o.absError));
  check(cOut.every((o) => o.agrees), `the committee: every checkpoint's bit is the clear answer (${cOut.length} checks, ${cOut.filter((o) => o.flag).length} liquidatable)`);
  check(cWorst < 1e-3, `the committee: the margin from the shares is within 1 stable unit of the clear margin (worst ${(cWorst * 1000).toExponential(2)} stable units)`);
  check(cOut.every((o) => o.committee.restZero), "the committee: every slot but the first decrypted to zero, at every check");
  check(cOut.every((o) => o.statisticalDistance === 0 && o.rotations === 0 && o.conjugations === 0), "the committee: a mask uniform modulo t, and no rotation or conjugation");
  check(cOut.every((o) => Math.abs(o.keyHolderViewMedianAbs - Math.abs(o.reconstructedMargin)) > 1e-3),
    `the committee: what Carol decrypted is never the margin itself (smallest |value| ${Math.min(...cOut.map((o) => o.keyHolderViewMedianAbs)).toFixed(1)} thousand, margins up to ${Math.max(...cOut.map((o) => Math.abs(o.clearMargin))).toFixed(1)})`);
  const pairs = cOut.map((o) => o.committee.active.join(" and "));
  check(cOut.every((o) => o.committee.active.length === 2 && new Set(o.committee.active).size === 2 && o.committee.active.every((i) => i >= 1 && i <= 3)) && new Set(pairs).size === 3,
    `the committee: two distinct members at each check, and every pair takes a turn (${[...new Set(pairs)].join("; ")})`);
  check(cOut.every((o) => o.messages.every((x) => x.from !== "Bob") && o.messages.filter((x) => x.to === "Carol").every((x) => x.what === "the re-encrypted masked margin, sealed")),
    "the committee: Bob sends nothing during any check, and only the re-encrypted masked margin reaches Carol");
  check(cOut.every((o) => o.messages.filter((x) => x.from.startsWith("member ")).length === 2 && o.messages.filter((x) => x.to.startsWith("member ")).length === 2),
    "the committee: each check sends c1 to the two active members and takes one share back from each");
  check(cOut.every((o) => o.bitSignatureVerified === true), "the committee: every signed bit verifies against Carol's public key");
  const signGates = cOut[0].andGates;
  check(cOut.every((o) => o.andGates === 648), `the committee: the sign circuit modulo t is ${signGates} AND gates at every check, as the crate's test pins`);
  const cMoved = cOut.filter((o) => partial.checkpoints.find((c) => c.hour === o.hour).settled.scaledRepaid > 0);
  check(cMoved.length >= 1 && cMoved.every((o) => o.agrees && o.absError < 1e-3), `the committee: after a settlement the pool's ciphertexts still give the clear margin (${cMoved.length} checks)`);
  cSession.close();
  for (const margin of [1, -1]) {
    const prices = cfg.initial_prices, thresholds = cfg.liquidation_thresholds, amounts = [10, 0.5];
    const collateral = amounts[0] * prices[0] * thresholds[0] + amounts[1] * prices[1] * thresholds[1];
    const debtValue = collateral - margin;
    const input = { hour: 200000 + margin, amounts, scaledDebt: debtValue, settled: { seized: [0, 0], scaledRepaid: 0 }, prices, thresholds, index: 1, debtValue, margin, healthFactor: collateral / debtValue };
    const one = await m.openCheckSession({ ...cInit, first: input });
    const o = await one.check(input);
    check(o.agrees && o.flag === (margin < 0), `the committee: a margin of ${margin} stable unit gets the bit ${margin < 0 ? "liquidatable" : "safe"} (reconstructed ${(o.reconstructedMargin * 1000).toFixed(3)})`);
    one.close();
  }
  const viaCommittee = await m.runLendingTasks(partial.checkpoints.slice(0, 3).map((input) => ({ kind: "check", input })), cInit, 1);
  check(viaCommittee.results.every((r) => r.outcome.agrees && r.outcome.bitSignatureVerified === true && r.committee?.members === 3),
    `the task path runs the committee's custody on one worker (${viaCommittee.results.length} checks)`);
  m.closeTaskSession();

  const csrc = readFileSync(join(WEB, "src/lib/lending/committee.ts"), "utf8");
  const bobC = csrc.slice(csrc.indexOf("// Bob, with the committee's public key only"), csrc.indexOf("// The pool: opens it"));
  check(bobC.length > 100 && !/committeeDecrypt|committeeCheck|secretKey(?!\))/.test(bobC.replace(/bob\.secretKey/g, "")), "Bob's part encrypts under the committee's key and calls nothing that decrypts");
  check((csrc.match(/committeeDecrypt\(/g) ?? []).length === 1 && /committeeDecrypt\(id, openedCt\)/.test(csrc), "Carol decrypts exactly one record, the one the pool sealed to her");

  const src = readFileSync(join(WEB, "src/lib/lending/privateCheck.ts"), "utf8");
  check(/M < 0/.test(src) && /public, and then Bob can compute M himself/.test(src), "the module states when the bilateral check protects nothing");
  const carolClass = src.slice(src.indexOf("class KeyHolderDecryptor"), src.indexOf("/**\n * One check in Carol's custody"));
  check(carolClass.length > 200 && !/Encryptor|Evaluator|evaluate\(/.test(carolClass), "Carol's class can decrypt and nothing else");
  check((src.match(/\.decryptMasked\(/g) ?? []).length === 1 && /carol\.decryptMasked\(opened\)/.test(src), "and the one record handed to it is what the pool sealed to her");
  const bobPart = src.slice(src.indexOf("// Bob, with Carol's public key only"), src.indexOf("const keys = checkKeys(o.sessionKey);\n    const signature"));
  check(bobPart.length > 100 && !/Decryptor|secretKey|SecretKey/.test(bobPart), "Bob's encryption uses her public key and nothing that could decrypt");
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
if (failed) { console.error(`${failed} check(s) failed`); process.exit(1); }
console.log("private lending check: all checks passed");
