// One party of the network benchmark: calibrates the link, then runs the demo's secure argmax under
// either engine against the other tab. Inputs come from a seed both tabs share, so both hold the two
// halves of the same additive sharing that web/src/lib/clientMPC.ts builds.
import initPolytune, { secure_argmax_party } from "../../../polytune-argmax/pkg/polytune_argmax.js";
import { Protocol } from "mpc-framework";
import { EmpWasmEngine } from "emp-wasm-engine";
import * as summon from "summon-ts";
import { connect } from "./transport.js";

const party = Number(new URLSearchParams(location.search).get("party") ?? 0);
document.getElementById("party").textContent = String(party);
const out = document.getElementById("out");

// Party 0 sends through the A->B relay and receives through B->A; party 1 the reverse.
const RELAYS = party === 0
  ? { out: "ws://127.0.0.1:8765", in: "ws://127.0.0.1:8766" }
  : { out: "ws://127.0.0.1:8766", in: "ws://127.0.0.1:8765" };

// clientMPC.ts: fixed-point scale and offset, clamp, share mask, boolify width.
const SCALE = 64, OFFSET = 8, QMAX = 2 * OFFSET * SCALE, BIG = 1 << 16, WIDTH = 24;

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shares(n, seed) {
  const u = rng(seed);
  const q = Array.from({ length: n }, () => Math.min(QMAX, Math.max(0, Math.round((u() * 6 - 3 + OFFSET) * SCALE))));
  const a = q.map(() => Math.floor(u() * BIG));
  const b = q.map((v, k) => v - a[k] + BIG);
  let want = 0;
  for (let k = 1; k < n; k++) if (q[k] > q[want]) want = k;
  return { a, b, want };
}

// The summon source clientMPC.ts compiles, unchanged.
function argmaxSrc(n) {
  const aIn = Array.from({ length: n }, (_, k) => `  const a${k} = io.input('alice', 'a${k}', summon.number());`).join("\n");
  const bIn = Array.from({ length: n }, (_, k) => `  const b${k} = io.input('bob', 'b${k}', summon.number());`).join("\n");
  const q = Array.from({ length: n }, (_, k) => `  const q${k} = a${k} + b${k} - ${BIG};`).join("\n");
  let cmp = "  let arg = 0;\n  let best = q0;\n";
  for (let k = 1; k < n; k++) cmp += `  if (q${k} > best) { best = q${k}; arg = ${k}; }\n`;
  return `export default (io: Summon.IO) => {\n${aIn}\n${bIn}\n${q}\n${cmp}  io.outputPublic('arg', arg);\n}`;
}

const summonReady = summon.init();
const polytuneReady = initPolytune();
const circuits = new Map();
async function empCircuit(n) {
  await summonReady;
  if (!circuits.has(n)) {
    circuits.set(n, summon.compile({ path: "circuit/main.ts", boolifyWidth: WIDTH,
      files: { "circuit/main.ts": argmaxSrc(n) } }).circuit);
  }
  return circuits.get(n);
}

async function runEmp(n, seed, room) {
  const circuit = await empCircuit(n);
  const { a, b, want } = shares(n, seed);
  const me = party === 0 ? "alice" : "bob";
  const other = party === 0 ? "bob" : "alice";
  const inputs = Object.fromEntries((party === 0 ? a : b).map((v, k) => [`${party === 0 ? "a" : "b"}${k}`, v]));
  const net = await connect(RELAYS, room);
  const t0 = performance.now();
  const session = new Protocol(circuit, new EmpWasmEngine()).join(me, inputs, (_to, msg) => net.send(msg));
  let live = true;
  (async () => { while (live) { const m = await net.recv(); if (live) session.handleMessage(other, m); } })();
  const o = await session.output();
  const ms = performance.now() - t0;
  live = false;
  const st = net.stats();
  net.close();
  return { engine: "emp-wasm", n, party, ms, arg: o.arg, want, correct: o.arg === want, ...st };
}

async function runPolytune(n, seed, room) {
  await polytuneReady;
  const { a, b, want } = shares(n, seed);
  const net = await connect(RELAYS, room);
  globalThis.qshNet = { send: (_to, data) => net.send(data), recv: (_from) => net.recv() };
  const t0 = performance.now();
  const r = await secure_argmax_party(party, Int32Array.from(party === 0 ? a : b));
  const ms = performance.now() - t0;
  const st = net.stats();
  net.close();
  return { engine: "polytune", n, party, ms, arg: r.arg, want, correct: r.arg === want, andOps: r.andOps, ...st };
}

// RTT by echo and goodput by a bulk one-way transfer, through the same relays the engines use, so the
// link's delay and bandwidth are measured rather than taken from the netem settings.
async function calibrate(room, bulkBytes) {
  const net = await connect(RELAYS, room);
  const ECHOES = 15;
  if (party === 0) {
    const rtts = [];
    for (let i = 0; i < ECHOES; i++) {
      const t = performance.now();
      net.send(crypto.getRandomValues(new Uint8Array(64)));
      await net.recv();
      rtts.push(performance.now() - t);
    }
    rtts.sort((x, y) => x - y);
    const t1 = performance.now();
    for (let s = 0; s < bulkBytes; s += 65536) {
      net.send(crypto.getRandomValues(new Uint8Array(Math.min(65536, bulkBytes - s))));
    }
    await net.recv();
    const bulkMs = performance.now() - t1;
    net.close();
    const rttMs = rtts[Math.floor(ECHOES / 2)];
    return { rttMs, rttMinMs: rtts[0], rttMaxMs: rtts[ECHOES - 1], bulkBytes, bulkMs,
      goodputMbps: (bulkBytes * 8) / ((bulkMs - rttMs / 2) * 1e3) };
  }
  for (let i = 0; i < ECHOES; i++) net.send(await net.recv());
  let got = 0;
  while (got < bulkBytes) got += (await net.recv()).length;
  net.send(new Uint8Array([1]));
  net.close();
  return { echoed: ECHOES, received: got };
}

// Both tabs call this with the same cfg; each step meets the other tab at the relay in its own room.
window.runConfig = async (cfg) => {
  const result = { cfg, party, calib: await calibrate(`${cfg.tag}-cal`, cfg.bulkBytes), runs: [] };
  // Interleaved (emp, polytune, emp, ...) rather than one engine's runs back to back, so that warm-up
  // or drift over a configuration cannot systematically favour whichever engine runs second.
  for (let i = 0; i < cfg.runs; i++) {
    for (const engine of cfg.engines) {
      const room = `${cfg.tag}-${engine}-${i}`;
      const run = engine === "polytune" ? runPolytune : runEmp;
      result.runs.push(await run(cfg.n, cfg.seed + i, room));
      out.textContent = JSON.stringify(result.runs.at(-1));
    }
  }
  window.__last = result;
  return result;
};

// The whole grid, in the same order in both tabs. Party 0 sets each link through the control server
// before joining the step's calibration room; party 1 simply joins, and the relay's "ready" holds it
// until party 0 arrives - by which time the new setting is in place.
window.runGrid = async (grid) => {
  window.__grid = { done: 0, total: grid.length, results: [], error: null };
  try {
    for (const cfg of grid) {
      let applied = null;
      if (party === 0) {
        const q = new URLSearchParams({ delay_ms: cfg.link.delayMs, rate_mbit: cfg.link.rateMbit, loss_pct: cfg.link.lossPct });
        applied = await (await fetch(`http://127.0.0.1:8779/shape?${q}`)).json();
      }
      const r = await window.runConfig(cfg);
      window.__grid.results.push({ ...r, applied });
      window.__grid.done += 1;
    }
  } catch (e) {
    window.__grid.error = String(e?.stack || e);
  }
  return window.__grid;
};
out.textContent = "ready";
