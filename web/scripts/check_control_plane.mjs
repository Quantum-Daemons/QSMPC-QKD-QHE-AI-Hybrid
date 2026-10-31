/**
 * The control-plane panel prints the run's own channel time, names its allocation, and says
 * "not yet run" before a run (Round 84).
 *
 * WHY THIS EXISTS
 *
 * Three defects on one panel, each a number that meant something other than its label:
 *   - `channel_time_s` was the sampled run's pulses over the clock (0.02 s on fibre, 0.2 s on the
 *     satellite) while the panel's "channel time" was the key block's (5.3 s, 408.5 s): one name,
 *     two quantities, one page (scratchpad audit82/live-checks.md).
 *   - The key budget column was the run's secure bits less 25%, from a bare `* 0.25` in the panel
 *     (104,400 against 139,200 on the satellite) - factcheck-satellite.md B3.
 *   - Before a run the panel printed "0 MHz": UseCasePage passed `source_rate_hz ?? 0` (B5).
 *
 * What this asserts, on the REAL pipeline stages and the REAL component rendered to HTML:
 *   1. on fibre and satellite BB84 runs `channel_time_s` equals `qkd_block_s`, and the sampled
 *      run's time is `run_channel_time_s` = qkd_pulses / source_rate_hz; on E91, which has no
 *      block, the two are one time;
 *   2. the panel, given what UseCasePage passes, prints that same channel time;
 *   3. the allocation is a named value the panel prints with its arithmetic, and no literal 0.25
 *      remains in the panel;
 *   4. an idle panel says "not yet run" for the rate and prints no "0 MHz".
 *
 * HOW IT LOADS THE TS: esbuild's API bundles the modules and the component, with the two imports
 * node cannot resolve stubbed (the gitignored wasm module and node-seal's ?url asset).
 *
 * Run: node web/scripts/check_control_plane.mjs
 */
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const esbuild = await import(pathToFileURL(join(WEB, "node_modules/esbuild/lib/main.js")).href);
const dir = mkdtempSync(join(tmpdir(), "qsh-control-plane-"));
const out = join(dir, "bundle.mjs");
const src = (p) => JSON.stringify(join(WEB, p));
await esbuild.build({
  stdin: {
    contents: [
      `export { runClientPipeline } from ${src("src/lib/clientCompute")};`,
      `export * as sdn from ${src("src/lib/clientSdnController")};`,
      `export { default as Panel } from ${src("src/components/SdnControlPlanePanel")};`,
      `export { createElement } from "react";`,
      `export { renderToStaticMarkup } from "react-dom/server.browser";`,
    ].join("\n"),
    resolveDir: WEB, loader: "tsx",
  },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error", jsx: "automatic",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "stub-unresolvable", setup(b) {
    b.onResolve({ filter: /qsh_mpc\.js$|\.wasm\?url$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
globalThis.fetch = async (url) => {
  const text = readFileSync(join(WEB, "public", String(url).replace(/^\//, "")), "utf8");
  return { ok: true, status: 200, headers: { get: () => "application/json" }, json: async () => JSON.parse(text), text: async () => text };
};
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });

/** The metrics a run's QKD stage emits; the run is stopped as soon as it moves past that stage. */
async function qkdMetrics(usecase, config) {
  const got = {};
  await new Promise((done) => {
    let ctl = null, finished = false;
    const finish = () => { if (!finished) { finished = true; clearTimeout(timer); ctl?.stop(); done(); } };
    const timer = setTimeout(finish, 90_000);
    ctl = m.runClientPipeline(usecase, { seed: 0, ...config }, {
      onEvent: (e) => {
        if (e.type === "metric") got[e.data.name] = e.data.value;
        if (e.stage && !["data", "pqc_handshake", "qkd_key_agreement", "satellite_qkd"].includes(e.stage)) finish();
      },
      onEnd: finish,
    });
  });
  return got;
}

const text = (html) => html.replace(/<[^>]+>/g, " ").replace(/&#x27;|&apos;/g, "'").replace(/\s+/g, " ");
/** The panel as UseCasePage renders it from a run's metrics (UseCasePage.tsx, the SdnControlPlanePanel element). */
const render = (usecase, metrics) => text(m.renderToStaticMarkup(m.createElement(m.Panel, {
  usecase, distanceKm: usecase === "satellite_comms" ? 800 : 50,
  maxSpanKm: usecase === "satellite_comms" ? Infinity : undefined,
  keyBudgetBits: Number(metrics.secure_key_bits ?? 0),
  secretKeyRatePerPulse: Number(metrics.skr_per_pulse ?? 0),
  sourceRateHz: metrics.source_rate_hz != null ? Number(metrics.source_rate_hz) : null,
  pulses: Number(metrics.qkd_block_pulses ?? metrics.qkd_pulses ?? 0),
  channelSeconds: metrics.channel_time_s != null ? Number(metrics.channel_time_s) : null,
  decoyStates: metrics.qkd_decoy_states, qkdEnabled: true,
})));
const fmtSeconds = (s) => (s >= 1 ? `${s.toFixed(1)} s` : `${(s * 1e3).toFixed(3)} ms`);

// 1 and 2. The two QKD stages, BB84 with a block and E91 without.
for (const [usecase, config, hasBlock] of [["telecom_ids", {}, true], ["satellite_comms", {}, true],
                                             ["telecom_ids", { qkd_protocol: "e91" }, false]]) {
  const label = `${usecase}${config.qkd_protocol ? ` ${config.qkd_protocol}` : ""}`;
  const r = await qkdMetrics(usecase, config);
  const runTime = r.qkd_pulses / r.source_rate_hz;
  check(label, typeof r.channel_time_s === "number", "the run emitted no channel_time_s");
  check(label, r.run_channel_time_s === runTime,
    `run_channel_time_s ${r.run_channel_time_s} is not qkd_pulses / source_rate_hz = ${runTime}`);
  if (hasBlock) {
    check(label, r.channel_time_s === r.qkd_block_s,
      `channel_time_s ${r.channel_time_s} s is not the key block's ${r.qkd_block_s} s - the name means two things again`);
  } else {
    check(label, r.qkd_block_s === undefined && r.channel_time_s === r.run_channel_time_s,
      "E91 has no block, so its key's time and the run's must be one number");
  }
  if (typeof r.channel_time_s === "number") {
    const html = render(usecase, r);
    check(label, html.includes(`channel time of the key block ${fmtSeconds(r.channel_time_s)}`),
      `the panel does not print the run's channel_time_s (${fmtSeconds(r.channel_time_s)})`);
  }
  // 3. The allocation, named and printed with its arithmetic.
  if (usecase === "satellite_comms") {
    const frac = m.sdn.SAE_DEMO_ALLOCATION_FRACTION;
    check("allocation", frac === 0.25, `SAE_DEMO_ALLOCATION_FRACTION is ${frac}, not exported as the panel's 25%`);
    // The panel uses the project config's value and this constant only as its default (Round 84):
    // two copies of one number, so they are held equal here rather than trusted to stay so.
    const cfgFrac = JSON.parse(readFileSync(join(WEB, "src/lib/config.defaults.json"), "utf8"))
      .web?.sdn_sae_allocation_fraction;
    check("allocation", cfgFrac === frac,
      `config web.sdn_sae_allocation_fraction is ${cfgFrac}, the panel's default ${frac}`);
    if (typeof frac === "number" && r.secure_key_bits > 0) {
      const given = Math.round(r.secure_key_bits * frac);
      const html = render(usecase, r);
      const said = `${r.secure_key_bits.toLocaleString()} secure bits less ${given.toLocaleString()} allocated leaves ${(r.secure_key_bits - given).toLocaleString()} bit`;
      check("allocation", html.includes(`${Math.round(frac * 100)}%`) && html.includes(said),
        `the panel does not state the allocation and its arithmetic ("${said}")`);
    }
  }
}
const panelSrc = readFileSync(join(WEB, "src/components/SdnControlPlanePanel.tsx"), "utf8")
  .replace(/\/\*[\s\S]*?\*\//g, "").split("\n").map((l) => l.replace(/\/\/.*$/, "")).join("\n");
check("allocation", !/\*\s*0\.25\b/.test(panelSrc), "the panel still multiplies by a literal 0.25");

// 4. Idle: no run, so no rate - and no "0 MHz".
for (const usecase of ["telecom_ids", "satellite_comms"]) {
  // As UseCasePage passes an idle run now (no rate), and as it used to (`?? 0`): neither says 0 MHz.
  for (const [how, metrics] of [["no rate", {}], ["a rate of 0", { source_rate_hz: 0 }]]) {
    const idle = render(usecase, metrics);
    check("idle", idle.includes("x source repetition rate not yet run"), `${usecase}, ${how}: the idle rate row does not say "not yet run"`);
    check("idle", !/\b0 MHz/.test(idle), `${usecase}, ${how}: the idle panel prints 0 MHz`);
  }
}
const pageSrc = readFileSync(join(WEB, "src/pages/UseCasePage.tsx"), "utf8");
check("idle", !/source_rate_hz \?\? 0/.test(pageSrc), "UseCasePage still passes `source_rate_hz ?? 0`");

if (failures.length) {
  console.error(`check_control_plane: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check_control_plane: OK - one channel time, a named allocation, and an idle panel that says so");
process.exit(0);
