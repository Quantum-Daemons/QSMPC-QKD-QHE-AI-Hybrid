/**
 * The relay's "pass pairs for Liao's 100 kB" counts at the page's own finite key (Round 84).
 *
 * WHY THIS EXISTS
 *
 * The satellite Network tab printed "Pass pairs for Liao's 100 kB: 2" from the ASYMPTOTIC key
 * (satNetwork.ts `trustedRelay`), while the same tab gives each pass's composable finite key, 241.4
 * kbit at the defaults, which needs 4 (scratchpad audit82/factcheck-satellite.md, A5). A label that
 * admitted the asymptotic key was a caveat on the wrong number: the figure a reader quotes should be
 * the one the page stands behind.
 *
 * What this asserts:
 *   1. recomputed here from satNetwork.ts and finiteKey.ts - each pass a block of its own, under the
 *      run's bound and eps_sec - the count at the defaults is 4, where the asymptotic key gives 2;
 *   2. the rendered Network tab prints that count on the pass-pairs row, and the asymptotic figure
 *      only as the comparison beside it.
 *
 * HOW IT LOADS THE TS: esbuild's API bundles the panel and the model modules for node.
 *
 * Run: node web/scripts/check_relay_pass_pairs.mjs
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const esbuild = await import(pathToFileURL(join(WEB, "node_modules/esbuild/lib/main.js")).href);
const dir = mkdtempSync(join(tmpdir(), "qsh-relay-"));
const out = join(dir, "bundle.mjs");
const src = (p) => JSON.stringify(join(WEB, p));
await esbuild.build({
  stdin: {
    contents: [
      `export * as net from ${src("src/lib/satNetwork")};`,
      `export * as fk from ${src("src/lib/finiteKey")};`,
      `export { satChannel } from ${src("src/lib/satLink")};`,
      `export { qkdCfg } from ${src("src/lib/config")};`,
      `export { default as SatNetworkPanel } from ${src("src/components/SatNetworkPanel")};`,
      `export { createElement } from "react";`,
      `export { renderToStaticMarkup } from "react-dom/server.browser";`,
    ].join("\n"),
    resolveDir: WEB, loader: "tsx",
  },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error", jsx: "automatic",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
});
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });

// 1. The count, from the model modules, at the defaults the tab opens with.
const params = {};
const link = m.satChannel({ ...params, sat_eve: 0 });
const el = Math.round(m.net.liaoPass(link).maxElevationDeg);
const station = (id) => m.net.LIAO_2018_STATIONS.find((s) => s.id === id);
const relay = m.net.trustedRelay(link, station("xinglong"), station("graz"), el, el);
const cfg = m.qkdCfg();
const fkSrc = m.fk.finiteKeySourceOf(link, Number(cfg.free_space_eps_sec), undefined, String(cfg.finite_key_deviation));
const finite = (pass) => m.fk.keyLength(...m.fk.blockCounts(pass.segments, fkSrc), fkSrc).keyBits;
const perPair = Math.min(finite(relay.a), finite(relay.b));
const pairs = Math.ceil(800_000 / perPair);
check("model", pairs === 4, `at the defaults the finite key per pass pair is ${(perPair / 1000).toFixed(1)} kbit, ${pairs} pairs - the fact-check found 241.4 kbit and 4`);
check("model", relay.passPairsFor100kB === 2, `the asymptotic count is ${relay.passPairsFor100kB}, expected 2 - the comparison moved`);

// 2. What the tab prints.
const html = m.renderToStaticMarkup(m.createElement(m.SatNetworkPanel, { params }))
  .replace(/<[^>]+>/g, " ").replace(/&#x27;|&apos;/g, "'").replace(/&quot;/g, '"').replace(/\s+/g, " ");
const row = html.match(/Pass pairs for Liao's 100 kB(.{0,400})/)?.[1] ?? "";
check("panel", row.length > 0, "the pass-pairs row was not rendered - this check is inert");
const shown = Number(row.trim().split(" ")[0]);
check("panel", shown === pairs, `the row shows ${row.trim().split(" ")[0]} pass pairs; the page's own finite key needs ${pairs}`);
check("panel", new RegExp(`asymptotic key would need ${relay.passPairsFor100kB}\\b`).test(row),
  "the asymptotic count is not named as the comparison beside it");

if (failures.length) {
  console.error(`check_relay_pass_pairs: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(`check_relay_pass_pairs: OK - ${pairs} pass pairs at ${(perPair / 1000).toFixed(1)} kbit per pair (asymptotic: ${relay.passPairsFor100kB})`);
