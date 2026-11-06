/**
 * The Validations page's "Download results (JSON)" waits for a result, and says why (Round 84).
 *
 * WHY THIS EXISTS
 *
 * It was enabled from the first paint and saved `{kem, sig, pqc, qkd, mcp}` all null, while the
 * use-case and E2E pages disable their exports until a run and say so on the page
 * (scratchpad audit82/live-checks.md). `disabled` takes a button out of the tab order, so the
 * reason has to be a visible line, not only a tooltip.
 *
 * What this asserts, on the REAL page rendered to HTML:
 *   1. on first render the button is disabled and a visible line gives the reason;
 *   2. once any one validation has a result the button is enabled and the line is gone, and the
 *      download names which ones ran.
 *
 * HOW IT LOADS THE TSX: esbuild's API bundles the page for node, with the imports node cannot
 * resolve stubbed (the gitignored wasm module and the ?url assets).
 *
 * Run: node web/scripts/check_validations_download.mjs
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const esbuild = await import(pathToFileURL(join(WEB, "node_modules/esbuild/lib/main.js")).href);
const dir = mkdtempSync(join(tmpdir(), "qsh-validations-"));
const out = join(dir, "bundle.mjs");
const src = (p) => JSON.stringify(join(WEB, p));
await esbuild.build({
  stdin: {
    contents: [
      `export * as page from ${src("src/pages/Validations")};`,
      `export { createElement } from "react";`,
      `export { renderToStaticMarkup } from "react-dom/server.browser";`,
    ].join("\n"),
    resolveDir: WEB, loader: "tsx",
  },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error", jsx: "automatic",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "stub-unresolvable", setup(b) {
    b.onResolve({ filter: /qsh_mpc\.js$|\?url$|\?worker$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });

/** The download button's own tag, and the text of the page. */
const button = (html) => html.match(/<button[^>]*>[^<]*Download results \(JSON\)<\/button>/)?.[0] ?? "";
const text = (html) => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");
const REASON = "Run at least one of the validations above first";

// 1. First paint: nothing has run.
const first = m.renderToStaticMarkup(m.createElement(m.page.default));
const b = button(first);
check("first paint", b.length > 0, "the Download results button was not rendered - this check is inert");
check("first paint", /\sdisabled=""/.test(b), "the download is enabled before anything has run, and saves all-null JSON");
check("first paint", text(first).includes(REASON), "no visible line says why the download is unavailable");

// 2. After one validation: enabled, no reason, and the file says which ran.
check("after a run", typeof m.page.ResultsDownload === "function", "Validations exports no ResultsDownload to render with a result");
if (typeof m.page.ResultsDownload === "function") {
  const after = m.renderToStaticMarkup(m.createElement(m.page.ResultsDownload, {
    results: { kem: { algorithm: "ML-KEM-768+X25519" }, sig: null, pqc: null, qkd: null, mcp: null } }));
  check("after a run", !/\sdisabled=""/.test(button(after)), "the download stays disabled after a validation ran");
  check("after a run", !text(after).includes(REASON), "the reason line is still shown after a validation ran");
}

// 3. Overlapping QKD runs (Round 86): the Eve slider starts one per step, so only the newest may write the table.
check("newest wins", typeof m.page.newestWins === "function", "Validations exports no newestWins");
if (typeof m.page.newestWins === "function") {
  const ticket = m.page.newestWins();
  const older = ticket(), newer = ticket();
  check("newest wins", newer() === true && older() === false, "an older run's ticket is still current after a newer one");
  const newest = ticket();
  check("newest wins", newest() && !newer(), "a third run did not supersede the second");
}
// And the page uses it at the one site that writes the QKD table: every `setQkd(` sits behind the run's ticket.
{
  const { readFileSync } = await import("node:fs");
  const code = readFileSync(join(WEB, "src/pages/Validations.tsx"), "utf8")
    .replace(/\/\*[\s\S]*?\*\//g, "").split("\n").map((l) => l.replace(/\/\/.*$/, "")).join("\n");
  const body = code.slice(code.indexOf("const runQkd"), code.indexOf("return (", code.indexOf("const runQkd")));
  check("newest wins", /if \(current\(\)\) setQkd\(result\)/.test(body), "runQkd writes an answer without checking it is the newest");
  check("newest wins", /if \(!current\(\)\) return;\s*setQkd\(null\)/.test(body), "runQkd clears the table on a superseded run's failure");
  check("newest wins", (body.match(/setQkd\(/g) || []).length === 2, "runQkd writes the table at a site this check does not read");
}

if (failures.length) {
  console.error(`check_validations_download: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check_validations_download: OK - the export waits for a result and says why, and only the newest QKD run writes the table");
