// The satellite page's residual card draws what the run reports, and credits ESA-ADB (Round 86).
//
// WHY THIS EXISTS (the owner's choice of 2026-11-23: the forecaster's residual against time). The card turns the
// run's per-row residuals into 8,640 canvas points, the flagged rows into crosses and the rows encryption changed into
// diamonds, and states the largest and the median residual. Each step can go wrong quietly: a flagged row drawn twice,
// a changed row drawn as an ordinary one, a residual of exactly 0 lost off a log axis, hours computed from the wrong
// step, or the figures in the caption not the ones the run logs. This checks `residualSummary` and `residualSeries`
// on a stated report, the rendered caption and credit, and that the pipeline fills the report from the evaluation's
// own arrays. tests/test_r86_satellite_student_ckks_in_node.py holds those arrays to the run's log figures.
//
// Run: node web/scripts/check_sat_residual_plot.mjs   (npm run check:sat-residual-plot; tests/test_r84_node_checks_run.py)

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const esbuild = await import(pathToFileURL(join(WEB, "node_modules/esbuild/lib/main.js")).href);
const dir = mkdtempSync(join(tmpdir(), "qsh-sat-residual-"));
const out = join(dir, "bundle.mjs");
const src = (p) => JSON.stringify(join(WEB, p));
await esbuild.build({
  stdin: {
    contents: [
      `export { default as SatForecastResidual, residualSummary, residualSeries } from ${src("src/components/SatForecastResidual")};`,
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

// A stated report: ten rows two minutes apart, rows 3 and 7 flagged in the clear, row 7's flag changed, row 5 exactly 0.
const report = {
  key: "satellite_comms.student", rows: 10, ciphertexts: 1, fheMs: 2100, maxAbsError: 9e-9, flagsChanged: 1,
  residuals: [1e-10, 2e-10, 3e-10, 9e-9, 4e-10, 0, 5e-10, 6e-10, 7e-10, 8e-10], changedRows: [7], flaggedRows: [3, 7],
  startMs: Date.UTC(2004, 0, 1), stepMs: 120_000, ranIn: "a Web Worker",
};
const s = m.residualSummary(report);
check("largest", s.worst === 9e-9, `${s.worst}, not 9e-9`);
check("median", s.median === (4e-10 + 5e-10) / 2, `${s.median}, not the mean of the 5th and 6th of the sorted ten`);
check("counts", s.changed === 1 && s.flagged === 2, JSON.stringify(s));
const series = m.residualSeries(report, 1e-11);
const named = Object.fromEntries(series.map((x) => [x.name, x.points]));
check("each row once", series.reduce((a, x) => a + x.points.length, 0) === 10 + 1,
      "every row is a circle or a cross, and the changed row also a diamond");
check("no flagged row among the circles", named["a forecast (circle)"].length === 8, `${named["a forecast (circle)"].length} circles`);
check("crosses are the flagged rows", JSON.stringify(named["flagged in the clear (cross)"].map((p) => p.x)) === JSON.stringify([3, 7].map((t) => (t * 120_000) / 3_600_000)),
      JSON.stringify(named["flagged in the clear (cross)"]));
check("the changed row is a diamond", (named["flag changed by encryption (diamond)"] ?? []).length === 1, "no diamond for row 7");
check("a zero residual sits at the floor", named["a forecast (circle)"].some((p) => p.y === 1e-11), "row 5's 0 vanished off the log axis");
check("hours from the start", named["a forecast (circle)"].at(-1).x === (9 * 120_000) / 3_600_000, `${named["a forecast (circle)"].at(-1).x}`);
const none = m.residualSeries({ ...report, changedRows: [], flagsChanged: 0 }, 1e-11);
check("no diamonds when nothing changed", none.length === 2, `${none.length} series`);

const html = m.renderToStaticMarkup(m.createElement(m.SatForecastResidual, { report }))
  .replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");
check("the caption states the run's figures", html.includes("The largest is 9.00e-9 and the median 4.50e-10"), html.slice(0, 700));
check("the caption counts the flags", html.includes("encryption changed 1 of them, the amber diamonds"), html.slice(0, 900));
check("ESA-ADB's credit", html.includes("Kotowski") && html.includes("CC BY 3.0 IGO") && html.includes("no telemetry value is drawn"), html.slice(-700));

// The pipeline fills the report from the evaluation's own arrays, and the page draws it on the satellite page.
const pipeline = readFileSync(join(WEB, "src", "lib", "clientCompute.ts"), "utf8");
check("residuals from the evaluation", /residuals: c\.residuals, changedRows: c\.changedRows, flaggedRows: flagged/.test(pipeline), "clientCompute.ts no longer copies them");
check("the stretch's clock", /startMs: sd\.stretch\.startMs, stepMs: sd\.stretch\.stepMs/.test(pipeline), "clientCompute.ts no longer copies the clock");
const page = readFileSync(join(WEB, "src", "pages", "UseCasePage.tsx"), "utf8");
check("the satellite page draws it", page.includes('{usecase === "satellite_comms" && artifacts?.forecast_residual ? <SatForecastResidual report={artifacts.forecast_residual as ForecastResidualReport} /> : null}'),
      "UseCasePage.tsx no longer draws the card");

if (failures.length) {
  console.error("check_sat_residual_plot: FAIL\n  " + failures.join("\n  "));
  process.exit(1);
}
console.log("check_sat_residual_plot: OK (10-row report, caption, credit, wiring)");
