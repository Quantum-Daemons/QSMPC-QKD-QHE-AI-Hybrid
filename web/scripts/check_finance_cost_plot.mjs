// The finance page's accuracy-against-cost plot puts each model where its record and this run say (Round 86).
//
// WHY THIS EXISTS (the owner's choice of 2026-11-23: plot accuracy against cost). The plot reads three kinds of number
// from three places: each model's AUPRC from its record (through the page's config), the student's encrypted batch from
// the run's metrics, and the MLP's and the compact GNN's one-transfer runs under MPC from other metrics. Each can go
// wrong without anything failing: a record key renamed (the student keeps its AUPRC in its calibration record, not as
// `auprc`), a metric renamed in the pipeline, a batch total plotted as if it were one transfer, or a model that did not
// run drawn anyway. This checks the RESULT of `costPoints` and the rendered table on stated metrics, and that every
// metric the plot reads is one the pipeline writes.
//
// Run: node web/scripts/check_finance_cost_plot.mjs   (npm run check:finance-cost-plot; tests/test_r84_node_checks_run.py)

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPO = resolve(WEB, "..");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const esbuild = await import(pathToFileURL(join(WEB, "node_modules/esbuild/lib/main.js")).href);
const dir = mkdtempSync(join(tmpdir(), "qsh-finance-cost-"));
const out = join(dir, "bundle.mjs");
const src = (p) => JSON.stringify(join(WEB, p));
await esbuild.build({
  stdin: {
    contents: [
      `export { default as FinanceAccuracyCost, costPoints, recordedAuprc, PRICED, PLAINTEXT_ONLY } from ${src("src/components/FinanceAccuracyCost")};`,
      `export { modelsFor } from ${src("src/lib/config")};`,
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

// 1. Each model's AUPRC is its record's, read from the page's config, and all are on one test window.
const metricsJson = JSON.parse(readFileSync(join(REPO, "models", "metrics.json"), "utf8"));
const specs = m.modelsFor("finance_aml");
const auprcOf = (key) => m.recordedAuprc(specs.find((s) => s.key === key)?.achieved);
// F7b's square network (Round 86) is not a registry model: its AUPRC is its site file's record, which the run reports.
const square = JSON.parse(readFileSync(join(WEB, "public", "data", "finance", "square_student_weights.json"), "utf8"));
const recorded = {
  "finance_aml.mlp_large": metricsJson["finance_aml.mlp_large"].auprc,
  "finance_aml.gnn_compact": metricsJson["finance_aml.gnn_compact"].auprc,
  "finance_aml.gbm": metricsJson["finance_aml.gbm"].auprc,
};
for (const [key, want] of Object.entries(recorded)) {
  check(`${key} AUPRC`, typeof want === "number" && auprcOf(key) === want, `the plot reads ${auprcOf(key)}, the record says ${want}`);
}
const windows = new Set([square.figures.test_rows, metricsJson["finance_aml.mlp_large"].n_test,
  metricsJson["finance_aml.gnn_compact"].n_test, metricsJson["finance_aml.gbm"].n_test]);
check("one test window", windows.size === 1, `the four records score ${[...windows].join(", ")} transfers`);
check("the square network's AUPRC is on the test split", /scored on the held-out test split/.test(square.calibration.fitted_on),
  `its calibration record says ${square.calibration.fitted_on}`);

// 2. A finished run with all three: the student is its batch over its rows, the two others one transfer each.
const full = {
  encrypted_model: "finance_aml.square_network", encrypted_model_test_auprc: square.figures.test_auprc,
  encrypted_rows_scored: 8192, encrypted_batch_ms: 17200, encrypted_rows_per_ciphertext: 4096, encrypted_ciphertexts: 228,
  mpc_mlp_ran: 1, mpc_mlp_model: "finance_aml.mlp_large", mpc_mlp_total_ms: 5400,
  mpc_gnn_ran: 1, mpc_gnn_model: "finance_aml.gnn_compact", mpc_gnn_total_ms: 31000,
};
const got = m.costPoints(full, auprcOf);
const at = (key) => got.points.find((p) => p.key === key);
check("three points", got.points.length === 3 && got.missing.length === 0, JSON.stringify(got));
check("the square network is per transfer", at("finance_aml.square_network")?.ms === 17200 / 8192, `${at("finance_aml.square_network")?.ms}`);
check("the square network's AUPRC is its file's", at("finance_aml.square_network")?.auprc === square.figures.test_auprc,
  `${at("finance_aml.square_network")?.auprc}, the file says ${square.figures.test_auprc}`);
check("the MLP is one transfer", at("finance_aml.mlp_large")?.ms === 5400, `${at("finance_aml.mlp_large")?.ms}`);
check("the GNN is one transfer", at("finance_aml.gnn_compact")?.ms === 31000, `${at("finance_aml.gnn_compact")?.ms}`);
check("the square network says how it was packed", /8,192 held-out transfers in 228 ciphertexts of 4096/.test(at("finance_aml.square_network")?.how ?? ""), at("finance_aml.square_network")?.how);

// 3. What did not run has no point, and says why.
const partial = m.costPoints({ ...full, mpc_mlp_ran: 0, encrypted_model: "finance_aml.scorecard" }, auprcOf);
check("no MLP point when it did not run", !partial.points.some((p) => p.key === "finance_aml.mlp_large")
  && partial.missing.some((s) => /^MLP .*did not run under MPC/.test(s)), JSON.stringify(partial.missing));
check("no square-network point when another model was batched", !partial.points.some((p) => p.key === "finance_aml.square_network")
  && partial.missing.some((s) => /^square network .*did not run/.test(s)), JSON.stringify(partial.missing));
const html = m.renderToStaticMarkup(m.createElement(m.FinanceAccuracyCost, { metrics: { ...full, mpc_gnn_ran: 0 } }))
  .replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");
check("the caption names the missing model", /No point: compact GNN \(CKKS and garbled circuits\): it did not run under MPC\./.test(html), html.slice(0, 600));
check("the table prints the square network's per-transfer time", html.includes("2.100"), "2.100 ms not in the table");
check("the table prints the plaintext-only model", html.includes(`${recorded["finance_aml.gbm"].toFixed(4)} none no encrypted path on this page`), "the GBM row is missing");

// 4. Every metric the plot reads is one the pipeline writes, under that name.
const pipeline = readFileSync(join(WEB, "src", "lib", "clientCompute.ts"), "utf8");
for (const name of ["encrypted_model", "encrypted_model_test_auprc", "encrypted_rows_scored", "encrypted_batch_ms", "encrypted_rows_per_ciphertext",
                    "encrypted_ciphertexts", "mpc_mlp_ran", "mpc_mlp_model", "mpc_mlp_total_ms", "mpc_gnn_ran", "mpc_gnn_model", "mpc_gnn_total_ms"]) {
  check(`the pipeline writes ${name}`, pipeline.includes(`"${name}"`), "no metric of that name in clientCompute.ts");
}
const page = readFileSync(join(WEB, "src", "pages", "UseCasePage.tsx"), "utf8");
check("the finance page draws it after a finished run", page.includes('{usecase === "finance_aml" && state === "done" ? <FinanceAccuracyCost metrics={metrics} /> : null}'),
  "UseCasePage.tsx no longer draws FinanceAccuracyCost on finance once the run is done");
check("the priced models are the three", m.PRICED.map((p) => p.key).join() === "finance_aml.square_network,finance_aml.mlp_large,finance_aml.gnn_compact"
  && m.PLAINTEXT_ONLY.key === "finance_aml.gbm", JSON.stringify(m.PRICED));

if (failures.length) {
  console.error("check_finance_cost_plot: FAIL\n  " + failures.join("\n  "));
  process.exit(1);
}
console.log("check_finance_cost_plot: OK (3 records and the square network's file, 3 priced paths, a partial run)");
