/**
 * The model cards say what they measured, name their table headers, and say which figure a run
 * reproduces (Round 84).
 *
 * WHY THIS EXISTS
 *
 *   - The finance student's card printed "0.6323 published baseline, NOT measured here" beside a
 *     measured "student minority_class_f1 0.0614": `baselineOrphaned` looked for a key spelled
 *     `minority_class_f1`, and a distillation record stores `student_metric` with `metric_name`
 *     saying which metric it is (scratchpad audit82/factcheck-finance-genomics.md, B1).
 *   - The metric mini-tables had no header cells, so Lighthouse's `td-has-header` failed on every
 *     use-case page (checklist UX38).
 *   - The OPS-SAT card printed the scikit-learn model's accuracy, 0.9660, while the run reproduced the
 *     ONNX graph's 0.9509 on the same 529 rows, and nothing on the page said which was which
 *     (factcheck-satellite.md, A3). Round 86 traced that gap to the sample's 6-decimal rounding; the
 *     sample now reproduces the card.
 *
 * What this asserts, on the REAL components rendered to HTML with the shipped registry:
 *   1. the finance student's card carries the baseline on its measured row and no "NOT measured
 *      here", while genomics, which measures no AUROC, still says so;
 *   2. every mini-table on every use case's model panel has column headers and a row header (or a
 *      single spanning note cell) in every body row;
 *   3. the verification card says nothing where the two figures agree, which OPS-SAT's own published
 *      sample now does. Where they differ on the same rows (the 0.9509 OPS-SAT's sample read until
 *      Round 86, constructed here) it names both figures and the rows between them. Where the card
 *      counted other rows, it names the rows as the reason.
 *
 * HOW IT LOADS THE TSX: esbuild's API bundles the components for node, with the imports node cannot
 * resolve stubbed (the gitignored wasm module and the ?url assets: node-seal's wasm, gif.js's worker).
 *
 * Run: node web/scripts/check_model_cards.mjs
 */
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = resolve(WEB, "..");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const esbuild = await import(pathToFileURL(join(WEB, "node_modules/esbuild/lib/main.js")).href);
const dir = mkdtempSync(join(tmpdir(), "qsh-model-cards-"));
const out = join(dir, "bundle.mjs");
const src = (p) => JSON.stringify(join(WEB, p));
await esbuild.build({
  stdin: {
    contents: [
      `export { ModelPanel } from ${src("src/components/ModelPanel")};`,
      `export * as page from ${src("src/pages/UseCasePage")};`,
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

const text = (html) => html.replace(/<[^>]+>/g, " ").replace(/&#x27;|&apos;/g, "'").replace(/&quot;/g, '"').replace(/\s+/g, " ");
const panel = (usecase) => m.renderToStaticMarkup(m.createElement(m.ModelPanel, { usecase }));
/** One model's block of a rendered panel, by its label. */
const block = (html, label) => {
  const at = html.indexOf(label);
  if (at < 0) return "";
  const next = html.indexOf('class="model-block"', at);
  return html.slice(at, next < 0 ? undefined : next);
};

// 1. The measured row carries the baseline; only an unmeasured metric says so.
{
  const spec = m.modelsFor("finance_aml").find((s) => s.key === "finance_aml.student");
  const html = panel("finance_aml");
  const b = text(block(html, spec.label));
  check("baseline", b.length > 0, "the finance student's block was not rendered - this check is inert");
  check("baseline", !b.includes("NOT measured here"),
    "the finance student's card says its baseline is NOT measured here, beside the row that measures it");
  check("baseline", new RegExp(`student minority_class_f1 [0-9.]+ baseline ${spec.metric.baseline}\\b`).test(b),
    `the student's minority_class_f1 row does not carry the baseline ${spec.metric.baseline}`);
  const g = m.modelsFor("genomics").find((s) => s.key === "genomics.prs");
  check("baseline", text(block(panel("genomics"), g.label)).includes("NOT measured here"),
    "genomics measures no AUROC, so its card must still say the baseline is not measured here");
}

// 2. Headers on every mini-table of every use case's panel.
for (const usecase of ["finance_aml", "medical_fl", "genomics", "telecom_ids", "satellite_comms"]) {
  const html = panel(usecase);
  const tables = html.match(/<table class="mini-table">[\s\S]*?<\/table>/g) ?? [];
  check("headers", tables.length > 0, `${usecase}: no mini-table rendered - this check is inert`);
  tables.forEach((t, i) => {
    check("headers", /<thead>[\s\S]*<th scope="col"/.test(t), `${usecase} mini-table ${i + 1} has no column headers`);
    const body = t.slice(t.indexOf("<tbody>"));
    for (const row of body.match(/<tr[\s\S]*?<\/tr>/g) ?? []) {
      const ok = /<th scope="row"/.test(row) || /^<tr[^>]*><td colSpan="\d+"|^<tr[^>]*><td colspan="\d+"/.test(row);
      if (!ok) { check("headers", false, `${usecase} mini-table ${i + 1} has a row with no header: ${text(row).slice(0, 80)}`); break; }
    }
  });
}

// 3. Which figure the run reproduces, from OPS-SAT's own published sample.
const page = m.page;
check("reproduces", typeof page.ReproducedFigure === "function", "UseCasePage exports no ReproducedFigure - the card says nothing");
if (typeof page.ReproducedFigure === "function") {
  const sample = JSON.parse(readFileSync(join(ROOT, "models/satellite_opssat/eval_sample.json"), "utf8"));
  const n = sample.labels.length, expected = sample.expected.accuracy;
  const card = m.modelsFor("satellite_comms").find((s) => s.key === "satellite_comms.opssat").achieved.accuracy;
  const say = (usecase, v) => text(m.renderToStaticMarkup(m.createElement(page.ReproducedFigure, { usecase, verification: v })));
  const opssat = say("satellite_comms", { modelKey: "satellite_comms.opssat", metricName: "accuracy", n, score: expected, expected });
  check("reproduces", Math.abs(expected - card) < 1e-12 && opssat.trim() === "",
    `OPS-SAT's sample should reproduce its card (${expected} against ${card}) and the page say nothing: "${opssat}"`);
  // The disagreement the statement exists for: the 0.9509 OPS-SAT's 6-decimal sample read until Round 86.
  const old = 0.9508506616257089;
  const differ = say("satellite_comms", { modelKey: "satellite_comms.opssat", metricName: "accuracy", n, score: old, expected: old });
  const net = Math.round(Math.abs(card - old) * n);
  check("reproduces", differ.includes(old.toFixed(4)) && differ.includes(card.toFixed(4)),
    `the statement does not show both ${old.toFixed(4)} (reproduced) and ${card.toFixed(4)} (card): "${differ}"`);
  check("reproduces", net === 8 && differ.includes(`a net ${net} of them`) && differ.includes(`same ${n} rows`),
    `the statement does not give the reason: the same ${n} rows, a net ${net} decided differently`);
  // Where the card and the graph agree there is nothing to say; where the card counted other rows,
  // the rows are the reason, not the model.
  const r50 = m.modelsFor("medical_fl").find((s) => s.key === "medical_fl.resnet50").achieved.accuracy;
  check("reproduces", say("medical_fl", { modelKey: "medical_fl.resnet50", metricName: "accuracy", n: 156, expected: r50 }).trim() === "",
    "a statement was rendered where the card and the reproduced figure agree");
  const gbm = say("finance_aml", { modelKey: "finance_aml.gbm", metricName: "accuracy", n: 8192, expected: 0.9818115234375 });
  check("reproduces", gbm.includes("not scored on the same rows"),
    "for a card scored over other rows the statement should name the rows as a reason");
}

if (failures.length) {
  console.error(`check_model_cards: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check_model_cards: OK - measured baselines, named headers, and the reproduced figure named");
