// Every published student's parameter count reaches the two places that read it.
//
// WHY THIS EXISTS (Round 86, found by a reader). The distiller records a student's size as `student_params`; telecom's
// student, written by another path, records `n_parameters`. Both readers asked for the first key only. So the
// encrypted-cost row sized telecom by an invented medium MLP, nine times the student's work, and multiplied the quantum
// row's latency by that, while the other use cases priced their students at scale 1; and telecom's use-case card printed
// no student size. `studentParamCount` reads either key, and this checks the RESULT at both sites, not the helper alone.
//
// Run: node web/scripts/check_student_cost_rows.mjs   (npm run check:student-cost-rows; tests/test_r84_node_checks_run.py)

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const esbuild = await import(pathToFileURL(join(WEB, "node_modules/esbuild/lib/main.js")).href);
const dir = mkdtempSync(join(tmpdir(), "qsh-student-cost-"));
const out = join(dir, "bundle.mjs");
const src = (p) => JSON.stringify(join(WEB, p));
await esbuild.build({
  stdin: {
    contents: [
      `export { benchScale } from ${src("src/lib/clientCompute")};`,
      `export { modelsFor, studentModel, studentParamCount } from ${src("src/lib/config")};`,
      `export { default as UseCases } from ${src("src/pages/UseCases")};`,
      `export { MemoryRouter } from "react-router";`,
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

// The helper itself, both keys and neither: a reader that returned 0 for a missing count would price it as "no student".
check("reads student_params", m.studentParamCount({ achieved: { student_params: 33 } }) === 33, "33 expected");
check("reads n_parameters", m.studentParamCount({ achieved: { n_parameters: 64 } }) === 64, "64 expected");
check("no count is null", m.studentParamCount({ achieved: {} }) === null && m.studentParamCount(null) === null, "null expected");

const html = m.renderToStaticMarkup(m.createElement(m.MemoryRouter, null, m.createElement(m.UseCases)))
  .replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");
const sized = [];
for (const usecase of ["finance_aml", "medical_fl", "genomics", "telecom_ids", "satellite_comms"]) {
  const n = m.studentParamCount(m.studentModel(usecase));
  if (n == null) continue;
  sized.push(usecase);
  // The student branch prices the student at every model size, so the scale against the lightweight row is 1.
  const scale = m.benchScale(usecase, "medium", "linear");
  check(`${usecase} cost row`, scale === 1, `benchScale is ${scale}, not 1: the row is sized by an invented MLP, not its ${n}-parameter student`);
  check(`${usecase} card`, html.includes(`encrypted student ${n} params`), `the use-case card does not print its ${n}-parameter student`);
}
// Telecom is the case this exists for; if its student lost its count the loop above would skip it and pass.
check("telecom is checked", sized.includes("telecom_ids"), `telecom's student has no recorded count; checked: ${sized.join(", ")}`);

if (failures.length) {
  console.error("check_student_cost_rows: FAIL\n  " + failures.join("\n  "));
  process.exit(1);
}
console.log(`check_student_cost_rows: OK (${sized.length} students: ${sized.join(", ")})`);
