#!/usr/bin/env node
/**
 * The IBM tab's teacher-student view pairs the right numbers (Round 86, the owner's choice of 2026-11-16).
 *
 * Through the page's own module (`src/lib/teacherStudent.ts`, bundled by esbuild), on the committed files: the MLP's and
 * the compact GNN's weights, and IBM's Multi-PNA+EU evaluation sample (`models/finance_aml_gnn/eval_sample.json`). It
 * asserts that each student's clear forward reproduces the logits its trainer recorded, that the three files hold the
 * same 16 test-window rows with the same labels, and that the pairing counts the teacher's decisions by its threshold
 * and the students' by argmax. Then it seeds three faults the view must refuse rather than draw: a student weight moved,
 * a teacher label flipped, and a teacher with none of the students' rows.
 *
 * Run: npm run check:teacher-student
 */
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPO = resolve(WEB, "..");
const { build } = createRequire(join(WEB, "package.json"))("esbuild");
const tmp = mkdtempSync(join(tmpdir(), "qsh-teacher-student-"));
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };
const sealWasm = join(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
try {
  await build({
    stdin: { contents: `export * from ${JSON.stringify(join(WEB, "src/lib/teacherStudent"))};`, resolveDir: WEB, loader: "ts" },
    bundle: true, format: "esm", platform: "node", outfile: join(tmp, "b.mjs"), logLevel: "error", absWorkingDir: WEB,
    define: { "import.meta.env": '{"BASE_URL":"/"}' },
    // clientHybridCnn.ts reaches SEAL's wasm through a Vite-only `?url` import; nothing here runs CKKS.
    plugins: [{ name: "wasm-url", setup(b) {
      b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
      b.onLoad({ filter: /.*/, namespace: "wasmurl" }, () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    } }],
  });
  const m = await import(pathToFileURL(join(tmp, "b.mjs")).href);
  const read = (p) => JSON.parse(readFileSync(join(REPO, p), "utf8"));
  const mlp = read("models/finance_aml_mlp_large/weights.json");
  const gnn = read("models/finance_aml_gnn_compact/weights.json");
  const sample = read("models/finance_aml_gnn/eval_sample.json");
  const teacher = { rows: sample.seeds.row_index_in_test_window, labels: sample.seeds.labels, logits: sample.expected.logits,
                    threshold: sample.expected.decision_threshold };

  const sm = m.mlpStudent("finance_aml.mlp_large", mlp), sg = m.gnnStudent("finance_aml.gnn_compact", gnn);
  check(sm.maxFixtureDiff <= m.FIXTURE_TOLERANCE, `the MLP's clear forward is within ${sm.maxFixtureDiff.toExponential(2)} of its recorded logits`);
  check(sg.maxFixtureDiff <= m.FIXTURE_TOLERANCE, `the compact GNN's clear forward is within ${sg.maxFixtureDiff.toExponential(2)} of its recorded logits`);
  const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
  check(same(sm.rows, sg.rows) && same(sm.rows, teacher.rows), `the three files hold the same ${teacher.rows.length} test-window rows, in the same order`);
  check(same(sm.labels, teacher.labels) && same(sg.labels, teacher.labels), "and the same labels");

  // The teacher's decisions, by the export's own rule: p(laundering) at or above the threshold.
  const p1 = (l) => 1 / (1 + Math.exp(l[0] - l[1]));
  const want = teacher.logits.map((l) => (p1(l) >= teacher.threshold ? 1 : 0));
  check(same(want, sample.expected.per_row.predicted), "the teacher's rule reproduces the decisions its sample records");
  for (const s of [sm, sg]) {
    const pr = m.pairUp(teacher, s);
    const studentSays = s.logits.map((l) => (l[1] > l[0] ? 1 : 0));
    const agree = want.filter((v, i) => v === studentSays[i]).length;
    check(!pr.refused && pr.n === 16 && pr.agree === agree, `${s.key}: ${pr.agree} of ${pr.n} decisions agree with the teacher, recounted ${agree}`);
    check(pr.points.every((q, i) => q.teacherSays === want[i] && q.studentSays === studentSays[i]), `${s.key}: each point carries both decisions`);
  }

  // Seeded: a moved weight, a flipped label, no shared row. Each must be refused, not drawn.
  const moved = structuredClone(mlp); moved.layers[0].weight[0][0] += 0.5;
  check(!!m.pairUp(teacher, m.mlpStudent("moved", moved)).refused, "a student whose weights no longer give its recorded logits is refused");
  const flipped = { ...teacher, labels: teacher.labels.map((v, i) => (i === 0 ? 1 - v : v)) };
  check(/labelled/.test(m.pairUp(flipped, sm).refused ?? ""), "a row labelled differently by the teacher's sample and the student's is refused");
  const elsewhere = { ...teacher, rows: teacher.rows.map((r) => r + 1e9) };
  check(/share no row/.test(m.pairUp(elsewhere, sm).refused ?? ""), "a teacher with none of the students' rows is refused");
  check(m.clipLogOdds(1e6) === m.LOG_ODDS_CLIP && m.clipLogOdds(-1e6) === -m.LOG_ODDS_CLIP, `log-odds are drawn clipped at ${m.LOG_ODDS_CLIP}`);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
if (failed) { console.log(`\n${failed} check(s) failed`); process.exit(1); }
console.log("\nall teacher-student checks pass");
