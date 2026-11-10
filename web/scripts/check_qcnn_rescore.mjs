#!/usr/bin/env node
/**
 * The trained QCNN re-scored by the module the page runs, on the served weights and held-out rows, against the record (Round 86).
 *
 * `scripts/data/build_qcnn_heldout.py` writes `public/data/<dir>/qcnn_heldout.json` only after the trainer's circuit reproduces every
 * recorded metric from those rows. This runs the BROWSER's circuit over the same rows and asserts the page's headline equals the
 * weights file's to six decimals for every served artifact, and that the check would notice a wrong re-score: one label flipped must
 * move the figure off the record.
 *
 * Run: npm run check:qcnn-rescore
 */
import { createRequire } from "node:module";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PUB = join(WEB, "public/data");
const { build } = createRequire(join(WEB, "package.json"))("esbuild");
const tmp = mkdtempSync(join(tmpdir(), "qsh-qcnn-rescore-"));
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };
try {
  await build({
    stdin: { contents: `export { rescoreQcnn, macroF1, minorityF1 } from ${JSON.stringify(join(WEB, "src/lib/qcnnRescore"))};`, resolveDir: WEB, loader: "ts" },
    bundle: true, format: "esm", platform: "node", outfile: join(tmp, "b.mjs"), logLevel: "error",
    define: { "import.meta.env": '{"BASE_URL":"/"}' },
  });
  const m = await import(pathToFileURL(join(tmp, "b.mjs")).href);
  // Every served QCNN, from the filesystem rather than a list kept by hand.
  const served = readdirSync(PUB).flatMap((d) => existsSync(join(PUB, d)) && !d.includes(".")
    ? readdirSync(join(PUB, d)).filter((f) => /^qcnn_weights(_q\d+)?\.json$/.test(f)).map((f) => [d, f]) : []);
  check(served.length >= 4, `${served.length} served QCNN weights files`);
  for (const [dir, file] of served) {
    const heldPath = join(PUB, dir, "qcnn_heldout.json");
    if (!existsSync(heldPath)) { check(false, `${dir}/${file}: no ${dir}/qcnn_heldout.json beside it`); continue; }
    const w = JSON.parse(readFileSync(join(PUB, dir, file), "utf8"));
    const held = JSON.parse(readFileSync(heldPath, "utf8"));
    check(held.rows.length === w.n_test, `${dir}/${file}: ${held.rows.length} held-out rows, the artifact's n_test ${w.n_test}`);
    const r = m.rescoreQcnn(w, held);
    check(r.matchesRecord, `${dir}/${file}: ${r.label} re-scored ${r.headline.toFixed(6)} over ${r.rows} rows, recorded ${r.recorded} (${r.ms.toFixed(0)} ms)`);
    check(Math.abs(r.accuracy - w.test_accuracy) < 5e-7, `${dir}/${file}: accuracy ${r.accuracy.toFixed(6)}, recorded ${w.test_accuracy}`);
    // Seeded: the first row's label moved to another class. A comparison that cannot see this is not checking the re-score.
    const flipped = { ...held, labels: held.labels.map((v, i) => (i === 0 ? (v === 0 ? 1 : 0) : v)) };
    check(!m.rescoreQcnn(w, flipped).matchesRecord, `${dir}/${file}: one flipped label moves the figure off the record`);
  }
  // scikit-learn's conventions, on cases with known answers: macro F1 averages over the labels in the truth or the predictions.
  check(Math.abs(m.macroF1([0, 0, 1, 2], [0, 1, 1, 1]) - (2 / 3 + 0.5 + 0) / 3) < 1e-12, "macro F1 averages over the union of labels, zero for an absent class");
  check(m.minorityF1([0, 0, 0], [0, 0, 0]) === 0, "minority F1 is 0 when class 1 is neither present nor predicted (zero_division=0)");
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
if (failed) { console.log(`\n${failed} check(s) failed`); process.exit(1); }
console.log("\nall QCNN re-score checks pass");
