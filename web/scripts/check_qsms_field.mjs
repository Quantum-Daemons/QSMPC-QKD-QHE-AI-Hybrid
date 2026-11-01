/**
 * The browser's secure-sum field comes from what the inputs CAN total, never from their total.
 *
 * WHY THIS EXISTS
 *
 * The genomics stage chose p = nextPrime(rawTotal + 1): the ANSWER picked the field. The cohorts'
 * 23 effect alleles ran over GF(29), so the prime alone placed the total in 23..28, and
 * `qsms_matches_plain_total` could not fail - the check compared a sum with itself by construction.
 * The comment above it said "max possible total", which is what the code did not do. The Python
 * twin had the opposite defect, a field too small (GF(5) wrapped 45 to 0); it is pinned by
 * tests/test_r84_c1_secure_sum_field.py. Round 84 (scratchpad audit82/factcheck-finance-genomics.md, A5).
 *
 * What this asserts:
 *   1. `qsmsField` returns a prime above the bound it is given, for every bound, and honours a
 *      requested prime;
 *   2. on the bundled 1000 Genomes slice every dosage is an allele count (0, 1 or 2), so twice the
 *      cohort is a true bound; the field from it holds the real total and the largest total the
 *      slice could carry, unreduced, and does not move when the counts do;
 *   3. the stage calls `qsmsField` with that bound and no longer derives p from the total.
 *
 * HOW IT LOADS THE TS: esbuild's API bundles clientCompute.ts, with the two imports node cannot
 * resolve (the gitignored wasm module and node-seal's ?url asset) stubbed, as they are unused here.
 *
 * Run: node web/scripts/check_qsms_field.mjs
 */
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const esbuild = await import(pathToFileURL(join(WEB, "node_modules/esbuild/lib/main.js")).href);
const dir = mkdtempSync(join(tmpdir(), "qsh-qsms-field-"));
const out = join(dir, "bundle.mjs");
await esbuild.build({
  stdin: {
    contents: `export { qsmsField, QSMS_STAND_IN_MAX } from ${JSON.stringify(join(WEB, "src/lib/clientCompute"))};\n`
      + `export { realAlleleCounts, PRS_PAYLOAD_PATH } from ${JSON.stringify(join(WEB, "src/lib/realData"))};\n`,
    resolveDir: WEB, loader: "ts",
  },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": "{}" },
  plugins: [{ name: "stub-unresolvable", setup(b) {
    b.onResolve({ filter: /qsh_mpc\.js$|\.wasm\?url$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });

const isPrime = (n) => { if (n < 2) return false; for (let i = 2; i * i <= n; i++) if (n % i === 0) return false; return true; };
const shareSum = (secrets, p) => secrets
  .map((s) => { const mask = Math.floor(Math.random() * p); return [((s - mask) % p + p) % p, mask]; })
  .reduce((acc, [a, b]) => (acc + a + b) % p, 0);

// 1. The field exceeds its bound, for every bound, and a requested prime wins.
check("export", typeof m.qsmsField === "function", "clientCompute.ts exports no qsmsField - the field is still chosen inline");
if (typeof m.qsmsField === "function") {
  for (let bound = 0; bound <= 600; bound++) {
    const p = m.qsmsField(bound, 0);
    if (!(isPrime(p) && p > bound)) { check("bound", false, `qsmsField(${bound}) = ${p}, not a prime above ${bound}`); break; }
  }
  check("paper", m.qsmsField(60, 0) === 61, `the bundled slice's bound 60 should give GF(61), got ${m.qsmsField(60, 0)}`);
  check("requested", m.qsmsField(60, 100) === 101, "a requested prime is snapped up to the next prime and honoured");
  check("stand-ins", m.QSMS_STAND_IN_MAX === 99, `stand-ins are floor(u() * 100), so their maximum is 99, not ${m.QSMS_STAND_IN_MAX}`);
}

// 2. The real slice: counts are allele counts, the bound is true, and the sum is exact.
const prs = JSON.parse(readFileSync(join(WEB, "public", m.PRS_PAYLOAD_PATH.replace(/^\/+/, "")), "utf8"));
const column = prs.dosage_samples.map((row) => row[0]);
check("dosages", column.every((v) => v === 0 || v === 1 || v === 2),
  "a dosage at the pooled variant is not 0, 1 or 2, so twice the cohort is not a bound on the count");
globalThis.fetch = async (url) => ({ ok: true, json: async () => JSON.parse(readFileSync(join(WEB, "public", String(url)), "utf8")) });
const alleles = await m.realAlleleCounts(4);
check("load", alleles && alleles.counts.length === 4, "realAlleleCounts did not return four cohorts");
if (alleles && typeof m.qsmsField === "function") {
  const total = alleles.counts.reduce((a, b) => a + b, 0);
  const bound = 2 * alleles.individuals;
  const p = m.qsmsField(bound, 0);
  check("holds", p > total && shareSum(alleles.counts, p) === total,
    `the pooled ${total} does not survive the sum over GF(${p}) unreduced`);
  // The extreme the bound allows: everybody homozygous for the effect allele.
  const sizes = alleles.counts.map((_, k) => Math.floor(alleles.individuals / 4) + (k < alleles.individuals % 4 ? 1 : 0));
  const extreme = sizes.map((s) => 2 * s);
  check("extreme", shareSum(extreme, p) === bound, `the largest possible pool, ${bound}, wraps over GF(${p})`);
  // The field is a function of the bound alone: other counts over the same cohort, same field.
  check("answer-free", m.qsmsField(bound, 0) === p && p !== m.qsmsField(total, 0),
    `GF(${p}) should not be the field the total alone would pick (GF(${m.qsmsField(total, 0)}))`);
}

// 3. The stage uses the bound. Comments stripped first, so a comment cannot satisfy the check.
const decomment = (text) => text.replace(/\/\*[\s\S]*?\*\//g, "").split("\n").map((l) => l.replace(/\/\/.*$/, "")).join("\n");
const src = decomment(readFileSync(join(WEB, "src/lib/clientCompute.ts"), "utf8"));
const at = src.indexOf('case "qsmpc_compute"');
const block = at >= 0 ? src.slice(at, src.indexOf('case "federated_training"', at)) : "";
check("stage", block.length > 0, "the qsmpc_compute case was not found - this check is inert");
check("stage", /qsmsField\(\s*maxTotal\s*,\s*askedPrime\s*\)/.test(block), "the stage does not call qsmsField(maxTotal, askedPrime)");
check("stage", /maxTotal\s*=\s*alleles\s*\?\s*2\s*\*\s*alleles\.individuals/.test(block),
  "the stage's bound is not two alleles per person for the genomics counts");
check("stage", !/nextPrime\(\s*rawTotal/.test(block), "the stage still derives p from the total");

if (failures.length) {
  console.error(`check_qsms_field: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check_qsms_field: OK - the field comes from the inputs' range and holds the pooled count unreduced");
