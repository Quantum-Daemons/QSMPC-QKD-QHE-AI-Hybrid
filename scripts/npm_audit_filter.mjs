#!/usr/bin/env node
// Reads `npm audit --json` on stdin and exits 0 only when every advisory it reports is let through by
// scripts/npm_audit_ignore.json and still within that entry's terms (Round 86, 2026-11-18). scripts/audit_deps.sh calls it
// only after npm audit has failed, and prints npm's own report when this fails too.
//
// An entry lets its advisory through when all of these hold, and the advisory fails the check otherwise:
// - today is on or before its `until` date;
// - every package npm reports is the entry's `package` or one of its `through`, so the advisory is reported along the path
//   the reason was written for and no other;
// - the advisory's range is a bare `<=X.Y.Z` or `<X.Y.Z`, and the newest published version of the package is inside it,
//   so no release fixes it yet; once one is published, the entry stops applying and the upgrade is due.
// Any other advisory, and any report that is not a version 2 audit report, fails.
//
//   node scripts/npm_audit_filter.mjs <checked directory> < report.json
// AUDIT_TODAY (YYYY-MM-DD) and NPM_LATEST (a JSON object of package to newest version) stand in for the date and for
// `npm view`, which is how tests/test_dependency_advisories_are_checked_on_a_schedule.py checks every failure path.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ignore = JSON.parse(readFileSync(join(here, "npm_audit_ignore.json"), "utf8"));
const where = process.argv[2] ?? "the project";
const today = process.env.AUDIT_TODAY ?? new Date().toISOString().slice(0, 10);
const latestStub = process.env.NPM_LATEST ? JSON.parse(process.env.NPM_LATEST) : null;

const newest = (pkg) => latestStub ? latestStub[pkg] ?? null
  : execFileSync("npm", ["view", pkg, "version"], { encoding: "utf8" }).trim();

const parts = (v) => (/^\d+\.\d+\.\d+$/.test(v ?? "") ? v.split(".").map(Number) : null);
const compare = (a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];

function inside(version, range) {
  const m = /^(<=?)\s*(\d+\.\d+\.\d+)$/.exec((range ?? "").trim());
  const v = parts(version);
  if (!m || !v) return null;                      // a range or a version this does not read: not let through
  const c = compare(v, parts(m[2]));
  return m[1] === "<=" ? c <= 0 : c < 0;
}

const input = readFileSync(0, "utf8");
let report;
try { report = JSON.parse(input); } catch { report = null; }
const problems = [];
if (!report || report.auditReportVersion !== 2 || typeof report.vulnerabilities !== "object") {
  problems.push("not a version 2 npm audit report" + (report?.error ? `: ${JSON.stringify(report.error)}` : ""));
}
const vulns = report?.vulnerabilities ?? {};
const used = new Map();                           // advisory id -> the packages reported, for the summary
for (const [name, v] of Object.entries(vulns)) {
  for (const via of v.via ?? []) {
    if (typeof via === "string") {
      if (!(via in vulns)) problems.push(`${name} is reported through ${via}, which the report does not list`);
      continue;
    }
    const id = /GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}/.exec(via.url ?? "")?.[0];
    const entry = id && ignore[id];
    if (!entry) { problems.push(`${name}: ${via.url ?? via.title ?? "an advisory"} is not let through`); continue; }
    if (via.name !== entry.package) problems.push(`${id} names ${via.name}, not ${entry.package}`);
    if (today > entry.until) problems.push(`${id} was let through until ${entry.until}, and today is ${today}`);
    const latest = newest(entry.package);
    const covered = inside(latest, via.range);
    if (covered !== true) {
      problems.push(covered === null
        ? `${id}: the range ${JSON.stringify(via.range)} or ${entry.package}'s newest version ${latest} cannot be read`
        : `${id}: ${entry.package} ${latest} is published outside the range ${via.range}, so a fix exists; upgrade`);
    }
    used.set(id, entry);
  }
}
for (const name of Object.keys(vulns)) {
  const allowed = [...used.values()].some((e) => name === e.package || e.through.includes(name));
  if (!allowed) problems.push(`${name} is reported, and no advisory let through names it`);
}
if (problems.length) {
  console.error(`npm audit in ${where}: not let through`);
  for (const p of problems) console.error(`  - ${p}`);
  process.exit(1);
}
for (const [id, e] of used) {
  console.log(`npm audit in ${where}: ${id} (${e.package}, through ${e.through.join(", ")}) let through until ${e.until}, `
    + "as scripts/npm_audit_ignore.json states; nothing else is reported");
}
if (!used.size) console.log(`npm audit in ${where}: nothing reported`);
