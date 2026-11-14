// The built page carries the notices of the npm packages in its JavaScript (Round 86).
//
// MIT, ISC, BSD and Apache-2.0 ask that their notice travel with copies of the code, and minification keeps only the
// comments a package marks for it: before this gate a built page carried React's, Three.js's and ONNX Runtime's and no
// other. `vite.config.ts` writes `licenses/js-bundle.txt` from the modules the bundle contains (the workers' included);
// this reads the BUILT file, not the config meant to produce it, and fails when
//   - the file is missing or names fewer packages than a page this size bundles;
//   - an entry has no licence text, or declares no licence;
//   - a package the page plainly ships is not listed (React, the noble libraries, node-seal, ONNX Runtime Web, recharts,
//     React Flow, three, html-to-image, gif.js, React Router).
//
//   node scripts/check_js_notices.mjs [dist/licenses/js-bundle.txt]
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const path = process.argv[2] ?? join(WEB, "dist", "licenses", "js-bundle.txt");
const REQUIRED = ["react", "react-dom", "@noble/post-quantum", "@noble/curves", "@noble/hashes", "node-seal",
  "onnxruntime-web", "recharts", "@xyflow/react", "three", "html-to-image", "gif.js", "react-router"];
const MIN_PACKAGES = 40;

const findings = [];
if (!existsSync(path)) {
  console.error(`no notices file at ${path}: build the page first (npm run build)`);
  process.exit(1);
}
const text = readFileSync(path, "utf8");
const entries = text.split(/^={78}$/m).slice(1).map((block) => {
  const lines = block.trim().split("\n");
  const [name, version] = (lines[0] ?? "").split(" ");
  const declared = (lines.find((l) => l.startsWith("Declared licence: ")) ?? "").slice("Declared licence: ".length);
  const body = lines.slice(3).join("\n").trim();
  return { name, version, declared, body };
});
if (entries.length < MIN_PACKAGES) findings.push(`only ${entries.length} packages listed; a build of this page bundles more than ${MIN_PACKAGES}`);
for (const e of entries) {
  if (!e.declared || e.declared === "none declared") findings.push(`${e.name} ${e.version}: no licence declared`);
  if (e.body.length < 100) findings.push(`${e.name} ${e.version}: no licence text`);
}
const names = new Set(entries.map((e) => e.name));
for (const r of REQUIRED) if (!names.has(r)) findings.push(`${r} is in the page and not in the notices`);
if (findings.length) {
  console.error(`check_js_notices: ${findings.length} finding(s) in ${path}`);
  for (const f of findings) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(`check_js_notices: ${entries.length} packages, each with its declared licence and its text`);
