// No built file points at a source map that the build does not ship.
//
// WHY THIS EXISTS
//
// `gif.js/dist/gif.worker.js?url` is copied into the build verbatim, and it ends with
// `//# sourceMappingURL=gif.worker.js.map`, a file that is not copied with it. Any open DevTools
// asks for it when the worker loads, at the start of every run. Visitors are not affected, but a
// build that names files it does not ship is wrong in a way nothing else checks. Found in the
// Round 74 walk. `vite.config.ts` now drops such comments, and this checks the output rather than
// the config.
//
// Run after `npm run build`: node web/scripts/check_no_dangling_source_maps.mjs [distDir]
// (npm run check:source-maps; CI gate, and step 8 of scripts/ci_local.sh)

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DIST = resolve(process.argv[2] ?? join(WEB, "dist"));

if (!existsSync(DIST)) {
  console.error(`no build at ${DIST} - run npm run build first; a check of nothing passes`);
  process.exit(1);
}

function* files(dir) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) yield* files(p);
    else if (/\.(m?js|css)$/.test(e.name)) yield p;
  }
}

const dangling = [];
let scanned = 0;
for (const f of files(DIST)) {
  scanned += 1;
  const text = readFileSync(f, "utf8");
  for (const m of text.matchAll(/[#@] sourceMappingURL=(\S+?)(?:\s*\*\/)?\s*$/gm)) {
    const ref = m[1];
    if (ref.startsWith("data:")) continue;                  // an inline map ships with the file
    const target = resolve(dirname(f), ref.split(/[?#]/)[0]);
    if (!existsSync(target) || !statSync(target).isFile()) {
      dangling.push(`${relative(DIST, f)} -> ${ref}`);
    }
  }
}

if (scanned < 10) {
  console.error(`only ${scanned} built scripts and stylesheets found in ${DIST} - is this the build?`);
  process.exit(1);
}
if (dangling.length) {
  console.error("built files reference source maps the build does not ship (DevTools logs a 404 for each):");
  for (const d of dangling) console.error(`  ${d}`);
  process.exit(1);
}
console.log(`PASS  ${scanned} built scripts and stylesheets: no source-map reference to a file that is not served`);
