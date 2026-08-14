// The browser's hybrid-KEM combiner must derive the SAME bytes as the Python one.
//
// WHY THIS EXISTS
//
// Round 38 fixed this combiner across what it believed were all its sites, twice, and was wrong both
// times. The second miss added the output-length tag to Python alone, one commit after a message
// reading "All four sites change together - they must, or backend and browser derive different
// keys." They did, and three shipped strings said otherwise - one of them rendered in the
// Validations panel, one printed into the run log. Nothing in either language's tests could see it:
// the Python test file never leaves Python, and the browser had no combiner test at all.
//
// A comment saying two implementations agree is not a check. This is: `scripts/export_combiner_
// vectors.py` calls the PRODUCTION `_combine` and commits its bytes; this bundles the PRODUCTION
// browser module and reproduces them. Neither side restates the other's info string, so the
// assertion survives a change to either.
//
// Modelled on `check_qfl.mjs`, which already proved this shape works: esbuild the real module graph
// into a temp dir and import it into Node, rather than re-implementing the thing under test. A
// re-implementation would have agreed with itself while the browser diverged, which is the exact
// failure being guarded.
//
// Run: node web/scripts/check_hybrid_combiner.mjs   (npm run check:combiner; CI gate)

import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = resolve(WEB, "..");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const hex = (u8) => Buffer.from(u8).toString("hex");
const unhex = (s) => Uint8Array.from(Buffer.from(s, "hex"));

function load() {
  const dir = mkdtempSync(join(tmpdir(), "qsh-combiner-"));
  const entry = join(dir, "entry.ts");
  writeFileSync(entry, `
export { hybridCombine, hybridCombinerInfo } from ${JSON.stringify(join(WEB, "src/lib/hybridCombiner"))};
`);
  const out = join(dir, "bundle.mjs");
  execFileSync("npx", ["esbuild", entry, "--bundle", "--format=esm", "--platform=node",
                       "--external:/node_modules/*", `--outfile=${out}`, "--log-level=error"],
               { cwd: WEB, stdio: "pipe" });
  return { out, dir };
}

const { out, dir } = load();
const m = await import(`file://${out}`);

// ---------------------------------------------------------------------------------------------
// 1. Every committed vector, reproduced byte for byte.
// ---------------------------------------------------------------------------------------------
const { vectors } = JSON.parse(
  readFileSync(join(WEB, "scripts/vectors/hybrid_combiner_vectors.json"), "utf8"));

check("vectors present", vectors.length >= 3,
      `only ${vectors.length} vectors; the L51 case is what exercises the length tag`);

for (const v of vectors) {
  const got = await m.hybridCombine(
    unhex(v.k1), unhex(v.k2),
    { c1: unhex(v.c1), c2: unhex(v.c2), ek1: unhex(v.ek1), ek2: unhex(v.ek2) },
    v.kem, v.length);
  check(`vector ${v.name}`, hex(got) === v.expected,
        `browser derived ${hex(got).slice(0, 32)}..., Python committed ${v.expected.slice(0, 32)}...`
        + " - the two implementations no longer agree; regenerate the vectors if Python changed"
        + " deliberately, otherwise the browser has drifted");
}

// ---------------------------------------------------------------------------------------------
// 2. The length tag is load-bearing, not decoration.
//
// HKDF-Expand for L bytes is a prefix of the expansion for any larger L'. `qhe/rate1_qfhe.py` is
// the caller that varies the length, so without the tag a QOTP pad would begin with exactly the
// bytes a transport session key over the same material would.
// ---------------------------------------------------------------------------------------------
{
  const v = vectors[0];
  const args = [unhex(v.k1), unhex(v.k2),
                { c1: unhex(v.c1), c2: unhex(v.c2), ek1: unhex(v.ek1), ek2: unhex(v.ek2) }, v.kem];
  const short = await m.hybridCombine(...args, 32);
  const long = await m.hybridCombine(...args, 51);
  check("length is separated", hex(long).slice(0, 64) !== hex(short),
        "the 32-byte derivation is a prefix of the 51-byte one; the output length is not in the info");
}

// ---------------------------------------------------------------------------------------------
// 3. The parameter set is in the separator, and the two sets do not share a derivation.
// ---------------------------------------------------------------------------------------------
{
  const a = new TextDecoder().decode(m.hybridCombinerInfo("ML-KEM-768", 32));
  const b = new TextDecoder().decode(m.hybridCombinerInfo("ML-KEM-1024", 32));
  check("768 named", a.includes("ML-KEM-768"), `separator is ${JSON.stringify(a)}`);
  check("1024 named", b.includes("ML-KEM-1024"), `separator is ${JSON.stringify(b)}`);
  check("sets differ", a !== b, "both parameter sets derive under one separator");
}

// ---------------------------------------------------------------------------------------------
// 4. The MCP agent scheme must STAY separate.
//
// A round themed "make the languages agree" is the round most likely to over-apply that principle.
// `clientMcp.ts` is a DIFFERENT composite scheme - static-static X25519, no ephemeral c2 - and
// Round 38 split it deliberately after briefly merging the two under one separator. This asserts
// the split holds, so a future unification cannot happen quietly.
// ---------------------------------------------------------------------------------------------
{
  const mcp = readFileSync(join(WEB, "src/lib/clientMcp.ts"), "utf8");
  const handshake = new TextDecoder().decode(m.hybridCombinerInfo("ML-KEM-768", 32));
  check("mcp separator distinct", mcp.includes("qsh/mcp-agent-kem/v1"),
        "clientMcp.ts no longer carries its own separator");
  check("mcp not merged", !mcp.includes(handshake.split("|")[0]),
        "clientMcp.ts now uses the handshake's separator; two composite schemes under one label is"
        + " what a domain separator exists to prevent");
}

// ---------------------------------------------------------------------------------------------
// 5. ONE definition of the separator in the browser.
//
// This is the check that would have caught the actual Round 38 failure. The bug was never a wrong
// string - it was three copies of a string, two of which were updated. A source-level assertion
// that the literal exists in exactly one module makes "did I get all the sites" answerable by grep
// instead of by memory.
// ---------------------------------------------------------------------------------------------
{
  // Tracked AND newly-added files, or a site introduced this session would not be counted.
  const argsets = [
    ["ls-files", "web/src"],
    ["ls-files", "--others", "--exclude-standard", "web/src"],
  ];
  const files = [...new Set(argsets.flatMap((args) =>
    execFileSync("git", args, { cwd: ROOT, encoding: "utf8" }).split("\n")))]
    .filter((f) => f.endsWith(".ts") || f.endsWith(".tsx"));
  // The literal as CODE, not as prose. `clientMcp.ts` discusses this separator in two comments
  // explaining why it deliberately does not use it - counting those would make the check fail for
  // documenting the very distinction it is asserting.
  const asCode = /["'`]qsh\/hybrid-kem\/v/;
  const carriers = files.filter((f) => {
    const src = readFileSync(join(ROOT, f), "utf8");
    return src.split("\n").some((line) =>
      asCode.test(line) && !/^\s*(\/\/|\*)/.test(line));
  });
  check("one definition", carriers.length === 1 && carriers[0].endsWith("hybridCombiner.ts"),
        `the handshake separator literal appears in ${JSON.stringify(carriers)};`
        + " it belongs in web/src/lib/hybridCombiner.ts alone, so a site that was not converted"
        + " fails here rather than diverging silently");
}

rmSync(dir, { recursive: true, force: true });

if (failures.length) {
  console.error("check:combiner FAILED");
  for (const f of failures) console.error("  - " + f);
  process.exit(1);
}
console.log(`check:combiner OK (${vectors.length} vectors reproduced from the Python combiner)`);
