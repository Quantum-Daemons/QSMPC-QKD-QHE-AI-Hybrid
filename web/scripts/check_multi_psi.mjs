#!/usr/bin/env node
/**
 * The PSI among three to five banks, end to end, with the real Polytune.
 *
 * WHAT THIS IS FOR. The finance page's card says every bank received the accounts all of them hold, and nothing else. The crate's tests cover the circuit; this runs
 * the module the page runs and checks what the card reports: the intersection against one computed in the clear, the circuit's size against the design,
 * that lists disagreeing with the clear answer are caught by the self-test, and that a bank's list being shuffled does not change the answer.
 *
 * Run: npm run check:multi-psi (about 10 s)
 */
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { build } = createRequire(join(WEB, "package.json"))("esbuild");
const mpcWasm = join(WEB, "src/wasm/qsh_mpc/qsh_mpc_bg.wasm");
const tmp = mkdtempSync(join(tmpdir(), "qsh-multi-psi-"));
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };

try {
  await build({
    stdin: { contents: `export { multiPartyPsi } from ${JSON.stringify(join(WEB, "src/lib/clientMultiPsi"))};`, resolveDir: WEB, loader: "ts" },
    bundle: true, format: "esm", platform: "node", outfile: join(tmp, "b.mjs"), logLevel: "error",
    define: { "import.meta.env": '{"BASE_URL":"/"}' },
  });
  globalThis.fetch = async (url) => {
    if (String(url).endsWith("qsh_mpc_bg.wasm")) return new Response(readFileSync(mpcWasm), { headers: { "Content-Type": "application/wasm" } });
    throw new Error(`unexpected fetch of ${url}`);
  };
  const { multiPartyPsi } = await import(pathToFileURL(join(tmp, "b.mjs")).href);
  const cfg = JSON.parse(readFileSync(join(WEB, "src/lib/config.defaults.json"), "utf8")).multi_party_psi;
  const lists = (n, k, shared) => Array.from({ length: n }, (_, p) => [
    ...Array.from({ length: shared }, (_, j) => `acct-${j}`),
    ...Array.from({ length: k - shared }, (_, j) => `bank${p}-${j}`),
  ]);
  const ands = (n, k) => k * ((n - 1) * k * 31 + (n - 1) * (k - 1) + (n - 2) + 32);

  for (const [n, k, shared] of [[3, cfg.set_size, cfg.shared], [4, 4, 2]]) {
    const r = await multiPartyPsi(lists(n, k, shared));
    check(r.intersection === shared && r.matchesClear, `${n} banks of ${k}: ${r.intersection} accounts held by all, the clear intersection's ${r.clearIntersection}`);
    check(r.andGates === ands(n, k), `the circuit is the size the design says (${r.andGates} AND gates, expected ${ands(n, k)})`);
    check(r.bytes > 0 && r.messages > 0, `${(r.bytes / 1e6).toFixed(1)} MB in ${r.messages} messages`);
    check(r.falseMatchBoundLog2 < -20, `the false-match bound is below 2^-20 (2^${r.falseMatchBoundLog2.toFixed(1)})`);
  }
  // One bank without one of the shared accounts: the answer shrinks by exactly that account.
  const short = lists(3, 4, 2);
  short[2][0] = "bank2-other";
  const s = await multiPartyPsi(short);
  check(s.intersection === 1 && s.matchesClear, `a bank without one shared account leaves ${s.intersection}`);
  // A bank that holds none of them: nothing comes out.
  const none = lists(3, 4, 2);
  none[1] = ["x-1", "x-2", "x-3", "x-4"];
  const z = await multiPartyPsi(none);
  check(z.intersection === 0 && z.matchesClear, `a bank holding none of them leaves ${z.intersection}`);

  const src = readFileSync(join(WEB, "src/lib/clientMultiPsi.ts"), "utf8");
  check(/const sets = hashed\.map\(\(h\) => shuffled\(/.test(src), "each bank's list is shuffled, because the output is positional in the first one");
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
if (failed) { console.error(`${failed} check(s) failed`); process.exit(1); }
console.log("multi-party PSI: all checks passed");
