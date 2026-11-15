#!/usr/bin/env node
/**
 * What IBM's Multi-GNN read for each transfer it scored: the page's counts against a hand-counted graph (Round 86).
 *
 * The finance page's IBM tab draws each scored transfer's two accounts and counts, from the graph sample it scores, the
 * transfers the model read and how many came after the scored one, in its batch of 2,048 or beyond it. The export
 * writes the same counts (scripts/train/ibm_multignn/neighbourhood.py) and the page reports a mismatch. Held here, on
 * scripts/vectors/graph_neighbourhood.json, which tests/test_r86_ibm_neighbourhood.py holds the Python side to:
 *   1. the page's counts are the hand counts, seed by seed;
 *   2. a changed count is reported, naming the seed and the count;
 *   3. the drawing: the scored transfer first, account A paying B, each other transfer at the two accounts once and
 *      marked earlier, later in its batch or beyond it, and a cap split between the two accounts, each keeping those
 *      nearest the scored transfer in time;
 *   4. given exported samples as arguments, their counts are the ones the file records, and a causal sample has none
 *      beyond its batch.
 *
 * Run: npm run check:graph-neighbourhood [-- eval_sample.json ...]
 */
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { build } = createRequire(join(WEB, "package.json"))("esbuild");
const tmp = mkdtempSync(join(tmpdir(), "qsh-graph-neighbourhood-"));
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };

try {
  await build({
    entryPoints: [join(WEB, "src/lib/graphNeighbourhood.ts")], bundle: true, format: "esm", platform: "node",
    outfile: join(tmp, "b.mjs"), logLevel: "error",
  });
  const { seedFields, fieldsMismatch, seedGraph } = await import(pathToFileURL(join(tmp, "b.mjs")).href);
  const f = JSON.parse(readFileSync(join(WEB, "scripts/vectors/graph_neighbourhood.json"), "utf8"));
  const ei = [...f.edge_index[0], ...f.edge_index[1]], n = f.edge_position.length;
  const view = { batch_rows: 2048, edge_position: f.edge_position, edge_seconds: f.edge_seconds, per_seed: f.expected };

  // 1. The hand counts.
  const got = seedFields(ei, n, view, f.seed_at);
  got.forEach((g, k) => check(JSON.stringify(g) === JSON.stringify(f.expected[k]), `seed ${k + 1}: ${JSON.stringify(g)}`));
  check(fieldsMismatch(got, f.expected) === null, "the page finds no mismatch with the hand counts");

  // 2. A changed count is named.
  const off = structuredClone(f.expected);
  off[1].later_in_batch += 1;
  check(fieldsMismatch(got, off) === "seed 2: later_in_batch is 2 here and 3 in the file", `a changed count: ${fieldsMismatch(got, off)}`);

  // 3. The drawing of seed 1 (edge 0, account 0 pays account 1 at 600 s).
  const g = seedGraph(ei, n, view, f.seed_at[0]);
  const drawn = g.edges.map((e) => `${e.from}>${e.to}:${e.when}:${e.minute}`).join(" ");
  check(drawn === "A>B:scored:0 A>A:earlier:-6 C>A:earlier:-5 A>D:in-batch:1 B>E:beyond:5", `the transfers at the two accounts, once each: ${drawn}`);
  check(g.from === "A" && g.to === "B" && g.shown === 4 && g.at_accounts === 4, "A pays B, and all 4 of the 4 at the accounts are drawn");
  const capped = seedGraph(ei, n, view, f.seed_at[0], 2);
  const kept = capped.edges.slice(1).map((e) => e.minute).join(",");
  check(kept === "1,5" && capped.shown === 2 && capped.at_accounts === 4,
    `a cap of 2 keeps the nearest at each account, the payer's 1 min after and the payee's 5 min after: ${kept}`);

  // 4. Exported samples given as arguments.
  for (const path of process.argv.slice(2)) {
    const s = JSON.parse(readFileSync(path, "utf8")), i = s.inputs;
    const derived = seedFields(i.edge_index, i.n_edges, s.view, s.seeds.positions_in_batch);
    check(fieldsMismatch(derived, s.view.per_seed) === null, `${s.protocol}: the counts are the file's (${path})`);
    if (s.protocol === "causal") check(derived.every((d) => d.beyond_batch === 0), "causal: nothing beyond a seed's batch of 2,048");
    derived.forEach((d, k) => console.log(`     seed ${String(k + 1).padStart(2)} label ${s.seeds.labels[k]}: read ${d.read}, later in its batch ${d.later_in_batch}, ` +
      `beyond it ${d.beyond_batch}, furthest ${(d.furthest_later_s / 60).toFixed(0)} min; payer out ${d.payer_out} in ${d.payer_in}, payee out ${d.payee_out} in ${d.payee_in}`));
  }
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
console.log(failed ? `${failed} FAILED` : "all graph-neighbourhood checks passed");
process.exit(failed ? 1 : 0);
