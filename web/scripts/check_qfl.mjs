#!/usr/bin/env node
/**
 * Assert that federated aggregation MASKS, and that the masks cancel exactly.
 *
 * WHAT THIS IS FOR. `TwoPartyDiagram` drew "federated train + pairwise masking (demo keys)" and
 * `OrchestrationGraph` drew "(secure aggregation)" while `clientQFL` summed plaintext updates: no
 * masks, no pairwise seeds, no key material, and `secureAgg: true` written as a literal on the
 * return line. A figure asserting a property of code in another file is unfalsifiable by
 * construction, so the property is asserted here instead, on the function.
 *
 * WHY A RING AND NOT A TOLERANCE. Pairwise masks cancel because +m and -m each appear once in the
 * sum. In doubles that cancellation depends on summation order, so it holds only to some epsilon -
 * and a check written against an epsilon is one that passes whether or not the masks are there. In
 * `Uint32` the cancellation is exact, so `masked-aggregate == plaintext-aggregate` can be asserted
 * as EQUALITY. That is the whole reason `secureAggregate` quantises.
 *
 * WHAT A PASSING RUN RULES OUT. Not just "the numbers look right": that the masks are applied at all
 * (a masked share must differ from its plaintext share), that they are keyed by the session key (a
 * different key must move them), and that they are not reused across rounds. Each of those is a way
 * the aggregation could be plaintext-in-effect while every arithmetic assertion still passed.
 *
 * Run: npm run check:qfl
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

function load() {
  const dir = mkdtempSync(join(tmpdir(), "qsh-qfl-"));
  const entry = join(dir, "entry.ts");
  writeFileSync(entry, `
export { clientQFL, secureAggregate } from ${JSON.stringify(join(WEB, "src/lib/clientQFL"))};
`);
  const out = join(dir, "bundle.mjs");
  // `clientCompute` reaches `clientFHE`, which imports the SEAL wasm through Vite's `?url` suffix
  // from a root-absolute path. esbuild has no such resolver, so that ONE specifier is left external:
  // it is a URL string handed to the FHE backend at call time, and nothing on this file's paths
  // calls it. Marking it external keeps the rest of the real module graph - including the actual
  // `secureAggregate` the browser runs - rather than re-implementing it here.
  execFileSync("npx", ["esbuild", entry, "--bundle", "--format=esm", "--platform=node",
                       "--external:/node_modules/*", `--outfile=${out}`, "--log-level=error"],
               { cwd: WEB, stdio: "pipe" });
  return { out, dir };
}

const { out, dir } = load();
const m = await import(`file://${out}`);

// A fixed, obviously-not-secret key: this asserts plumbing and algebra, not key quality.
const KEY = Uint8Array.from({ length: 32 }, (_, i) => (i * 7 + 3) & 0xff);
const KEY2 = Uint8Array.from({ length: 32 }, (_, i) => (i * 11 + 5) & 0xff);

// Four clients, four coordinates, deliberately uneven shard sizes so the weights are all distinct
// and a mask that failed to cancel could not be hidden by symmetry.
const UPDATES = [
  [0.8125, -1.5, 0.03125, 2.0],
  [-0.25, 0.5, 1.125, -0.75],
  [1.5, 0.25, -0.5, 0.0625],
  [-0.0625, 2.25, 0.75, -1.375],
];
const SIZES = [37, 21, 29, 13];
const TOT = SIZES.reduce((a, b) => a + b, 0);
const W = SIZES.map((n) => n / TOT);

// The aggregator's own definition, with the masks set aside: quantise each weighted contribution,
// sum in the ring, map back through the signed half. If the masks cancel, `secureAggregate` returns
// exactly this. Written here rather than exported from the module so the two are independent.
const SCALE = 1 << 20;
const RING = 2 ** 32;
function plainAggregate(updates, weights) {
  const P = updates[0].length;
  return Array.from({ length: P }, (_, k) => {
    let acc = 0;
    for (let u = 0; u < updates.length; u++) {
      acc = (acc + ((Math.round(updates[u][k] * weights[u] * SCALE) | 0) >>> 0)) >>> 0;
    }
    return (acc >= RING / 2 ? acc - RING : acc) / SCALE;
  });
}

// --------------------------------------------------------------------------------------------- //
//  1. The masks cancel EXACTLY
// --------------------------------------------------------------------------------------------- //
{
  const { aggregate } = m.secureAggregate(UPDATES, W, KEY, 0);
  const reference = plainAggregate(UPDATES, W);
  check("masks-cancel-exactly",
    aggregate.length === reference.length && aggregate.every((v, k) => Object.is(v, reference[k])),
    `masked aggregate ${JSON.stringify(aggregate)} != plaintext aggregate ` +
    `${JSON.stringify(reference)}; the pair terms did not cancel`);

  // The result must also be the weighted mean it claims to be, to the quantisation and no worse.
  const exact = UPDATES[0].map((_, k) => UPDATES.reduce((s, wl, u) => s + wl[k] * W[u], 0));
  const worst = Math.max(...aggregate.map((v, k) => Math.abs(v - exact[k])));
  check("aggregate-is-the-weighted-mean", worst <= UPDATES.length / SCALE,
    `worst coordinate error ${worst} exceeds ${UPDATES.length}/2^20 - the ring is not recovering ` +
    `the mean, so agreeing with plainAggregate would only mean both are wrong the same way`);
}

// --------------------------------------------------------------------------------------------- //
//  2. The masks are actually APPLIED
// --------------------------------------------------------------------------------------------- //
// Without this, every arithmetic assertion above is satisfied by a function that masks with zero.
{
  const { maskedShares } = m.secureAggregate(UPDATES, W, KEY, 0);
  for (let u = 0; u < UPDATES.length; u++) {
    const plain = UPDATES[u].map((v) => (Math.round(v * W[u] * SCALE) | 0) >>> 0);
    check(`masked-share-differs-${u}`,
      maskedShares[u].some((v, k) => v !== plain[k]),
      `client ${u}'s transmitted share equals its plaintext share ${JSON.stringify(plain)} - ` +
      `no mask was added, so the aggregator sees the update in the clear`);
  }
  // And the masks must be the ONLY difference: shares summing to the plaintext total is checked
  // above, so a share that matched plaintext everywhere would be caught here rather than silently.
  check("every-client-is-masked",
    maskedShares.length === UPDATES.length,
    `${maskedShares.length} shares for ${UPDATES.length} clients`);
}

// --------------------------------------------------------------------------------------------- //
//  3. The masks are keyed, and per round
// --------------------------------------------------------------------------------------------- //
// The failure this rules out is the one that made the fEC check necessary: an input that is accepted
// and then ignored. A mask derived from a constant would satisfy sections 1 and 2 perfectly.
{
  const a = m.secureAggregate(UPDATES, W, KEY, 0).maskedShares;
  const b = m.secureAggregate(UPDATES, W, KEY, 0).maskedShares;
  check("shares-are-deterministic",
    a.every((s, u) => s.every((v, k) => v === b[u][k])),
    "two identical calls produced different shares; HKDF is deterministic, so something " +
    "non-derived entered the mask");

  const other = m.secureAggregate(UPDATES, W, KEY2, 0).maskedShares;
  check("session-key-reaches-the-mask",
    a.some((s, u) => s.some((v, k) => v !== other[u][k])),
    "changing the session key left every share identical - the key is not an input to the mask");

  const round1 = m.secureAggregate(UPDATES, W, KEY, 1).maskedShares;
  check("masks-are-not-reused-across-rounds",
    a.some((s, u) => s.some((v, k) => v !== round1[u][k])),
    "round 0 and round 1 produced identical masks; reusing a pad across two updates of the same " +
    "weights leaks their difference");

  // Cancellation must survive both, or the key is an input to the mask but not to a correct one.
  for (const [label, agg] of [["key2", m.secureAggregate(UPDATES, W, KEY2, 0).aggregate],
                              ["round1", m.secureAggregate(UPDATES, W, KEY, 1).aggregate]]) {
    const reference = plainAggregate(UPDATES, W);
    check(`masks-cancel-under-${label}`,
      agg.every((v, k) => Object.is(v, reference[k])),
      `aggregate under ${label} is ${JSON.stringify(agg)}, not ${JSON.stringify(reference)}`);
  }
}

// --------------------------------------------------------------------------------------------- //
//  3b. The ring bound rejects what would overflow it, and admits what fits
// --------------------------------------------------------------------------------------------- //
// The bound depends on the client count: with K clients the SUM has to stay inside the signed half,
// so each share is capped at 2^31/K. It was written as a flat 2^30, which four clients could exceed
// in aggregate - a guard that admitted the exact overflow it existed to reject. Real weights are
// order 1 and never approach either number, which is why only an assertion finds this.
{
  const K = UPDATES.length;
  const justOver = (2 ** 31 / K) / SCALE * 1.01;      // one per cent past the per-client cap
  const big = UPDATES.map((row, u) => (u === 0 ? [justOver, ...row.slice(1)] : row));
  let threw = false;
  try { m.secureAggregate(big, W.map(() => 1), KEY, 0); } catch { threw = true; }
  check("ring-bound-rejects-an-overflowing-share", threw,
    `a share of ${justOver * SCALE} quantised units was accepted; with ${K} clients anything past ` +
    `2^31/${K} can push the sum out of the signed half, where the recovered mean is a wrong number ` +
    `that still looks like a number`);

  // And it must not reject ordinary work: a bound set too tight is a different way to be broken.
  let ok = true;
  try { m.secureAggregate(UPDATES, W, KEY, 0); } catch { ok = false; }
  check("ring-bound-admits-real-updates", ok,
    "the ordinary four-client update was rejected by the range guard");
}

// --------------------------------------------------------------------------------------------- //
//  4. There is no plaintext mode to fall back into
// --------------------------------------------------------------------------------------------- //
// How the original claim survived: written once, and the path that would have contradicted it was
// the one always taken. `clientQFL` must refuse rather than aggregate in the clear.
{
  let threw = false;
  try { m.clientQFL(0, { clients: 4, rounds: 1 }); } catch { threw = true; }
  check("no-plaintext-fallback", threw,
    "clientQFL aggregated without a session key. A silent plaintext path is exactly how the " +
    "diagram came to claim masking that did not run");
}

// --------------------------------------------------------------------------------------------- //
//  5. The pair count is measured, and the pipeline's accuracy is pinned
// --------------------------------------------------------------------------------------------- //
{
  const q = m.clientQFL(0, { clients: 4, rounds: 6, sessionKey: KEY });
  check("pair-count-is-C(K,2)", q.pairs === 6,
    `4 clients need C(4,2)=6 pairwise masks, reported ${q.pairs}`);
  check("aggregation-names-the-scheme", /pairwise masks/.test(q.aggregation),
    `qfl_aggregation reads ${JSON.stringify(q.aggregation)}, which is what the diagram renders`);
  check("single-client-does-not-claim-masking",
    m.clientQFL(0, { clients: 1, rounds: 1, sessionKey: KEY }).pairs === 0,
    "one client reported a nonzero pair count");
}

rmSync(dir, { recursive: true, force: true });

if (failures.length) {
  console.error(`check:qfl FAILED (${failures.length})`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check:qfl OK - pairwise masks are applied, keyed by the session key, fresh per round, " +
            "and cancel to a bit-identical weighted mean; there is no plaintext path");
