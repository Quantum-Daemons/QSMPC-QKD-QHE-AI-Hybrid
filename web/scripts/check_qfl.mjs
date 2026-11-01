#!/usr/bin/env node
/**
 * Assert that federated aggregation MASKS, that each pair's masks come from that pair's own X25519
 * secret, and that the masks cancel exactly.
 *
 * WHAT THIS IS FOR. `TwoPartyDiagram` drew "federated train + pairwise masking (demo keys)" and
 * `OrchestrationGraph` drew "(secure aggregation)" while `clientQFL` summed plaintext updates: no
 * masks, no pairwise seeds, no key material, and `secureAgg: true` written as a literal on the
 * return line. A figure asserting a property of code in another file is unfalsifiable by
 * construction, so the property is asserted here instead, on the function. Since Round 84 the masks
 * are no longer keyed by the PQC session key, which the aggregator holds: each pair of clients agrees
 * its seed by X25519 and HKDF-SHA256 (Bonawitz et al., CCS 2017), and the checks below say so.
 *
 * WHY A RING AND NOT A TOLERANCE. Pairwise masks cancel because +m and -m each appear once in the
 * sum. In doubles that cancellation depends on summation order, so it holds only to some epsilon -
 * and a check written against an epsilon is one that passes whether or not the masks are there. In
 * `Uint32` the cancellation is exact, so `masked-aggregate == plaintext-aggregate` can be asserted
 * as EQUALITY. That is the whole reason `secureAggregate` quantises.
 *
 * WHAT A PASSING RUN RULES OUT. Not just "the numbers look right": that the masks are applied at all
 * (a masked share must differ from its plaintext share), that each pair's mask is the expansion of
 * that pair's X25519 secret (recomputed here with node's own OpenSSL, not the module's noble and
 * WebCrypto), that the session key does not reach them, that a client's key pair reaches exactly the
 * pairs it is in, and that they are not reused across rounds. Each of those is a way the
 * aggregation could be plaintext-in-effect, or unmaskable by the aggregator, while every arithmetic
 * assertion still passed.
 *
 * Run: npm run check:qfl
 */
import { execFileSync } from "node:child_process";
import { createHash, createPrivateKey, createPublicKey, diffieHellman, hkdfSync } from "node:crypto";
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
export * from ${JSON.stringify(join(WEB, "src/lib/clientQFL"))};
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

// Fixed, obviously-not-secret private keys: this asserts plumbing and algebra, not key quality.
const fixedKeys = (salt) => Array.from({ length: 4 }, (_, c) =>
  Uint8Array.from({ length: 32 }, (_, i) => (i * salt + 3 * c + 1) & 0xff));
const KEYS = fixedKeys(7);
// Two stand-ins for a run's PQC session key, which must not reach the masks.
const SESSION_A = Uint8Array.from({ length: 32 }, (_, i) => (i * 7 + 3) & 0xff);
const SESSION_B = Uint8Array.from({ length: 32 }, (_, i) => (i * 11 + 5) & 0xff);

/** The clients' key pairs and each client's seeds with its peers, as `clientQFL` derives them. */
async function agree(keys) {
  const kps = m.secaggKeyPairs(keys.length, keys);
  const publicKeys = kps.map((kp) => kp.publicKey);
  const seeds = await Promise.all(kps.map((kp, u) => m.secaggClientSeeds(u, kp.privateKey, publicKeys)));
  return { publicKeys, seeds };
}
const A = await agree(KEYS);

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
const K = UPDATES.length;
const P = UPDATES[0].length;

// The aggregator's own definition, with the masks set aside: quantise each weighted contribution,
// sum in the ring, map back through the signed half. If the masks cancel, `secureAggregate` returns
// exactly this. Written here rather than exported from the module so the two are independent.
const SCALE = 1 << 20;
const RING = 2 ** 32;
const plainShare = (u) => UPDATES[u].map((v) => (Math.round(v * W[u] * SCALE) | 0) >>> 0);
function plainAggregate(updates, weights) {
  return Array.from({ length: updates[0].length }, (_, k) => {
    let acc = 0;
    for (let u = 0; u < updates.length; u++) {
      acc = (acc + ((Math.round(updates[u][k] * weights[u] * SCALE) | 0) >>> 0)) >>> 0;
    }
    return (acc >= RING / 2 ? acc - RING : acc) / SCALE;
  });
}

// The construction recomputed with node's OpenSSL: X25519 from a PKCS#8 private key, HKDF-SHA256 and
// SHAKE-256 from node:crypto. The labels are written out, not read from the module.
const enc = new TextEncoder();
const SALT = enc.encode("qsh/secagg/salt/v1");
const PAIR = enc.encode("qsh/secagg/pair/v1");
const MASK = enc.encode("qsh/secagg/mask/v1");
const u32be = (n) => { const b = Buffer.alloc(4); b.writeUInt32BE(n); return b; };
const PKCS8_X25519 = Buffer.from("302e020100300506032b656e04220420", "hex");
const osslPrivate = (sk) => createPrivateKey({ key: Buffer.concat([PKCS8_X25519, Buffer.from(sk)]), format: "der", type: "pkcs8" });
const osslPublicRaw = (sk) => Buffer.from(createPublicKey(osslPrivate(sk)).export({ format: "jwk" }).x, "base64url");
const osslPublic = (raw) => createPublicKey({ key: { kty: "OKP", crv: "X25519", x: Buffer.from(raw).toString("base64url") }, format: "jwk" });
function osslSeed(keys, i, j) {
  const shared = diffieHellman({ privateKey: osslPrivate(keys[i]), publicKey: osslPublic(osslPublicRaw(keys[j])) });
  return Buffer.from(hkdfSync("sha256", shared, SALT, Buffer.concat([PAIR, u32be(i), u32be(j)]), 32));
}
function osslMask(seed, round, n) {
  const b = createHash("shake256", { outputLength: 4 * n }).update(Buffer.concat([seed, MASK, u32be(round)])).digest();
  return Array.from({ length: n }, (_, k) => b.readUInt32LE(4 * k));
}
const sameBytes = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
const hex = (u) => Buffer.from(u).toString("hex");

// --------------------------------------------------------------------------------------------- //
//  1. The masks cancel EXACTLY
// --------------------------------------------------------------------------------------------- //
{
  const { aggregate } = m.secureAggregate(UPDATES, W, A.seeds, 0);
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
  const { maskedShares } = m.secureAggregate(UPDATES, W, A.seeds, 0);
  for (let u = 0; u < K; u++) {
    const plain = plainShare(u);
    check(`masked-share-differs-${u}`,
      maskedShares[u].some((v, k) => v !== plain[k]),
      `client ${u}'s transmitted share equals its plaintext share ${JSON.stringify(plain)} - ` +
      `no mask was added, so the aggregator sees the update in the clear`);
  }
  check("every-client-is-masked", maskedShares.length === K, `${maskedShares.length} shares for ${K} clients`);
}

// --------------------------------------------------------------------------------------------- //
//  3. Each pair's mask is the expansion of THAT PAIR's X25519 secret, through HKDF
// --------------------------------------------------------------------------------------------- //
// Recomputed with node's OpenSSL. A mask derived from anything the aggregator holds - the session
// key, a constant, the public keys - would satisfy sections 1 and 2 perfectly and fail here.
{
  check("public-keys-are-x25519", A.publicKeys.every((pk, c) => sameBytes(pk, osslPublicRaw(KEYS[c]))),
    "the relayed public keys are not the X25519 public keys of the clients' private keys");
  for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
    const want = osslSeed(KEYS, i, j);
    check(`pair-seed-${i}${j}-agreed-by-both`, sameBytes(A.seeds[i][j], A.seeds[j][i]),
      `clients ${i} and ${j} derived different seeds, so their masks cannot cancel`);
    check(`pair-seed-${i}${j}-is-hkdf-of-x25519`, sameBytes(A.seeds[i][j], want),
      `seed ${hex(A.seeds[i][j])} is not HKDF-SHA256(X25519(sk_${i}, pk_${j}), "qsh/secagg/salt/v1", ` +
      `"qsh/secagg/pair/v1"||${i}||${j}) = ${hex(want)}`);
    for (const r of [0, 1, 5]) {
      check(`mask-${i}${j}-r${r}-is-shake-of-the-seed`, sameBytes(m.secaggMaskWords(A.seeds[i][j], r, P), osslMask(want, r, P)),
        `the round-${r} mask of pair (${i}, ${j}) is not SHAKE-256(seed || "qsh/secagg/mask/v1" || ${r})`);
    }
  }
  // And the shares are exactly the plain shares plus those masks, signed by the pair's order.
  for (const r of [0, 3]) {
    const { maskedShares } = m.secureAggregate(UPDATES, W, A.seeds, r);
    for (let u = 0; u < K; u++) {
      const want = plainShare(u);
      for (let v = 0; v < K; v++) {
        if (v === u) continue;
        const mk = osslMask(osslSeed(KEYS, Math.min(u, v), Math.max(u, v)), r, P);
        for (let k = 0; k < P; k++) want[k] = (v > u ? want[k] + mk[k] : want[k] - mk[k]) >>> 0;
      }
      check(`share-${u}-r${r}-is-plain-plus-pair-masks`, sameBytes(maskedShares[u], want),
        `client ${u}'s round-${r} share ${JSON.stringify(maskedShares[u])} is not its plain share plus its ` +
        `pairs' X25519 masks ${JSON.stringify(want)}`);
    }
  }
}

// --------------------------------------------------------------------------------------------- //
//  4. The session key does not reach the masks
// --------------------------------------------------------------------------------------------- //
// What the aggregator receives under two different session keys and the same client key pairs. The
// option is passed and must be ignored: nothing in the derivation reads it.
async function aggregatorView(keys, extra) {
  const seen = [];
  await m.clientQFL(0, { clients: keys.length, rounds: 2, privateKeys: keys, ...extra,
    onAggregatorInput: (r, v) => seen.push({ r, publicKeys: v.publicKeys.map(hex), shares: v.maskedShares }) });
  return seen;
}
{
  const a = await aggregatorView(KEYS, { sessionKey: SESSION_A });
  const b = await aggregatorView(KEYS, { sessionKey: SESSION_B });
  check("aggregator-sees-every-round", a.length === 2 && b.length === 2,
    `clientQFL showed the aggregator ${a.length} and ${b.length} rounds of 2`);
  check("session-key-does-not-reach-the-mask", JSON.stringify(a) === JSON.stringify(b),
    "changing the session key changed a share: the masks are keyed by a key the aggregator holds");
  check("aggregator-receives-public-keys-only",
    a.every((s) => s.publicKeys.every((pk, c) => pk === hex(A.publicKeys[c]) && pk !== hex(KEYS[c]))),
    "the aggregator was handed something other than the clients' X25519 public keys");
}

// --------------------------------------------------------------------------------------------- //
//  5. One client's key pair reaches exactly the pairs it is in
// --------------------------------------------------------------------------------------------- //
{
  const C = 2;
  const keysB = KEYS.map((k, c) => (c === C ? Uint8Array.from(k, (x) => x ^ 0x5a) : k));
  const B = await agree(keysB);
  for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
    const inPair = i === C || j === C;
    const moved = !sameBytes(A.seeds[i][j], B.seeds[i][j]);
    check(`rekey-${C}-pair-${i}${j}`, moved === inPair,
      `replacing client ${C}'s key pair ${moved ? "moved" : "left"} the seed of pair (${i}, ${j}), ` +
      `which ${inPair ? "contains" : "does not contain"} it`);
  }
  // On the shares: every difference is the change in the masks of the pairs containing C, and
  // nothing else; client C's share moves through all of its pairs, every other share through one.
  const r = 1;
  const sa = m.secureAggregate(UPDATES, W, A.seeds, r).maskedShares;
  const sb = m.secureAggregate(UPDATES, W, B.seeds, r).maskedShares;
  for (let u = 0; u < K; u++) {
    const want = Array.from({ length: P }, () => 0);
    for (let v = 0; v < K; v++) {
      if (v === u || (u !== C && v !== C)) continue;
      const lo = Math.min(u, v), hi = Math.max(u, v);
      const ma = m.secaggMaskWords(A.seeds[lo][hi], r, P), mb = m.secaggMaskWords(B.seeds[lo][hi], r, P);
      for (let k = 0; k < P; k++) {
        const d = (mb[k] - ma[k]) >>> 0;
        want[k] = (v > u ? want[k] + d : want[k] - d) >>> 0;
      }
    }
    const got = sb[u].map((x, k) => (x - sa[u][k]) >>> 0);
    check(`rekey-${C}-share-${u}`, sameBytes(got, want),
      `client ${u}'s share moved by ${JSON.stringify(got)}, not by the change in its pairs with client ${C} ` +
      `${JSON.stringify(want)}`);
  }
  const ref = plainAggregate(UPDATES, W);
  check("rekey-still-cancels", m.secureAggregate(UPDATES, W, B.seeds, r).aggregate.every((v, k) => Object.is(v, ref[k])),
    "after one client's new key pair the masks no longer cancel");
}

// --------------------------------------------------------------------------------------------- //
//  6. The masks are fresh per round, and cancel in every round
// --------------------------------------------------------------------------------------------- //
{
  const a = m.secureAggregate(UPDATES, W, A.seeds, 0).maskedShares;
  const b = m.secureAggregate(UPDATES, W, A.seeds, 0).maskedShares;
  check("shares-are-deterministic", a.every((s, u) => s.every((v, k) => v === b[u][k])),
    "two identical calls produced different shares; the derivation is deterministic, so something " +
    "non-derived entered the mask");
  const round1 = m.secureAggregate(UPDATES, W, A.seeds, 1).maskedShares;
  check("masks-are-not-reused-across-rounds", a.every((s, u) => s.some((v, k) => v !== round1[u][k])),
    "round 0 and round 1 produced an identical share; reusing a pad across two updates of the same " +
    "weights leaks their difference");
  const reference = plainAggregate(UPDATES, W);
  for (const r of [1, 2, 7]) {
    const agg = m.secureAggregate(UPDATES, W, A.seeds, r).aggregate;
    check(`masks-cancel-in-round-${r}`, agg.every((v, k) => Object.is(v, reference[k])),
      `aggregate in round ${r} is ${JSON.stringify(agg)}, not ${JSON.stringify(reference)}`);
  }
}

// --------------------------------------------------------------------------------------------- //
//  6b. The ring bound rejects what would overflow it, and admits what fits
// --------------------------------------------------------------------------------------------- //
// The bound depends on the client count: with K clients the SUM has to stay inside the signed half,
// so each share is capped at 2^31/K. It was written as a flat 2^30, which four clients could exceed
// in aggregate - a guard that admitted the exact overflow it existed to reject. Real weights are
// order 1 and never approach either number, which is why only an assertion finds this.
{
  const justOver = (2 ** 31 / K) / SCALE * 1.01;      // one per cent past the per-client cap
  const big = UPDATES.map((row, u) => (u === 0 ? [justOver, ...row.slice(1)] : row));
  let threw = false;
  try { m.secureAggregate(big, W.map(() => 1), A.seeds, 0); } catch { threw = true; }
  check("ring-bound-rejects-an-overflowing-share", threw,
    `a share of ${justOver * SCALE} quantised units was accepted; with ${K} clients anything past ` +
    `2^31/${K} can push the sum out of the signed half, where the recovered mean is a wrong number ` +
    `that still looks like a number`);

  // And it must not reject ordinary work: a bound set too tight is a different way to be broken.
  let ok = true;
  try { m.secureAggregate(UPDATES, W, A.seeds, 0); } catch { ok = false; }
  check("ring-bound-admits-real-updates", ok, "the ordinary four-client update was rejected by the range guard");
}

// --------------------------------------------------------------------------------------------- //
//  7. There is no plaintext mode to fall back into
// --------------------------------------------------------------------------------------------- //
// How the original claim survived: written once, and the path that would have contradicted it was
// the one always taken. There is no key to be missing any more, so the two ways back to the clear are
// a pair seed that is skipped rather than refused, and a run whose shares are not masked at all.
{
  const holed = A.seeds.map((row, u) => row.map((s, v) => ((u === 0 && v === 1) ? null : s)));
  let threw = false;
  try { m.secureAggregate(UPDATES, W, holed, 0); } catch { threw = true; }
  check("a-missing-pair-seed-is-refused", threw,
    "a client with no seed for one peer aggregated anyway: skipping that mask leaves the pair " +
    "uncancelled, or with both members skipping it, the pair in the clear");
  // A plain share of the logistic head is a small signed number (|w| * size weight * 2^20); a masked
  // one is uniform over the ring. Every client's share must carry a coordinate far from zero.
  const seen = await aggregatorView(KEYS, {});
  const far = (x) => Math.min(x, RING - x) > 2 ** 26;
  check("clientQFL-sends-masked-shares", seen.every((s) => s.shares.every((sh) => sh.some(far))),
    "clientQFL handed the aggregator a share whose every coordinate is a small number: plaintext");
}

// --------------------------------------------------------------------------------------------- //
//  8. The pair count is measured, and fresh keys are drawn per run
// --------------------------------------------------------------------------------------------- //
{
  const q = await m.clientQFL(0, { clients: 4, rounds: 6 });
  const q2 = await m.clientQFL(0, { clients: 4, rounds: 6 });
  check("pair-count-is-C(K,2)", q.pairs === 6, `4 clients need C(4,2)=6 pairwise masks, reported ${q.pairs}`);
  check("aggregation-names-the-scheme", /pairwise masks from X25519-agreed pair seeds/.test(q.aggregation),
    `qfl_aggregation reads ${JSON.stringify(q.aggregation)}, which is what the diagram renders`);
  check("fresh-key-pairs-per-run", q.publicKeys.length === 4 && q.publicKeys.every((pk) => !q2.publicKeys.includes(pk)),
    "two runs drew a common public key: the key pairs are not fresh per run");
  check("figures-do-not-depend-on-the-keys", JSON.stringify(q.roundAcc) === JSON.stringify(q2.roundAcc),
    `two runs with different keys reported ${JSON.stringify(q.roundAcc)} and ${JSON.stringify(q2.roundAcc)}`);
  check("single-client-does-not-claim-masking",
    (await m.clientQFL(0, { clients: 1, rounds: 1 })).pairs === 0,
    "one client reported a nonzero pair count");
}

rmSync(dir, { recursive: true, force: true });

if (failures.length) {
  console.error(`check:qfl FAILED (${failures.length})`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check:qfl OK - pairwise masks are applied, keyed by each pair's X25519 secret through HKDF, " +
            "untouched by the session key, fresh per round, and cancel to a bit-identical weighted mean; " +
            "there is no plaintext path");
