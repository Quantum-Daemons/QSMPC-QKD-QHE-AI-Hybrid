/**
 * Federated learning in the browser ("Quantum Federated Learning" in the stage names), and its
 * masked aggregation.
 *
 * NOTHING QUANTUM IS COMPUTED HERE. The "quantum feature map" is `qcnnExpectation`, the closed form
 * tanh(mean(sin(x_i + 0.5 i))) - no circuit and no state vector - and the model is a 4-weight
 * logistic head over three such features and a bias.
 *
 * SPLIT OUT OF `clientCompute.ts` so it can be exercised on its own. The property this module
 * provides - that the aggregator recovers the weighted mean from MASKED shares, the masks cancelling
 * exactly - is worth an executable check. Each pair of clients agrees its own mask seed by X25519
 * (`secaggPairSeed`), so the aggregator, which receives the public keys and the masked shares and
 * nothing else, cannot remove a mask on its own. What that does and does not give is stated at
 * `secureAggregate`. And `clientCompute` transitively reaches `clientFHE`, which imports the
 * SEAL wasm through a Vite-only `?url` specifier that no plain bundler resolves. Keeping federated
 * aggregation here means `npm run check:qfl` loads the real code rather than a re-implementation of
 * it. It is also a distinct concern from a 2,000-line stage dispatcher.
 */
import { x25519 } from "@noble/curves/ed25519.js";
import { shake256 } from "@noble/hashes/sha3.js";

import { federatedCfg } from "./config";
import { qcnnExpectation, rng } from "./quantumSim";

const r2 = (x: number) => Math.round(x * 100) / 100;

// A GENUINELY-computed, in-browser FedAvg of a classical model: a fixed feature map
// (`qcnnExpectation`, a closed-form function despite its name, not a simulated circuit) feeds a
// logistic head that is federated across K simulated clients with size-weighted FedAvg. The full
// deployment's PennyLane HybridQNN trains a real variational circuit; this does not.
// Round accuracy is REALLY computed on a held-out split (not a canned constant).
//
// This comment used to end "under quantum-safe secure aggregation", and the diagram said "pairwise
// masking (demo keys)". Neither was true of this function: it summed plaintext updates. `aggregation`
// replaces the `secureAgg: true` literal that stood here - a boolean hardcoded to the answer it was
// supposed to report is not a report. The stage publishes it as `qfl_aggregation` and the diagram
// renders that, so the figure cannot outrun the algorithm again.
export interface QFLResult {
  roundAcc: number[]; finalAcc: number; clients: number; rounds: number; aggregation: string;
  pairs: number;
  /** Held-out rows `finalAcc` was measured on. Reported because it is small enough to matter. */
  evalRows: number;
  /** What the same held-out rows score with untrained weights - the constant all-class-0 predictor. */
  baselineAcc: number;
  /** Measured wall time of this run's key agreement: every client's X25519 key pair and every
   *  client's seeds with its peers (`secaggKeyPairs` + `secaggClientSeeds`), in milliseconds. */
  keyAgreementMs: number;
  /** The clients' X25519 public keys for this run, hex: with the masked shares, all the aggregator
   *  receives. No private key and no seed is part of the result. */
  publicKeys: string[];
}

/** What the aggregator receives in one round: the clients' public keys, which it relays to every
 *  client before the first round, and each client's masked share. Nothing else reaches it. */
export interface AggregatorInput {
  publicKeys: readonly Uint8Array[];
  maskedShares: readonly number[][];
}

/**
 * Fixed-point ring for secure aggregation. NOT a rounding convenience - a correctness requirement.
 *
 * Pairwise masks cancel because +m and -m appear once each in the sum. Floating-point addition is
 * not associative, so in doubles they cancel only to a tolerance that depends on summation order,
 * and a check written against that tolerance is one that cannot fail. In a ring the cancellation is
 * exact and `secureAggregate` can be asserted bit-identical to a plaintext weighted sum.
 *
 * 2^20 gives ~1e-6 resolution per coordinate; the logistic head's weights are order 1, so the
 * quantisation is far below anything `r2()` reports. 2^32 is the ring, matching Uint32 semantics so
 * `>>> 0` IS the reduction rather than an approximation of it.
 */
const QFL_SCALE = 1 << 20;
const QFL_RING = 2 ** 32;
/**
 * Largest quantised per-client contribution that cannot overflow the signed half of the ring.
 *
 * It depends on K. Written as a flat 2^30 first, which is a guard that does not guard: with four
 * clients the sum could reach 4 * 2^30 = 2^32, twice the 2^31 the two's-complement recovery can
 * represent, so the bound admitted exactly the overflow it existed to reject. In practice the
 * logistic head's weights are order 1 and a share is around 2^20, nowhere near either number - which
 * is precisely why a wrong bound here would never have been noticed.
 */
const qflMaxShare = (clients: number) => 2 ** 31 / Math.max(1, clients);

// ------------------------------------------------------------------------------------------------
//  Pairwise key agreement (Bonawitz et al., CCS 2017)
// ------------------------------------------------------------------------------------------------
//
// THE CONSTRUCTION, byte for byte, so the full deployment can reproduce it (src/qsh/ai/federated.py,
// asserted equal in tests/test_r84_pairwise_masks.py):
//
//   key pair    each client u: a fresh X25519 private key per run (32 bytes from the CSPRNG) and
//               its public key, RFC 7748.
//   pair seed   for the pair i < j, both clients compute s = X25519(sk_i, pk_j) = X25519(sk_j, pk_i)
//               and seed_ij = HKDF-SHA256(IKM = s, salt = SECAGG_SALT,
//                                         info = SECAGG_PAIR_LABEL || u32be(i) || u32be(j), L = 32)
//               (RFC 5869). The salt is the 18 ASCII bytes "qsh/secagg/salt/v1", the label the 18
//               ASCII bytes "qsh/secagg/pair/v1", so the info is 26 bytes.
//   round mask  mask_ij(r) = the first 4P bytes of
//               SHAKE-256(seed_ij || SECAGG_MASK_LABEL || u32be(r)), read as P little-endian uint32
//               words; the label is the 18 ASCII bytes "qsh/secagg/mask/v1". A fresh mask every
//               round, from one seed per run.
//   share       client u adds mask_uv(r) for every v > u and subtracts mask_vu(r) for every v < u,
//               modulo 2^32, so every pair's mask appears once with each sign and the sum cancels.
//
// This is the masking of Bonawitz et al.'s Protocol 3 (arXiv:1611.04482, the workshop version of
// "Practical Secure Aggregation for Privacy-Preserving Machine Learning", ACM CCS 2017,
// doi:10.1145/3133956.3133982): each pair agrees a secret by Diffie-Hellman and expands it with a
// PRG, the lower-indexed user adding the expansion and the higher-indexed one subtracting it.
// X25519, HKDF-SHA256 and SHAKE-256 are this project's instantiation of their AGREE and PRG.
// The PQC session key is not an input to any of it.

const enc = new TextEncoder();
/** The fixed HKDF salt of the pair-seed derivation: the 18 ASCII bytes "qsh/secagg/salt/v1". */
export const SECAGG_SALT = enc.encode("qsh/secagg/salt/v1");
/** The HKDF info is this 18-byte label followed by the pair's two indices, i < j, each a 4-byte
 *  big-endian unsigned integer. */
export const SECAGG_PAIR_LABEL = enc.encode("qsh/secagg/pair/v1");
/** The per-round mask's domain label, between the seed and the round in SHAKE-256's input. */
export const SECAGG_MASK_LABEL = enc.encode("qsh/secagg/mask/v1");

const u32be = (n: number): Uint8Array => {
  if (!Number.isInteger(n) || n < 0 || n > 0xffffffff) throw new Error(`secagg: ${n} is not a uint32`);
  const b = new Uint8Array(4);
  new DataView(b.buffer).setUint32(0, n, false);
  return b;
};
const concat = (...parts: Uint8Array[]): Uint8Array => {
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
};
/** A copy in a plain ArrayBuffer: WebCrypto's BufferSource typing rejects a view over a shared or
 *  resizable buffer, and a copy is what `clientMcp.ts` hands it for the same reason. */
const ab = (u: Uint8Array): ArrayBuffer => { const b = new ArrayBuffer(u.byteLength); new Uint8Array(b).set(u); return b; };

/** The HKDF info of the pair (i, j): SECAGG_PAIR_LABEL || u32be(i) || u32be(j), with i < j. */
export function secaggPairInfo(i: number, j: number): Uint8Array {
  if (!(Number.isInteger(i) && Number.isInteger(j) && 0 <= i && i < j)) {
    throw new Error(`secagg: a pair is named by its two client indices in order, got (${i}, ${j})`);
  }
  return concat(SECAGG_PAIR_LABEL, u32be(i), u32be(j));
}

/** One client's X25519 key pair for one run. The private key never leaves the client's own code
 *  path: the aggregator is handed `publicKey` alone. */
export interface SecaggKeyPair {
  readonly privateKey: Uint8Array;
  readonly publicKey: Uint8Array;
}

/**
 * Every client's X25519 key pair for this run: fresh from the CSPRNG (`x25519.utils.randomSecretKey`
 * reads `crypto.getRandomValues`), or the given 32-byte private keys, which is what a test or a
 * reproduction passes. A caller that supplies keys takes responsibility for their secrecy: a fixed
 * key is a public one.
 */
export function secaggKeyPairs(K: number, fixedPrivateKeys?: readonly Uint8Array[]): SecaggKeyPair[] {
  if (fixedPrivateKeys && fixedPrivateKeys.length !== K) {
    throw new Error(`secagg: ${fixedPrivateKeys.length} private keys for ${K} clients`);
  }
  return Array.from({ length: K }, (_, u) => {
    const privateKey = fixedPrivateKeys ? Uint8Array.from(fixedPrivateKeys[u]) : x25519.utils.randomSecretKey();
    return { privateKey, publicKey: x25519.getPublicKey(privateKey) };
  });
}

/**
 * The seed of the pair (i, j), i < j, as ONE of its two clients computes it: from its own private
 * key and the other client's public key. The other client reaches the same 32 bytes from its own
 * private key, which is the whole point - nobody else can, including the aggregator that relayed
 * both public keys. HKDF runs on WebCrypto. A low-order public key is refused by noble's X25519
 * before the ladder runs, rather than yielding the all-zero secret RFC 7748 section 6.1 warns of.
 */
export async function secaggPairSeed(
  ownPrivateKey: Uint8Array, peerPublicKey: Uint8Array, i: number, j: number,
): Promise<Uint8Array> {
  const info = secaggPairInfo(i, j);
  const shared = x25519.getSharedSecret(ownPrivateKey, peerPublicKey);
  const ikm = await crypto.subtle.importKey("raw", ab(shared), "HKDF", false, ["deriveBits"]);
  shared.fill(0);
  const bits = await crypto.subtle.deriveBits(
    { name: "HKDF", hash: "SHA-256", salt: ab(SECAGG_SALT), info: ab(info) }, ikm, 256);
  return new Uint8Array(bits);
}

/** Client u's seeds with every other client (index v holds the seed of the pair {u, v}; u's own
 *  slot is null), from u's private key and the public keys the aggregator relays. */
export async function secaggClientSeeds(
  u: number, ownPrivateKey: Uint8Array, publicKeys: readonly Uint8Array[],
): Promise<(Uint8Array | null)[]> {
  return Promise.all(publicKeys.map((pk, v) =>
    (v === u ? null : secaggPairSeed(ownPrivateKey, pk, Math.min(u, v), Math.max(u, v)))));
}

/**
 * The mask of one pair at one round: P little-endian uint32 words of
 * SHAKE-256(seed || SECAGG_MASK_LABEL || u32be(round)).
 *
 * Deriving per round is what stops a mask from being reused across updates of the same weights:
 * with one mask for two rounds, differencing a client's two shares gives its update delta with no
 * key at all (tests/test_federated_mask_is_per_round.py states the attack). SHAKE-256 rather than
 * HKDF-Expand because the full deployment masks whole parameter vectors, and HKDF-SHA256 stops at
 * 255 x 32 = 8,160 bytes (RFC 5869 section 2.3); one derivation serves both tiers byte for byte.
 */
export function secaggMaskWords(seed: Uint8Array, round: number, P: number): number[] {
  if (P === 0) return [];
  const bytes = shake256(concat(seed, SECAGG_MASK_LABEL, u32be(round)), { dkLen: 4 * P });
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from({ length: P }, (_, k) => dv.getUint32(k * 4, true));
}

/**
 * One client's masked share for one round, computed with nothing but that client's own seeds.
 *
 * It weights its own update, quantises it into the ring, adds `mask_uv` for every peer above it and
 * subtracts `mask_vu` for every peer below. WEIGHT BEFORE MASKING, not after. Masks cancel under a
 * SUM. Applying the size weights to already-masked shares would scale each mask by a different
 * factor and they would not cancel - the weighted mean is recovered by having each client scale its
 * own contribution.
 */
/** The masks' fixed point, 2^20, for an aggregation beside them that must agree to the bit (Round 86). */
export const QFL_FIXED_POINT = QFL_SCALE;

/**
 * Client u's weighted update in the masks' fixed point, the integers its masked share hides. Split out of
 * `maskedShare` in Round 86 so the threshold-BGV aggregation (clientFedAgg.ts) encrypts exactly these integers:
 * its decrypted sum is then the masks' sum, bit for bit, which `npm run check:fed-agg` asserts.
 */
export function qflQuantise(u: number, update: readonly number[], weight: number, K: number): number[] {
  const maxShare = qflMaxShare(K);
  return update.map((x, k) => {
    const q = Math.round(x * weight * QFL_SCALE);
    if (!Number.isFinite(q) || Math.abs(q) > maxShare) {
      // Loud rather than silent: past this magnitude the ring wraps into the wrong half and the
      // recovered mean is garbage that still looks like a number.
      throw new Error(`QFL share out of ring range: client ${u}, coord ${k}, q=${q}`);
    }
    return q;
  });
}

export function maskedShare(
  u: number, update: number[], weight: number, ownSeeds: readonly (Uint8Array | null)[], round: number,
): number[] {
  return maskIntegers(u, qflQuantise(u, update, weight, ownSeeds.length), ownSeeds, round);
}

/**
 * Client u's integers (each within the ring's signed half, as qflQuantise keeps them), masked with its pair masks for
 * `round`. Split out of `maskedShare` in Round 86 so the record-level DP arm (clientFedHeart.ts) can mask integers
 * that already carry its noise share; `sumMaskedIntegers` is the aggregator's half. A pair's seed must not mask two
 * messages in the same round: the DP arm agrees fresh seeds for each of its settings.
 */
export function maskIntegers(
  u: number, q: readonly number[], ownSeeds: readonly (Uint8Array | null)[], round: number,
): number[] {
  const K = ownSeeds.length;
  const P = q.length;
  const maxShare = qflMaxShare(K);
  const share = q.map((v, k) => {
    if (!Number.isSafeInteger(v) || Math.abs(v) > maxShare) {
      throw new Error(`QFL share out of ring range: client ${u}, coord ${k}, q=${v}`);
    }
    return (v | 0) >>> 0;      // two's complement, i.e. reduction mod 2^32
  });
  for (let v = 0; v < K; v++) {
    if (v === u) continue;
    const seed = ownSeeds[v];
    if (!seed) throw new Error(`secagg: client ${u} has no seed with client ${v}; the round cannot cancel`);
    const m = secaggMaskWords(seed, round, P);
    for (let k = 0; k < P; k++) {
      share[k] = (v > u ? share[k] + m[k] : share[k] - m[k]) >>> 0;
    }
  }
  return share;
}

/** The aggregator's half for integers: the masked shares' sum mod 2^32, read through the signed half (Round 86). */
export function sumMaskedIntegers(maskedShares: readonly number[][]): number[] {
  const P = maskedShares[0]?.length ?? 0;
  const sum = new Array<number>(P).fill(0);
  for (let k = 0; k < P; k++) {
    let acc = 0;
    for (const share of maskedShares) acc = (acc + share[k]) >>> 0;
    sum[k] = acc >= QFL_RING / 2 ? acc - QFL_RING : acc;
  }
  return sum;
}

/**
 * The aggregator's whole computation: the sum of the masked shares modulo 2^32, mapped back through
 * the signed half. It is handed the masked shares and nothing else - no private key, no seed, no
 * session key - and the pair masks cancel in the sum, leaving the weighted mean.
 */
export function aggregateMaskedShares(maskedShares: readonly number[][]): number[] {
  return sumMaskedIntegers(maskedShares).map((s) => s / QFL_SCALE);
}

/**
 * Size-weighted FedAvg under pairwise additive masking, with each pair's mask seed agreed by that
 * pair alone (Bonawitz et al., CCS 2017; the construction is spelled out above).
 *
 * `pairSeeds[u]` is client u's row from `secaggClientSeeds`. Each client masks with its own row only
 * (`maskedShare`); the aggregator sums what it receives (`aggregateMaskedShares`); every pair's mask
 * cancels exactly in the ring and the weighted mean is left.
 *
 * WHAT THIS GIVES. An aggregator that follows the protocol, colludes with no client, and sees only
 * the public keys and the masked shares learns the sum and not an individual update: each share
 * carries K - 1 masks from seeds it cannot compute, so short of breaking X25519, HKDF or SHAKE-256 it
 * looks uniform over the ring apart from the sum. The PQC session key seeds no mask, so holding it
 * gives nothing here either.
 *
 * WHAT IT DOES NOT, stated so the masks are not read as more than they are:
 *   - No dropout recovery. Bonawitz et al. Shamir-share each user's key-agreement private key, so the
 *     server can remove a dropped user's masks and finish the round, and add a second, per-user mask,
 *     shared the same way, so that this recovery cannot expose a user who did not drop (Protocols 1
 *     to 3 of the arXiv version). Neither is implemented, so a client that drops leaves its masks
 *     uncancelled and breaks the round.
 *   - No malicious security. The full version (IACR ePrint 2017/281) proves its protocol in the
 *     malicious setting as well; this is the honest-but-curious case only.
 *   - No authentication of the public keys. The aggregator relays them, so one that substitutes its
 *     own keys can agree a seed with each client and unmask it - the man-in-the-middle the arXiv
 *     version's Protocol 4 names. That adversary is out of scope here.
 *   - Notional parties. The demo runs all K clients in one tab, so the tab holds every private key;
 *     the aggregator's code path (`aggregateMaskedShares`) never reads one, which is what the run
 *     demonstrates - not mutually distrusting hosts.
 *
 * What a complete version adds, and the citation for it: arXiv:2607.28191 (v1 2026-07-30, v2
 * 2026-09-07; to appear at HotDiSec 2026, an ESORICS workshop; no DOI or journal_ref as of
 * 2026-11-08) builds Bell et al.'s SecAgg+ on Flower for clinical EEG: pairwise masks over a sparse
 * neighbour graph plus a self-mask, Shamir-shared dropout recovery, local update clipping, and
 * semi-honest and malicious variants. That is more than this file's all-pairs masks, and not the same
 * protocol. It reports the semi-honest variant as the cheapest of its secure configurations, with
 * the malicious and notary variants adding computation, communication and round duration. Its runs
 * injected no client dropouts and no active server, so recovery and the malicious checks are built
 * there but not measured. It corroborates the scope note above rather than contradicting it: the
 * masking is the core, and dropout recovery is the next layer, not a correction to this one.
 */
export function secureAggregate(
  updates: number[][], weights: number[], pairSeeds: readonly (readonly (Uint8Array | null)[])[], round: number,
): { aggregate: number[]; maskedShares: number[][] } {
  if (pairSeeds.length !== updates.length) {
    throw new Error(`secagg: ${pairSeeds.length} clients' seeds for ${updates.length} updates`);
  }
  const maskedShares = updates.map((wl, u) => maskedShare(u, wl, weights[u], pairSeeds[u], round));
  return { aggregate: aggregateMaskedShares(maskedShares), maskedShares };
}

/**
 * One federated run: local training on each client's shard, then each round's size-weighted FedAvg
 * under pairwise masks whose seeds the clients agree among themselves at the start of the run.
 *
 * `privateKeys` fixes the clients' X25519 private keys (32 bytes each), for a test or a reproduction;
 * without it every client draws a fresh key pair. `onAggregatorInput` is shown, each round, exactly
 * what the aggregator receives. Neither changes a reported figure: the masks cancel exactly in the
 * ring, so the accuracies are those of the unmasked FedAvg whatever keys the clients hold.
 */
export async function clientQFL(
  seed = 0,
  opts?: { clients?: number; rounds?: number; data?: { X: number[][]; Y: number[] } | null;
           privateKeys?: readonly Uint8Array[];
           onAggregatorInput?: (round: number, input: AggregatorInput) => void },
): Promise<QFLResult> {
  const u = rng(seed + 11);
  // A DELIBERATE lighter schedule than the server's, now read from config rather than hardcoded.
  // Round 23b found the browser running 6 rounds at lr 0.15 while `config.federated` said 8 at 0.05,
  // with no way to reconcile them because FederatedConfig was not exported to the frontend at all.
  // Round 24 exports it, including the browser overrides and the reason for them, so the override is
  // a visible decision instead of two unrelated literals that happen to differ.
  const fed = federatedCfg();
  const K = opts?.clients ?? fed.hospitals;
  const rounds = opts?.rounds ?? fed.browser_rounds;
  const X: number[][] = [];
  const Y: number[] = [];
  if (opts?.data?.X?.length) {
    // Federate over the page's REAL bundled sample. Without this the function always built its own
    // two-cluster toy, so "federated accuracy" was the same number on every industry page for a
    // given seed -- a metric that described no dataset the page was showing.
    const src = opts.data;
    for (let i = 0; i < src.X.length; i++) { X.push(src.X[i].slice()); Y.push(src.Y[i]); }
  } else {
    const D = 4;
    const N = 240;
    for (let i = 0; i < N; i++) {
      const c = i % 2;                                 // two overlapping clusters (non-trivial -> a real curve)
      X.push(Array.from({ length: D }, () => (c ? 0.5 : -0.5) + (u() * 2 - 1) * 1.0));
      Y.push(c);
    }
  }
  const N = X.length;
  // feature map: `qcnnExpectation` (closed form, no circuit) on three deterministic views of x
  const phi = (x: number[]) => [
    qcnnExpectation(x).z,
    qcnnExpectation(x.map((v) => v + 0.7)).z,
    qcnnExpectation(x.map((v) => v * 1.3 - 0.4)).z,
    1,                                                 // bias
  ];
  const P = 4;
  const idx = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) { const j = Math.floor(u() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  // `fed.test_split`, not a literal. 0.75 stood here while `config.federated.test_split` was 0.2 -
  // so the two DISAGREED, and the config key was exported all the way to the frontend's type
  // (`FederatedConfig.test_split`) with no reader anywhere in `web/src`. Same defect as
  // `error_correction_eff` in the QKD model: a key that looks wired, a literal that runs, and the
  // only reason nobody noticed is that both values are plausible.
  const cut = Math.floor(N * (1 - fed.test_split));
  const tr = idx.slice(0, cut), te = idx.slice(cut);
  const parts: number[][] = Array.from({ length: K }, () => []);
  tr.forEach((id, i) => parts[i % K].push(id));        // shard the training set across K clients

  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
  const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
  const localTrain = (ids: number[], w0: number[], epochs = fed.local_epochs,
                      lr = fed.browser_lr) => {
    const w = [...w0];
    for (let e = 0; e < epochs; e++) for (const id of ids) {       // local SGD on this client's shard
      const p = phi(X[id]); const g = sigmoid(dot(w, p)) - Y[id];
      for (let k = 0; k < P; k++) w[k] -= lr * g * p[k];
    }
    return w;
  };
  const acc = (w: number[]) => {
    let c = 0;
    for (const id of te) if ((sigmoid(dot(w, phi(X[id]))) > 0.5 ? 1 : 0) === Y[id]) c++;
    return c / te.length;
  };
  // What the SAME held-out rows score before any training: w = 0 gives sigmoid(0) = 0.5, which is
  // not above 0.5, so every row is predicted 0 and this is the all-class-0 rate - not the majority
  // rate wherever class 1 is the majority of the held-out rows. Reported because "accuracy 0.74"
  // says nothing on its own about whether the federated rounds did anything, and the round-by-round
  // series cannot say either once the problem is well conditioned enough to converge in one round.
  // It is the `C` (constant predictor) discipline this project already applies to its other models.
  const baselineAcc = r2(acc(new Array(P).fill(0)));

  // The key agreement, once per run. Each client draws its key pair; the aggregator relays the
  // public keys; each client derives its seed with every peer from its OWN private key. No plaintext
  // branch: every round below goes through `secureAggregate`, because a fallback that aggregates in
  // the clear is how the diagram once came to claim masking that did not run.
  const t0 = performance.now();
  const keyPairs = secaggKeyPairs(K, opts?.privateKeys);
  const publicKeys = keyPairs.map((kp) => kp.publicKey);
  const pairSeeds = await Promise.all(keyPairs.map((kp, c) => secaggClientSeeds(c, kp.privateKey, publicKeys)));
  const keyAgreementMs = performance.now() - t0;
  let w = new Array(P).fill(0);
  const tot = tr.length;
  const sizeWeights = parts.map((ids) => ids.length / tot);
  const roundAcc: number[] = [];
  for (let r = 0; r < rounds; r++) {
    const updates = parts.map((ids) => localTrain(ids, w));        // each client trains locally
    const { aggregate, maskedShares } = secureAggregate(updates, sizeWeights, pairSeeds, r);
    opts?.onAggregatorInput?.(r, { publicKeys, maskedShares });
    w = aggregate;
    roundAcc.push(r2(acc(w)));
  }
  // MEASURED, not asserted: C(K,2) is the number of masks that actually had to cancel. With one
  // client there are none, and the string says so rather than claiming a property of an empty set.
  const pairs = (K * (K - 1)) / 2;
  // The DENOMINATOR travels with the accuracy. It was added when the bundled samples were 16 rows
  // (finance) and 12 (medical), so a score was over 4 or 3 held-out rows and "0.25" alone read as a
  // model worse than chance. The samples have since been enlarged to 400 rows each, so the held-out
  // split is 80 rows (Round 80 corrected this note, which still described the small samples); the
  // denominator stays, because a fraction without one says nothing about how much it rests on.
  return { roundAcc, finalAcc: roundAcc[roundAcc.length - 1], clients: K, rounds, pairs,
    evalRows: te.length, baselineAcc, keyAgreementMs,
    publicKeys: publicKeys.map((pk) => Array.from(pk, (b) => b.toString(16).padStart(2, "0")).join("")),
    aggregation: pairs > 0
      ? `size-weighted FedAvg + ${pairs} pairwise masks from X25519-agreed pair seeds, fixed-point mod 2^32`
      : "size-weighted FedAvg, 1 client - no pair to mask" };
}
