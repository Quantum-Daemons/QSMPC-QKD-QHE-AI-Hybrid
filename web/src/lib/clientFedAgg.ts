/**
 * The four-hospital federation's aggregation, five ways side by side, and its DP arm under threshold BGV (Round 86,
 * the owner's choices of 2026-11-13).
 *
 * The medical page's federation (clientFedHeart.ts) aggregates under pairwise masks. This module reruns the same
 * federation - the same initial model, the same local SGD streams, the same 15 rounds - and changes only how each
 * round's size-weighted FedAvg is aggregated:
 *
 *   masks            the page's own: X25519-agreed pairwise masks in the ring mod 2^32, at the fixed point 2^20.
 *   ckks-shared-key  NVIDIA FLARE's design (its TenSEAL settings: CKKS, N 8192, moduli [60, 40, 40], scale 2^40):
 *                    every hospital holds the one secret key, the aggregator adds ciphertexts. Here on node-seal.
 *   ckks-demo        threshold CKKS on Lattigo v6.2.0 (wasm/qsh-fedthresh): no party holds the key; each round's sum
 *                    is decrypted once by all four hospitals' shares, each flooded with noise of sigma 2^30, the value
 *                    Lattigo's multiparty examples use. That is about 16 bits of statistical security: a demonstration.
 *   ckks-l40         the same with 40 bits of statistical security (sigma 2^54, scale 2^80).
 *   bgv-128          threshold BGV, exact, with 128 bits; it encrypts the masks' own fixed-point integers, so its
 *                    decrypted sum is the masks' sum bit for bit, and so is the model it trains.
 *   bgv-128-dp       a sixth row, not FedAvg: the page's record-level DP arm (clientFedHeart.trainFedHeartDp) at
 *                    epsilon 1, run once more with each step's four noisy vectors summed under threshold BGV, the
 *                    noise inside the encryption, and under the masks beside it, so every step's two sums are compared.
 *
 * What a design changes is who can read one hospital's update, what the aggregator sees, the bytes a hospital sends,
 * the time, and how far each round's aggregate is from the exact FedAvg of that round's updates in float64. Every
 * party runs in this tab, so every separation here is notional; nothing below authenticates a party or survives a
 * dropout. The Lattigo module is about 3 MB compressed and the run takes seconds, so the page runs this in a Web
 * Worker, and only when asked (clientFedAgg.worker.ts).
 */
import sealWasmUrl from "/node_modules/node-seal/dist/seal_throws.wasm?url";
import { cryptoRandBelow, epsilonOf, rhoOf, varianceFor } from "./clientDp";
import { accuracy, FED_HEART, FED_HEART_DP, type HeartSites, initTheta, trainFedHeartDp, trainSgdSteps } from "./clientFedHeart";
import { maskIntegers, QFL_FIXED_POINT, qflQuantise, secaggClientSeeds, secaggKeyPairs, secureAggregate,
         sumMaskedIntegers } from "./clientQFL";
import { rng } from "./quantumSim";

/** globalThis.qshFedThresh, as wasm/qsh-fedthresh/cmd/wasm installs it. */
export interface FedThreshApi {
  settings(): string;
  open(kind: string, parties: number, crs: Uint8Array): { id?: number; setting?: string; keyShareBytes?: number;
    keyGenMs?: number; error?: string };
  round(id: number, flat: Uint8Array, parties: number, n: number): { aggregate?: Uint8Array; stats?: string;
    error?: string };
  close(id: number): void;
}

export type AggregationKind = "masks" | "ckks-shared-key" | "ckks-demo" | "ckks-l40" | "bgv-128" | "bgv-128-dp";

/** The DP row's target: the strongest setting in the record-level arm's table. */
export const DP_ROW_EPSILON = 1;

export interface AggregationDesign {
  kind: AggregationKind;
  label: string;
  scheme: string;
  /** Who can read one hospital's update. */
  whoDecrypts: string;
  aggregatorSees: string;
  /** log2(smudging sigma / the aggregate's noise bound), for a threshold decryption; null where there is none. */
  statSecBits: number | null;
  federated: number[];
  mean: number;
  /** Largest |aggregate - exact float64 FedAvg of the same round's updates|, over every round and coordinate; null
   *  for the DP row, which is not FedAvg. */
  maxDiffFromExact: number | null;
  /** Every round's aggregate equal to the masks' run's, to the bit (null for the masks themselves); for the DP row,
   *  every step's sum under BGV equal to the masks' sum of the same noisy vectors. */
  identicalToMasks: boolean | null;
  /** The DP row only. */
  dp?: { epsilon: number; epsilonAchieved: number; delta: number; rho: number; stepsEqual: number; steps: number };
  /** What one hospital sends in one round: its masked share, or its ciphertext (and decryption share). */
  bytesPerHospitalPerRound: number;
  keySetupMs: number;
  msPerRound: number;
}

export interface AggregationComparison {
  seed: number; rounds: number; hospitals: number; designs: AggregationDesign[]; ms: number;
}

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
const asBytes = (f: Float64Array) => new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
const asF64 = (u: Uint8Array) => new Float64Array(u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength));

type Aggregator = (updates: number[][], weights: number[], round: number) => Promise<number[]> | number[];

/** The federation of clientFedHeart.ts with its aggregation step swapped; the rest is its federated arm exactly. */
async function federate(sites: HeartSites, seed: number, aggregate: Aggregator) {
  const cs = sites.centres;
  const tests = cs.map((c) => c.test);
  const sizes = cs.map((c) => c.train.y.length);
  const total = sizes.reduce((a, b) => a + b, 0);
  const weights = sizes.map((n) => n / total);
  const streams = cs.map((_, k) => rng(seed * 1000 + 200 + k));
  let glob = initTheta(seed, sites.feature_names.length);
  let maxDiff = 0;
  const aggregates: number[][] = [];
  const t0 = performance.now();
  for (let r = 0; r < FED_HEART.rounds; r++) {
    const updates = cs.map((c, k) => trainSgdSteps(glob, c.train.X, c.train.y, streams[k]));
    const exact = updates[0].map((_, i) => updates.reduce((a, u, k) => a + weights[k] * u[i], 0));
    const agg = await aggregate(updates, weights, r);
    for (let i = 0; i < agg.length; i++) maxDiff = Math.max(maxDiff, Math.abs(agg[i] - exact[i]));
    aggregates.push(agg);
    glob = agg;
  }
  const federated = tests.map((t) => accuracy(glob, t.X, t.y));
  return { federated, mean: mean(federated), maxDiff, aggregates, ms: performance.now() - t0 };
}

/** NVIDIA FLARE's homomorphic aggregation, on node-seal: one key pair, held by every hospital. */
async function sharedKeyCkks() {
  const SEAL = (await import("node-seal/throws")).default;
  const seal = await SEAL({ locateFile: (p: string) => (p.endsWith(".wasm") ? sealWasmUrl : p) });
  const t0 = performance.now();
  const parms = new seal.EncryptionParameters(seal.SchemeType.ckks);
  parms.setPolyModulusDegree(8192);
  parms.setCoeffModulus(seal.CoeffModulus.Create(8192, Int32Array.from([60, 40, 40])));
  const context = new seal.SEALContext(parms, true, seal.SecLevelType.tc128);
  const keyGen = new seal.KeyGenerator(context);
  const encryptor = new seal.Encryptor(context, keyGen.createPublicKey());
  const decryptor = new seal.Decryptor(context, keyGen.secretKey());
  const encoder = new seal.CKKSEncoder(context);
  const evaluator = new seal.Evaluator(context);
  const keySetupMs = performance.now() - t0;
  let ctBytes = 0;
  const aggregate: Aggregator = (updates, weights) => {
    // Every object a round makes is freed when it returns: the WASM heap is not a pressure the JS collector sees.
    const live: { delete(): void }[] = [];
    const hold = <T extends { delete(): void }>(o: T): T => { live.push(o); return o; };
    try {
      const cts = updates.map((u, k) => {
        const v = Float64Array.from(u, (x) => x * weights[k]);
        if (!v.every(Number.isFinite)) throw new Error(`CKKS aggregation: hospital ${k}'s update is not finite`);
        const pt = hold(new seal.Plaintext());
        encoder.encode(v, 2 ** 40, pt);
        const ct = hold(new seal.Ciphertext());
        encryptor.encrypt(pt, ct);
        return ct;
      });
      ctBytes = cts[0].saveToArray(seal.ComprModeType.none).length;
      const sum = hold(new seal.Ciphertext());
      evaluator.add(cts[0], cts[1], sum);
      for (const ct of cts.slice(2)) evaluator.add(sum, ct, sum);
      const out = hold(new seal.Plaintext());
      decryptor.decrypt(sum, out);   // by a hospital: the aggregator holds no secret key
      return Array.from(encoder.decodeFloat64(out).slice(0, updates[0].length));
    } finally {
      for (const o of live) o.delete();
    }
  };
  const close = () => { for (const o of [encryptor, decryptor, encoder, evaluator, keyGen, context, parms]) o.delete(); };
  return { aggregate, keySetupMs, bytes: () => ctBytes, close };
}

/** One Lattigo threshold setting, through the wasm module. */
function threshold(api: FedThreshApi, kind: "ckks-demo" | "ckks-l40" | "bgv-128", K: number, crs: Uint8Array) {
  const o = api.open(kind, K, crs);
  if (o.error || o.id == null) throw new Error(`qsh-fedthresh: ${o.error ?? "no session"}`);
  const id = o.id;
  const setting = JSON.parse(o.setting ?? "{}");
  let bytes = 0;
  const aggregate: Aggregator = (updates, weights) => {
    const n = updates[0].length;
    const flat = new Float64Array(K * n);
    updates.forEach((u, k) => {
      // BGV takes the masks' own integers; CKKS the weighted floats.
      const v = kind === "bgv-128" ? qflQuantise(k, u, weights[k], K) : u.map((x) => x * weights[k]);
      flat.set(v, k * n);
    });
    const res = api.round(id, asBytes(flat), K, n);
    if (res.error || !res.aggregate) throw new Error(`qsh-fedthresh round: ${res.error ?? "no aggregate"}`);
    const stats = JSON.parse(res.stats ?? "{}");
    bytes = (stats.ciphertextBytes ?? 0) + (stats.decryptShareBytes ?? 0);
    const out = Array.from(asF64(res.aggregate));
    return kind === "bgv-128" ? out.map((x) => x / QFL_FIXED_POINT) : out;
  };
  return { aggregate, keySetupMs: o.keyGenMs ?? 0, statSecBits: Number(setting.statSecBits), bytes: () => bytes,
           close: () => api.close(id) };
}

const TEXT: Record<AggregationKind, Pick<AggregationDesign, "label" | "scheme" | "whoDecrypts" | "aggregatorSees">> = {
  masks: {
    label: "Pairwise masks (the page's default)",
    scheme: "X25519 pair seeds, HKDF-SHA256 masks, mod 2^32 at 2^20",
    whoDecrypts: "nobody: a masked share hides its update until all four are summed",
    aggregatorSees: "masked shares; their sum is the FedAvg in the clear",
  },
  "ckks-shared-key": {
    label: "CKKS with one shared key (NVIDIA FLARE's design)",
    scheme: "node-seal 7.0.0, N 8192, moduli [60, 40, 40], scale 2^40",
    whoDecrypts: "every hospital: all four hold the one secret key",
    aggregatorSees: "ciphertexts only; a hospital decrypts the sum, and the aggregator with any one hospital reads every update",
  },
  "ckks-demo": {
    label: "Threshold CKKS, demonstration smudging",
    scheme: "Lattigo v6.2.0, N 8192, Q 2x60 bits, scale 2^60, sigma 2^30",
    whoDecrypts: "no single party: all four hospitals' decryption shares are needed",
    aggregatorSees: "ciphertexts, then four decryption shares, from which it combines the sum",
  },
  "ckks-l40": {
    label: "Threshold CKKS, 40-bit smudging",
    scheme: "Lattigo v6.2.0, N 8192, Q 2x60 bits, scale 2^80, sigma 2^54",
    whoDecrypts: "no single party: all four hospitals' decryption shares are needed",
    aggregatorSees: "ciphertexts, then four decryption shares, from which it combines the sum",
  },
  "bgv-128": {
    label: "Threshold BGV, 128-bit smudging, exact",
    scheme: "Lattigo v6.2.0, N 8192, Q 3x60 bits, t a 33-bit prime, sigma 2^142",
    whoDecrypts: "no single party: all four hospitals' decryption shares are needed",
    aggregatorSees: "ciphertexts, then four decryption shares, from which it combines the sum",
  },
  "bgv-128-dp": {
    label: `Threshold BGV with DP noise inside, epsilon ${DP_ROW_EPSILON}`,
    scheme: "the record-level DP arm's 50 clipped full-batch steps; Lattigo v6.2.0 as the row above",
    whoDecrypts: "no single party, and the sum the four shares decrypt already carries the DP noise",
    aggregatorSees: "ciphertexts, then four decryption shares, from which it combines a noisy sum",
  },
};

/**
 * The DP row: one more record-level DP run at DP_ROW_EPSILON, from this tab's CSPRNG, whose every step is summed under
 * threshold BGV - the model trains on those sums - and, from the same noisy vectors, under fresh pairwise masks.
 */
async function dpUnderThresholdBgv(sites: HeartSites, seed: number, api: FedThreshApi, crs: Uint8Array): Promise<AggregationDesign> {
  const { rounds, clip, fixedPointBits, delta } = FED_HEART_DP;
  const K = sites.centres.length;
  const dim = sites.feature_names.length + 1;
  const sensitivity = clip * 2 ** fixedPointBits + Math.sqrt(dim);
  const V = varianceFor(DP_ROW_EPSILON, rounds, sensitivity, K, dim, delta);
  const tk = performance.now();
  const keyPairs = secaggKeyPairs(K);
  const pairSeeds = await Promise.all(keyPairs.map((kp, c) => secaggClientSeeds(c, kp.privateKey,
    keyPairs.map((p) => p.publicKey))));
  const o = api.open("bgv-128", K, crs);
  if (o.error || o.id == null) throw new Error(`qsh-fedthresh: ${o.error ?? "no session"}`);
  const id = o.id;
  const keySetupMs = performance.now() - tk;
  let stepsEqual = 0, bytes = 0;
  const t0 = performance.now();
  let theta: number[];
  try {
    theta = await trainFedHeartDp(sites, seed, V, cryptoRandBelow, (noisy, r) => {
      const masked = sumMaskedIntegers(noisy.map((q, k) => maskIntegers(k, q, pairSeeds[k], r)));
      const flat = new Float64Array(K * dim);
      noisy.forEach((q, k) => flat.set(q, k * dim));
      const res = api.round(id, asBytes(flat), K, dim);
      if (res.error || !res.aggregate) throw new Error(`qsh-fedthresh round: ${res.error ?? "no aggregate"}`);
      const stats = JSON.parse(res.stats ?? "{}");
      bytes = (stats.ciphertextBytes ?? 0) + (stats.decryptShareBytes ?? 0);
      const sum = Array.from(asF64(res.aggregate));
      if (sum.every((x, j) => x === masked[j])) stepsEqual++;
      return sum;
    });
  } finally {
    api.close(id);
  }
  const msPerRound = (performance.now() - t0) / rounds;
  const federated = sites.centres.map((c) => accuracy(theta, c.test.X, c.test.y));
  return {
    kind: "bgv-128-dp", ...TEXT["bgv-128-dp"], statSecBits: 128, federated, mean: mean(federated), maxDiffFromExact: null,
    identicalToMasks: stepsEqual === rounds, bytesPerHospitalPerRound: bytes, keySetupMs, msPerRound,
    dp: { epsilon: DP_ROW_EPSILON, epsilonAchieved: epsilonOf(V, rounds, sensitivity, K, dim, delta), delta,
          rho: rhoOf(V, rounds, sensitivity, K, dim), stepsEqual, steps: rounds },
  };
}

/**
 * Runs the federation once per design and reports each. `fedThresh` is the loaded wasm module; the page's worker
 * loads it from /wasm/qsh_fedthresh, and `npm run check:fed-agg` from the files scripts/build_wasm.sh exports.
 */
export async function runAggregationComparison(
  sites: HeartSites, seed: number, fedThresh: FedThreshApi,
  onProgress?: (kind: AggregationKind, done: number, of: number) => void,
): Promise<AggregationComparison> {
  const t0 = performance.now();
  const K = sites.centres.length;
  const order: Exclude<AggregationKind, "bgv-128-dp">[] = ["masks", "ckks-shared-key", "ckks-demo", "ckks-l40", "bgv-128"];
  const designs: AggregationDesign[] = [];
  let masksAggregates: number[][] = [];
  // The crs is public: every hospital derives the same common reference polynomials from it.
  const crs = new TextEncoder().encode(`qsh/fedthresh/crs/v1/seed-${seed}`);
  for (const [i, kind] of order.entries()) {
    onProgress?.(kind, i, order.length + 1);
    let agg: Aggregator, keySetupMs = 0, statSecBits: number | null = null, bytes = () => 0, close = () => {};
    if (kind === "masks") {
      const tk = performance.now();
      const keyPairs = secaggKeyPairs(K);
      const pairSeeds = await Promise.all(keyPairs.map((kp, c) => secaggClientSeeds(c, kp.privateKey,
        keyPairs.map((p) => p.publicKey))));
      keySetupMs = performance.now() - tk;
      agg = (updates, weights, r) => secureAggregate(updates, weights, pairSeeds, r).aggregate;
      bytes = () => sites.feature_names.length * 4 + 4;   // one 32-bit word per coordinate, the bias included
    } else if (kind === "ckks-shared-key") {
      const s = await sharedKeyCkks();
      ({ aggregate: agg, keySetupMs, bytes, close } = s);
    } else {
      const s = threshold(fedThresh, kind, K, crs);
      ({ aggregate: agg, keySetupMs, statSecBits, bytes, close } = s);
    }
    const r = await federate(sites, seed, agg);
    close();
    if (kind === "masks") masksAggregates = r.aggregates;
    designs.push({
      kind, ...TEXT[kind], statSecBits, federated: r.federated, mean: r.mean, maxDiffFromExact: r.maxDiff,
      identicalToMasks: kind === "masks" ? null
        : r.aggregates.every((a, j) => a.every((x, c) => x === masksAggregates[j][c])),
      bytesPerHospitalPerRound: bytes(), keySetupMs, msPerRound: r.ms / FED_HEART.rounds,
    });
  }
  onProgress?.("bgv-128-dp", order.length, order.length + 1);
  designs.push(await dpUnderThresholdBgv(sites, seed, fedThresh, crs));
  onProgress?.("bgv-128-dp", order.length + 1, order.length + 1);
  return { seed, rounds: FED_HEART.rounds, hospitals: K, designs, ms: performance.now() - t0 };
}

/** One sum of integers, one per party, under threshold BGV: what the genomics page's button reports. */
export interface ThresholdSum {
  sum: number; parties: number; statSecBits: number; plaintextModulus: number;
  keyGenMs: number; roundMs: number; ciphertextBytes: number; decryptShareBytes: number;
}

/**
 * The genomics page's noisy cohort contributions summed under threshold BGV (Round 86, the owner's choice of
 * 2026-11-13): each cohort encrypts its count in the fixed point 1/16 plus its share of the DP noise under the cohorts'
 * collective key, the sum is decrypted once by all of their shares, and the result is the same integer the masked
 * secure sum gave. A fresh key per call, so no cohort ever makes a second decryption share for a ciphertext.
 */
export function sumUnderThresholdBgv(api: FedThreshApi, values: readonly number[], crs: Uint8Array): ThresholdSum {
  const K = values.length;
  if (K < 2 || !values.every(Number.isSafeInteger)) throw new Error("threshold BGV sums two or more whole numbers");
  const o = api.open("bgv-128", K, crs);
  if (o.error || o.id == null) throw new Error(`qsh-fedthresh: ${o.error ?? "no session"}`);
  const id = o.id;
  try {
    const setting = JSON.parse(o.setting ?? "{}");
    const t = Number(setting.plaintextModulus);
    if (!(values.reduce((a, v) => a + Math.abs(v), 0) < t / 2)) throw new Error(`the sum could wrap the plaintext modulus ${t}`);
    const t0 = performance.now();
    const res = api.round(id, asBytes(Float64Array.from(values)), K, 1);
    const roundMs = performance.now() - t0;
    if (res.error || !res.aggregate) throw new Error(`qsh-fedthresh round: ${res.error ?? "no aggregate"}`);
    const stats = JSON.parse(res.stats ?? "{}");
    return { sum: asF64(res.aggregate)[0], parties: K, statSecBits: Number(setting.statSecBits), plaintextModulus: t,
             keyGenMs: o.keyGenMs ?? 0, roundMs, ciphertextBytes: stats.ciphertextBytes ?? 0,
             decryptShareBytes: stats.decryptShareBytes ?? 0 };
  } finally {
    api.close(id);
  }
}

/** Loads Go's wasm_exec.js and the module from `base` (a URL prefix ending in "/") into this realm. */
export async function loadFedThresh(base: string): Promise<FedThreshApi> {
  const g = globalThis as unknown as { Go?: new () => { importObject: WebAssembly.Imports; run(i: WebAssembly.Instance): void };
                                       qshFedThresh?: FedThreshApi };
  if (!g.qshFedThresh) {
    if (!g.Go) await import(/* @vite-ignore */ `${base}wasm_exec.js`);
    const go = new g.Go!();
    const res = await fetch(`${base}qsh_fedthresh.wasm`);
    if (!res.ok) throw new Error(`qsh_fedthresh.wasm: HTTP ${res.status}`);
    const { instance } = await WebAssembly.instantiate(await res.arrayBuffer(), go.importObject);
    void go.run(instance);   // installs globalThis.qshFedThresh, then waits on its exported functions
  }
  if (!g.qshFedThresh) throw new Error("qsh_fedthresh.wasm did not install qshFedThresh");
  return g.qshFedThresh;
}
