/**
 * Four real hospitals, each alone, federated and pooled, in the browser: FLamby's Fed-Heart-Disease benchmark (Round 86).
 *
 * WHY THIS EXISTS. The medical page federated round-robin shards of one corpus (WDBC), so its federation had no hospital
 * in it. UCI Heart Disease holds four real ones (Cleveland, Budapest, Zurich and Basel, Long Beach), CC BY 4.0, and
 * FLamby (du Terrail et al., NeurIPS 2022 Datasets and Benchmarks) made them a federated benchmark. This module runs
 * that benchmark's baseline on the page's copy of FLamby's preparation (`web/public/data/medical/heart_disease_sites.json`,
 * `scripts/data/build_heart_disease_sites.py`): logistic regression on 13 standardised features; each hospital alone
 * and all four pooled with Adam (lr 0.001, batches of 4, 50 epochs); FedAvg for 15 rounds of 100 local SGD steps at lr
 * 0.001, each hospital weighted by its training rows; accuracy at 0.5 on each hospital's own test rows, averaged.
 *
 * THE AGGREGATION IS THE PAGE'S SECURE AGGREGATION: every round goes through `secureAggregate`, the pairwise masks agreed
 * by X25519 that `clientQFL` uses, so the aggregator receives masked shares and nothing else. The masks cancel exactly in
 * the ring, so the federated model is the size-weighted FedAvg of the quantised updates.
 *
 * A TWIN of `qsh.ai.fed_heart`: the same random streams (`rng`, mulberry32), the same order of operations in float64,
 * and the bias corrections as running products rather than powers, so `tests/test_r86_fed_heart_twin.py` can hold the
 * page's figures to Python's.
 */
import { secaggClientSeeds, secaggKeyPairs, secureAggregate } from "./clientQFL";
import { rng } from "./quantumSim";

export const FED_HEART = { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, batch: 4, epochs: 50, rounds: 15, localSteps: 100 };

export interface HeartCentre {
  key: string; name: string;
  train: { X: number[][]; y: number[] };
  test: { X: number[][]; y: number[] };
  mean: number[]; std: number[];
}
export interface HeartSites {
  corpus: string; provenance: string; licence: string; licence_url: string; attribution: string; modifications: string;
  feature_names: string[]; label: string; split: string; centres: HeartCentre[];
}
export interface FedHeartResult {
  centres: string[];
  /** alone[k][j]: hospital k's model, trained on its own rows only, on hospital j's test rows. */
  alone: number[][];
  federated: number[];
  pooled: number[];
  /** The shared initial model on each hospital's test rows: what training starts from. */
  initial: number[];
  /** FLamby's metric: the mean over the four hospitals' test accuracies. */
  mean: { alone: number[]; federated: number; pooled: number; initial: number };
  rounds: number;
  /** The federated model's FLamby metric after each round. */
  roundMean: number[];
  testRows: number[]; trainRows: number[];
  aggregation: string; pairs: number; keyAgreementMs: number; ms: number;
  /** The hospitals' X25519 public keys for this run, hex: what the aggregator receives besides the masked shares. */
  publicKeys: string[];
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** The page's copy of FLamby's preparation, or null when it cannot be read (the caller says so). */
export async function loadHeartSites(): Promise<HeartSites | null> {
  try {
    const r = await fetch(`${import.meta.env.BASE_URL}data/medical/heart_disease_sites.json`);
    if (!r.ok) return null;
    const j = (await r.json()) as HeartSites;
    return Array.isArray(j?.centres) && j.centres.length > 1 ? j : null;
  } catch {
    return null;
  }
}

/** Fisher-Yates from the end, as clientQFL shuffles. */
function shuffled(n: number, u: () => number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(u() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  return idx;
}

function grad(theta: number[], X: number[][], y: number[], batch: number[]): number[] {
  const d = theta.length - 1;
  const g = new Array<number>(theta.length).fill(0);
  for (const i of batch) {
    let z = theta[d];
    for (let k = 0; k < d; k++) z += theta[k] * X[i][k];
    const r = sigmoid(z) - y[i];
    for (let k = 0; k < d; k++) g[k] += r * X[i][k];
    g[d] += r;
  }
  return g.map((v) => v / batch.length);
}

export function initTheta(seed: number, d = 13): number[] {
  const u = rng(seed * 1000 + 7);
  const bound = 1 / Math.sqrt(d);
  return Array.from({ length: d + 1 }, () => (u() * 2 - 1) * bound);
}

export function trainAdam(theta0: number[], X: number[][], y: number[], u: () => number): number[] {
  const { lr, beta1, beta2, eps, batch, epochs } = FED_HEART;
  const th = theta0.slice(), m = new Array<number>(th.length).fill(0), v = new Array<number>(th.length).fill(0);
  let b1t = 1, b2t = 1;
  for (let e = 0; e < epochs; e++) {
    const order = shuffled(y.length, u);
    for (let s = 0; s < order.length; s += batch) {
      const g = grad(th, X, y, order.slice(s, s + batch));
      b1t *= beta1; b2t *= beta2;
      const bc1 = 1 - b1t, bc2 = 1 - b2t;
      for (let k = 0; k < th.length; k++) {
        m[k] = beta1 * m[k] + (1 - beta1) * g[k];
        v[k] = beta2 * v[k] + (1 - beta2) * g[k] * g[k];
        const denom = Math.sqrt(v[k]) / Math.sqrt(bc2) + eps;
        th[k] -= (lr / bc1) * m[k] / denom;
      }
    }
  }
  return th;
}

/** `localSteps` batches, a fresh shuffle each time the hospital's rows run out, as a DataLoader is iterated. */
export function trainSgdSteps(theta0: number[], X: number[][], y: number[], u: () => number): number[] {
  const { lr, batch, localSteps } = FED_HEART;
  const th = theta0.slice();
  let order = shuffled(y.length, u), pos = 0;
  for (let s = 0; s < localSteps; s++) {
    if (pos >= order.length) { order = shuffled(y.length, u); pos = 0; }
    const g = grad(th, X, y, order.slice(pos, pos + batch));
    pos += batch;
    for (let k = 0; k < th.length; k++) th[k] -= lr * g[k];
  }
  return th;
}

export function accuracy(theta: number[], X: number[][], y: number[]): number {
  const d = theta.length - 1;
  let hits = 0;
  for (let i = 0; i < y.length; i++) {
    let s = 0;
    for (let k = 0; k < d; k++) s += theta[k] * X[i][k];
    if ((sigmoid(theta[d] + s) > 0.5 ? 1 : 0) === y[i]) hits++;
  }
  return hits / y.length;
}

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;

/**
 * The two comparisons the page states, counted from one run. By FLamby's metric (one model over the four hospitals'
 * test rows) the federation is above most hospitals alone; on its own test rows a hospital's own model is above the
 * federated one at most hospitals here. FedHeartPanel and the run's log both print these counts rather than assert
 * either comparison, and `tests/test_r86_fed_heart_twin.py` holds them to Python's.
 */
export function fedHeartComparisons(r: FedHeartResult) {
  const K = r.centres.length;
  const aboveAlone = r.mean.alone.filter((a) => r.mean.federated > a).length;
  const ownBeatsFederated = r.centres.filter((_, k) => r.alone[k][k] > r.federated[k]).length;
  let offDiagonalMin = 1;
  for (let k = 0; k < K; k++) for (let j = 0; j < K; j++) if (j !== k) offDiagonalMin = Math.min(offDiagonalMin, r.alone[k][j]);
  return { aboveAlone, ownBeatsFederated, offDiagonalMin, abovePooled: r.mean.federated > r.mean.pooled };
}

export async function fedHeart(sites: HeartSites, seed = 0,
                               opts?: { privateKeys?: readonly Uint8Array[] }): Promise<FedHeartResult> {
  const t0 = performance.now();
  const cs = sites.centres;
  const tests = cs.map((c) => c.test);
  const theta0 = initTheta(seed, sites.feature_names.length);
  const initial = tests.map((t) => accuracy(theta0, t.X, t.y));
  const alone = cs.map((c, k) => {
    const th = trainAdam(theta0, c.train.X, c.train.y, rng(seed * 1000 + 100 + k));
    return tests.map((t) => accuracy(th, t.X, t.y));
  });
  const pooledModel = trainAdam(theta0, cs.flatMap((c) => c.train.X), cs.flatMap((c) => c.train.y), rng(seed * 1000 + 300));
  const pooled = tests.map((t) => accuracy(pooledModel, t.X, t.y));
  // The federation: keys agreed once, then every round's size-weighted FedAvg under the masks.
  const K = cs.length;
  const tk = performance.now();
  const keyPairs = secaggKeyPairs(K, opts?.privateKeys);
  const publicKeys = keyPairs.map((kp) => kp.publicKey);
  const pairSeeds = await Promise.all(keyPairs.map((kp, c) => secaggClientSeeds(c, kp.privateKey, publicKeys)));
  const keyAgreementMs = performance.now() - tk;
  const sizes = cs.map((c) => c.train.y.length);
  const total = sizes.reduce((a, b) => a + b, 0);
  const weights = sizes.map((n) => n / total);
  const streams = cs.map((_, k) => rng(seed * 1000 + 200 + k));
  let glob = theta0.slice();
  const roundMean: number[] = [];
  for (let r = 0; r < FED_HEART.rounds; r++) {
    const updates = cs.map((c, k) => trainSgdSteps(glob, c.train.X, c.train.y, streams[k]));
    glob = secureAggregate(updates, weights, pairSeeds, r).aggregate;
    roundMean.push(mean(tests.map((t) => accuracy(glob, t.X, t.y))));
  }
  const federated = tests.map((t) => accuracy(glob, t.X, t.y));
  const pairs = (K * (K - 1)) / 2;
  return {
    centres: cs.map((c) => c.name), alone, federated, pooled, initial,
    mean: { alone: alone.map(mean), federated: mean(federated), pooled: mean(pooled), initial: mean(initial) },
    rounds: FED_HEART.rounds, roundMean, testRows: tests.map((t) => t.y.length), trainRows: sizes,
    aggregation: `size-weighted FedAvg + ${pairs} pairwise masks from X25519-agreed pair seeds, fixed-point mod 2^32`,
    pairs, keyAgreementMs, ms: performance.now() - t0,
    publicKeys: publicKeys.map((pk) => Array.from(pk, (b) => b.toString(16).padStart(2, "0")).join("")),
  };
}
