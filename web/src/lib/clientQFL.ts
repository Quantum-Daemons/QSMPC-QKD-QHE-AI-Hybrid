/**
 * Quantum Federated Learning in the browser, and its secure aggregation.
 *
 * SPLIT OUT OF `clientCompute.ts` so it can be exercised on its own. The security property this
 * module provides - that the aggregator recovers the weighted mean from MASKED shares alone - is
 * worth an executable check, and `clientCompute` transitively reaches `clientFHE`, which imports the
 * SEAL wasm through a Vite-only `?url` specifier that no plain bundler resolves. Keeping federated
 * aggregation here means `npm run check:qfl` loads the real code rather than a re-implementation of
 * it. It is also a distinct concern from a 2,000-line stage dispatcher.
 */
import { hkdf } from "@noble/hashes/hkdf.js";
import { sha256 } from "@noble/hashes/sha2.js";

import { federatedCfg } from "./config";
import { qcnnExpectation, rng } from "./quantumSim";

const r2 = (x: number) => Math.round(x * 100) / 100;

// A GENUINELY-computed, in-browser FedAvg of a HYBRID quantum-classical model: a fixed quantum
// feature map (the quantumSim QCNN expectation) feeds a classical logistic head that is federated
// across K simulated clients with size-weighted FedAvg — mirroring the full deployment's
// Flower + PennyLane HybridQNN. Round accuracy is REALLY computed on a held-out split (not a canned
// constant). Honest scope: a small simulated VQC feature map + a federated classical head (not
// fault-tolerant QFL); the full path also trains the VQC weights.
//
// This comment used to end "under quantum-safe secure aggregation", and the diagram said "pairwise
// masking (demo keys)". Neither was true of this function: it summed plaintext updates. `aggregation`
// replaces the `secureAgg: true` literal that stood here — a boolean hardcoded to the answer it was
// supposed to report is not a report. The stage publishes it as `qfl_aggregation` and the diagram
// renders that, so the figure cannot outrun the algorithm again.
export interface QFLResult {
  roundAcc: number[]; finalAcc: number; clients: number; rounds: number; aggregation: string;
  pairs: number;
  /** Held-out rows `finalAcc` was measured on. Reported because it is small enough to matter. */
  evalRows: number;
  /** What the same held-out rows score with untrained weights - the constant predictor. */
  baselineAcc: number;
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

/**
 * The mask vector shared by the ordered pair (u, v), u < v, at one round.
 *
 * HKDF-SHA256 from the run's PQC session key, with the pair and round in the info string - the same
 * primitive and the same domain-separation style the hybrid handshake uses. Deriving per round is
 * what stops a mask from being reused across updates of the same weights.
 */
function qflPairMask(sessionKey: Uint8Array, u: number, v: number, round: number, P: number): number[] {
  const info = new TextEncoder().encode(`qsh/qfl-pairwise-mask/v1|${u}|${v}|r${round}`);
  const bytes = hkdf(sha256, sessionKey, undefined, info, P * 4);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from({ length: P }, (_, k) => dv.getUint32(k * 4, true));
}

/**
 * Size-weighted FedAvg under pairwise additive masking. Bonawitz et al., CCS 2017.
 *
 * Each client weights its own update, quantises it, and adds `+mask(u,v)` for every peer above it
 * and `-mask(u,v)` for every peer below. The aggregator sees only masked shares; summing them
 * cancels every pair exactly and leaves the weighted mean.
 *
 * WEIGHT BEFORE MASKING, not after. Masks cancel under a SUM. Applying the size weights to already-
 * masked shares would scale each mask by a different factor and they would not cancel - the weighted
 * mean is recovered by having each client scale its own contribution, which is what Bonawitz does.
 *
 * Honest scope: the demo runs all K clients in one tab, so this demonstrates that the aggregate is
 * recoverable from masked shares alone - it is not a threat model in which the shares are held by
 * mutually distrusting parties. There is no dropout-recovery secret sharing (Bonawitz section 4),
 * so a client that vanished mid-round would leave its masks uncancelled.
 */
export function secureAggregate(
  updates: number[][], weights: number[], sessionKey: Uint8Array, round: number,
): { aggregate: number[]; maskedShares: number[][] } {
  const K = updates.length;
  const P = updates[0]?.length ?? 0;
  const maxShare = qflMaxShare(K);
  const maskedShares = updates.map((wl, u) => {
    const share = new Array<number>(P);
    for (let k = 0; k < P; k++) {
      const q = Math.round(wl[k] * weights[u] * QFL_SCALE);
      if (!Number.isFinite(q) || Math.abs(q) > maxShare) {
        // Loud rather than silent: past this magnitude the ring wraps into the wrong half and the
        // recovered mean is garbage that still looks like a number.
        throw new Error(`QFL share out of ring range: client ${u}, coord ${k}, q=${q}`);
      }
      share[k] = (q | 0) >>> 0;      // two's complement, i.e. reduction mod 2^32
    }
    for (let v = 0; v < K; v++) {
      if (v === u) continue;
      const m = v > u ? qflPairMask(sessionKey, u, v, round, P)
                      : qflPairMask(sessionKey, v, u, round, P);
      for (let k = 0; k < P; k++) {
        share[k] = (v > u ? share[k] + m[k] : share[k] - m[k]) >>> 0;
      }
    }
    return share;
  });
  const aggregate = new Array<number>(P).fill(0);
  for (let k = 0; k < P; k++) {
    let acc = 0;
    for (const share of maskedShares) acc = (acc + share[k]) >>> 0;
    aggregate[k] = (acc >= QFL_RING / 2 ? acc - QFL_RING : acc) / QFL_SCALE;
  }
  return { aggregate, maskedShares };
}

export function clientQFL(
  seed = 0,
  opts?: { clients?: number; rounds?: number; data?: { X: number[][]; Y: number[] } | null;
           sessionKey?: Uint8Array },
): QFLResult {
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
  // quantum feature map: the QCNN expectation on a few deterministic views of x → 3 quantum features
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
  // What the SAME held-out rows score before any training: w = 0 gives sigmoid(0) = 0.5, so every
  // row is predicted 1 and this is the constant-predictor rate. Reported because "accuracy 0.74"
  // says nothing on its own about whether the federated rounds did anything, and the round-by-round
  // series cannot say either once the problem is well conditioned enough to converge in one round.
  // It is the `C` (constant predictor) discipline this project already applies to its other models.
  const baselineAcc = r2(acc(new Array(P).fill(0)));

  // No plaintext branch. A `sessionKey ?? aggregate in the clear` fallback is how the diagram came
  // to claim masking that did not run: the claim is written once and the fallback is taken forever.
  // Both flows that reach this stage run `pqc_handshake` first, so the key is always there.
  const sessionKey = opts?.sessionKey;
  if (!sessionKey || sessionKey.length === 0) {
    throw new Error("clientQFL needs the PQC session key: secure aggregation has no plaintext mode");
  }
  let w = new Array(P).fill(0);
  const tot = tr.length;
  const sizeWeights = parts.map((ids) => ids.length / tot);
  const roundAcc: number[] = [];
  for (let r = 0; r < rounds; r++) {
    const updates = parts.map((ids) => localTrain(ids, w));        // each client trains locally
    w = secureAggregate(updates, sizeWeights, sessionKey, r).aggregate;
    roundAcc.push(r2(acc(w)));
  }
  // MEASURED, not asserted: C(K,2) is the number of masks that actually had to cancel. With one
  // client there are none, and the string says so rather than claiming a property of an empty set.
  const pairs = (K * (K - 1)) / 2;
  // The DENOMINATOR travels with the accuracy, because on the bundled samples it is 4 rows (finance,
  // 16 total) and 3 rows (medical, 12). A "0.25" with no denominator reads as a model performing
  // worse than chance; a "0.25 on 4 rows" reads as one held-out row, which is what it is. Both pages'
  // round accuracies are also CONSTANT across all six rounds on those samples - at that size the
  // decision on every held-out row is fixed before training moves the head. Enlarging and scaling the
  // bundled federated samples is the real fix and is tracked separately; reporting a bare fraction as
  // an accuracy was the part that had to stop immediately.
  return { roundAcc, finalAcc: roundAcc[roundAcc.length - 1], clients: K, rounds, pairs,
    evalRows: te.length, baselineAcc,
    aggregation: pairs > 0
      ? `size-weighted FedAvg + ${pairs} pairwise masks (Bonawitz CCS 2017), fixed-point mod 2^32`
      : "size-weighted FedAvg, 1 client - no pair to mask" };
}
