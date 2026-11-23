/**
 * BB84 with decoy intensities, pulse by pulse, with an intercept-resend eavesdropper (Round 86, the owner's choice of
 * 2026-11-22: a per-photon BB84 table feeding the QBER chart). The page's QKD figures come from closed-form models
 * (quantumSim.ts); this draws the pulses themselves, so a reader can watch a QBER arise and compare it with the model.
 *
 * Each pulse is a signal or a decoy (`sourceOf`), carries a Poisson number of photons at that intensity, and encodes
 * Alice's random bit in her random basis. With probability `eve`, a non-empty pulse is intercepted: Eve measures in a
 * random basis, gets Alice's bit when the bases agree and a coin flip otherwise, and resends as many photons as she
 * took, in her basis, so the gains stay as they were. Each photon reaches Bob's detector with the channel's
 * transmittance; Bob measures in his random basis, getting the state's bit when the bases agree and a coin flip
 * otherwise, and misalignment flips what he gets. Dark counts are left out, so misalignment and Eve are the only
 * errors, and the expected QBER is exactly `composeIndependentErrors(misalignment, 0.25 eve)`, the page's model with
 * its dark-count term at zero: an intercepted pulse errs with probability 1/4 + e/2 and any other with e, which is
 * e + eve/4 - eve e/2. The pulses are simulated; there is no hardware.
 */
import { composeIndependentErrors, rng } from "./quantumSim";

export type Basis = "Z" | "X";
export interface Pulse {
  kind: "signal" | "decoy"; photons: number; aliceBit: number; aliceBasis: Basis;
  eve: null | { basis: Basis; bit: number };
  bobBasis: Basis; clicked: boolean; bobBit: number | null; sifted: boolean; error: boolean;
}
export interface Bb84Options { pulses: number; eve: number; misalignment: number; transmittance: number;
  mu: number; nu: number; pSignal: number; seed: number; keep?: number }
export interface Bb84Tally { sent: number; detected: number; sifted: number; errors: number }
export interface Bb84Result {
  /** The first pulses Bob detected, and how many pulses were sent until the last of them. */
  first: Pulse[];
  firstSpan: number;
  signal: Bb84Tally; decoy: Bb84Tally;
  qber: number; ci95: [number, number];
  /** The page's closed-form QBER for the same channel with no dark counts. */
  model: number;
}

function poisson(mean: number, u: () => number): number {
  const limit = Math.exp(-mean);
  let k = 0, p = u();
  while (p > limit) { k += 1; p *= u(); }
  return k;
}

/** The Wilson score interval at 95 percent for k of n. */
export function wilson(k: number, n: number): [number, number] {
  if (n === 0) return [0, 1];
  const z = 1.959963984540054, p = k / n, d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d, h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

export function bb84Expected(misalignment: number, eve: number): number {
  return composeIndependentErrors(misalignment, 0.25 * eve);
}

export function simulateBb84(o: Bb84Options): Bb84Result {
  const u = rng(o.seed);
  const coin = () => (u() < 0.5 ? 1 : 0);
  const basis = (): Basis => (u() < 0.5 ? "Z" : "X");
  const keep = o.keep ?? 20;
  const first: Pulse[] = [];
  let firstSpan = 0;
  const tally = { signal: { sent: 0, detected: 0, sifted: 0, errors: 0 }, decoy: { sent: 0, detected: 0, sifted: 0, errors: 0 } };
  for (let i = 0; i < o.pulses; i++) {
    const kind = u() < o.pSignal ? "signal" : "decoy";
    const mean = kind === "signal" ? o.mu : o.nu;
    const photons = poisson(mean, u);
    const aliceBit = coin(), aliceBasis = basis();
    let stateBit = aliceBit, stateBasis = aliceBasis, n = photons;
    let eve: Pulse["eve"] = null;
    if (photons > 0 && u() < o.eve) {
      const eb = basis();
      const bit = eb === aliceBasis ? aliceBit : coin();
      eve = { basis: eb, bit };
      // She resends as many photons as she took, in her basis: the pulse's detection chance is unchanged, so her
      // pulses are as likely to be kept as any other. A fresh Poisson draw here would often send nothing and hide her.
      stateBit = bit; stateBasis = eb; n = photons;
    }
    let arrived = 0;
    for (let k = 0; k < n; k++) if (u() < o.transmittance) arrived += 1;
    const bobBasis = basis();
    const clicked = arrived > 0;
    let bobBit: number | null = null;
    if (clicked) {
      bobBit = bobBasis === stateBasis ? stateBit : coin();
      if (u() < o.misalignment) bobBit ^= 1;
    }
    const sifted = clicked && bobBasis === aliceBasis;
    const error = sifted && bobBit !== aliceBit;
    const t = tally[kind];
    t.sent += 1;
    if (clicked) t.detected += 1;
    if (sifted) t.sifted += 1;
    if (error) t.errors += 1;
    if (clicked && first.length < keep) {
      first.push({ kind, photons, aliceBit, aliceBasis, eve, bobBasis, clicked, bobBit, sifted, error });
      firstSpan = i + 1;
    }
  }
  const sifted = tally.signal.sifted + tally.decoy.sifted, errors = tally.signal.errors + tally.decoy.errors;
  return { first, firstSpan, signal: tally.signal, decoy: tally.decoy, qber: sifted ? errors / sifted : 0, ci95: wilson(errors, sifted),
           model: bb84Expected(o.misalignment, o.eve) };
}
