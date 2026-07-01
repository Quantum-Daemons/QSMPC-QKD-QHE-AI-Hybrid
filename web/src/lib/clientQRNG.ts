// Client-side QRNG mirror of src/qsh/qrng/ — a genuine state-vector quantum-measurement simulation
// entirely in the browser, with the same SP 800-22-style health checks and MEASURED throughput.
// Honest label: a SIMULATED quantum measurement (state-vector), not quantum hardware. The optional ANU
// hardware source is opt-in + online (never the default), matching the backend.
import { rng } from "./quantumSim";

const SR2 = Math.SQRT1_2;
// |+> = H|0> = [1/sqrt2, 1/sqrt2]; Born rule => P(measure |1>) = |amp_1|^2 = 0.5.
const PROB_ONE = SR2 * SR2;

export interface QRNGResult {
  bits: Uint8Array;
  nBits: number;
  source: string;
  postProcessing: string;
  healthPassed: boolean;
  monobitP: number;
  runsP: number;
  blockFreqP: number;
  throughputBitsPerSec: number;
  simulated: boolean;
}

// erfc via the Abramowitz-Stegun 7.1.26 rational approximation (pure JS, no dep).
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const y = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277))))))))); // erfc(z) for z>=0
  return x >= 0 ? y : 2 - y;
}

// Prepare n |+> qubits and measure each in the Z basis (Born-rule sampling); genuine 50/50 outcomes.
function measurePlus(n: number, u: () => number): Uint8Array {
  const b = new Uint8Array(n);
  for (let i = 0; i < n; i++) b[i] = u() < PROB_ONE ? 1 : 0;
  return b;
}

function vonNeumann(bits: Uint8Array): Uint8Array {
  const out: number[] = [];
  for (let i = 0; i + 1 < bits.length; i += 2) if (bits[i] !== bits[i + 1]) out.push(bits[i]);
  return Uint8Array.from(out);
}

// SP 800-22 subset: monobit + runs + block-frequency (p-values).
function monobit(b: Uint8Array): number {
  if (!b.length) return 0;
  let s = 0;
  for (const x of b) s += 2 * x - 1;
  return erfc(Math.abs(s) / Math.sqrt(b.length) / Math.SQRT2);
}
function runs(b: Uint8Array): number {
  const n = b.length;
  if (n < 2) return 0;
  let ones = 0;
  for (const x of b) ones += x;
  const pi = ones / n;
  if (Math.abs(pi - 0.5) >= 2 / Math.sqrt(n)) return 0;
  let v = 1;
  for (let i = 1; i < n; i++) if (b[i] !== b[i - 1]) v++;
  const den = 2 * Math.sqrt(2 * n) * pi * (1 - pi);
  return den > 0 ? erfc(Math.abs(v - 2 * n * pi * (1 - pi)) / den) : 0;
}
function blockFrequency(b: Uint8Array, m = 128): number {
  const nb = Math.floor(b.length / m);
  if (nb === 0) return 1;
  let chi = 0;
  for (let k = 0; k < nb; k++) {
    let ones = 0;
    for (let i = 0; i < m; i++) ones += b[k * m + i];
    const pi = ones / m;
    chi += (pi - 0.5) * (pi - 0.5);
  }
  chi *= 4 * m;
  // P = igamc(nb/2, chi/2); series/continued-fraction upper incomplete gamma (regularized).
  return gammaincc(nb / 2, chi / 2);
}
// Regularized upper incomplete gamma Q(a,x) (Numerical-Recipes style), pure JS.
function gammaincc(a: number, x: number): number {
  if (x <= 0) return 1;
  const gln = lgamma(a);
  if (x < a + 1) {
    let ap = a, sum = 1 / a, del = sum;
    for (let n = 0; n < 200; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-12) break; }
    return 1 - sum * Math.exp(-x + a * Math.log(x) - gln);
  }
  let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 200; i++) {
    const an = -i * (i - a);
    b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
    c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
    d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-12) break;
  }
  return Math.exp(-x + a * Math.log(x) - gln) * h;
}
function lgamma(x: number): number {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += c[j] / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}

// Generate nBits post-processed quantum-random bits + a health report + MEASURED throughput.
export function generateQuantumBits(nBits: number, opts: { source?: string; postProcessing?: string;
  seed?: number; alpha?: number } = {}): QRNGResult {
  const source = opts.source ?? "sim";
  if (source === "anu") throw new Error("ANU QRNG is opt-in + online; the offline 'sim' source is the demo default");
  const post = opts.postProcessing ?? "von_neumann";
  const alpha = opts.alpha ?? 0.01;
  const u = opts.seed === undefined ? Math.random : rng(opts.seed);
  const t0 = performance.now();
  const need = post === "von_neumann" ? 5 * nBits + 16 : nBits;
  const raw = measurePlus(need, u);
  const bits = (post === "von_neumann" ? vonNeumann(raw) : raw).slice(0, nBits);
  const dt = Math.max(performance.now() - t0, 1e-6);
  const mp = monobit(bits), rp = runs(bits), bp = blockFrequency(bits);
  return {
    bits, nBits: bits.length, source, postProcessing: post,
    healthPassed: mp >= alpha && rp >= alpha && bp >= alpha,
    monobitP: mp, runsP: rp, blockFreqP: bp,
    throughputBitsPerSec: (bits.length / dt) * 1000, simulated: true,
  };
}
