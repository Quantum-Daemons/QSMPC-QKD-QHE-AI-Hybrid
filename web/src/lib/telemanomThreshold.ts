/**
 * Telemanom's non-parametric dynamic thresholding, per channel, as ESA-ADB's TelemanomThresholding applies it
 * (Round 86). A port of `src/qsh/models/telemanom_threshold.py`, which flags exactly the rows that class flags; a test
 * bundles this file, runs both on the same inputs and requires identical flags.
 *
 * The arithmetic follows the Python's operation for operation, because the procedure compares smoothed errors with
 * thresholds and a last-bit difference can move a flag:
 * - the moving average is pandas' `ewm(span).mean()` with `adjust=True`, in pandas' own update order;
 * - the percentiles are numpy's linear interpolation, with numpy's two-sided lerp;
 * - standard deviations are population ones (numpy's default).
 */

export interface TelemanomConfig {
  batchSize: number;
  smoothingWindowSize: number;
  smoothingPerc: number;
  errorBuffer: number;
  p: number;
  windowSize: number;
  sdLim: number;
  minErrorValue: number;
  adjustStart: boolean;
}

export const DEFAULT_CONFIG: TelemanomConfig = {
  batchSize: 70, smoothingWindowSize: 30, smoothingPerc: 0.05, errorBuffer: 100, p: 0.13, windowSize: 250,
  sdLim: 12.0, minErrorValue: 0.0, adjustStart: false,
};

/** pandas' `Series.ewm(span=span).mean()` (adjust=True, no missing values), in the order pandas' kernel updates it. */
export function ewmMean(x: ArrayLike<number>, span: number): Float64Array {
  const n = x.length;
  const out = new Float64Array(n);
  if (n === 0) return out;
  const alpha = 2 / (span + 1);
  const oldWtFactor = 1 - alpha;
  const newWt = 1;
  let weighted = x[0];
  let oldWt = 1;
  out[0] = weighted;
  for (let i = 1; i < n; i++) {
    const cur = x[i];
    oldWt *= oldWtFactor;
    if (weighted !== cur) {
      weighted = oldWt * weighted + newWt * cur;
      weighted /= oldWt + newWt;
    }
    oldWt += newWt;
    out[i] = weighted;
  }
  return out;
}

/** numpy's pairwise summation of a contiguous float64 array (`pairwise_sum_DOUBLE`), so a mean here is the same double
 *  numpy's is: blocks of up to 128 summed with 8 accumulators, longer runs split in two at a multiple of 8. */
function pairwiseSum(a: ArrayLike<number>, lo: number, n: number): number {
  if (n < 8) {
    let r = 0;
    for (let i = 0; i < n; i++) r += a[lo + i];
    return r;
  }
  if (n <= 128) {
    const r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[lo + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwiseSum(a, lo, n2) + pairwiseSum(a, lo + n2, n - n2);
}

function mean(a: ArrayLike<number>): number {
  return pairwiseSum(a, 0, a.length) / a.length;
}

/** numpy's population standard deviation: the square root of the mean squared deviation, summed as numpy sums. */
function std(a: ArrayLike<number>): number {
  const m = mean(a);
  const sq = Float64Array.from(a, (v) => (v - m) * (v - m));
  return Math.sqrt(pairwiseSum(sq, 0, sq.length) / sq.length);
}

/** numpy's `percentile(a, q)` with the default linear method, including its two-sided lerp. */
function percentile(a: ArrayLike<number>, q: number): number {
  const v = Float64Array.from(a).sort();
  const n = v.length;
  const idx = (q / 100) * (n - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, n - 1);
  const t = idx - lo;
  const diff = v[hi] - v[lo];
  return t >= 0.5 ? v[hi] - diff * (1 - t) : v[lo] + diff * t;
}

/** Runs of consecutive integers in a sorted unique array, as [first, last]. */
function groups(idx: number[]): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i < idx.length; i++) {
    if (i > 0 && idx[i] === idx[i - 1] + 1) out[out.length - 1][1] = idx[i];
    else out.push([idx[i], idx[i]]);
  }
  return out;
}

/** Each index with every offset 1..maxOffset either side added, clipped to [0, n), sorted and unique. */
function widen(iAnom: number[], n: number, maxOffset: number): number[] {
  const mark = new Uint8Array(n);
  for (const i of iAnom) {
    for (let d = -maxOffset; d <= maxOffset; d++) {
      const j = i + d;
      if (j >= 0 && j < n) mark[j] = 1;
    }
  }
  const out: number[] = [];
  for (let j = 0; j < n; j++) if (mark[j]) out.push(j);
  return out;
}

/** The threshold that maximises Telemanom's objective over z in 2.5, 3.0, ..., below `sdLim`. */
function findEpsilon(e: Float64Array, m: number, sd: number, cfg: TelemanomConfig): number {
  let epsilon = m + cfg.sdLim * sd;
  if (sd === 0) return epsilon;
  let best = -Infinity;
  for (let k = 0; ; k++) {
    const z = 2.5 + 0.5 * k;
    if (!(z < cfg.sdLim)) break;
    const eps = m + sd * z;
    const pruned: number[] = [];
    const above: number[] = [];
    for (let i = 0; i < e.length; i++) {
      if (e[i] < eps) pruned.push(e[i]);
      if (e[i] >= eps) above.push(i);
    }
    const iAnom = widen(above, e.length, cfg.errorBuffer - 1);
    if (iAnom.length === 0) break;
    const seqs = groups(iAnom).filter(([a, b]) => a !== b);
    const meanDrop = m ? (m - mean(pruned)) / m : 0;
    const sdDrop = (sd - std(pruned)) / sd;
    const score = (meanDrop + sdDrop) / (seqs.length ** 2 + iAnom.length);
    if (score >= best && seqs.length <= 5 && iAnom.length < 0.5 * e.length) {
      best = score;
      epsilon = eps;
    }
  }
  return epsilon;
}

/** Drop the sequences whose maximum is within `p` of the next one down, unless a large drop follows them. */
function prune(seqs: [number, number][], e: Float64Array, nonAnomMax: number, p: number): [number, number][] {
  if (seqs.length === 0) return seqs;
  const maxima = seqs.map(([a, b]) => {
    let mx = -Infinity;
    for (let i = a; i <= b; i++) mx = Math.max(mx, e[i]);
    return mx;
  });
  const ordered = [...maxima].sort((x, y) => y - x);
  ordered.push(nonAnomMax);
  let remove: number[] = [];
  for (let i = 0; i < ordered.length - 1; i++) {
    const drop = ordered[i] ? (ordered[i] - ordered[i + 1]) / ordered[i] : 0;
    if (drop < p) {
      maxima.forEach((v, k) => { if (v === ordered[i]) remove.push(k); });
    } else {
      remove = [];
    }
  }
  const gone = new Set(remove);
  return seqs.filter((_, k) => !gone.has(k));
}

/** Telemanom's anomalous timesteps for one channel, from its smoothed errors and its values. */
export function flagChannel(eS: Float64Array, values: ArrayLike<number>, cfg: TelemanomConfig = DEFAULT_CONFIG): Uint8Array {
  const n = eS.length;
  const flags = new Uint8Array(n);
  const width = cfg.smoothingWindowSize;
  const nWindows = Math.ceil(n / cfg.batchSize);
  const numToIgnore = n < 1800 ? 0 : n < 2500 ? cfg.windowSize : 2 * cfg.windowSize;
  const found = new Set<number>();
  for (let w = 0; w < nWindows; w++) {
    const lo = w * cfg.batchSize;
    const hi = w === nWindows - 1 ? n : Math.min(n, width * cfg.batchSize + w * cfg.batchSize);
    const win = eS.subarray(lo, hi);
    const y = Array.prototype.slice.call(values, lo, hi) as number[];
    const m = mean(win), sd = std(win);
    const inter = percentile(y, 95) - percentile(y, 5);
    let winMax = -Infinity;
    for (let i = 0; i < win.length; i++) winMax = Math.max(winMax, win[i]);
    if (!(sd > 0.05 * std(y) || winMax > 0.05 * inter) || win.length < cfg.errorBuffer
        || !(winMax > cfg.minErrorValue)) continue;
    const keepAll: number[][] = [];
    const reflected = Float64Array.from(win, (v) => m + (m - v));
    for (const e of [win, reflected]) {
      const eps = findEpsilon(e, m, sd, cfg);
      const hits: number[] = [];
      for (let i = 0; i < e.length; i++) if (e[i] >= eps && e[i] > 0.05 * inter) hits.push(i);
      if (hits.length === 0) continue;
      let iAnom = widen(hits, e.length, cfg.errorBuffer);
      iAnom = w === 0 ? iAnom.filter((i) => i >= numToIgnore) : iAnom.filter((i) => i >= e.length - cfg.batchSize);
      if (iAnom.length === 0) continue;
      const excluded = new Set<number>(iAnom);
      for (const f of found) if (f >= lo && f < hi) excluded.add(f - lo);
      let nonAnomMax = -Infinity, any = false;
      for (let i = 0; i < e.length; i++) {
        if (!excluded.has(i)) { nonAnomMax = Math.max(nonAnomMax, e[i]); any = true; }
      }
      if (!any) nonAnomMax = 0;
      let seqs = groups(iAnom).filter(([a, b]) => a !== b);
      if (seqs.length === 0) {
        // No sequence of two or more: ESA-ADB's prune_anoms returns before pruning, so these points stay.
        keepAll.push(iAnom);
        continue;
      }
      seqs = prune(seqs, e, nonAnomMax, cfg.p);
      if (seqs.length === 0) continue;
      const kept = iAnom.filter((i) => seqs.some(([a, b]) => i >= a && i <= b));
      keepAll.push(kept);
    }
    for (const arr of keepAll) for (const i of arr) found.add(i + lo);
  }
  const sorted = [...found].sort((a, b) => a - b);
  for (const [a, b] of groups(sorted)) {
    if (a !== b) for (let i = a; i < b; i++) flags[i] = 1;        // [first, last), as ESA-ADB writes a sequence
  }
  return flags;
}

/** Smoothed errors per channel: absolute errors in, the moving average out, with the optional start adjustment. */
export function smoothErrors(errors: ArrayLike<number>, cfg: TelemanomConfig = DEFAULT_CONFIG): Float64Array {
  const span = Math.trunc(cfg.batchSize * cfg.smoothingWindowSize * cfg.smoothingPerc);
  const abs = Float64Array.from(errors, (v) => Math.abs(v));
  const eS = ewmMean(abs, span);
  const w = cfg.windowSize;
  if (cfg.adjustStart && eS.length >= 2 * w) {
    const m = mean(eS.subarray(0, 2 * w));
    for (let i = 0; i < w; i++) eS[i] = m;
  }
  return eS;
}

/** Per-channel flags and their union. `errors[c]` and `values[c]` are channel c's absolute errors and values. */
export function flag(errors: ArrayLike<number>[], values: ArrayLike<number>[], cfg: TelemanomConfig = DEFAULT_CONFIG):
    { perChannel: Uint8Array[]; any: Uint8Array } {
  const perChannel = errors.map((e, c) => flagChannel(smoothErrors(e, cfg), values[c], cfg));
  const n = perChannel[0]?.length ?? 0;
  const any = new Uint8Array(n);
  for (const f of perChannel) for (let i = 0; i < n; i++) if (f[i]) any[i] = 1;
  return { perChannel, any };
}
