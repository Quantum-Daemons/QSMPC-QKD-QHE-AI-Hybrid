// Client-side QML-IDS mirror — trains a genuine multiclass softmax logistic classifier in-browser on the
// 8-class QKD attack signatures from src/qsh/qml_ids/attacks.py (Al-Kuwari et al. 2026, arXiv:2509.14282),
// and reports MEASURED accuracy + binary attack-F1 on a held-out split. Honest scope: cited-methodology
// synthetic telemetry (no commercial-OK real QKD-attack dataset exists); a lightweight linear model, not
// the backend's RF/QSVC — so the demo's IDS numbers are measured, not a canned constant.
import { idsCfg } from "./config";
import { rng } from "./quantumSim";

const ATTACKS = ["normal", "intercept_resend", "photon_number_splitting", "trojan_horse",
  "detector_blinding", "wavelength_trojan", "rng_manipulation", "combined"];
// (qber_mean, qber_std, sift_ratio, multiphoton_rate, timing_jitter) — the rest derived, matching attacks.py
const SIG: Record<string, number[]> = {
  normal: [0.015, 0.004, 0.50, 0.02, 0.05],
  intercept_resend: [0.135, 0.020, 0.50, 0.02, 0.06],
  photon_number_splitting: [0.030, 0.008, 0.42, 0.18, 0.05],
  trojan_horse: [0.022, 0.006, 0.49, 0.05, 0.20],
  detector_blinding: [0.005, 0.003, 0.62, 0.03, 0.08],
  wavelength_trojan: [0.028, 0.010, 0.48, 0.06, 0.15],
  rng_manipulation: [0.018, 0.030, 0.50, 0.02, 0.05],
  combined: [0.090, 0.035, 0.45, 0.12, 0.18],
};
const N_FEAT = 7, N_CLASS = 8;
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const gauss = (u: () => number) => { const a = Math.max(1e-12, u()), b = u(); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b); };

function genData(nPer: number, seed: number): { X: number[][]; y: number[] } {
  const u = rng(seed + 11);
  const X: number[][] = [], y: number[] = [];
  ATTACKS.forEach((name, cls) => {
    const [qm, qs, sr, mp, tj] = SIG[name];
    for (let i = 0; i < nPer; i++) {
      X.push([
        clamp(qm + qs * gauss(u), 0, 0.5),
        clamp(qs + qs * 0.3 * gauss(u), 1e-4, 0.1),
        clamp(sr + 0.02 * gauss(u), 0.2, 0.8),
        clamp(mp + (mp * 0.3 + 1e-3) * gauss(u), 0, 1),
        clamp(tj + 0.02 * gauss(u), 0, 1),
        clamp(1e-5 * (name === "detector_blinding" ? 3 : 1) + 5e-6 * gauss(u), 0, 1),
        clamp((name === "detector_blinding" ? 0.5 : 0.02) + 0.03 * gauss(u), 0, 1),
      ]);
      y.push(cls);
    }
  });
  for (let i = X.length - 1; i > 0; i--) {           // Fisher-Yates shuffle (seeded)
    const j = Math.floor(u() * (i + 1));
    [X[i], X[j]] = [X[j], X[i]]; [y[i], y[j]] = [y[j], y[i]];
  }
  return { X, y };
}

export interface IDSResult { accuracy: number; binaryF1: number; nParams: number; nTest: number; }

export function clientTrainScoreIDS(seed = 0): IDSResult {
  const cfg = idsCfg();
  const { X, y } = genData(cfg.demo_n_per_class ?? 120, seed);
  const nTest = Math.floor(X.length * (cfg.test_fraction ?? 0.25));
  const Xtr = X.slice(nTest), ytr = y.slice(nTest), Xte = X.slice(0, nTest), yte = y.slice(0, nTest);

  const mean = new Array(N_FEAT).fill(0), std = new Array(N_FEAT).fill(0);
  for (const r of Xtr) for (let f = 0; f < N_FEAT; f++) mean[f] += r[f];
  for (let f = 0; f < N_FEAT; f++) mean[f] /= Xtr.length;
  for (const r of Xtr) for (let f = 0; f < N_FEAT; f++) std[f] += (r[f] - mean[f]) ** 2;
  for (let f = 0; f < N_FEAT; f++) std[f] = Math.sqrt(std[f] / Xtr.length) || 1;
  const norm = (r: number[]) => r.map((v, f) => (v - mean[f]) / std[f]);
  const Ztr = Xtr.map(norm), Zte = Xte.map(norm);

  // softmax multiclass logistic regression via batch gradient descent (genuine training, not canned)
  const W = Array.from({ length: N_CLASS }, () => new Array(N_FEAT).fill(0));
  const b = new Array(N_CLASS).fill(0);
  const lr = 0.5, iters = 250;
  for (let it = 0; it < iters; it++) {
    const gW = Array.from({ length: N_CLASS }, () => new Array(N_FEAT).fill(0));
    const gb = new Array(N_CLASS).fill(0);
    for (let n = 0; n < Ztr.length; n++) {
      const logits = W.map((w, c) => b[c] + w.reduce((a, wi, f) => a + wi * Ztr[n][f], 0));
      const mx = Math.max(...logits);
      const exps = logits.map((l) => Math.exp(l - mx));
      const s = exps.reduce((a, e) => a + e, 0);
      for (let c = 0; c < N_CLASS; c++) {
        const err = exps[c] / s - (ytr[n] === c ? 1 : 0);
        gb[c] += err;
        for (let f = 0; f < N_FEAT; f++) gW[c][f] += err * Ztr[n][f];
      }
    }
    for (let c = 0; c < N_CLASS; c++) {
      b[c] -= (lr * gb[c]) / Ztr.length;
      for (let f = 0; f < N_FEAT; f++) W[c][f] -= (lr * gW[c][f]) / Ztr.length;
    }
  }
  const predict = (z: number[]) => {
    const logits = W.map((w, c) => b[c] + w.reduce((a, wi, f) => a + wi * z[f], 0));
    let best = 0;
    for (let c = 1; c < N_CLASS; c++) if (logits[c] > logits[best]) best = c;
    return best;
  };
  let correct = 0, tp = 0, fp = 0, fn = 0;
  for (let n = 0; n < Zte.length; n++) {
    const pred = predict(Zte[n]);
    if (pred === yte[n]) correct++;
    const pa = pred !== 0, ta = yte[n] !== 0;
    if (pa && ta) tp++; else if (pa && !ta) fp++; else if (!pa && ta) fn++;
  }
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0, rec = tp + fn > 0 ? tp / (tp + fn) : 0;
  return {
    accuracy: Zte.length ? correct / Zte.length : 0,
    binaryF1: prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0,
    nParams: N_CLASS * N_FEAT + N_CLASS, nTest: Zte.length,
  };
}
