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

export interface IDSResult {
  /** Headline: mean per-class F1 over the 8 classes present in the held-out split. */
  macroF1: number;
  /** The weakest single class — what macro F1 averages over and what an operator actually feels. */
  worstClassF1: number;
  /** Companions, kept because they are the numbers this field usually quotes, not because they lead. */
  accuracy: number;
  binaryF1: number;
  nParams: number; nTest: number;
  /** Samples/class actually trained on after clamping — not necessarily what was requested. */
  nPerClassUsed: number;
  /**
   * One scored held-out sample and the linear model that classified it, so the run can attribute
   * the decision. This classifier IS linear - softmax over `b[c] + W[c].z` - so exact linear SHAP
   * applies to it directly, and it is the model that produces the class the page reports. No
   * surrogate and no invented method for a quantum circuit is involved.
   *
   * `W` is returned as a one-vs-rest PAIR for the predicted class: row 1 is W[pred], row 0 is the
   * mean of the other classes. `w_eff = W[1] - W[0]` is then exactly the one-vs-rest contrast
   * `explainability.py` computes for the multiclass case.
   */
  explain: {
    W: number[][]; b: number[]; x: number[]; predicted: number;
    className: string; featureNames: string[];
    /** The same row BEFORE this module's z-scoring, for a consumer that carries its own constants. */
    xRaw: number[]; trueClass: number;
  } | null;
}

/** Feature order must match `genData` below, and mirrors `qml_ids/attacks.py:FEATURES`. */
export const IDS_FEATURE_NAMES = [
  "qber_mean", "qber_std", "sift_ratio", "multiphoton_rate", "timing_jitter",
  "dark_count_rate", "click_asymmetry",
];

export function clientTrainScoreIDS(seed = 0, nPerClass?: number): IDSResult {
  const cfg = idsCfg();
  // samples-per-attack-class is a run-config knob (config.n_per_class): more training data per class
  // shifts the measured accuracy / attack-F1 (bounded >=20 for a usable split; robust to NaN/undefined).
  //
  // Clamped at BOTH ends. Only the lower bound was here, while the Parameters field advertised
  // max=2000 - and `max` on a number input does not constrain a typed value, it only styles the
  // spinner. Typing 100000 asked for 800k rows and 250 full-batch passes over them on the main
  // thread, which does not return.
  const nP = Number(nPerClass);
  const nPer = Number.isFinite(nP) && nP > 0
    ? Math.min(2000, Math.max(20, Math.round(nP)))
    : (cfg.demo_n_per_class ?? 120);
  const { X, y } = genData(nPer, seed);
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
  // Per-class confusion counts. Macro F1 is computed from these because it is the only one of the
  // three numbers here that degrades when a single attack class collapses: binary F1 folds all seven
  // attack types into "attack" and aggregate accuracy averages the failure away. Engelen et al.
  // (WTMC 2021) is the standing example — correcting CIC-IDS2017's labels moved Bot F1 from 0.60 to
  // 0.99 while aggregate accuracy sat at 0.99 throughout. scripts/train/telecom_ids.py has reported
  // macro F1 as the headline since Round 17; this is the browser catching up to it.
  const tpC = new Array(N_CLASS).fill(0), fpC = new Array(N_CLASS).fill(0), fnC = new Array(N_CLASS).fill(0);
  let correct = 0, tp = 0, fp = 0, fn = 0;
  for (let n = 0; n < Zte.length; n++) {
    const pred = predict(Zte[n]);
    if (pred === yte[n]) { correct++; tpC[pred]++; } else { fpC[pred]++; fnC[yte[n]]++; }
    const pa = pred !== 0, ta = yte[n] !== 0;
    if (pa && ta) tp++; else if (pa && !ta) fp++; else if (!pa && ta) fn++;
  }
  /** One-vs-rest contrast for a representative held-out sample: an ATTACK if the split has one,
   *  since attributing a normal-traffic verdict says little. Returns null on an empty split. */
  const explainSample = () => {
    if (!Zte.length) return null;
    let idx = Zte.findIndex((_, n) => yte[n] !== 0);
    if (idx < 0) idx = 0;
    const z = Zte[idx];
    const pred = predict(z);
    const rest = [...Array(N_CLASS).keys()].filter((c) => c !== pred);
    const restW = new Array(N_FEAT).fill(0).map(
      (_, f) => rest.reduce((a2, c) => a2 + W[c][f], 0) / rest.length);
    const restB = rest.reduce((a2, c) => a2 + b[c], 0) / rest.length;
    return {
      W: [restW, W[pred].slice()], b: [restB, b[pred]], x: z.slice(), predicted: pred,
      className: ATTACKS[pred] ?? `class_${pred}`, featureNames: IDS_FEATURE_NAMES,
      // The RAW (un-standardised) row as well as the z-scored one. The published CKKS student ships
      // its OWN mean and scale, fitted on its own training split, so it has to standardise from raw
      // features - feeding it `x` would apply this function's normalisation twice with the wrong
      // constants and produce a number that looks like a score.
      xRaw: Xte[idx].slice(), trueClass: yte[idx],
    };
  };

  const f1 = (t: number, f: number, m: number) => {
    const pr = t + f > 0 ? t / (t + f) : 0, rc = t + m > 0 ? t / (t + m) : 0;
    return pr + rc > 0 ? (2 * pr * rc) / (pr + rc) : 0;
  };
  // Averaged over classes actually present in the held-out split, so a class the split happens to
  // omit cannot be scored 0 and drag the mean down as if the model had missed it.
  const present = [...Array(N_CLASS).keys()].filter((c) => tpC[c] + fnC[c] > 0);
  const perClassF1 = present.map((c) => f1(tpC[c], fpC[c], fnC[c]));
  return {
    accuracy: Zte.length ? correct / Zte.length : 0,
    macroF1: present.length ? perClassF1.reduce((a, v) => a + v, 0) / present.length : 0,
    worstClassF1: present.length ? Math.min(...perClassF1) : 0,
    binaryF1: f1(tp, fp, fn),
    nParams: N_CLASS * N_FEAT + N_CLASS, nTest: Zte.length, nPerClassUsed: nPer,
    explain: explainSample(),
  };
}
