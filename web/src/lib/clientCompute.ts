// Client-side compute for the public demo: in-browser equivalents of the backend
// benchmark + E2E pipeline, so the demo runs entirely in the browser (PQC via @noble,
// QKD/QBER via quantumSim, private inference via onnxruntime-web, PSI via WebCrypto HMAC).
// The security-score and quantum-advantage formulas mirror src/qsh/benchmark/{security,
// quantum_advantage}.py so demo numbers track the server.
import { ml_dsa65, ml_dsa87 } from "@noble/post-quantum/ml-dsa.js";
import { ml_kem768, ml_kem1024 } from "@noble/post-quantum/ml-kem.js";
import { x25519 } from "@noble/curves/ed25519.js";
import { hkdf } from "@noble/hashes/hkdf.js";
import { sha256 } from "@noble/hashes/sha2.js";
import {
  benchCfg, costPerOp, dataCfg, demoFeatureDim, genericArchitecture, idsCfg, medicalCnn, qkdCfg, qrngCfg,
  demoSample, federatedCfg, qsmpcCfg, quantumAdvantageCfg, securityScoringCfg, studentModel, tierArchitectures, webCfg,
  workloadMultipliers,
} from "./config";
import { encryptedConv2d, encryptedLinear, encryptedStudentInference, warmFHE,
  type CnnStudentWeights } from "./clientFHE";
import { clientTrainScoreIDS } from "./clientIDS";
import { clientQFL } from "./clientQFL";
import { secureArgmax, warmMPC } from "./clientMPC";
import { optimizeQKD } from "./clientOptiQKD";
import { generateQuantumBits } from "./clientQRNG";
import { bb84, channelLossDb, e91, FIBER, FREE_SPACE, h2, intrinsicQber, qcnnExpectation,
  qcnnInferTrained, qotpQcnnInfer, rng, taggedFraction,
  type QcnnWeights } from "./quantumSim";
import { type Provenance } from "../components/ProvenanceBadge";
import { browserModels, evaluatePublishedModel, onnxStatus, runPublishedModel } from "./onnxInference";
import { realInput, realSatelliteAnomaly, realFederatedSet, type RealInput } from "./realData";

const r1 = (x: number) => Math.round(x * 10) / 10;
const r2 = (x: number) => Math.round(x * 100) / 100;
const argmax = (a: number[]) => a.reduce((bi, v, i, arr) => (v > arr[bi] ? i : bi), 0);
const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));
// small prime utilities for the QSMS secure-sum field Z_p (config.prime; 0 = auto-select a prime > the total)
/**
 * The largest field modulus this browser mirror will accept.
 *
 * Two independent limits meet just above it, so it is the honest ceiling rather than a round number:
 *
 *   - Arithmetic. `(acc + s0 + s1) % p` accumulates in a double. Past 2^53 the intermediate sum
 *     stops being exact and the "secure sum" is silently wrong -- no exception, just a wrong number
 *     under a correct-looking label.
 *   - Time. `isPrime` is trial division, so cost grows as sqrt(p). At 2^31 that is ~46k iterations
 *     per candidate and imperceptible; at 1e16 it is 1e8 per candidate and the tab stops responding.
 *
 * 2^31-1 is also exactly the modulus the MCP allele-pooling path needs, which is why the paper's
 * n <= d <= 2n bound is recorded rather than enforced (see qsmpc/secure_summation.py).
 */
const MAX_FIELD_PRIME = 2147483647;          // 2^31 - 1, itself prime

const isPrime = (n: number) => {
  if (!Number.isFinite(n) || n < 2) return false;
  for (let i = 2; i * i <= n; i++) if (n % i === 0) return false;
  return true;
};

/**
 * The next prime at or above `n`, bounded.
 *
 * The bound is the point. Typing 1e400 into the Parameters field produced `Infinity`, and
 * `isPrime(Infinity)` never terminates: `i * i <= Infinity` is always true and `Infinity % i` is
 * NaN, so it is an infinite loop rather than an exception. Nothing could catch it -- not the
 * try/catch around the stage, and not the Stop button, which never gets a turn on the event loop.
 */
const nextPrime = (n: number) => {
  if (!Number.isFinite(n)) return MAX_FIELD_PRIME;
  let p = Math.min(MAX_FIELD_PRIME, Math.max(2, Math.floor(n)));
  while (p <= MAX_FIELD_PRIME && !isPrime(p)) p++;
  return p > MAX_FIELD_PRIME ? MAX_FIELD_PRIME : p;
};

// --------------------------------------------------------------------------- //
//  Security score (mirror of qsh/benchmark/security.py)
// --------------------------------------------------------------------------- //
const KEM_CATEGORY: Record<string, number> = {
  "ml-kem-512": 1, "ml-kem-768": 3, "ml-kem-1024": 5,
  "ml-kem-768+x25519": 3, "ml-kem-1024+x25519": 5,
};
export function kemCategory(kem?: string | null): number {
  return kem ? (KEM_CATEGORY[kem.toLowerCase()] ?? 3) : 3;
}

// Crypto-agility selectors threaded from the run config (KEM / signature) — so picking ml-kem-1024
// (NIST cat-5) vs ml-kem-768 (cat-3) genuinely runs the larger KEM AND moves the security score, and
// ml-dsa-87 vs ml-dsa-65 genuinely signs with the chosen level. Defaults track the config default kem/sig
// (never a magic string): unknown/undefined → the cat-3 / level-65 default, matching DEMO_CRYPTO.defaults.
export function kemName(config: any): string {
  return String(config?.kem ?? "ml-kem-768+x25519");
}
export function kemImpl(kem?: string) {
  return kem && kem.includes("1024") ? ml_kem1024 : ml_kem768;
}
export function sigImpl(sig?: string) {
  return sig && sig.includes("87")
    ? { fn: ml_dsa87, name: "ML-DSA-87" }
    : { fn: ml_dsa65, name: "ML-DSA-65" };
}
// The benchmark rows' security score reflects the run's chosen KEM (cat-3/cat-5) and whether QKD is mixed
// in (config.qkd === "pqc-only" drops the ≥256-bit BB84 key ⇒ no defence-in-depth bonus). Both are
// crypto-agility selections, so swapping them visibly moves the surfaced score.
function secFromConfig(
  config: any, metrics?: Record<string, any> | null,
): { kem?: string; qkdBits?: number | null; qber?: number | null } {
  // `qber` is carried now, and its absence was not a small omission. The consumer at the QSMPC row
  // guards on `typeof sec?.qber === "number"` under a comment claiming the Eve slider had been wired
  // through - but this function returned only `{kem, qkdBits}`, so the guard was ALWAYS false and the
  // fallback `qkdCfg().misalignment` (0.015) was used unconditionally. One literal replaced another.
  //
  // Measured on the deployed demo before this fix: Eve 0.6 drove QBER to 0.16, past the 0.11 abort
  // threshold, and drove `secure_key_bits` to ZERO - and the security score went UP, 96 to 100. The
  // row scored full marks on a run whose quantum channel produced no key at all.
  //
  // `secure_key_bits` is the honest source for the key-bits term too: the run either produced key
  // material or it did not, and a pqc-only profile is not the only way to end up with none.
  const measuredQber = typeof metrics?.qber === "number" ? metrics.qber : null;
  const measuredBits = typeof metrics?.secure_key_bits === "number" ? metrics.secure_key_bits : null;
  return {
    kem: kemName(config),
    qkdBits: config?.qkd === "pqc-only" ? 0 : (measuredBits ?? 256),
    qber: measuredQber,
  };
}
/**
 * Simulated qubit count, clamped to the range the Parameters field advertises.
 *
 * The state vector is 2^n doubles, so this is the difference between a run and a dead tab. The
 * benchmark row clamped it and the QCNN inference path did not, which is the wrong way round: the
 * benchmark is a cost model, the inference actually allocates. Measured before the clamp: 22 qubits
 * took 4.4 s, 24 allocated 268 MB, and a negative value threw RangeError inside
 * `new Float64Array(1 << -1)`, aborting the stage and leaving `explanation: null` in the signed
 * payload. `min`/`max` on a number input do not constrain a typed value, so the clamp has to be here.
 */
const qubitsOf = (config: any) => {
  const n = Number(config?.qubits ?? benchCfg().quantum_qubits);
  if (!Number.isFinite(n)) return Number(benchCfg().quantum_qubits);
  return Math.min(14, Math.max(2, Math.round(n)));
};

export interface SecOpts {
  securityModel: string;
  pqcHybrid?: boolean;
  kem?: string;
  qkdKeyBits?: number | null;
  qber?: number | null;
  leakageBitsPerWeight?: number | null;
}

/** One reported security fact: a label, what the run measured, and where the claim comes from. */
export interface SecurityFact {
  label: string;
  value: string;
  /** The standard, specification or measurement this rests on. Empty is not allowed. */
  source: string;
  /** "standard" = checkable against a published document; "measured" = produced by this run. */
  kind: "standard" | "measured";
  /** Set when the run contradicts the claim, e.g. a QKD profile that produced no key. */
  warn?: string;
}

/**
 * What this run's security posture actually is, reported rather than synthesised.
 *
 * This replaces the 0-100 composite as the headline. The composite's weights summed to exactly 100
 * by construction, fourteen of seventeen carried no rationale, and the 68-to-96 gap it produced was
 * mostly an artefact of which arguments two functions chose to supply - +10 of it because one call
 * passed `qber=None` while the other passed a literal, and +6 the same way for leakage. It also
 * moved the wrong way: measured on the deployed demo, Eve at 0.6 with QBER 0.16 and ZERO secure key
 * bits scored 100 out of 100.
 *
 * Every entry below is either checkable against a published standard or produced by this run, and
 * each says which. No entry is a weight. The NIST category leads because it is the one figure here
 * that is comparable with another system: FIPS 203 anchors ML-KEM-768 at Category 3 and ML-KEM-1024
 * at Category 5 against AES/SHA reference problems.
 *
 * Deliberately NOT adopted: QERS (arXiv:2601.13399) normalises against an ESP32/Wi-Fi testbed and
 * needs RSSI, packet loss and energy that a browser has none of, with author-chosen weights and no
 * calibration; CARS (ePrint 2026/1467) publishes no formula, derives weights from a 12-engineer
 * Delphi round, and scores a codebase's migration readiness rather than a run. Importing either
 * would swap invented weights for different invented weights and add a citation implying a rigour
 * that does not transfer.
 */
export function securityFacts(config: any, metrics: Record<string, any>): SecurityFact[] {
  const s = securityScoringCfg();
  const kem = kemName(config);
  const cat = kemCategory(kem);
  const sig = String(config?.sig ?? "ml-dsa-65");
  const sigLevel = sig.includes("87") ? 5 : 3;
  const pqcOnly = config?.qkd === "pqc-only";
  const keyBits = typeof metrics.secure_key_bits === "number" ? metrics.secure_key_bits : null;
  const qber = typeof metrics.qber === "number" ? metrics.qber : null;
  const skr = typeof metrics.skr === "number" ? metrics.skr : null;

  const facts: SecurityFact[] = [
    { label: "KEM", value: `${kem} - NIST Category ${cat}`, kind: "standard",
      source: "FIPS 203; categories anchored to AES/SHA reference problems" },
    { label: "Signature", value: `${sig.toUpperCase()} - NIST Category ${sigLevel}`, kind: "standard",
      source: "FIPS 204" },
    { label: "KEM ciphertext", value: `${metrics.kem_ciphertext_bytes ?? "-"} bytes`, kind: "measured",
      source: "this run" },
    { label: "Signature size", value: `${metrics.signature_bytes ?? "-"} bytes`, kind: "measured",
      source: "this run" },
  ];

  if (pqcOnly) {
    facts.push({ label: "QKD key material", value: "none - PQC-only profile", kind: "measured",
      source: "operator selection; no quantum channel in this run" });
  } else {
    // The warning is the point. A QKD profile that produced no key is the case the composite scored
    // 100 on, because nothing checked whether the channel actually delivered.
    facts.push({
      label: "QKD key material",
      value: keyBits == null ? "not reported" : `${keyBits} secure bits`,
      kind: "measured",
      source: String(config?.qkd_protocol ?? qkdCfg().protocol) === "e91"
        ? "E91 / CHSH run, GLLP secure fraction"
        : "BB84 decoy-state run, GLLP secure fraction",
      warn: keyBits === 0
        ? "the QKD profile is selected but this channel produced NO secure key: the session rests on "
          + "the PQC hybrid alone"
        : undefined,
    });
    if (skr != null) {
      facts.push({ label: "Secret-key rate", value: `${skr} per pulse`, kind: "measured",
        source: "GLLP bound over the decoy-state estimate" });
    }
    // What the decoy states are actually buying on THIS channel. The runtime model is asymptotic
    // decoy-state, which already assumes the single-photon contribution has been bounded - so the
    // assumption is invisible unless the counterfactual is computed. Delta >= 1 means no detection
    // could be certified single-photon and the same hardware would yield nothing without decoys.
    const mu = qkdCfg().mu_signal;
    // The channel model AND the parameters fed to it both have to follow the use case.
    //
    // Round 29 fixed half of this. It emitted `usecase` so the MODEL switched to free-space on the
    // satellite page - and left the parameter reads on `qkd_distance_km` / `qkd_misalignment` /
    // `qkd_eve`, which are the FIBRE page's controls. Those keys are always present (they default from
    // `qkd.*` config), so the satellite page evaluated the free-space model at 50 km with fibre
    // misalignment: the tagged fraction went from 7.26 to 2.89 rather than to the free-space figure,
    // and the three controls that page actually renders - `sat_distance_km`, `sat_eve`, `sat_pulses` -
    // still could not move it. A half-corrected number is harder to catch than an uncorrected one,
    // because it moved.
    const sat = config?.usecase === "satellite_comms";
    const chan = sat ? FREE_SPACE : FIBER;
    const delta = taggedFraction(mu, { ...chan,
      distanceKm: Number((sat ? config?.sat_distance_km : config?.qkd_distance_km) ?? chan.distanceKm),
      // No misalignment control is rendered on the satellite page, so the free-space channel's own
      // 0.025 is the honest value there. Reading `qkd_misalignment` unconditionally would have
      // substituted the fibre page's 0.015 - always, since that key is never absent.
      misalignment: Number((sat ? undefined : config?.qkd_misalignment) ?? chan.misalignment),
      eveFraction: Number((sat ? config?.sat_eve : config?.qkd_eve) ?? 0) });
    facts.push({
      label: "Decoy-state necessity",
      // Always print the number. ">= 1" was a needless loss of information: on the default fibre
      // link Delta is 7.26, and "7.26" says the decoy states are carrying the entire link, where
      // ">= 1" only says they are carrying some of it.
      value: `tagged fraction ${delta.toFixed(2)} at mu ${mu}`,
      kind: "measured",
      source: "GLLP tagged fraction over this channel; Ma PRA 72 012326, GLLP QIC 4 325",
      warn: delta >= 1
        ? "without decoy states this link yields NO key. The worst-case bound on Eve-known "
          + "(multi-photon) detections exceeds the total detection probability, so not one "
          + "detection could be certified single-photon and the whole sifted block is discarded. "
          + "The decoy states are what make this link work at all."
        : undefined,
    });
  }

  if (qber != null) {
    // Two thresholds, and they disagree with each other. 0.11 is the textbook BB84 figure; this
    // repository's own GLLP model puts the zero-key point at 0.098106, BELOW it - so a QBER in
    // [0.098, 0.11) used to score full eavesdropper-detection marks on a channel already dead.
    // Report against the model that actually ran, and name the textbook figure beside it.
    const dead = skr != null ? skr <= 0 : qber >= 0.098106;
    facts.push({
      label: "QBER",
      value: `${qber}${dead ? " - no secure key at this error rate" : ""}`,
      kind: "measured",
      source: "this run's channel model; GLLP zero-key point 0.098106, textbook BB84 bound "
        + `${s.qber_secure_threshold}`,
      warn: dead ? "above this channel model's zero-key point" : undefined,
    });
  }
  return facts;
}

export function securityScore(o: SecOpts) {
  const s = securityScoringCfg();        // weights/thresholds from the central config (single source)
  const c: Record<string, number> = {};
  // Mirror of security.py: PQC-only hybrid scores the cap minus the bonus, QKD-mixed reaches the
  // cap. The previous min(cap, cap + bonus) made the QKD selector inert -- see the note there.
  c.hndl_pqc_hybrid =
    o.pqcHybrid === false ? 0
    : (o.qkdKeyBits && o.qkdKeyBits >= 256) ? s.hndl_pqc_hybrid
    : Math.max(0, s.hndl_pqc_hybrid - s.qkd_bonus);
  const it = (o.securityModel || "").toLowerCase().includes("information-theoretic");
  c.security_model = it ? s.info_theoretic : s.computational;
  const cat = kemCategory(o.kem);
  c.key_strength = (s.key_strength_by_category as Record<string, number>)[String(cat)] ?? 16;
  // Mirrors `benchmark/security.py`: 0.11 is the Shor-Preskill BB84 bound, but this project's GLLP
  // model yields zero secure key from 0.098106, so a QBER in between used to score FULL marks on a
  // dead channel. The fact display right below was already guarded against this; the SCORE was not.
  // `qkdKeyBits` is the field that carries this: `secFromConfig` fills it from the run's measured
  // `secure_key_bits`, so zero here means the protocol emitted no key, not merely that QKD was off.
  const deadKey = o.qkdKeyBits != null && o.qkdKeyBits <= 0;
  if (o.qber == null) c.eavesdropper_detection = 0;
  else if (deadKey) c.eavesdropper_detection = s.eavesdrop_warn_pts;
  else if (o.qber < s.qber_secure_threshold) c.eavesdropper_detection = s.eavesdrop_secure_pts;
  else if (o.qber < s.qber_danger_threshold) c.eavesdropper_detection = s.eavesdrop_warn_pts;
  else c.eavesdropper_detection = 0;
  const lk = o.leakageBitsPerWeight;
  if (lk == null) c.leakage_bound = it ? s.leakage_pts_default_it : s.leakage_pts_default_comp;
  else if (lk <= s.leakage_low) c.leakage_bound = s.leakage_pts_low;
  else if (lk <= s.leakage_mid) c.leakage_bound = s.leakage_pts_mid;
  else c.leakage_bound = s.leakage_pts_high;
  const total = r1(Math.min(100, Object.values(c).reduce((a, b) => a + b, 0)));
  return {
    total,
    components: Object.fromEntries(Object.entries(c).map(([k, v]) => [k, r1(v)])),
    kem_category: cat,
    summary:
      `${it ? "information-theoretic" : "computational"} security, NIST cat-${cat} KEM, ` +
      `HNDL-resistant hybrid${o.qkdKeyBits && o.qkdKeyBits >= 256 ? ", QKD-mixed" : ""}`,
  };
}

// --------------------------------------------------------------------------- //
//  Quantum advantage (mirror of qsh/benchmark/quantum_advantage.py)
// --------------------------------------------------------------------------- //
export function quantumAdvantage(classical: any, quantum: any, w?: { comm: number; latency: number; security: number }) {
  const qa = quantumAdvantageCfg();      // weights + tiers from the central config (single source)
  const ww = w ?? { comm: qa.w_comm, latency: qa.w_latency, security: qa.w_security };
  const cct = classical.ciphertext_plaintext_ratio, qct = quantum.ciphertext_plaintext_ratio;
  const comm = cct && qct && qct > 0 ? cct / qct : 1;
  const lat = classical.latency_ms && quantum.latency_ms > 0 ? classical.latency_ms / quantum.latency_ms : 1;
  const sec = (quantum.security_model || "").toLowerCase().includes("information-theoretic") ? qa.info_theoretic_multiplier : 1;
  const composite = comm ** ww.comm * Math.max(lat, 1e-6) ** ww.latency * sec ** ww.security;
  const raw = Math.log10(composite + 1) * qa.log_scale + qa.log_offset;
  const score = Math.max(0, Math.min(10, raw));
  const tier = score >= qa.tier_dominant ? "dominant" : score >= qa.tier_strong ? "strong"
    : score >= qa.tier_moderate ? "moderate" : "marginal";
  return {
    score: r2(score),
    // Uncapped composite, reported beside the clamped one so a 10/10 is distinguishable from a run
    // well past the ceiling. Never feeds the tier.
    score_raw: r2(raw), clamped: raw > 10 || raw < 0,
    tier, communication_ratio: r1(comm), latency_ratio: r2(lat),
    security_multiplier: sec,
    interpretation:
      `Quantum path is ~${Math.round(comm)}x more communication-compact` +
      (lat > 1 ? ` and ${lat.toFixed(1)}x faster` : "") + `; ${tier} advantage.`,
  };
}

// --------------------------------------------------------------------------- //
//  Keyed-hash PSI over WebCrypto HMAC-SHA256 (mirror of qsmpc/qot.py)
// --------------------------------------------------------------------------- //
async function hmacHex(key: Uint8Array, msg: string): Promise<string> {
  const k = await crypto.subtle.importKey(
    "raw", key as unknown as BufferSource, { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const data = new TextEncoder().encode(msg) as unknown as BufferSource;
  const sig = await crypto.subtle.sign("HMAC", k, data);
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function keyedHashPSI(setA: string[], setB: string[], key: Uint8Array) {
  const hb = new Map<string, string>();
  for (const e of setB) hb.set(await hmacHex(key, e), e);
  const inter: string[] = [];
  for (const e of setA) if (hb.has(await hmacHex(key, e))) inter.push(e);
  return { intersection: inter, size_a: setA.length, size_b: setB.length, intersection_size: inter.length };
}

// --------------------------------------------------------------------------- //
//  Private inference: published ONNX model -> bundled real model -> placeholder
// --------------------------------------------------------------------------- //
// A last-resort weight set, used only when neither a published ONNX model nor a bundled real model
// is available for the use case. It is arbitrary, so it is reported as SIMULATED: a number produced
// by these weights describes nothing. Before Round 17 this path ran on *every* page load — the ONNX
// model it was nominally backing up could never exist, because `*.onnx` is git-ignored and
// `web/public/models/` shipped empty — and the page presented its output as an inference result.
const PLACEHOLDER_W = [
  [0.12, -0.34, 0.08, 0.21, -0.15, 0.4, -0.05, 0.18],
  [-0.1, 0.27, -0.22, 0.05, 0.33, -0.18, 0.12, -0.07],
];
const PLACEHOLDER_B = [0.05, -0.05];

export type InferenceResult = {
  logits: number[];
  prediction: number;
  backend: string;
  provenance: Provenance;
  modelKey: string | null;
  note?: string;
};

/**
 * Private inference for one feature vector.
 *
 * Two tiers, and which one ran is reported rather than left to be inferred from the output:
 *   1. the bundled real distilled model shipped with the demo   -> MEASURED (plain JS arithmetic)
 *   2. placeholder weights                                      -> SIMULATED
 *
 * There used to be a third above these - the published ONNX model - and it was unreachable. See the
 * note in the body: it ran on an 8-element vector no published model accepts.
 */
export async function privateInference(
  usecase: string, features: number[], realModel?: { W: number[][]; b: number[] } | null,
): Promise<InferenceResult> {
  const f = features.slice(0, 8);
  while (f.length < 8) f.push(0);

  // The published-ONNX tier used to be attempted HERE, on this 8-element vector, and it could never
  // succeed: no published model takes [1,8]. It takes [1,32], [1,1,28,28], [1,250,6], [1,19] or
  // [1,7]. So every run fetched a model - 11 MB and 1975 ms of the 2966 ms this function reported as
  // `client_compute_ms` - created a session, failed on input rank, and fell through to the line
  // below. A measured number that is two thirds download for an inference that cannot happen.
  //
  // The published model now runs in `runPublishedForStage`, on a row of its own corpus at its own
  // input shape, and reports its own latency. This function is what it always actually was: the
  // page's explainable path over its 8 real features, which is what the SHAP attribution and the
  // CKKS evaluation both consume.
  if (realModel?.W?.length) {
    const logits = realModel.W.map((row, j) =>
      row.reduce((a, wij, i) => a + wij * (f[i] ?? 0), 0) + (realModel.b?.[j] ?? 0));
    return {
      logits, prediction: argmax(logits), backend: "js-linear (bundled real model)",
      // Not measured-wasm: this loop is plain JS numbers. Claiming WASM here made every pipeline run
      // on every page render "MEASURED (local WASM)" without a wasm module ever being instantiated.
      provenance: "measured-js", modelKey: null,
      note: "bundled distilled linear model, evaluated in plain JS arithmetic",
    };
  }

  const logits = PLACEHOLDER_W.map((row, j) =>
    row.reduce((a, wij, i) => a + wij * f[i], 0) + PLACEHOLDER_B[j]);
  return {
    logits, prediction: argmax(logits), backend: "js-placeholder",
    provenance: "simulated", modelKey: null,
    note: onnxStatus().error
      ? `no model available: ${onnxStatus().error}`
      : "no published model registered for this use case; placeholder weights",
  };
}

// Exact linear-SHAP (mirror of qsh/ai/explainability.py): phi_i = w_eff_i * (x_i - baseline_i),
// w_eff = W[1] - W[0] (one-vs-rest margin), base = w_eff . baseline + (b1 - b0).
export function linearShap(
  W: number[][], b: number[], x: number[], names: string[], topK = 5, baseline?: number[],
) {
  const n = W[0].length;
  const xs = x.slice(0, n);
  while (xs.length < n) xs.push(0);
  const base = baseline ?? new Array(n).fill(0);
  const wEff = W[1].map((w1, i) => w1 - W[0][i]);
  const bEff = b[1] - b[0];
  const phi = wEff.map((w, i) => w * (xs[i] - base[i]));
  const margin = wEff.reduce((a, w, i) => a + w * xs[i], 0) + bEff;
  const baseValue = wEff.reduce((a, w, i) => a + w * base[i], 0) + bEff;
  const attrs = phi.map((c, i) => ({
    feature: names[i] ?? `feature_${i}`, value: r2(xs[i]), contribution: Math.round(c * 1e6) / 1e6,
  }));
  const top = [...attrs].sort((a, b2) => Math.abs(b2.contribution) - Math.abs(a.contribution)).slice(0, topK);
  return {
    method: "exact-linear-shap", prediction: margin >= 0 ? 1 : 0,
    base_value: Math.round(baseValue * 1e6) / 1e6, margin: Math.round(margin * 1e6) / 1e6,
    top_features: top,
    // COMPUTED, not asserted. This was the literal `0`, which is algebraically right for this closed
    // form - sum_i w_eff_i (x_i - base_i) + (w_eff . base + db) collapses to w_eff . x + db, exactly
    // the margin - but a field named after a measurement and set to a constant cannot fail. If the
    // attribution rule ever changes and stops satisfying completeness, a hardcoded 0 goes on
    // claiming the axiom holds. Now it is the residual, so the claim is checkable at runtime.
    efficiency_gap: Math.abs(phi.reduce((a, c) => a + c, 0) + baseValue - margin),
  };
}

// Per-use-case feature names for the XAI attribution chart (mirror the backend).
const FEATURE_NAMES: Record<string, string[]> = {
  finance_aml: ["amount", "velocity_24h", "cross_border", "night_tx",
    "new_payee", "amount_zscore", "device_risk", "geo_mismatch"],
  medical_fl: ["mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "worst_radius", "worst_concavity", "worst_symmetry"],
};

// --------------------------------------------------------------------------- //
//  Benchmark (client) — faithful MPC-FHE hybrid mirror: linear→FHE, non-linear→MPC.
//  Mirrors /api/benchmark (the full deployment runs CKKS + MPyC; here the linear layers are
//  JS/WASM matmul standing in for the encrypted-GPU stage, and the non-linear layers run a
//  2-party additive-secret-sharing ReLU/argmax — same split, in the browser, no server.
// --------------------------------------------------------------------------- //
// Tier dims / per-use-case architectures / workload multipliers / per-op cost ALL come from the central
// config (config.ts ← /api/config ← config/defaults.py) — the demo cost model can no longer drift from
// the backend (single source of truth). N_IN = the demo feature width; cost_per_op are representative
// encrypted-op surrogates (plaintext-JS wall-clock can't reproduce CKKS cost, so the demo reports these;
// the full deployment measures real latency — the demo still *executes* the matmul + secret-sharing).
const tierDims = (size: string) => [demoFeatureDim(), ...(genericArchitecture()[size] ?? [16]), 2];
const MPC_COMM_BYTES_PER_COMPARE = () => costPerOp().mpc_comm_bytes_per_cmp;
const FHE_MS_PER_DOT = () => costPerOp().fhe_ms_per_dot;   // one CKKS encrypted linear dot per output neuron
const MPC_MS_PER_CMP = () => costPerOp().mpc_ms_per_cmp;   // one secret-sharing secure comparison (ReLU/argmax)
const tierOps = (dims: number[]) => dims.slice(1).reduce((a, x) => a + x, 0);  // output neurons = dots = comparisons

function archDims(usecase: string | undefined, modelSize: string, workload: string) {
  const wm = workloadMultipliers()[workload] ?? workloadMultipliers().linear;
  const [mult, wtag] = wm;
  // Genuine-CNN representative for medical (mirrors the backend's REAL MedMNIST convolution under FHE):
  // the conv layer's per-position×channel outputs dominate the encrypted cost (each is a dot_plain).
  if (usecase === "medical_fl" && workload === "cnn") {
    const mc = medicalCnn();
    const ch = (mc.channels as number[])[0] ?? 4;
    const pixels = mc.img_size * mc.img_size;
    const convOut = pixels * ch;                              // encrypted conv patch-dots (FHE cost)
    const dims = [pixels, convOut, mc.hidden, 2];             // cost model: conv outputs dominate tierOps
    const label = `diagnostic CNN (MedMNIST-scale, ${(mc.channels as number[]).length}-block) · ${modelSize}`;
    const convParams = ch * (1 * 3 * 3) + ch;                // shared 3x3 kernels (not pixels×convOut)
    const poolOut = Math.floor(mc.img_size / 2) ** 2 * ch;
    const params = convParams + poolOut * mc.hidden + mc.hidden + mc.hidden * 2 + 2;
    return { dims, label, params };
  }
  // The REAL encrypted model, when this use case has one. `tier_architectures` describes MLPs that
  // nothing ever trained -- 2,722 parameters for medical against a student that genuinely has
  // 3,226 and a teacher with 11.17 million -- and Round 19 deleted the selector that chose between
  // them without removing what it selected, leaving the benchmark silently costing a fiction at
  // "medium". Prefer the t1_crypto student's measured parameter count: it is the model CKKS
  // actually evaluates, so it is the right thing for an encrypted-cost row to describe.
  const student = usecase ? studentModel(usecase) : null;
  const studentParams = Number((student?.achieved as { student_params?: number } | undefined)?.student_params ?? 0);
  if (studentParams > 0) {
    const inDim = demoFeatureDim();
    // Two layers whose parameter count matches the student's, so tierOps tracks a real model's
    // width instead of an invented one. The student is linear for three of the four use cases and
    // one small conv block for medical; either way its parameter count is what was measured.
    const width = Math.max(2, Math.round((studentParams - 2) / Math.max(1, inDim + 1)));
    const dims = [inDim, width, 2];
    const params = studentParams;
    return { dims, label: `${student?.label ?? "encrypted student"} (${params} params, measured)`, params };
  }
  const a = usecase ? tierArchitectures()[usecase] : undefined;
  const base = a ? (a.hidden[modelSize] ?? [16]) : (genericArchitecture()[modelSize] ?? [16]);
  const hidden = base.map((h) => Math.max(2, Math.round(h * mult)));
  const dims = [demoFeatureDim(), ...hidden, 2];
  const label = (a ? a.label : "MLP") + ` · ${modelSize}${wtag} (cost model, not a trained model)`;
  const params = dims.slice(0, -1).reduce((s, inp, i) => s + inp * dims[i + 1] + dims[i + 1], 0);
  return { dims, label, params };
}

// The demo's REAL in-browser run measures a small (n->2) model; this scale = (selected architecture's op count)
// / (that measured model's op count) lets both the measured classical row AND the quantum-sim row genuinely
// track lightweight/medium/large × linear/cnn/transformer, while keeping the real per-op measurement as the
// unit cost. lightweight+linear === the measured model -> scale 1.0 (reports the raw measurement).
function benchScale(usecase: string | undefined, modelSize: string, workload: string): number {
  const ops = tierOps(archDims(usecase, modelSize, workload).dims);
  const baseOps = Math.max(1, tierOps(archDims(usecase, "lightweight", "linear").dims));
  return ops / baseOps;
}

// 2-party additive-secret-sharing non-linear (representative LAN sim): ReLU then (last layer) argmax.
function mpcNonlinear(vec: number[], isLast: boolean, u: () => number) {
  // additive share x = s0 + s1 with s0 pseudo-random; a real 2PC (MPyC on the server) computes
  // ReLU/argmax via secure comparison on the shares — here we model the shares + count the
  // communication, and reconstruct in-sim (representative, not cryptographic).
  const shares = vec.map((x) => { const s0 = (u() * 2 - 1) * 4; return [s0, x - s0]; });
  const recon = shares.map(([a, b]) => a + b);
  const out = isLast ? [recon.indexOf(Math.max(...recon))] : recon.map((v) => Math.max(0, v));
  return { out, compares: vec.length, commBytes: vec.length * MPC_COMM_BYTES_PER_COMPARE() };
}

function hybridForward(dims: number[], u: () => number) {
  // deterministic weights; linear layers timed as the FHE stage, non-linear as the MPC stage
  const layers = dims.slice(0, -1).map((inp, i) => ({
    W: Array.from({ length: dims[i + 1] }, () => Array.from({ length: inp }, () => (u() * 2 - 1) * 0.1)),
    b: Array.from({ length: dims[i + 1] }, () => (u() * 2 - 1) * 0.05),
  }));
  let h = Array.from({ length: dims[0] }, () => u() * 2 - 1);
  let fheMs = 0, mpcMs = 0, comm = 0, compares = 0;
  layers.forEach((L, i) => {
    let t = performance.now();
    const z = L.W.map((row, j) => row.reduce((a, w, k) => a + w * h[k], 0) + L.b[j]); // FHE-linear
    fheMs += performance.now() - t;
    t = performance.now();
    const nl = mpcNonlinear(z, i === layers.length - 1, u);                              // MPC-non-linear
    mpcMs += performance.now() - t;
    comm += nl.commBytes; compares += nl.compares;
    h = i === layers.length - 1 ? z : (nl.out as number[]);
  });
  return { fheMs, mpcMs, comm, compares };
}

// Measured real client-side FHE/MPC (node-seal CKKS + secret-sharing) — when present, mpcFheRow reports
// these REAL wall-clock numbers instead of the representative cost model.
export interface MeasuredHybrid {
  fheMs: number; mpcMs: number; commBytes: number; featDim: number;
  /** null when the serialized size could not be read. Not 0: a ciphertext is always larger than
   *  its plaintext, so a ct/pt ratio of 0 is impossible rather than small, and it was reaching
   *  the benchmark as a measured value. */
  ctBytes: number | null;
  /** Set when the run performed a real encrypted CONVOLUTION rather than a dot product. Counted,
   *  not estimated: rotations and the key-switch each triggers are what dominate CKKS cost. */
  conv?: { rotations: number; multiplies: number; patchesPerCiphertext: number;
           outH: number; outW: number; channels: number; convMs: number; maxAbsError: number;
           /** True when the PUBLISHED student's weights drove the computation, false when only its
            *  shape did. The two must never be reported as the same thing. */
           publishedWeights: boolean; totalRotations: number;
           studentParams?: number; studentPrediction?: number };
  /** Rotations and multiply-plain for the DENSE layer, so the two workloads compare like for like. */
  dense?: { rotations: number; multiplies: number };
}

export interface RealHybridRun extends MeasuredHybrid {
  prediction: number;
  maxAbsError: number;
  /** The decrypted logits and the plaintext reference the CKKS path was measured against.
   *  Carried so every use case can report the same encrypted-verification figures telecom got in
   *  Round 33 - a top-1 mismatch COUNT and bits of precision - instead of only a raw error magnitude
   *  whose scale a reader has no way to judge. */
  logits: number[];
  plainLogits: number[];
}

let _fheWarmed = false;
/** Whether the one-time CKKS/MPC warm-up has already run in this tab. */
const fheReady = () => _fheWarmed && _mpcWarmed;
let _mpcWarmed = false;

// Run a GENUINE encrypted hybrid inference entirely in the browser: a small linear model evaluated under
// REAL node-seal CKKS (clientFHE) for the linear layer + a REAL emp-wasm authenticated-garbling 2PC
// (clientMPC) for the non-linear argmax, all measured. Returns null only if the real crypto is genuinely
// unavailable here (e.g. a browser without WASM) — the caller then reports a clearly-labelled
// representative figure as an honest last resort (logged with the cause, not silently masked).
/** Fetch a published student's readable weights from its own pinned HF revision.
 *
 *  Returns null rather than throwing when the model publishes none: three of the four students are
 *  sklearn-linear and ship a different JSON shape, and a null here selects the shape-only path with
 *  the run saying so. No fallback weights are invented.
 */
const _studentWeightCache = new Map<string, CnnStudentWeights | null>();
async function publishedStudentWeights(usecase: string | undefined): Promise<CnnStudentWeights | null> {
  if (!usecase) return null;
  if (_studentWeightCache.has(usecase)) return _studentWeightCache.get(usecase) ?? null;
  let out: CnnStudentWeights | null = null;
  try {
    const spec = studentModel(usecase);
    const art = spec?.artifacts?.find((a) => a.path.endsWith("weights.json"));
    if (spec?.hf_repo_id && spec.hf_revision && art) {
      const url = `https://huggingface.co/${spec.hf_repo_id}/resolve/${spec.hf_revision}/${art.path}`;
      // Bounded for the same reason as the two in `onnxInference` - see HF_FETCH_TIMEOUT_MS there.
      const res = await fetch(url, { mode: "cors", signal: AbortSignal.timeout(30_000) });
      if (res.ok) {
        const j = await res.json();
        // Only the structural CNN encoding is evaluable here; a linear student's JSON is a
        // different shape and must not be coerced into this one.
        if (j && j.kind === "cnn-1conv-2dense") out = j as CnnStudentWeights;
      }
    }
  } catch {
    out = null;                       // offline or blocked: the shape-only path reports itself
  }
  _studentWeightCache.set(usecase, out);
  return out;
}

export async function realClientHybrid(features: number[], u: () => number,
                                      convolve = false,
                                      usecase?: string): Promise<RealHybridRun | null> {
  try {
    if (!_fheWarmed) { await warmFHE(); _fheWarmed = true; }     // one-time CKKS keygen
    if (!_mpcWarmed) { await warmMPC(); _mpcWarmed = true; }     // one-time summon compile + emp-wasm init
    const n = features.length;
    // small representative 2-class linear model (class 0 = zeros; class 1 = a fixed signed pattern). The
    // ENCRYPTION + computation are genuinely real (node-seal CKKS); the weights are a small demo model.
    const w1 = Array.from({ length: n }, (_, i) => ((i % 3) - 1) * 0.15);
    const W = [Array(n).fill(0), w1];
    const b = [0, -0.05];
    const fhe = await encryptedLinear(W, b, features);           // REAL CKKS, measured
    const mpc = await secureArgmax(fhe.logits, u);              // REAL garbled-circuit 2PC argmax, measured

    // The CNN workload now runs a REAL encrypted convolution instead of multiplying this dot
    // product's latency by a cost model. Shape comes from the trained student (`medical_cnn`),
    // not from `tier_architectures`: 14x14 in, 3x3 kernels, the student's own channel count.
    let conv: MeasuredHybrid["conv"];
    if (convolve) {
      const mc = medicalCnn();
      const size = Number(mc.img_size);
      const img = Array.from({ length: size }, (_, y) =>
        Array.from({ length: size }, (_, x) => features[(y * size + x) % features.length] ?? 0));
      // The PUBLISHED student's own weights when they can be fetched. Running the real network is
      // what turns this from a cost model of the right shape into the model's actual encrypted
      // inference, which is what makes crypto_fidelity checkable in a browser.
      const wts = await publishedStudentWeights(usecase);
      if (wts) {
        const r = await encryptedStudentInference(wts, img);
        conv = { rotations: r.rotations, multiplies: r.multiplies,
                 patchesPerCiphertext: r.patchesPerCiphertext, outH: r.outH, outW: r.outW,
                 channels: wts.conv.out_ch, convMs: r.fheMs, maxAbsError: r.logitMaxAbsError,
                 publishedWeights: true, totalRotations: r.totalRotations,
                 studentParams: wts.n_parameters, studentPrediction: r.prediction };
      } else {
        // No fetch, no invented weights: the shape is real but the kernels are a stated pattern,
        // and the run says which of the two it did rather than letting them look alike.
        const chans = Number((mc.channels as number[])[0] ?? 4);
        const kernels = Array.from({ length: chans }, (_, k) =>
          Array.from({ length: 3 }, (_, ky) =>
            Array.from({ length: 3 }, (_, kx) => ((ky * 3 + kx + k) % 5 - 2) * 0.1)));
        const cres = await encryptedConv2d(img, kernels, Array(chans).fill(0));
        conv = { rotations: cres.rotations, multiplies: cres.multiplies,
                 patchesPerCiphertext: cres.patchesPerCiphertext, outH: cres.outH, outW: cres.outW,
                 channels: chans, convMs: cres.fheMs, maxAbsError: cres.maxAbsError,
                 publishedWeights: false, totalRotations: cres.rotations };
      }
    }
    return {
      prediction: mpc.out[0], fheMs: fhe.fheMs + (conv?.convMs ?? 0), mpcMs: mpc.mpcMs,
      commBytes: mpc.commBytes, ctBytes: fhe.ctBytes, featDim: n,
      maxAbsError: fhe.maxAbsError, plainLogits: fhe.plainLogits, logits: fhe.logits, conv,
      dense: { rotations: fhe.rotations, multiplies: fhe.multiplies },
    };
  } catch (e) {
    // Surface the cause (no-fallback policy: a fixable defect must not hide behind the representative path).
    console.warn("[realClientHybrid] real in-browser FHE/MPC unavailable:", e);
    return null;
  }
}

// MPC-FHE row — uses REAL measured client-side FHE/MPC when ``measured`` is supplied, else a representative
// cost model (e.g. the standalone clientBenchmark with no live run). Shared by clientBenchmark + assembleBenchmark.
function mpcFheRow(usecase: string | undefined, workload: string, modelSize: string, measured?: MeasuredHybrid | null,
  // `qber` belongs here even though this row is the CLASSICAL path. Both rows run under ONE transport
  // configuration, so both must be scored under it. This parameter used to omit it, so this row's
  // eavesdropper term was structurally zero while its counterpart threaded the value through - and
  // the 68-vs-96 gap on screen was that asymmetry, not a property of the cryptography.
  sec?: { kem?: string; qkdBits?: number | null; qber?: number | null }) {
  const { dims, label, params } = archDims(usecase, modelSize, workload);
  const cost = costPerOp();
  // ctpt is nullable: the measured branch withholds it when the serialized ciphertext size could
  // not be read, rather than reporting an impossible 0 or borrowing the cost model's literature
  // figure into a row labelled measured.
  let fhe: number, mpc: number, comm: number, backend: string, ctpt: number | null,
    projected: boolean, device: string;
  if (measured) {
    // The real run measures a small (n->2) model; scale that MEASURED per-op cost by the SELECTED architecture's
    // op count so the latency genuinely tracks lightweight/medium/large × linear/cnn/transformer (the unit cost
    // is real node-seal CKKS / emp-wasm; only the op count is the chosen architecture's).
    // When a real encrypted convolution ran, its latency is MEASURED and nothing is scaled: the
    // whole point of running one is to stop multiplying a dot product by a cost model. The scaling
    // path below survives only for the linear workload, where the measured model really is an
    // (n -> 2) layer standing in for a wider one.
    const scale = measured.conv ? 1 : benchScale(usecase, modelSize, workload);
    fhe = measured.fheMs * scale; mpc = measured.mpcMs * scale; comm = measured.commBytes * scale;
    projected = !measured.conv && modelSize === "large";
    backend = measured.conv
      ? `node-seal CKKS + emp-wasm 2PC (measured: real ${measured.conv.channels}-channel encrypted `
        + `convolution, ${measured.conv.rotations} rotations over `
        + `${measured.conv.patchesPerCiphertext} packed patches)`
      : "node-seal CKKS + emp-wasm 2PC (measured per-op, scaled to the selected architecture)";
    // null propagates. Previously a failed size read became 0 and this printed "ct/pt 0" beside
    // rows carrying real ratios - an impossible value in a MEASURED column. The cost-model
    // default is NOT substituted here either: this branch is the measured one, and quietly
    // borrowing a literature figure would relabel an estimate as a measurement.
    ctpt = measured.ctBytes == null || measured.featDim <= 0
      ? null
      : r2(measured.ctBytes / (measured.featDim * 8));
    device = "cpu";                                   // node-seal CKKS is WASM-CPU (no commercial-OK WebGPU FHE in 2026)
  } else {
    // The COST-MODEL branch: nothing here was measured. It used to name whatever execution provider
    // `onnxBackend()` happened to report, which was never this row's - and is now flatly wrong,
    // because the only session on the page belongs to the published model running on a different
    // corpus. A modelled row does not get to borrow a measured row's backend.
    const ops = tierOps(dims);
    fhe = ops * cost.fhe_ms_per_dot; mpc = ops * cost.mpc_ms_per_cmp; comm = ops * cost.mpc_comm_bytes_per_cmp;
    backend = "ckks + mpyc-shamir (cost model, not measured)";
    ctpt = cost.demo_ct_pt_ratio; projected = modelSize === "large";
    device = "cpu";
  }
  return {
    path: "MPC-FHE", backend, workload,
    model_size: modelSize, model_arch: label, model_params: params, projected,
    latency_ms: r2(fhe + mpc), throughput_ips: r2(1000 / Math.max(fhe + mpc, 1e-3)),
    // NOT measured in the browser. The backend row this mirrors computes accuracy as
    // (decisions matching plaintext)/n in classical_mpc_fhe.py; nothing here evaluates a model, so
    // reporting 1.0 as if it were a result would be the flattering-number pattern this project
    // rules out. null makes the panel print an em dash instead of a perfect score.
    accuracy: null, max_abs_error: cost.demo_max_abs_error, ciphertext_plaintext_ratio: ctpt,
    fhe_linear_ms: r2(fhe), mpc_nonlinear_ms: r2(mpc), mpc_comm_kb: r2(comm / 1024),
    t_gate_count: null, qubits: null,
    security_model: "computational (RLWE) + secret-sharing MPC",
    security_score: securityScore({
      securityModel: "computational (RLWE / FIPS-203 family)", kem: sec?.kem,
      qkdKeyBits: sec?.qkdBits,
      qber: typeof sec?.qber === "number" ? sec.qber : null,
    }).total,
    device,
  };
}

// QSMPC-FHE row — GENUINE state-vector QOTP + Clifford+T QCNN inference (quantumSim.qotpQcnnInfer):
// `latency_ms` is the REAL-measured wall-clock of the simulation, `t_gate_count`/`qubits` are COUNTED from
// the actual circuit, and ct/pt = 1.0 is the QOTP perfect-hiding property (Rate-1 QFHE's ~1.3 is the
// classical-key-included literature figure). A state-vector SIMULATION, not quantum hardware.
function qsmpcFheRow(workload: string, modelSize: string, u: () => number, samples = 4, scale = 1,
  nQubits = benchCfg().quantum_qubits,
  sec?: { kem?: string; qkdBits?: number | null; qber?: number | null }) {
  // qubit count is a run-config knob (config.qubits) — it drives the 2^n state-vector size, the counted
  // T-gate total (n per conv round) and the measured latency, so the QCNN-qubits control genuinely moves
  // the quantum row (clamped >=2 for a valid circuit).
  // Clamped at both ends. The state vector is exact, so cost is 2^n and there is no graceful
  // degradation past ~14 -- the tab simply stops responding. The UI caps the field too; this is the
  // backstop for a value arriving any other way (URL, restored state, API).
  const nq = Math.min(14, Math.max(2, Math.round(nQubits)));
  qotpQcnnInfer(Array.from({ length: nq }, () => u() * 2 - 1), 0, nq);   // warm-up (drop the JIT spike)
  let last = qotpQcnnInfer(Array.from({ length: nq }, () => u() * 2 - 1), 1, nq);
  let total = 0;
  for (let i = 0; i < samples; i++) {
    last = qotpQcnnInfer(Array.from({ length: nq }, () => u() * 2 - 1), i + 2, nq);
    total += last.latencyMs;
  }
  // measured per-inference state-vector sim (fixed 8-qubit reference circuit), scaled to the selected
  // architecture so the quantum row tracks lightweight/medium/large × linear/cnn/transformer like the classical row.
  const latency = r2((total / samples) * scale);
  return {
    path: "QSMPC-FHE", backend: "qotp+qcnn (state-vector sim, not quantum HW)", workload, model_size: modelSize,
    // Names what this row IS, because the model column read "—" beside two rows that name real
    // models, which invites the reader to assume the blank is an oversight rather than the point.
    // There are no learned parameters here: qotpQcnnInfer applies a fixed H/X/Z/T/RY structure whose
    // angles come from its input, and the input is random.
    model_arch: `QOTP + ${nq}-qubit QCNN circuit (no trained weights)`,
    latency_ms: latency, throughput_ips: r2(1000 / Math.max(latency, 0.001)),
    // NOT 1.0, which is what this reported for three rounds while the MPC-FHE row eight lines above
    // carries a comment forbidding exactly that: "reporting 1.0 as if it were a result would be the
    // flattering-number pattern this project rules out". The two rows contradicted each other, and
    // the fabricated one was the flattering side of the contradiction - a reader saw the quantum
    // path as perfectly accurate, fastest and most secure, and one of those three was invented.
    //
    // There is nothing here to be accurate ABOUT. The circuit carries no trained weights and its
    // input is `u() * 2 - 1`, uniform noise. What this row legitimately measures is the state-vector
    // simulation's latency, its counted T-gate total, its qubit count and its security model.
    accuracy: null, max_abs_error: null,
    ciphertext_plaintext_ratio: 1.0,                  // QOTP: ciphertext = same #qubits as plaintext (genuine)
    t_gate_count: last.tGates, qubits: last.qubits,   // counted from the actual gate list
    security_model: "information-theoretic (QOTP perfect hiding)",
    security_score: securityScore({
      securityModel: "information-theoretic (QOTP perfect hiding)", kem: sec?.kem,
      // qber was the literal 0.0128 here while the panel described the score as reflecting "QBER
      // eavesdrop margin" for this run — so moving the Eve slider changed the QKD panel and left this
      // number untouched. Use the run's measured QBER when the pipeline produced one.
      qkdKeyBits: sec?.qkdBits === undefined ? 256 : sec.qkdBits,
      // null, not a config constant. `qkdCfg().misalignment` is the channel's baseline optical
      // error, not this run's QBER, and substituting it made an unmeasured term look measured -
      // scoring the full eavesdropper-detection points on every run including ones with no key.
      // compute_security_score already treats null as "not measured" and awards zero.
      qber: typeof sec?.qber === "number" ? sec.qber : null,
      // Was a hardcoded 0.1 - the PRX leakage bound - which is a LITERATURE figure, not a property
      // of this run. Nothing in the browser measures leakage, so asserting it bought this row the
      // most generous bucket (10 pts) over its counterpart's default (4). That was +6 of the gap.
      // null scores as unmeasured, which is what it is.
      leakageBitsPerWeight: null,
    }).total,
    device: "cpu-sim", simulated: true,
  };
}

export async function clientBenchmark(workload = "linear", samples = 20, modelSize = "medium") {
  const u = rng(7);
  const dims = tierDims(modelSize);
  // light + medium genuinely execute the hybrid mechanism client-side (matmul + secret-sharing,
  // zero /api); large is projected (not executed). Reported ms = the representative cost model.
  if (modelSize !== "large") { const reps = Math.min(samples, 6); for (let i = 0; i < reps; i++) hybridForward(dims, u); }
  const classical = mpcFheRow(undefined, workload, modelSize);
  const quantum = qsmpcFheRow(workload, modelSize, u, Math.min(samples, 6), benchScale(undefined, modelSize, workload));
  return { workload, model_size: modelSize, results: [classical, quantum],
    advantage: quantumAdvantage(classical, quantum) };
}

// Inference use cases produce both an MPC-FHE row + a QSMPC-FHE comparison; telecom/satellite get the
// quantum row only (compact, no advantage). Mirrors the backend QuantumComparisonStage shape.
const INFERENCE_USECASES = new Set(["finance_aml", "medical_fl", "genomics"]);

// Assemble the unified benchmark from a finished CLIENT run (demo, zero /api). The MPC-FHE split is
// the representative cost model tied to the run's model_size/workload; client_compute_ms is the real
// in-browser wall-clock of the private-inference stage (the "client compute" figure shown in the UI).
/**
 * The published model's own plaintext latency, as a benchmark row.
 *
 * Every other row in this table is either a cost model or the bundled 8-feature student, so the
 * table had no plaintext reference measured on a model anyone can download - the thing an FHE
 * overhead is actually a multiple OF. This row is that reference: one real forward pass of the
 * pinned Hugging Face model, on its own corpus, on the execution provider that ran it.
 *
 * Deliberately null for everything it did not measure. It is plaintext, so there is no FHE time, no
 * MPC time, no communication and no ciphertext ratio; it is one row, so there is no accuracy. The
 * panel prints em dashes for those, which is the correct reading.
 */
function publishedModelRow(m: any) {
  if (m?.published_inference_ms == null || m?.published_model_status === "failed") return null;
  return {
    // Short on purpose: this is a chart category as well as a table cell, and recharts drops a tick
    // whose neighbours it would overlap - "plaintext (published model)" was long enough to push the
    // QSMPC-FHE label off the latency chart entirely, leaving three bars with two labels. The model
    // column beside it names the model, so the extra words were never carrying anything.
    path: "plaintext",
    backend: m.published_model_backend ?? "onnx",
    workload: "published",
    model_size: null, model_arch: m.published_model ?? null, model_params: null, projected: false,
    latency_ms: r2(m.published_inference_ms),
    throughput_ips: r2(1000 / Math.max(Number(m.published_inference_ms), 1e-3)),
    accuracy: null, max_abs_error: null, ciphertext_plaintext_ratio: null,
    fhe_linear_ms: null, mpc_nonlinear_ms: null, mpc_comm_kb: null,
    t_gate_count: null, qubits: null,
    security_model: "none (plaintext reference)",
    // No security score: this row carries no cryptography, and scoring it would put a number in a
    // column that invites comparison with rows where the number means something.
    security_score: null,
    input_shape: m.published_input_shape ?? null,
    revision: m.published_model_revision ?? null,
  };
}

export function assembleBenchmark(
  usecase: string, config: any,
  opts: { metrics?: any; artifacts?: any; stageMs?: Record<string, number> },
) {
  const modelSize = config?.model_size ?? "medium";
  const workload = config?.workload ?? "linear";
  // Pass the run's metrics: this is the call site whose score the benchmark panel renders, and
  // without them the QBER and secure-key-bits terms fall back to constants.
  const sec = secFromConfig(config, opts?.metrics);
  const u = rng(7);
  const q = qsmpcFheRow(workload, modelSize, u, 6, benchScale(usecase, modelSize, workload), qubitsOf(config), sec);
  const clientComputeMs = opts?.metrics?.client_compute_ms ?? opts?.stageMs?.private_inference ?? null;
  // Present on every use case that ran one, including telecom and satellite, whose flows have no
  // private_inference stage and so never reach the MPC-FHE row below.
  const pub = publishedModelRow(opts?.metrics);
  if (!INFERENCE_USECASES.has(usecase)) {
    return { workload, model_size: modelSize, results: pub ? [q, pub] : [q], advantage: null,
      live: true, client_compute_ms: clientComputeMs,
      security_facts: securityFacts(config, opts?.metrics ?? {}) };
  }
  // use the REAL measured client-side FHE/MPC from the run (node-seal CKKS + secret-sharing) when present
  const measured: MeasuredHybrid | null = opts?.artifacts?._measured_hybrid ?? null;
  const c = mpcFheRow(usecase, workload, modelSize, measured, sec);
  return { workload, model_size: modelSize, results: pub ? [c, q, pub] : [c, q],
    advantage: quantumAdvantage(c, q),
    live: true, client_compute_ms: clientComputeMs, measured: !!measured,
    security_facts: securityFacts(config, opts?.metrics ?? {}) };
}

// --------------------------------------------------------------------------- //
//  Alice <-> Bob two-party E2E (mirror of qsh/benchmark/alice_bob_e2e.py)
// --------------------------------------------------------------------------- //
export async function clientAliceBob(seed = 0, nShared = 8) {
  const u = rng(seed + 11);
  const timing: Record<string, number> = {};
  const acct = () => `ACCT-${Math.floor(u() * 9e6 + 1e6)}`;

  let t = performance.now();
  const bob = ml_kem768.keygen();
  timing.bob_keygen_ms = r2(performance.now() - t);

  t = performance.now();
  const qkd = bb84(webCfg().demo_alice_bob_pulses, { ...FIBER, distanceKm: qkdCfg().distance_km }, seed);
  timing.qkd_negotiation_ms = r2(performance.now() - t);

  t = performance.now();
  const enc = ml_kem768.encapsulate(bob.publicKey);
  timing.alice_pqc_handshake_ms = r2(performance.now() - t);
  t = performance.now();
  const recovered = ml_kem768.decapsulate(enc.cipherText, bob.secretKey);
  timing.bob_pqc_handshake_ms = r2(performance.now() - t);
  const agree = enc.sharedSecret.length === recovered.length &&
    enc.sharedSecret.every((b, i) => b === recovered[i]);
  const sessionKey = enc.sharedSecret.slice(0, 32);

  const shared = Array.from({ length: nShared }, acct);
  const aliceList = [...shared, ...Array.from({ length: 200 }, () => `A-${Math.floor(u() * 9e6 + 1e6)}`)];
  const bobList = [...shared, ...Array.from({ length: 180 }, () => `B-${Math.floor(u() * 9e6 + 1e6)}`)];
  t = performance.now();
  const psi = await keyedHashPSI(aliceList, bobList, sessionKey);
  timing.psi_ms = r2(performance.now() - t);

  t = performance.now();
  const inf = await privateInference("demo", Array.from({ length: 8 }, () => u() * 2 - 1));
  timing.private_inference_ms = r2(performance.now() - t);

  const sec = securityScore({
    securityModel: "computational + QKD (two-party, TTP-free)", kem: "ml-kem-768",
    qkdKeyBits: qkd.secureKeyBits, qber: qkd.qber,
  });
  const totalMs = r2(Object.values(timing).reduce((a, b) => a + b, 0));
  const uniqShared = new Set(shared).size;
  return {
    parties: ["Alice", "Bob"],
    phases: [
      { name: "Bob key generation", party: "Bob", ms: timing.bob_keygen_ms },
      { name: "QKD (BB84)", party: "Alice+Bob", ms: timing.qkd_negotiation_ms },
      { name: "PQC handshake (initiate)", party: "Alice", ms: timing.alice_pqc_handshake_ms },
      { name: "PQC handshake (respond)", party: "Bob", ms: timing.bob_pqc_handshake_ms },
      { name: "Private set intersection", party: "Alice", ms: timing.psi_ms },
      { name: "Private inference (ONNX)", party: "Bob", ms: timing.private_inference_ms },
    ],
    timing_ms: timing, total_ms: totalMs,
    session_key_agreement: agree, kem: "ML-KEM-768",
    qber: r2(qkd.qber), qkd_secure_key_bits: qkd.secureKeyBits, qkd_mixed: qkd.secureKeyBits >= 256,
    psi_intersection_size: psi.intersection_size, psi_ground_truth: uniqShared,
    psi_correct: psi.intersection_size === uniqShared,
    private_inference_class: inf.prediction, private_inference_backend: inf.backend,
    security_score: sec.total, security_components: sec.components, security_summary: sec.summary,
    ttp_free: true, raw_data_shared: false,
  };
}

// --------------------------------------------------------------------------- //
//  In-browser E2E pipeline with Pause / Resume / Step / Stop
// --------------------------------------------------------------------------- //
// ---- Quantum Federated Learning (QFL) — client-side mirror (demo) ------------------------------

/**
 * The free-space channel this run selected, in one place.
 *
 * `satellite_qkd` and `qsdc_telemetry` each spread `FREE_SPACE` and overrode a subset of its fields
 * inline - four sites, three different subsets. `channelLossDb` was called WITHOUT `sat_eve` (right,
 * loss does not depend on Eve) and `intrinsicQber` WITH it (also right), which is exactly the kind
 * of distinction that survives only while one person holds all four sites in their head. Adding the
 * two link-budget keys would have made it eight overrides across four sites.
 *
 * `eveFraction` stays an explicit argument because the two callers genuinely differ: the counterfactual
 * that decides `link_budget_limited` re-evaluates the SAME link with Eve removed.
 */
function satChannel(config: Record<string, any>, eveFraction?: number) {
  const fs = FREE_SPACE.freeSpace!;
  return {
    ...FREE_SPACE,
    distanceKm: Number(config.sat_distance_km ?? FREE_SPACE.distanceKm),
    eveFraction: eveFraction ?? Number(config.sat_eve ?? FREE_SPACE.eveFraction),
    freeSpace: {
      ...fs,
      atmosphericDb: Number(config.sat_atmospheric_db ?? fs.atmosphericDb),
      pointingDb: Number(config.sat_pointing_db ?? fs.pointingDb),
    },
  };
}

export interface ClientEvent {
  type: string; stage: string; message: string; data: Record<string, any>; t: number;
}
export interface PipelineControl { pause(): void; resume(): void; step(): void; stop(): void; }
export interface PipelineHandlers {
  onEvent(e: ClientEvent): void;
  onEnd(status: { state: string; metrics: Record<string, any>; artifacts: Record<string, any> }): void;
}

// stage sequences match OrchestrationGraph + the backend stage.name values
const FLOWS: Record<string, string[]> = {
  finance_aml: ["data", "pqc_handshake", "qkd_key_agreement", "qsmpc_compute", "private_inference", "quantum_comparison", "result", "sign_result"],
  medical_fl: ["data", "pqc_handshake", "qkd_key_agreement", "federated_training", "private_inference", "quantum_comparison", "result", "sign_result"],
  genomics: ["data", "pqc_handshake", "qkd_key_agreement", "qsmpc_compute", "private_inference", "quantum_comparison", "result", "sign_result"],
  telecom_ids: ["data", "pqc_handshake", "qkd_key_agreement", "qml_ids", "quantum_comparison", "result", "sign_result"],
  satellite_comms: ["data", "pqc_handshake", "satellite_qkd", "qsdc_telemetry", "quantum_comparison", "result", "sign_result"],
  demo: ["data", "pqc_handshake", "qkd_key_agreement", "qsmpc_compute", "private_inference", "quantum_comparison", "result", "sign_result"],
};

export function clientFlow(usecase: string): string[] {
  return FLOWS[usecase] ?? FLOWS.demo;
}
// Trained QCNN weights, fetched once per use case. These three sit at MODULE scope deliberately.
// They were written flush-left INSIDE the `private_inference` case block, which reads as top level
// and is not: an ES module is always strict mode, so a function declaration in a block is scoped to
// that block. Two consequences, both invisible until something outside that case needed them - the
// telecom stage could not call `loadQcnnWeights` at all (TS2304), and `_qcnnCache` was re-created on
// every entry to the case, so the memo never survived a run and the fetch repeated. The HTTP cache
// hid the second one.
// Trained QCNN weights, fetched once per use case. Absent file is the normal case for a use case
// that has no trained circuit yet, and the caller degrades to CKKS rather than inventing a class.
const _qcnnCache: Record<string, QcnnWeights | null> = {};

/** Directory under `public/data` for a use case's bundled artefacts.
 *
 *  `ParameterPanel` decides whether to OFFER the QCNN option, and it must agree with whether the
 *  weights actually load or the control goes inert again. The two live in different modules, so
 *  `tests/test_qcnn_usecases.py` compares `QCNN_USECASES` against the files on disk using this same
 *  mapping rather than relying on the two staying in step by inspection. */
function dataDirFor(usecase: string): string {
  return usecase === "medical_fl" ? "medical" : usecase.split("_")[0];        // finance_aml -> finance
}

async function loadQcnnWeights(usecase: string): Promise<QcnnWeights | null> {
  if (usecase in _qcnnCache) return _qcnnCache[usecase];
  try {
    const r = await fetch(`${import.meta.env.BASE_URL}data/${dataDirFor(usecase)}/qcnn_weights.json`);
    // `r.ok` is NOT the test. This is an SPA: the host serves index.html with HTTP 200 for any path
    // that does not exist, so `r.ok` is true for a file that was never deployed. The old guard only
    // appeared to work because `r.json()` then threw on HTML and the catch swallowed it - a check
    // that reads as handling the case while the catch does the work. Content-type is the
    // discriminator, which is the same rule the verification checklist applies to `curl`.
    const json = (r.headers.get("content-type") ?? "").includes("application/json");
    _qcnnCache[usecase] = r.ok && json ? ((await r.json()) as QcnnWeights) : null;
  } catch { _qcnnCache[usecase] = null; }
  return _qcnnCache[usecase];
}


export function runClientPipeline(
  usecase: string, config: Record<string, any>, h: PipelineHandlers, startPaused = false,
): PipelineControl {
  let stages = clientFlow(usecase);
  // Quantum Federated Learning is a per-use-case WebUI option: when toggled on, inject a
  // federated_training stage (if the flow lacks one) right before the inference/compute step, so EVERY
  // simulator can run a real client-side QFL round (medical_fl already has the stage in its flow).
  if (config.qfl && !stages.includes("federated_training")) {
    const anchorIdx = stages.findIndex((s) =>
      ["private_inference", "qml_ids", "qsdc_telemetry", "quantum_comparison"].includes(s));
    const at = anchorIdx >= 0 ? anchorIdx : stages.length;
    stages = [...stages.slice(0, at), "federated_training", ...stages.slice(at)];
  }
  const seed = Number(config.seed ?? 0);
  const u = rng(seed + 3);
  const t0 = performance.now();
  const now = () => (performance.now() - t0) / 1000;
  const metrics: Record<string, any> = {};
  const artifacts: Record<string, any> = {};
  const state = { paused: startPaused, step: 0, stopped: false };

  const emit = (type: string, stage: string, message: string, data: Record<string, any> = {}) =>
    h.onEvent({ type, stage, message, data, t: now() });
  const metric = (stage: string, name: string, value: any) => {
    metrics[name] = value;
    emit("metric", stage, "", { name, value });
  };

  /**
   * Run this use case's PUBLISHED model here, on a row of its own corpus, and report what happened.
   *
   * Called from whichever stage carries this use case's inference -- `private_inference` for finance,
   * medical and genomics, `qml_ids` for telecom, `qsdc_telemetry` for satellite. It is a helper
   * rather than a stage of its own because every use case already has an inference stage, and adding
   * a seventh box to five diagrams to say "and also the real model" would describe the plumbing
   * rather than the protocol.
   *
   * This is a SECOND inference, not a replacement for the stage's own. The stage runs the page's
   * explainable path over real Wisconsin / PGS / PaySim / CIC-IDS features, which is what the SHAP
   * attribution and the CKKS evaluation both depend on. The published model is trained on a
   * different corpus (BreastMNIST 28x28, AMLworld 32-feature, ESA 250x6), so feeding it the stage's
   * features would be the category error this project refuses elsewhere. Both are real, both are
   * measured, and each says which corpus it is on.
   *
   * Until this existed the Models panel's Run buttons were the only thing in the product that ever
   * executed a pinned model: the pipeline truncated its features to 8, every published input rank
   * disagreed, the ONNX tier failed on all five use cases, and the run reported the bundled linear
   * model under a MEASURED badge.
   */
  /** Evaluate the published telecom student's 8 class scores under REAL CKKS, in this tab.
   *
   * This is the verification figure for telecom's encrypted tier, and the numbers it reports are the
   * two conventions the field actually uses rather than ones invented here. The top-1 mismatch COUNT
   * follows arXiv:2605.22237, whose Table IX reports "the integer number of top-1 mismatches between
   * plaintext polynomial evaluation and CKKS evaluation" and expects zero; the agreement percentage
   * follows Concrete-ML, which prints exactly that as "Similarity". Both are reported because at this
   * sample count a rounded 100% and "0 of 1" carry different information.
   *
   * Precision is reported in BITS, which is the unit OpenFHE and HElib both use - OpenFHE's own
   * examples print |log2(error)| floored - with the raw decimal kept beside it because seeing
   * -1.999998 against an exact -2 is what makes the approximation concrete. The error here is a MAX
   * and is labelled as one: OpenFHE averages, so a max is the stricter claim.
   *
   * The argmax is NOT computed under encryption and the log says so. CKKS has no native comparison;
   * all 8 scores are decrypted and the maximum is taken in the clear, which is the mainstream
   * construction and is what Concrete-ML's deployment model does. A homomorphic argmax costs 12.83 s
   * for 128 elements (Mazzone et al., USENIX Security 2025), and routing it through the MPC argmax
   * this repo already implements would demonstrate an architecture rather than establish a security
   * property, because both MPC parties share one JS heap - `docs/threat-model.md` records that.
   */
  async function runEncryptedTelecomStudent(stage: string, raw: number[] | undefined) {
    try {
      const r = await fetch(`${import.meta.env.BASE_URL}data/telecom/student_weights.json`);
      if (!r.ok) { emit("log", stage, `encrypted tier: weights unavailable (HTTP ${r.status})`); return; }
      const w = await r.json();
      if (!Array.isArray(w?.W) || !Array.isArray(raw)) {
        emit("log", stage, "encrypted tier: no input row to score, skipped");
        return;
      }
      // The student's OWN mean/scale, shipped in the same file as the weights it was fitted with.
      const mean = w.input?.mean ?? [], scale = w.input?.scale ?? [];
      const z = raw.map((v: number, i: number) => (v - (mean[i] ?? 0)) / (scale[i] || 1));

      if (!_fheWarmed) { await warmFHE(); _fheWarmed = true; }
      const fhe = await encryptedLinear(w.W, w.b, z);
      const plain = w.W.map((row: number[], j: number) =>
        row.reduce((a, wij, i) => a + wij * (z[i] ?? 0), 0) + (w.b?.[j] ?? 0));
      const encCls = argmax(fhe.logits), plainCls = argmax(plain);
      const maxAbs = Math.max(...fhe.logits.map((v, i) => Math.abs(v - plain[i])));
      // Bits of precision is -log2(error), and the sign is the whole point. This read
      // `Math.abs(Math.log2(maxAbs))` until Round 34, which made the best and the worst case print
      // IDENTICALLY: an error of 4.294e-9 gives 27 bits, and an error of 2^27 - a total CKKS blow-up,
      // eight orders of magnitude larger than the logits themselves - also gave "about 27 bits of
      // precision". Folding the sign turned a catastrophe into a healthy-looking number in the one
      // metric that exists to detect it. An error at or above 1 has no bits of precision to report,
      // so it reports as a failure instead of as a plausible positive.
      const bits = maxAbs > 0 && maxAbs < 1 ? Math.floor(-Math.log2(maxAbs)) : null;
      const precisionFailed = maxAbs >= 1;

      metric(stage, "encrypted_student_params", Number(w.n_parameters) || null);
      metric(stage, "encrypted_student_class", w.classes?.[encCls] ?? encCls);
      metric(stage, "encrypted_student_ms", r2(fhe.fheMs));
      metric(stage, "encrypted_top1_mismatches", encCls === plainCls ? 0 : 1);
      metric(stage, "encrypted_scores_compared", w.W.length);
      metric(stage, "encrypted_max_abs_error", maxAbs);
      metric(stage, "encrypted_precision_bits", bits);
      metric(stage, "encrypted_argmax_location", "client, plaintext");
      emit("log", stage, `encrypted tier (REAL CKKS, ${w.n_parameters} params, `
        + `${w.W.length} dot products): class "${w.classes?.[encCls] ?? encCls}" in ${r2(fhe.fheMs)} ms. `
        + `Top-1 mismatches vs the plaintext reference: ${encCls === plainCls ? 0 : 1} of `
        + `${w.W.length ? 1 : 0} scored inputs. Max abs error ${maxAbs.toExponential(3)}`
        + (bits !== null ? ` (about ${bits} bits of precision, MAX not mean)` : "")
        + (precisionFailed
          ? ` -- PRECISION FAILURE: the error is at or above 1, so there are no bits of precision to `
            + `report. The decrypted result is not a usable approximation of the plaintext one.`
          : "") + ".");
      emit("log", stage, `the ARGMAX is NOT encrypted: all ${w.W.length} class scores are decrypted `
        + `and the maximum taken in the clear, because CKKS has no native comparison. That is the `
        + `standard construction and it reveals nothing here, since in one tab the client is the user.`);
    } catch (e) {
      // No invented numbers on failure. A skipped encrypted tier is reported as skipped.
      emit("log", stage, `encrypted tier skipped: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function runPublishedForStage(stage: string) {
    // The visitor's choice where they made one, else the registry's first fetchable model. Three use
    // cases publish more than one and could previously only ever exercise the first: medical ships
    // ResNet-18, ResNet-50 and the CKKS student, and satellite ships four across two tasks that are
    // not comparable with each other.
    let chosen = String(config.published_model ?? "") || undefined;
    // Ignore a selection that belongs to a different use case. The UI resets on a use-case change,
    // so this is the belt to that braces: a stale key reaching here would otherwise resolve to no
    // target at all and be reported as "this use case has no published model", which is a much
    // stronger and quite wrong claim.
    if (chosen && !browserModels(usecase).some((m) => m.spec.key === chosen)) {
      emit("log", stage, `ignoring published-model selection "${chosen}": it belongs to another use `
        + `case. Running this one's default instead.`);
      chosen = undefined;
    }
    const pub = await runPublishedModel(usecase, {
      rowIndex: Math.floor(u() * 4096), modelKey: chosen,
    });
    if (pub.kind === "ran") {
      const p = pub.run;
      metric(stage, "published_model", p.modelKey);
      metric(stage, "published_model_revision", p.revision.slice(0, 12));
      metric(stage, "published_model_backend", p.backend);
      metric(stage, "published_inference_ms", r2(p.latencyMs));
      metric(stage, "published_input_shape", p.inputShape.join("x"));
      // A forecaster has no class. Reporting an argmax over its 6 continuous channel outputs as a
      // "predicted class" printed a channel index dressed up as a verdict.
      let outcome: string;
      if (p.task === "forecast") {
        if (p.rowMae != null) metric(stage, "published_row_mae", p.rowMae);
        outcome = p.rowMae == null
          ? `Forecast ${p.logits.length} channels (the sample ships no target for this row)`
          : `Forecast absolute error ${p.rowMae} on this row, over ${p.logits.length} channels`;
      } else {
        metric(stage, "published_predicted_class", p.predictedClass);
        if (p.trueClass != null) metric(stage, "published_true_class", p.trueClass);
        outcome = `Predicted ${p.className ?? p.predictedClass}`
          + (p.trueClass != null ? ` (true ${p.trueClassName ?? p.trueClass})` : "");
      }
      emit("log", stage,
        `published model ${p.modelKey} ran here on ${p.backend}: row ${p.rowIndex} of its own `
        + `held-out sample, input ${p.inputShape.join("x")}, ${r2(p.latencyMs)} ms. ${outcome}`
        + `. Weights from ${p.repoId} at ${p.revision.slice(0, 12)}, ${p.provenance}`);
    } else if (pub.kind === "failed") {
      // Deliberately NOT substituted. A published model that exists and did not run is a reportable
      // failure, not an occasion to quietly show a different model's number in its place.
      metric(stage, "published_model", pub.modelKey);
      metric(stage, "published_model_status", "failed");
      emit("log", stage, `published model ${pub.modelKey} did NOT run here: ${pub.error}. `
        + `No substitute figure is shown for it.`);
    } else {
      // Recorded as a metric, not only logged: the diagram has to tell "this use case has no
      // published model" apart from "the run has not reached this stage yet", and both otherwise
      // render as an empty field.
      metric(stage, "published_model_status", "unavailable");
      emit("log", stage, `no published model for this use case: ${pub.reason}`);
    }

    // ---- Verification, DURING the run rather than beside it ---------------------------------
    // The block above infers ONE row and reports its prediction. That demonstrates the model
    // executes; it does not demonstrate the model is the one whose numbers this page prints. Those
    // are different claims, and until now only the first was made at run time while the metric
    // panel asserted the second statically.
    //
    // So re-score the whole published held-out sample here and compare against the figure recorded
    // when it was trained. A mismatch is surfaced as a FAILED verification rather than a smaller
    // number, because a model that does not reproduce its own published metric is not a weaker
    // result - it is a different model, a different sample, or a broken pipeline, and none of those
    // should be presented as a measurement.
    // EVERY published model for this use case, not only the one the run selected. medical publishes
    // three and satellite four; verifying just the selected one left the others asserted-but-never-
    // checked, and the standalone "run this model" button that used to cover them was removed in
    // favour of this. A visitor should not have to know which control re-checks which model.
    for (const target of browserModels(usecase)) {
      const key = target.spec.key;
      const isSelected = pub.kind === "ran" && pub.run.modelKey === key;
      try {
        const v = await evaluatePublishedModel(usecase, 0.05, key);
        if (v && v.ran && v.score != null && v.expected != null) {
          // Only the selected model's numbers become run metrics; the rest are logged. Otherwise
          // four models would overwrite each other's `verified_score` and the panel would show
          // whichever finished last.
          if (isSelected) {
            metric(stage, "verified_metric", v.metricName);
            metric(stage, "verified_score", r2(v.score));
            metric(stage, "verified_expected", r2(v.expected));
            metric(stage, "verified_n", v.n);
            metric(stage, "verification", v.matchesExpected ? "PASS" : "FAIL");
            // Carried as an artifact so the page can DRAW what was measured. A single number is the
            // right thing to gate on and the wrong thing to reason from: "minority-class F1 0.51"
            // does not say whether the classes separate or the threshold sat somewhere lucky.
            artifacts.verification = { modelKey: key, ...v };
          }
          const delta = Math.abs(v.score - v.expected);
          const tag = isSelected ? "" : " (not the selected model)";
          emit("log", stage, v.matchesExpected
            ? `verification PASS ${key}${tag}: re-scored all ${v.n} published held-out rows in this `
              + `tab and got ${v.metricName} ${r2(v.score)} against the recorded ${r2(v.expected)} `
              + `(|delta| ${delta.toExponential(2)}). The number this page prints was reproduced `
              + `here, not asserted.`
            : `verification FAIL ${key}${tag}: re-scoring ${v.n} rows gave ${v.metricName} `
              + `${r2(v.score)} but the recorded figure is ${r2(v.expected)} (|delta| `
              + `${delta.toExponential(2)}, tolerance 0.05). Treat every number on this page for `
              + `${key} as unverified until this is explained.`);
          // A model that cannot beat a trivial predictor is worth saying out loud even when it
          // reproduces its own figure exactly - reproducing a bad number is still reproducing it.
          if (!v.beatsBaseline && v.baseline != null) {
            emit("log", stage, `note: this model does not beat the trivial baseline on that sample `
              + `(${r2(v.score)} against ${r2(v.baseline)}). Reproducing a figure and being useful `
              + `are separate questions.`);
          }
        } else if (v && v.error) {
          if (isSelected) metric(stage, "verification", "ERROR");
          emit("log", stage, `verification could not run for ${key}: ${v.error}`);
        } else {
          // Distinct from a failure: some models publish no evaluation sample, and saying nothing
          // would let "not checked" read as "checked and fine".
          if (isSelected) metric(stage, "verification", "no-sample");
          emit("log", stage, `no published evaluation sample for ${key}, so its recorded metric `
            + `is NOT reproduced in this tab.`);
        }
      } catch (e) {
        if (isSelected) metric(stage, "verification", "ERROR");
        emit("log", stage, `verification threw for ${key}: ${(e as Error).message}`);
      }
    }
  }

  // `gate()` is awaited BETWEEN stages, so a pause or stop requested mid-stage does not take effect
  // until that stage finishes - and `private_inference` is seconds long. The UI used to flip its badge
  // to "paused"/"stopped" the instant the button was pressed, so the badge asserted a state the run
  // was demonstrably not in while the log kept streaming and metric rows kept appearing. The request
  // and its effect are now two different events, and the badge distinguishes them.
  async function gate() {
    let announced = false;
    while (!state.stopped) {
      if (!state.paused) return;
      if (state.step > 0) { state.step--; return; }
      if (!announced) { announced = true; emit("pipeline_pause_effective", "", "paused"); }
      await sleep(50);
    }
  }

  async function runStage(stage: string) {
    switch (stage) {
      case "data": {
        // These describe the BUNDLED DEMO SAMPLE, which is not always the corpus the published model
        // was trained on — finance is the clearest case: the demo ships PaySim while the published
        // model is trained on IBM AMLworld. The registry (ModelPanel below) shows the training
        // corpus; this line shows what is being scored in the browser right now.
        //
        // Five of these strings were wrong and had to be corrected against the primary sources,
        // because this is the line a visitor actually reads: PaySim is a *simulator* (Kaggle titles
        // it "Synthetic Financial Datasets For Fraud Detection"), PGS Catalog is per-score under
        // EMBL-EBI terms rather than blanket CC-BY, IGSR grants no licence, NASA SMAP/MSL carries no
        // licence statement from any provider, and UCI relicensed WDBC to CC BY 4.0 — so calling it
        // "public domain" dropped an attribution requirement.
        //
        // The text is NOT stored here. It comes from src/qsh/models/registry.py via config, because
        // it previously existed as three independent copies: this map, the shipped JSON files, and
        // the generator script. They drifted, and a claim the project had already published a
        // correction for went on being printed by the live demo.
        const ds = demoSample(usecase);
        if (ds) emit("log", stage, `dataset (bundled demo sample): ${ds.provenance}`);
        else emit("log", stage, "synthetic dataset prepared (client-side; no commercial-OK real set for this use case)");
        break;
      }
      case "pqc_handshake": {
        // Hybrid KEM selected via crypto-agility (config.kem): ml-kem-768 (cat-3) or ml-kem-1024 (cat-5).
        // BOTH halves are computed. This comment previously read "the +X25519 half is the classical
        // hedge (label)" and that was exactly the problem: every diagram, the Overview and the
        // Validations page said "ML-KEM-768 + X25519" while only ML-KEM ran, so the session key had
        // none of the property that makes a hybrid worth having — security if EITHER primitive holds.
        const kemSel = kemName(config);
        const kem = kemImpl(kemSel);
        const bob = kem.keygen();
        const e = kem.encapsulate(bob.publicKey);
        const rec = kem.decapsulate(e.cipherText, bob.secretKey);
        const pqOk = e.sharedSecret.every((b, i) => b === rec[i]);

        // Classical half: X25519 ECDH, then a concatenate-then-hash combiner over both secrets and
        // the transcript. SHA3-256 rather than a bare XOR so neither half can be cancelled out by an
        // attacker who controls the other.
        const bobX = x25519.utils.randomSecretKey(), aliceX = x25519.utils.randomSecretKey();
        const bobXPub = x25519.getPublicKey(bobX), aliceXPub = x25519.getPublicKey(aliceX);
        const xA = x25519.getSharedSecret(aliceX, bobXPub), xB = x25519.getSharedSecret(bobX, aliceXPub);
        // Same HKDF-SHA256 combiner and domain-separation string as src/qsh/pqc/hybrid_kem.py, so a
        // demo run and a backend run derive the session key the same way.
        const INFO = new TextEncoder().encode("qsh/hybrid-kem/v1 ML-KEM-768+X25519|");
        const bind = (pq: Uint8Array, cl: Uint8Array) => {
          const tr = new Uint8Array(bob.publicKey.length + bobXPub.length + aliceXPub.length);
          tr.set(bob.publicKey, 0);
          tr.set(bobXPub, bob.publicKey.length);
          tr.set(aliceXPub, bob.publicKey.length + bobXPub.length);
          const info = new Uint8Array(INFO.length + tr.length);
          info.set(INFO, 0); info.set(tr, INFO.length);
          const ikm = new Uint8Array(pq.length + cl.length);
          ikm.set(pq, 0); ikm.set(cl, pq.length);
          return hkdf(sha256, ikm, undefined, info, 32);
        };
        const sessA = bind(e.sharedSecret, xA), sessB = bind(rec, xB);
        const ok = pqOk && sessA.every((b: number, i: number) => b === sessB[i]);

        (config as any)._sessionKey = sessA.slice(0, 32);
        metric(stage, "session_key_bits", sessA.length * 8);
        metric(stage, "kem_category", kemCategory(kemSel));
        metric(stage, "kem_ciphertext_bytes", e.cipherText.length);
        metric(stage, "classical_public_key_bytes", aliceXPub.length);
        // kemSel already carries the "+x25519" suffix from the crypto-agility selector, so naming
        // X25519 again here read as "ml-kem-768+x25519 + X25519".
        emit("log", stage, `hybrid KEM = ${kemSel} (NIST cat-${kemCategory(kemSel)}) — `
          + `agreement=${ok}; ${e.cipherText.length} B ML-KEM ciphertext + ${aliceXPub.length} B X25519 `
          + `ephemeral public key, combined with HKDF-SHA256(pq_ss || x25519_ss), info="qsh/hybrid-kem/v1" — the same combiner the backend uses`);
        // Quantum entropy source (simulated state-vector measurement, not hardware) — measured health + throughput.
        const qr = generateQuantumBits(Number(config.qrng_bits ?? qrngCfg().bits),
          { source: "sim", postProcessing: String(config.qrng_post ?? qrngCfg().post_processing),
            epsilonLog2: Number(config.qrng_eps_log2 ?? qrngCfg().security_epsilon_log2), seed });
        metric(stage, "qrng_health_passed", qr.healthPassed ? 1 : 0);
        metric(stage, "qrng_throughput_kbps", r1(qr.throughputBitsPerSec / 1000));
        metric(stage, "qrng_min_entropy_per_bit", r2(qr.minEntropyPerBit));
        metric(stage, "qrng_bits_delivered", qr.nBits);
        metric(stage, "qrng_extraction_rate", r2(qr.extractionRate));
        // The leftover-hash-lemma ceiling. Reported on every run because it is the only observable
        // the LHL eps parameter moves: delivered output is min(bound, requested), and requested is
        // normally the smaller, so eps -1 against eps -128 produced byte-identical bits and the
        // control read as inert. It was never inert - the bound was simply never shown.
        if (qr.lhlMaxBits != null) {
          metric(stage, "qrng_lhl_max_bits", qr.lhlMaxBits);
          if (qr.nBits < qr.requestedBits) {
            emit("log", stage, `QRNG delivered ${qr.nBits} bits, not the ${qr.requestedBits} requested: `
              + `at eps 2^${Number(config.qrng_eps_log2 ?? qrngCfg().security_epsilon_log2)} the leftover `
              + `hash lemma permits only ${qr.lhlMaxBits} from this much raw min-entropy. Truncated to `
              + `the bound rather than padded past it.`);
          }
        }
        // INCONCLUSIVE is a third outcome, not a shade of FAIL. Below the adaptive-proportion
        // window both continuous tests return "passed" by short-circuit, so a 1-bit output used to
        // report health PASS beside a min-entropy of 0.
        metric(stage, "qrng_health_conclusive", qr.healthConclusive ? 1 : 0);
        if (!qr.healthConclusive) {
          emit("log", stage, `QRNG health tests INCONCLUSIVE on ${qr.nBits} bits: SP 800-90B 4.4 needs `
            + `at least 1024 for the adaptive-proportion window. Both tests short-circuit to "passed" `
            + `below it, so this is reported as inconclusive rather than as a pass.`);
        }
        emit("log", stage, `QRNG (simulated quantum measurement): ${qr.nBits} bits, health `
          + `${qr.healthConclusive ? (qr.healthPassed ? "PASS" : "FAIL") : "INCONCLUSIVE"} per SP 800-90B 4.4 `
          + `(repetition-count ${qr.repetitionCountLongest}/${qr.repetitionCountCutoff}, `
          + `adaptive-proportion ${qr.adaptiveProportionMax}/${qr.adaptiveProportionCutoff}); `
          + `SP 800-22 reported not gating: ${qr.sp80022Passed ? "pass" : "fail"} (monobit p=${qr.monobitP.toFixed(3)}), `
          + `${r1(qr.throughputBitsPerSec / 1000)} kbit/s`);
        break;
      }
      case "qkd_key_agreement": {
        if (config.qkd === "pqc-only") { emit("log", stage, "QKD disabled (PQC-only profile)"); break; }
        // Every field from config, not three of five. `fiber_loss_db_per_km` and
        // `detector_efficiency` are exported to the browser and were sitting unread in the same
        // qkdCfg() object this line already calls, while the values actually used came from the
        // FIBER literal in quantumSim.ts. They happen to agree today (0.2 and 0.25), which is
        // exactly why nobody noticed: editing defaults.py would have moved the Python channel model
        // and left the browser on the old constant, silently, with no error anywhere.
        const chan = { ...FIBER,
          distanceKm: Number(config.qkd_distance_km ?? qkdCfg().distance_km),
          lossDbPerKm: Number(qkdCfg().fiber_loss_db_per_km ?? FIBER.lossDbPerKm),
          detectorEff: Number(qkdCfg().detector_efficiency ?? FIBER.detectorEff),
          darkCount: Number(qkdCfg().dark_count_prob ?? FIBER.darkCount),
          misalignment: Number(config.qkd_misalignment ?? qkdCfg().misalignment),
          eveFraction: Number(config.qkd_eve ?? qkdCfg().eve_intercept_fraction) };
        // config.qkd_protocol genuinely selects which physical QKD model runs (BB84 decoy-state vs E91/CHSH),
        // so switching it visibly moves QBER / SKR / secure-key-bits — independently of the OptiQKD toggle.
        const protocol = String(config.qkd_protocol ?? qkdCfg().protocol);
        const pulses = Number(config.qkd_pulses ?? qkdCfg().pulses);
        const eRes = protocol === "e91" ? e91(pulses, chan, seed, qkdCfg().e91_visibility) : null;   // typed with chshS
        const res = eRes ?? bb84(pulses, chan, seed);
        // The eavesdropper fraction that actually ran. The telecom diagram labelled its QKD arrow
        // "(w/ eavesdropper)" unconditionally while `qkd_eve` defaults to 0.0, so a default run drew
        // an attack it did not simulate. Publishing the value lets the label read the run.
        metric(stage, "qkd_eve_fraction", chan.eveFraction);
        metric(stage, "qber", r2(res.qber));
        metric(stage, "skr", Math.round(res.secretKeyRate * 1e5) / 1e5);
        metric(stage, "secure_key_bits", res.secureKeyBits);
        if (eRes) metric(stage, "chsh_s", r2(eRes.chshS));
        emit("log", stage, eRes
          ? `E91 (entanglement / CHSH S=${r2(eRes.chshS)}): QBER=${res.qber.toFixed(4)}, ${res.secureKeyBits} secure bits`
          : `BB84 (decoy-state): QBER=${res.qber.toFixed(4)}, ${res.secureKeyBits} secure bits`);
        if (config.qkd_optimize) {   // AI-QKD parameter optimization (OptiQKD, WS1) — measured SKR gain
          const opt = optimizeQKD(chan, protocol,
            Number(config.qkd_opt_samples ?? qkdCfg().optimize_n_samples));
          if (opt.improvementPct !== null) metric(stage, "qkd_skr_improvement_pct", r1(opt.improvementPct));
          // BOTH ends of the ratio, and named for the model they come from. The optimiser scores a
          // SURROGATE (clientOptiQKD's analytic skrModel); the `skr` metric above comes from the
          // decoy-state bb84() simulation. They are different models, so a lone "optimized SKR"
          // printed beside `skr` read as the optimiser making things worse -- 0.0035 against
          // 0.00906 -- while the same panel claimed +48.8%. Emitting the surrogate's own baseline
          // makes the percentage checkable in place, and the names stop the two being compared.
          metric(stage, "qkd_surrogate_skr_default", Math.round(opt.baselineSkr * 1e6) / 1e6);
          metric(stage, "qkd_surrogate_skr_tuned", Math.round(opt.bestSkr * 1e6) / 1e6);
          metric(stage, "qkd_convergence_evals", opt.convergenceEvals);
          // Only quote the decoy intensities for protocols whose model reads them. The E91/CHSH
          // model ignores muSignal/muDecoy entirely, so printing "tuned" values there described a
          // search that never happened.
          const tuned = protocol === "e91"
            ? `basis p=${r2(opt.bestParams.basisProb)}`
            : `basis p=${r2(opt.bestParams.basisProb)}, mu_signal=${r2(opt.bestParams.muSignal)}, `
              + `mu_decoy=${r2(opt.bestParams.muDecoy)}`;
          emit("log", stage, opt.baselineIsDead
            ? `OptiQKD (coordinate ascent): baseline SKR is 0 — no secure key to improve, so no `
              + `percentage is reported (${tuned}; ${opt.nEvals} evals — simulated)`
            : `OptiQKD (coordinate ascent): surrogate SKR ${opt.baselineSkr.toExponential(2)} -> `
              + `${opt.bestSkr.toExponential(2)}, +${r1(opt.improvementPct as number)}% `
              + `(${tuned}; ${opt.nEvals} evals, converged in ${opt.convergenceEvals}). This is the `
              + `optimiser's own analytic model, NOT the decoy-state run above - do not compare it `
              + `with that run's skr, which a different model produced.`);
        }
        break;
      }
      case "satellite_qkd": {
        // Same guard the fibre stage carries. Without it, selecting pqc-only on satellite hid the
        // quantum arrow in the diagram while the run went on emitting qber, channel_loss_db and
        // secure_key_bits for a channel the page had just said was not there.
        if (config.qkd === "pqc-only") { emit("log", stage, "free-space QKD disabled (PQC-only profile)"); break; }
        const satChan = satChannel(config);
        const satProtocol = String(config.qkd_protocol ?? qkdCfg().protocol);
        const satPulses = Number(config.sat_pulses ?? webCfg().demo_sat_pulses);
        const satERes = satProtocol === "e91" ? e91(satPulses, satChan, seed, qkdCfg().e91_visibility) : null;
        const res = satERes ?? bb84(satPulses, satChan, seed);
        // Same key the fibre stage uses: one QKD stage runs per use case, and both report the same
        // quantity. The diagram's eavesdropper tag reads it rather than asserting one.
        metric(stage, "qkd_eve_fraction", satChan.eveFraction);
        metric(stage, "qber", r2(res.qber));
        metric(stage, "channel_loss_db", res.channelLossDb);
        metric(stage, "secure_key_bits", res.secureKeyBits);
        if (satERes) metric(stage, "chsh_s", r2(satERes.chshS));
        (config as any)._satQber = res.qber;
        emit("log", stage, satERes
          ? `free-space E91 (CHSH S=${r2(satERes.chshS)}): ${res.channelLossDb}dB loss, QBER=${res.qber.toFixed(4)}, ${res.secureKeyBits} bits`
          : `free-space BB84: ${res.channelLossDb}dB loss, QBER=${res.qber.toFixed(4)}, ${res.secureKeyBits} bits`);
        // The fibre stage returns before its optimise block under the PQC-only profile; this one
        // had no such guard, so the same checkbox did work here and nothing there.
        if (config.qkd_optimize && String(config.qkd ?? "") !== "pqc-only") {
          const opt = optimizeQKD(satChan, satProtocol,
            Number(config.qkd_opt_samples ?? qkdCfg().optimize_n_samples));
          if (opt.improvementPct !== null) metric(stage, "qkd_skr_improvement_pct", r1(opt.improvementPct));
          metric(stage, "qkd_surrogate_skr_default", Math.round(opt.baselineSkr * 1e6) / 1e6);
          metric(stage, "qkd_surrogate_skr_tuned", Math.round(opt.bestSkr * 1e6) / 1e6);
          metric(stage, "qkd_convergence_evals", opt.convergenceEvals);
          const tunedSat = satProtocol === "e91"
            ? `basis p=${r2(opt.bestParams.basisProb)}`
            : `basis p=${r2(opt.bestParams.basisProb)}, mu_signal=${r2(opt.bestParams.muSignal)}, `
              + `mu_decoy=${r2(opt.bestParams.muDecoy)}`;
          emit("log", stage, opt.baselineIsDead
            ? `OptiQKD (free-space): baseline SKR is 0 — no secure key to improve, so no percentage `
              + `is reported (${tunedSat}; ${opt.nEvals} evals — simulated)`
            : `OptiQKD (free-space, coordinate ascent): surrogate SKR `
              + `${opt.baselineSkr.toExponential(2)} -> ${opt.bestSkr.toExponential(2)}, `
              + `+${r1(opt.improvementPct as number)}% vs default `
              + `(${tunedSat}; ${opt.nEvals} evals, converged in ${opt.convergenceEvals} — simulated)`);
        }
        break;
      }
      case "qsdc_telemetry": {
        // The QSDC and free-space QKD figures come from an analytic channel model. There is no
        // browser API for quantum communication and there cannot be one, so this is SIMULATED and
        // the page says so rather than presenting a modelled number as a measurement.
        // The same guard the free-space QKD stage above carries. Without it, selecting pqc-only on
        // satellite logged "free-space QKD disabled (PQC-only profile)" and then, one stage later,
        // emitted qsdc_error_rate, qsdc_secrecy_capacity and qsdc_eavesdrop_detected - delivering
        // telemetry over the quantum channel the run had just said did not exist. QSDC is a quantum
        // direct-communication protocol; there is no PQC-only version of it to fall back to.
        if (config.qkd === "pqc-only") {
          emit("log", stage, "QSDC direct telemetry skipped (PQC-only profile): QSDC carries the "
            + "message on the quantum channel itself, so with no quantum channel there is nothing "
            + "to send it over. The payload travels under the PQC session key instead.");
          break;
        }
        metric(stage, "inference_provenance", "simulated");
        metric(stage, "inference_backend", "analytic free-space channel model (quantumSim)");
        // Same channel the satellite_qkd stage above built, distance included. Reading only sat_eve
        // here meant the downlink range moved the QKD QBER but not the QSDC error rate, so one
        // control produced two inconsistent pictures of the same link.
        const e = intrinsicQber(satChannel(config));
        // QSDC aborts on SECRECY CAPACITY, not on a fixed QBER. Qi et al., Light Sci. Appl. 8, 22
        // (2019), protocol step 2: "If the secrecy capacity C_s is zero, they conclude the channel
        // is insecure and terminate the process."
        //
        //     C_s = 1 - h(e) - g*h(e_chk),   g = Q_Eve / Q_Bob = 10^(loss_dB / 10)
        //
        // The scalar 0.11 used here before is the g = 1 (LOSS-FREE) root of that expression, which
        // is why it coincided exactly with the Shor-Preskill BB84 bound. Real links are lossy and g
        // grows with loss, so the true threshold falls fast: Qi 2019 measured g = 2.57 at 4.1 dB,
        // giving ~0.048. This page's own free-space downlink measures around 24.5 dB, where the
        // threshold is ~0.0003 - so 0.11 was wrong here by three orders of magnitude, and wrong in
        // the PERMISSIVE direction: it delivered telemetry over channels whose secrecy capacity was
        // already zero.
        //
        // The founding QSDC papers (Long-Liu 2002, Deng-Long-Liu 2003, DL04 2004) give no numeric
        // threshold at all - only "if the error rate is high, halt" - and no standards body has ever
        // specified one: ETSI, ITU-T, ISO/IEC and IEEE quantum series are all key distribution.
        const lossDb = channelLossDb(satChannel(config));
        const g = 10 ** (lossDb / 10);
        const secrecy = 1 - h2(e) - g * h2(e);
        const abort = secrecy <= 0;
        // WHY the abort fired matters more than THAT it fired, and reporting both as
        // `qsdc_eavesdrop_detected` was wrong. Re-evaluate the same link with Eve removed: if it
        // still cannot close, the limit is the LINK BUDGET, and the page was accusing a clean
        // channel of carrying an eavesdropper on every single run.
        //
        // It always does still cannot close, at every setting this demo offers. The fixed 5 dB
        // atmospheric + 3 dB pointing budget alone gives g = 6.31, and C_s > 0 there needs
        // QBER < ~0.019 while the misalignment floor is 0.025 - so even at zero range, with a
        // perfect pointing solution and no eavesdropper, secrecy capacity is negative. Moving
        // `sat_distance_km` cannot produce a delivery. That is not a bug in this code, it is the
        // finding: QSDC does not close over a LEO free-space downlink under the published
        // criterion. Qi et al. demonstrated it at 4.1 dB on FIBRE; this link is 24.5 dB.
        const secrecyNoEve = (() => {
          const e0 = intrinsicQber(satChannel(config, 0));
          return 1 - h2(e0) - g * h2(e0);
        })();
        const lossLimited = secrecyNoEve <= 0;
        metric(stage, "qsdc_error_rate", r2(e));
        // SIGNED. `Math.max(0, secrecy)` stood here, so a link missing by 51 bit/carrier and one
        // missing by 0.001 both reported "0" - and 0 also reads as marginal-but-not-negative, the
        // one thing it is not. Measured at 1 km with Eve at 1.0 the true value is -5.2033 and the
        // page displayed 0. How badly a link fails is the quantity that distinguishes "widen the
        // aperture" from "QSDC does not close over this class of link".
        metric(stage, "qsdc_secrecy_capacity", r2(secrecy));
        metric(stage, "qsdc_eve_gain_g", r2(g));
        metric(stage, "qsdc_link_budget_limited", lossLimited ? 1 : 0);
        // Reserved for what the name says: an abort this link would NOT have suffered on its own.
        metric(stage, "qsdc_eavesdrop_detected", abort && !lossLimited ? 1 : 0);
        artifacts.telemetry_delivered = !abort;
        emit("log", stage, !abort
          ? `QSDC delivered telemetry (secrecy ${secrecy.toFixed(4)} bit/carrier, g=${g.toFixed(1)})`
          : lossLimited
            ? `QSDC NOT AVAILABLE on this link — secrecy capacity ${secrecy.toFixed(4)} bit/carrier `
              + `at ${lossDb.toFixed(1)} dB (g=${g.toFixed(1)}), and it is still `
              + `${secrecyNoEve.toFixed(4)} with NO eavesdropper. The limit is the link budget, not `
              + `an attack: C_s > 0 requires h(QBER) < 1/(1+g) = ${(1 / (1 + g)).toExponential(2)}, `
              + `and h(${e.toFixed(4)}) = ${h2(e).toFixed(4)} here. `
              + `Telemetry falls back to the PQC session key.`
            : `QSDC ABORTED: secrecy capacity ${secrecy.toFixed(4)} <= 0 at error ${e.toFixed(4)} `
              + `with g=${g.toFixed(1)} (${lossDb.toFixed(1)} dB) — this link closes at `
              + `${secrecyNoEve.toFixed(4)} without Eve, so the loss is attributable — telemetry withheld`);
        {
          // REAL NASA SMAP/MSL telemetry + the real linear anomaly model, scored in-browser. The
          // satellite flow has no private_inference stage, so this is where the anomaly verdict is
          // surfaced.
          //
          // OUTSIDE the delivery branch, for exactly the reason the published-model block below was
          // moved out: this step scores a window of the BUNDLED corpus, not the telemetry the channel
          // just delivered, so it has no dependency on whether QSDC aborted.
          //
          // Gating it was not a harmless over-caution, because QSDC essentially never delivers here.
          // `C_s > 0` requires `h(QBER) < 1/(1+g)`, and `g` scales with link loss: measured on the live
          // site, secrecy capacity is -51.58 bit/carrier at the default 800 km (24.5 dB, g=280) and
          // still -2.15 at 200 km (12.4 dB, g=17.5), the shortest downlink the panel offers. So the
          // anomaly verdict, the model's provenance line and the whole SHAP panel were unreachable
          // through the UI - a page advertising a model whose output no user action could produce.
          // The abort itself is correct physics and is unchanged; only the false dependency is gone.
          const anom = await realSatelliteAnomaly(u);
          if (anom) {
            artifacts.telemetry_anomaly = anom.isAnomaly;
            metric(stage, "telemetry_anomaly_score", anom.anomalyScore);
            emit("log", stage, `private telemetry anomaly detection (${anom.provenance}): `
              + `${anom.isAnomaly ? "ANOMALY" : "nominal"} (score ${anom.anomalyScore})`);
            // Satellite had NO attribution in the browser, while the Python pipeline has computed
            // one since Round 18 that nothing rendered. The model here is genuinely linear - a 2xN
            // margin over a NASA telemetry window - so exact linear SHAP applies unchanged and no
            // new method is introduced. Reference point is E[x] over the bundled windows, the same
            // choice the three inference use cases make.
            artifacts.explanation = linearShap(
              anom.model.W, anom.model.b, anom.window, anom.featureNames, 5, anom.baseline);
            artifacts.explanation_provenance = "real bundled telemetry model";
            artifacts.explanation_baseline = "mean of the bundled NASA telemetry windows";
            // This page has NO private_inference stage, so the model explained here is the model
            // that produced the verdict the page reports - the bars ARE an attribution of it.
            // Declared as a flag rather than inferred from `primary_inference` being absent: that
            // inference would hold only by coincidence of which stage writes which artifact, and
            // this project has twice shipped a wrong number that came from exactly that coupling.
            // `tests/test_explanation_scope.py` ties this flag to FLOWS membership.
            artifacts.explanation_is_of_reported_class = true;
          }
        }
        // Satellite's flow has no private_inference stage either, so its published model runs here.
        //
        // OUTSIDE the delivery branch, which is a correction. It was inside, on the reasoning that an
        // aborted channel leaves no telemetry to analyse - but this step never analyses the delivered
        // telemetry. It runs the pinned model on a row of the model's OWN held-out corpus, which has
        // nothing to do with whether this run's channel aborted. Gating it implied a dependency that
        // does not exist, made the model benchmark unavailable exactly when a visitor is exploring
        // the eavesdrop behaviour, and left the diagram reading "published model · —" with a done
        // tick and no reason.
        await runPublishedForStage(stage);
        break;
      }
      case "qsmpc_compute": {
        const key = (config as any)._sessionKey ?? new Uint8Array(32);
        // genomics defaults to QSMS secure summation; others to keyed-hash PSI (matches the backend)
        const op = config.qsmpc_op || (usecase === "genomics" ? "sum" : "psi");
        if (op === "sum") {
          // QSMS additive secret-sharing over a prime field Z_p (config.prime; 0/auto = smallest prime above
          // the max possible total, so the true sum is preserved). Each party masks its secret; the masks
          // cancel mod p on reconstruction — a smaller p genuinely wraps the sum (demonstrates the field).
          // Party count from config, so the browser and the backend describe the same computation
          // instead of the browser inventing 5 while the backend ran n_parties.
          const nParties = Math.max(2, Number(qsmpcCfg().n_parties ?? 5));
          const secrets = Array.from({ length: nParties }, () => Math.floor(u() * 100));
          const rawTotal = secrets.reduce((a, b) => a + b, 0);
          // Snapped, not trusted. The field is labelled "QSMS prime" and used to accept any
          // integer, so entering 100 shared additively over Z_100 -- which is not a field, while
          // the log still printed "over Z_100" as though it were.
          const askedPrime = Number(config.prime);
          const p = askedPrime > 1
            ? nextPrime(Math.floor(askedPrime))
            : nextPrime(rawTotal + 1);
          // Say so when the request was not honoured. A clamp that reports nothing is how a run ends
          // up computing over a different field than the panel above it displays.
          if (askedPrime > MAX_FIELD_PRIME || (askedPrime > 1 && !Number.isFinite(askedPrime))) {
            emit("log", stage, `requested field modulus ${Number.isFinite(askedPrime) ? askedPrime : "infinite"} `
              + `exceeds the ${MAX_FIELD_PRIME} ceiling and was clamped: past 2^53 the modular sum `
              + `stops being exact, and trial division at that size does not return`);
          }
          const shares = secrets.map((s) => { const mask = Math.floor(u() * p); return [((s - mask) % p + p) % p, mask % p]; });
          const total = shares.reduce((acc, [s0, s1]) => (acc + s0 + s1) % p, 0) % p;
          metric(stage, "secure_sum", total);
          metric(stage, "qsms_prime", p);
          metric(stage, "qsms_parties", nParties);
          // Named for what the BROWSER does. The Shamir + d-level QFT + GHZ construction runs in
          // the Python backend (qsh/qsmpc/secure_summation.py); this mirror is its classical
          // equivalent - additive shares over the same prime field - because there is no browser
          // API for qudit state preparation and there will not be one. Saying "GHZ" here would
          // describe a computation this tab did not perform.
          emit("log", stage, `secure summation = ${total} over Z_${p} (${nParties} parties, additive `
            + `shares mod p - the classical equivalent of the backend's Shamir + d-level QFT over a `
            + `GHZ state, which needs qudits no browser can prepare)`);
        } else {
          // finance PSI: the counterparty set size scales with the transaction volume (config.aml_transactions)
          // — more transactions ⇒ larger PSI sets ⇒ more HMAC compute + a larger intersection (bounded for
          // demo responsiveness; the cap is an honest perf guard, not a silent fallback).
          const nTx = Number(config.aml_transactions);
          const txv = Number.isFinite(nTx) && nTx > 0 ? nTx : dataCfg().aml_transactions;
          const perParty = usecase === "finance_aml" ? Math.max(20, Math.min(2000, Math.round(txv / 20))) : 40;
          const nShared = Math.max(4, Math.round(perParty * 0.3));
          const shared = Array.from({ length: nShared }, () => `acct-${Math.floor(u() * 1e6)}`);
          const a = [...shared, ...Array.from({ length: perParty }, () => `a-${Math.floor(u() * 1e6)}`)];
          const b = [...shared, ...Array.from({ length: perParty }, () => `b-${Math.floor(u() * 1e6)}`)];
          const psi = await keyedHashPSI(a, b, key);
          artifacts.psi_intersection = psi.intersection;
          metric(stage, "psi_intersection_size", psi.intersection_size);
          metric(stage, "psi_set_size", a.length);
          emit("log", stage, `PSI complete: |A∩B|=${psi.intersection_size} over ${a.length}x${b.length} accounts`);
        }
        break;
      }
      case "federated_training": {
        // REAL client-side Quantum Federated Learning: FedAvg of a quantum-classical model across
        // simulated clients, round accuracy genuinely computed in-browser (mirrors the full
        // Flower+PennyLane HybridQNN).
        // Federate over this page's REAL bundled sample where one exists, so the accuracy
        // describes the data the page is actually showing rather than an internal toy.
        //
        // The `if (config.qfl)` that wrapped this, and its else branch, are gone. Neither could run
        // any other way: `qfl` is `useState(true)` with no writer (`setQfl` is declared and returned
        // and never called), `buildConfig` sets `config.qfl` for every use case in
        // FEDERATED_USECASES, and medical_fl - the only flow that DECLARES this stage rather than
        // having it injected - is in that set. So the else branch could not execute, and its log
        // still told the visitor to "toggle Federated (QFL)", a checkbox removed in Round 18. A
        // flow that declares this stage should run it; the guard added nothing but an unreachable
        // second answer to "what is federated_accuracy?".
        const qflData = await realFederatedSet(usecase);
        // The PQC handshake's session key, which every flow reaching this stage has already run.
        // Before this it was the ONLY consumer-less key in the pipeline outside `qsmpc_compute`:
        // the diagram claimed pairwise masking while no stage but PSI read `_sessionKey` at all.
        const q = clientQFL(seed, { data: qflData, sessionKey: (config as any)._sessionKey });
        metric(stage, "federated_accuracy", q.finalAcc);
        metric(stage, "qfl_rounds", q.rounds);
        metric(stage, "qfl_clients", q.clients);
        metric(stage, "qfl_mask_pairs", q.pairs);
        metric(stage, "federated_eval_rows", q.evalRows);
        // What the aggregation ACTUALLY was, published so the diagram can render it instead of
        // carrying its own claim. See the QFLResult comment for why this replaced `secureAgg`.
        metric(stage, "qfl_aggregation", q.aggregation);
        metric(stage, "federated_accuracy_source",
          qflData ? `real bundled ${usecase} sample, ${qflData.X.length} rows`
                  : "synthetic 2-cluster toy (no bundled sample for this use case)");
        artifacts.qfl = q;
        emit("log", stage, `Quantum Federated Learning (client): ${q.clients} clients × ${q.rounds} `
          + `rounds (quantum-classical FedAvg; ${q.aggregation}) over `
          + `${qflData ? `the real bundled ${usecase} sample (${qflData.X.length} rows)` : "a synthetic toy"}`
          + ` — accuracy ${q.roundAcc[0].toFixed(3)}→${q.finalAcc.toFixed(3)}`);
        break;
      }
      case "qml_ids": {
        // clientIDS genuinely trains an 8-class softmax in this tab, so the timing and the F1 are
        // measured. The *data* is synthetic, which the run logs separately — measured computation
        // over synthetic input is not the same claim as measured performance on real traffic.
        metric(stage, "inference_provenance", "measured-wasm");
        metric(stage, "inference_backend", "in-browser softmax training (clientIDS), synthetic telemetry");
        // genuine in-browser IDS: softmax logistic trained + scored on the Al-Kuwari-2026 attack signatures.
        // Training volume per class = config.n_per_class (Parameters panel), so it genuinely shifts acc/F1.
        const ids = clientTrainScoreIDS(seed, Number(config.n_per_class ?? idsCfg().n_per_class));
        metric(stage, "ids_macro_f1", r2(ids.macroF1));
        metric(stage, "ids_worst_class_f1", r2(ids.worstClassF1));
        metric(stage, "ids_binary_f1", r2(ids.binaryF1));
        metric(stage, "ids_accuracy", r2(ids.accuracy));
        metric(stage, "ids_params", ids.nParams);
        // clientIDS clamps this to >=20 and falls back on a non-finite input, so the requested value
        // is not always the value used. Report what the run actually trained on.
        metric(stage, "ids_train_per_class", ids.nPerClassUsed);
        // Telecom had NO attribution anywhere - browser or Python - despite being the one use case
        // whose whole point is classifying an attack. The classifier trained above is linear
        // (softmax over b[c] + W[c].z), so exact linear SHAP applies to it directly. clientIDS
        // returns W as a one-vs-rest PAIR for the predicted class, so `W[1] - W[0]` here is the
        // same contrast explainability.py computes for the multiclass case - the method is shared,
        // not reinvented. Baseline is the zero vector because the features are z-scored inside
        // clientIDS, so E[x] is 0 by construction; that is stated rather than left implied.
        if (ids.explain) {
          artifacts.explanation = linearShap(
            ids.explain.W, ids.explain.b, ids.explain.x, ids.explain.featureNames, 5);
          artifacts.explanation_provenance =
            `in-browser IDS classifier, one-vs-rest for ${ids.explain.className}`;
          artifacts.explanation_baseline = "zero vector (features are z-scored, so E[x] = 0)";
          // Telecom has NO private_inference stage: the class this page reports comes from this
          // very softmax, so the bars ARE an attribution of the reported class. See the satellite
          // site above for why this is a declared flag and not inferred from an absent artifact.
          artifacts.explanation_is_of_reported_class = true;
        }
        emit("log", stage, `QML-IDS (client, softmax logistic on Al-Kuwari-2026 signatures): `
          + `macro-F1=${r2(ids.macroF1)} over 8 classes, worst class ${r2(ids.worstClassF1)} `
          + `— measured on ${ids.nTest} held-out samples (${ids.nPerClassUsed}/class)`);
        emit("log", stage, `companions (NOT the headline): binary attack-vs-normal F1=${r2(ids.binaryF1)}, `
          + `aggregate accuracy=${r2(ids.accuracy)}. Both collapse the seven attack types together, which is `
          + `why they read high while a single class can be failing — see scripts/train/telecom_ids.py`);
        // Telecom's flow has no private_inference stage, so its published model runs here.
        await runPublishedForStage(stage);
        // ...and so does its ENCRYPTED tier. Telecom was the only use case with no CKKS evaluation
        // of any kind, which meant `runs_encrypted` was false on every telecom spec and the
        // per-industry private-inference question had no artifact to judge here.
        //
        // This runs inside `qml_ids` rather than in a new `private_inference` stage on purpose.
        // `runs_encrypted` is a property of the MODEL - whether its inference is evaluated under
        // homomorphic encryption - not of the stage that hosts it, and renaming the flow would
        // change the orchestration graph, the Python pipeline and this page's reported-class
        // semantics for no gain in what is actually computed.
        //
        // What the page REPORTS still comes from the in-tab softmax above, so
        // `explanation_is_of_reported_class` stays true and the SHAP bars keep explaining the
        // deciding model. The encrypted run is reported as VERIFICATION of the encrypted tier -
        // its scores, its agreement with the plaintext reference, its precision - which is the
        // honest description of what a dual-path comparison establishes. It is empirical fidelity,
        // not cryptographic integrity: verifiable-FHE proofs establish that a server ran the
        // circuit it was asked to run, and this establishes something different and complementary.
        await runEncryptedTelecomStudent(stage, ids.explain?.xRaw);

        // The 8-class QCNN, where the visitor selected it. This lives here rather than in
        // `private_inference` because telecom has no such stage - and putting the weights on disk and
        // registering the use case WITHOUT this made the QCNN option appear in the Parameters panel
        // and do nothing. An inert control is the defect class this file's own checklist opens with
        // (Round 20 audited use-case pages and found five), and the only thing that surfaced it was
        // running the deployed page: every local gate was green.
        if (String(config.inference || "ckks") === "qcnn" && ids.explain?.xRaw) {
          const w = await loadQcnnWeights(usecase);
          if (w) {
            const qc = qcnnInferTrained(ids.explain.xRaw.slice(0, w.n_qubits), w);
            artifacts.primary_inference = { backend: "qcnn-trained", class: qc.cls,
                                            qubits: qc.qubits, expectation: r2(qc.expectation) };
            metric(stage, "primary_inference_backend", "qcnn-trained");
            metric(stage, "primary_inference_class", w.class_names?.[qc.cls] ?? qc.cls);
            metric(stage, "qcnn_classes_available", w.readout?.n_classes ?? 2);
            emit("log", stage, `QCNN (trained, ${w.readout?.n_classes ?? 2}-class basis readout on `
              + `${w.readout?.readout_qubits ?? 0} of ${w.n_qubits} qubits): class `
              + `"${w.class_names?.[qc.cls] ?? qc.cls}". Class index is ${w.readout?.class_index}, `
              + `${w.readout?.bit_order} - reading those bits in another order permutes the class map `
              + `while leaving every aggregate metric identical, which is why the convention ships in `
              + `the weights.`);
            if (typeof w.test_macro_f1 === "number" || w.per_class_f1) {
              emit("log", stage, `and it LOSES: macro F1 ${w.test_macro_f1 ?? "n/a"} against a `
                + `classical logistic regression at ${w.classical_lr_macro_f1 ?? "n/a"} on the same `
                + `split, with ${w.n_classes_predicted ?? "?"} of ${w.readout?.n_classes ?? 2} classes `
                + `predicted. Reported because it is the measured result, not because it flatters.`);
            }
          } else {
            emit("log", stage, "QCNN selected but no trained weights are published for this use case");
          }
        }
        break;
      }
      case "private_inference": {
        // Use REAL commercial-OK data + a REAL model where one is shipped (medical = Wisconsin breast-cancer
        // + a real logistic regression; genomics = PGS-Catalog PRS weights) — matching the full deployment.
        // finance/telecom/satellite have no shippable commercial-OK real dataset → synthetic, logged honestly.
        const realIn: RealInput | null = await realInput(usecase, u);
        const feats = realIn ? realIn.values : Array.from({ length: 8 }, () => u() * 2 - 1);
        emit("log", stage, realIn
          ? `data: ${realIn.provenance}`
          : "data: synthetic features (no commercial-OK real dataset shipped for this use case)");
        if (realIn?.metrics) for (const [k, v] of Object.entries(realIn.metrics)) metric(stage, k, v);
        const t = performance.now();
        // Published ONNX model where one exists for this use case, else the bundled real distilled
        // model, else placeholder weights. inf.provenance records which, and the UI shows it.
        const inf = await privateInference(usecase, feats, realIn?.model ?? null);
        const clientMs = r2(performance.now() - t);
        artifacts.client_compute_ms = clientMs;
        metric(stage, "client_compute_ms", clientMs);
        // GENUINE encrypted hybrid IN THE BROWSER: real node-seal CKKS linear + real secret-sharing argmax,
        // entirely client-side (zero VPS). The real run measures a small (n->2) model; the reported FHE-linear /
        // MPC-non-linear ms scale that MEASURED per-op cost to the SELECTED architecture (size × workload) so the
        // time genuinely tracks lightweight/medium/large × linear/cnn/transformer. client_compute_ms above is the
        // raw ONNX wall-clock; _measured_hybrid keeps the raw per-op measurement (the benchmark panel scales it too).
        // Surface which tier actually produced the inference, so a page that fell back to
        // placeholder weights cannot look like one that ran the published model.
        metric(stage, "inference_backend", inf.backend);
        metric(stage, "inference_provenance", inf.provenance);
        if (inf.modelKey) metric(stage, "inference_model", inf.modelKey);
        if (inf.note) emit("log", stage, `inference: ${inf.note}`);

        await runPublishedForStage(stage);

        // The medical CNN workload runs a genuine encrypted convolution.
        // Medical always evaluates its published student's convolution; the other use cases have
        // linear students, for which the packed dot product IS the real model.
        // Announce the one-time warm-up BEFORE it blocks. SEAL key generation (public + Galois keys at
        // N=8192) runs synchronously inside WASM on the main thread, and `clientFHE` records that this
        // "made the first click feel like a hang"; summon compile and emp-wasm init follow. The tab
        // cannot repaint through any of it, so a visitor's first Run looked frozen with the badge on
        // `running`. The fix that was applied then was to make keygen cheaper; saying it is happening
        // costs one line and is what a visitor actually needs.
        if (!fheReady()) {
          emit("log", stage, "first run in this tab: generating CKKS keys and compiling the "
            + "garbled-circuit backend locally. This happens once and blocks briefly - no key "
            + "material leaves the browser.");
          await sleep(0);        // yield once so the log line paints before WASM takes the thread
        }
        const real = await realClientHybrid(feats, u, usecase === "medical_fl", usecase);
        if (real) {
          // A real encrypted convolution is not scaled by anything: its latency IS the measurement.
          const bscale = real.conv
            ? 1
            : benchScale(usecase, config.model_size ?? "medium", config.workload ?? "linear");
          artifacts.fhe_prediction = real.prediction;
          artifacts._measured_hybrid = { fheMs: real.fheMs, mpcMs: real.mpcMs, commBytes: real.commBytes,
            ctBytes: real.ctBytes, featDim: real.featDim, conv: real.conv, dense: real.dense };
          metric(stage, "fhe_max_abs_error", r2(real.maxAbsError) || real.maxAbsError);
          // The encrypted-verification figures, on the same footing telecom got in Round 33 - where
          // they existed at exactly ONE call site and the other four use cases reported only a raw
          // error magnitude whose scale a reader has no way to judge.
          //
          // Two conventions, both published, and both reported because at n=1 they say different
          // things. The top-1 mismatch COUNT is arXiv:2605.22237 Table IX, which reports it as an
          // integer expected to be zero; the agreement percentage is Concrete-ML's "Similarity".
          // "0 of 1" and "100.0%" are not the same claim.
          //
          // Precision in BITS because that is the unit OpenFHE and HElib both use, and the sign is
          // load-bearing: bits are -log2(error), so an error at or above 1 has no bits to report and
          // must surface as a failure rather than fold to a plausible positive. That exact fold was a
          // live defect on the telecom path until Round 34.
          {
            const encCls = argmax(real.logits), plainCls = argmax(real.plainLogits);
            const mism = encCls === plainCls ? 0 : 1;
            const bits = real.maxAbsError > 0 && real.maxAbsError < 1
              ? Math.floor(-Math.log2(real.maxAbsError)) : null;
            metric(stage, "encrypted_top1_mismatches", mism);
            metric(stage, "encrypted_scores_compared", real.logits.length);
            metric(stage, "encrypted_agreement", mism === 0 ? 1 : 0);
            metric(stage, "encrypted_precision_bits", bits);
            emit("log", stage, `encrypted-vs-plaintext verification: ${mism} of 1 top-1 mismatch`
              + `es over ${real.logits.length} class scores. Max abs error `
              + `${real.maxAbsError.toExponential(3)}`
              + (bits !== null ? ` (about ${bits} bits of precision, MAX not mean)` : "")
              + (real.maxAbsError >= 1
                ? " -- PRECISION FAILURE: the error is at or above 1, so the decrypted result is not"
                  + " a usable approximation of the plaintext one."
                : "") + ".");
          }
          // Counted on BOTH paths, so Linear vs CNN is a measured comparison of key-switch cost -
          // the quantity the 2026 CKKS-inference literature identifies as dominant - rather than a
          // wall-clock difference that could be anything.
          if (real.dense) {
            metric(stage, "fhe_dense_rotations", real.dense.rotations);
            metric(stage, "fhe_dense_multiplies", real.dense.multiplies);
          }
          metric(stage, "fhe_linear_ms", r2(real.fheMs * bscale));
          metric(stage, "mpc_nonlinear_ms", r2(real.mpcMs * bscale));
          metric(stage, "mpc_comm_kb", r2((real.commBytes * bscale) / 1024));
          if (real.conv) {
            // Counted during the run. Rotations and the key-switch each one triggers are what
            // dominate CKKS inference cost, so this is the quantity worth surfacing - not a
            // latency multiplier that stands in for it.
            metric(stage, "fhe_conv_rotations", real.conv.rotations);
            metric(stage, "fhe_conv_multiplies", real.conv.multiplies);
            metric(stage, "fhe_conv_patches_packed", real.conv.patchesPerCiphertext);
            metric(stage, "fhe_conv_output_shape", `${real.conv.channels}x${real.conv.outH}x${real.conv.outW}`);
            metric(stage, "fhe_conv_ms", r2(real.conv.convMs));
            // The convolution's own approximation error against a plaintext reference computed on
            // the same inputs. Reported separately from the dense layer's so a regression in one
            // cannot hide behind the other.
            metric(stage, "fhe_conv_max_abs_error", real.conv.maxAbsError);
            // Whether the PUBLISHED student's weights drove this, or only its shape did. These are
            // different claims and the run must never let them look alike.
            metric(stage, "fhe_weights", real.conv.publishedWeights
              ? "published student (HF, pinned revision)" : "student shape, stated kernel pattern");
            metric(stage, "fhe_total_rotations", real.conv.totalRotations);
            if (real.conv.studentParams != null) metric(stage, "fhe_student_params", real.conv.studentParams);
            if (real.conv.studentPrediction != null) metric(stage, "fhe_student_class", real.conv.studentPrediction);
            emit("log", stage, (real.conv.publishedWeights
              ? `private inference (REAL in-browser FHE, encrypted CONVOLUTION over the PUBLISHED `
                + `student's own weights fetched from its pinned HF revision, `
                + `${real.conv.studentParams} params): `
              : `private inference (REAL in-browser FHE, encrypted CONVOLUTION on the student's `
                + `SHAPE with a stated kernel pattern - the published weights could not be `
                + `fetched, and none were invented): `)
              + `class=${real.prediction}; ${real.conv.channels}x3x3 kernels over a `
              + `${real.conv.outH}x${real.conv.outW} output, ${real.conv.patchesPerCiphertext} im2col `
              + `patches packed into one ciphertext, ${real.conv.rotations} rotations and `
              + `${real.conv.multiplies} multiply-plain in ${r2(real.conv.convMs)} ms `
              + `(+ ${r2(real.fheMs - real.conv.convMs)} ms dense layer, MPC ${r2(real.mpcMs)} ms). `
              + `Nothing here is scaled by a cost model.`);
          } else {
            emit("log", stage, `private inference (REAL in-browser FHE, ${config.model_size ?? "medium"}/${config.workload ?? "linear"}): `
              + `class=${real.prediction}; node-seal CKKS ${r2(real.fheMs * bscale)} ms + secret-sharing MPC ${r2(real.mpcMs * bscale)} ms `
              + `(measured per-op x${r1(bscale)} arch scale; real unit ${r2(real.fheMs)}/${r2(real.mpcMs)} ms, ct=${real.ctBytes}B), ONNX ${clientMs} ms`);
          }
        } else {
          // honest last resort: node-seal genuinely unavailable here (e.g. no WASM) — representative figure, logged
          artifacts.fhe_prediction = inf.prediction;
          const split = mpcFheRow(usecase, config.workload ?? "linear", config.model_size ?? "medium", null, secFromConfig(config, metrics));
          metric(stage, "fhe_max_abs_error", costPerOp().demo_max_abs_error);
          metric(stage, "fhe_linear_ms", split.fhe_linear_ms);
          metric(stage, "mpc_nonlinear_ms", split.mpc_nonlinear_ms);
          metric(stage, "mpc_comm_kb", split.mpc_comm_kb);
          emit("log", stage, `private inference: class=${inf.prediction}; in-browser CKKS unavailable here — `
            + `representative split ${split.fhe_linear_ms}/${split.mpc_nonlinear_ms} ms, client compute ${clientMs} ms`);
        }

        // Inference backend selection (config.inference): "qcnn" routes the reported class through the
        // quantum QOTP+QCNN readout; "ckks"/default keeps the CKKS(FHE)+MPC path. Both are computed for the
        // comparison — this selects which one is surfaced as the primary prediction (genuinely changes it).
        const inferBackend = String(config.inference || "ckks");
        if (inferBackend === "qcnn") {
          const nq = qubitsOf(config);
          // TRAINED variational circuit where weights exist for this use case; the untrained
          // Clifford+T path is not a predictor. Measured: over 24 vectors x 8 seeds its class flipped
          // for 100% of inputs, mean |<Z0>| 0.076, correlation with mean(feature) -0.070. It stays as
          // the QOTP hiding + T-count demonstration, which is what it can honestly support.
          const w = await loadQcnnWeights(usecase);
          if (w) {
            const qc = qcnnInferTrained(feats.slice(0, w.n_qubits), w);
            artifacts.primary_inference = { backend: "qcnn-trained", class: qc.cls,
                                            qubits: qc.qubits, expectation: r2(qc.expectation) };
            metric(stage, "primary_inference_class", qc.cls);
            metric(stage, "primary_inference_backend", "qcnn-trained");
            metric(stage, "qcnn_expectation_z0", r2(qc.expectation));
            // Report the metric the CORPUS warrants, not always accuracy. At finance's 6% fraud
            // rate accuracy reads 0.97 against a 0.95 majority baseline and says almost nothing,
            // while minority-class F1 is 0.71 against an all-positive 0.10. The trainer decides
            // which is the headline and ships it in `headline_metric`.
            // Three regimes, matching `headline_metric` in the artifact. The multiclass arm reads
            // the uniform-random floor rather than the constant-predictor one: it is the higher of
            // the two the trainer ships, so the comparison is against the harder baseline.
            const head = w.headline_metric;
            const macroLed = head === "macro_f1", f1Led = head === "minority_class_f1";
            const mine = macroLed ? w.test_macro_f1 : f1Led ? w.test_minority_f1 : w.test_accuracy;
            const theirs = macroLed ? w.classical_lr_macro_f1
                         : f1Led ? w.classical_lr_minority_f1 : w.classical_lr_accuracy;
            const base = macroLed ? w.macro_f1_floors?.uniform_random
                       : f1Led ? w.all_positive_f1 : w.majority_baseline;
            const label = macroLed ? "macro F1" : f1Led ? "minority-class F1" : "accuracy";
            emit("log", stage, `inference backend = QCNN (QHE, TRAINED): class=${qc.cls} `
              + `(<Z0>=${qc.expectation.toFixed(4)}, ${w.layers}x${w.n_qubits} trained angles, `
              + `state-vector sim). Held-out ${label} ${(mine ?? 0).toFixed(4)} against a baseline of `
              + `${(base ?? 0).toFixed(4)} and logistic regression at ${(theirs ?? 0).toFixed(4)} on `
              + `the same split - the classical model is ahead, and this reports that rather than `
              + `implying quantum advantage. Test rows were never used to pick the checkpoint.`);
          } else {
            // No trained weights for this use case: refuse to surface a class rather than pass off
            // the untrained circuit's coin flip as a prediction.
            const qc = qotpQcnnInfer(feats.slice(0, nq), seed, nq);
            artifacts.primary_inference = { backend: "ckks", class: artifacts.fhe_prediction ?? inf.prediction };
            metric(stage, "primary_inference_class", artifacts.fhe_prediction ?? inf.prediction);
            metric(stage, "primary_inference_backend", "ckks");
            emit("log", stage, `QCNN selected but no trained weights ship for ${usecase}: the `
              + `untrained Clifford+T circuit is a hiding and T-count demonstration `
              + `(${qc.tGates} T-gates on ${qc.qubits} qubits), NOT a classifier - its class flips `
              + `with the run seed for every input measured. Falling back to CKKS for the reported `
              + `class.`);
          }
        } else {
          const cls = artifacts.fhe_prediction ?? inf.prediction;
          artifacts.primary_inference = { backend: "ckks", class: cls };
          metric(stage, "primary_inference_class", cls);
          metric(stage, "primary_inference_backend", "ckks");
          emit("log", stage, `inference backend = CKKS (FHE): class=${cls}`);
        }
        // XAI: exact linear-SHAP attribution (faithful linear surrogate), signed downstream
        const names = realIn?.names ?? FEATURE_NAMES[usecase] ?? Array.from({ length: 8 }, (_, i) => `feature_${i}`);
        // XAI uses the REAL model where one is shipped (real breast-cancer LR / real PGS PRS weights),
        // else the linear surrogate — so the attribution on real data reflects the real model.
        // The baseline is passed now. Defaulting it to the zero vector measured every attribution
        // against a reference case that does not occur - PaySim's `amount` column has a log-scale
        // mean of 11.5, so its bar was read against "a transaction of log-amount zero" and inflated
        // against every other feature. `efficiency_gap` cannot detect this: completeness holds for
        // any baseline, which is exactly why it went unnoticed.
        artifacts.explanation = linearShap(
          realIn?.model?.W ?? PLACEHOLDER_W, realIn?.model?.b ?? PLACEHOLDER_B, feats, names, 5,
          realIn?.baseline);
        artifacts.explanation_baseline = realIn?.baseline
          ? "sample mean E[x] over the shipped rows"
          : "zero vector (no sample mean available for this use case)";
        // SHAP is exact for the model it is given. With placeholder weights the attribution is
        // exact for weights that mean nothing, so say which case this is rather than let a reader
        // assume the explanation describes the industry model.
        artifacts.explanation_provenance = realIn?.model?.W ? "real bundled model" : "placeholder weights";
        // These pages DO have an encrypted stage, and it evaluates a different model from the one
        // explained here - so the bars are not an attribution of the class the page reports. This is
        // the case the panel's caveat was written for; the two above are the case it was wrong about.
        artifacts.explanation_is_of_reported_class = false;
        metric(stage, "explanation_model", artifacts.explanation_provenance);
        metric(stage, "explanation_top_contribution", artifacts.explanation.top_features[0]?.contribution ?? 0);
        emit("log", stage, `XAI: top driver = ${artifacts.explanation.top_features[0]?.feature}`);
        break;
      }
      case "quantum_comparison": {
        // run the QSMPC-FHE alternative (real client QCNN readout) so the benchmark comparison +
        // quantum advantage are produced by an actual run — mirrors the backend QuantumComparisonStage.
        const ms = config.model_size ?? "medium", wl = config.workload ?? "linear";
        // `metrics` goes in here. This used to call secFromConfig(config) with no metrics and then
        // re-thread `qber` by hand on the next line - so `secure_key_bits` fell to the ?? 256
        // fallback, and the row claimed QKD-mixed key material on runs that produced none. One field
        // wired, its neighbour left to a constant, from the same object already in scope.
        const sec = secFromConfig(config, metrics);
        const isInf = ["finance_aml", "medical_fl", "genomics"].includes(usecase);
        // `benchScale(usecase, ...)`, NOT 1. The scale argument was the other half of the same defect
        // the comment below describes. Fixing the classical denominator aligned one input and left
        // this one hardcoded, so the quantum row here was costed at scale 1 while `assembleBenchmark`
        // costs it at `benchScale(usecase, ...)`. That is 1 by coincidence for finance and medical -
        // `archDims` takes the `studentParams > 0` branch and ignores `modelSize` - and 33 for
        // genomics, which has no t1_crypto student and falls through to the tier architecture. So the
        // diagram bar read 10/10 on the genomics page while the benchmark panel two cards down read
        // about 9.1 from the same run. One quantity, two values, both on screen.
        const qrow = qsmpcFheRow(wl, ms, rng(7), 6, benchScale(usecase, ms, wl), qubitsOf(config), sec);
        // Costed against the MEASURED hybrid, the same input assembleBenchmark uses.
        //
        // This passed `null` and therefore scored the quantum path against a COST MODEL of the
        // classical one, while the benchmark panel above scored it against the encrypted convolution
        // the run had actually just performed. Same quantity, two denominators, both on screen: the
        // badge read 10/10 while the metrics table two panels down read 9.6 and the log line agreed
        // with the table. `private_inference` runs before this stage, so the measurement is already
        // in hand - there was never a reason to model it.
        const measuredHybrid: MeasuredHybrid | null = artifacts._measured_hybrid ?? null;
        const adv = isInf
          ? quantumAdvantage(mpcFheRow(usecase, wl, ms, measuredHybrid, sec), qrow)
          : null;
        if (adv) metric(stage, "quantum_advantage_score", adv.score);
        metric(stage, "qsmpc_t_gates", qrow.t_gate_count);
        emit("log", stage, `QSMPC-FHE comparison (QOTP+QCNN, ${qrow.t_gate_count} T-gates)`
          + (adv ? `: quantum advantage ${adv.score}/10 (${adv.tier})` : " (security baseline)"));
        break;
      }
      case "result": {
        artifacts.summary = { usecase, metrics: { ...metrics } };
        emit("log", stage, "pipeline complete");
        break;
      }
      case "sign_result": {
        // signature algorithm selected via crypto-agility (config.sig): ML-DSA-65 or the larger ML-DSA-87 —
        // genuinely signs with the chosen level (ML-DSA-87 produces a visibly larger signature).
        const dsa = sigImpl(String(config.sig ?? "ml-dsa-65"));
        const sk = dsa.fn.keygen(crypto.getRandomValues(new Uint8Array(32)));
        // Sign the RUN, not just its name. This used to be JSON.stringify({ usecase }) -- a
        // signature over the string {"usecase":"finance_aml"} and nothing else, so every metric,
        // every algorithm selection and the explanation could be altered without invalidating the
        // "tamper-evident audit record" the UI advertises. Canonicalised with sorted keys so the
        // same run produces the same payload.
        const canon = (v: unknown): unknown =>
          Array.isArray(v) ? v.map(canon)
          : v && typeof v === "object"
            ? Object.fromEntries(Object.entries(v as Record<string, unknown>).sort(
                ([a], [b]) => a.localeCompare(b)).map(([k, x]) => [k, canon(x)]))
            : v;
        const signedPayload = canon({
          usecase,
          // Every crypto-agility selection that shaped the run. `mpc` and `qkd_protocol` were both
          // missing: the payload recorded which FHE backend was chosen but not which MPC one, and
          // recorded the QKD family but not the protocol that actually ran - so an E91 run and a
          // BB84 run signed byte-identical config records.
          config: { kem: config.kem ?? null, qkd: config.qkd ?? null, sig: config.sig ?? null,
                    fhe: config.fhe ?? null, mpc: config.mpc ?? null,
                    qkd_protocol: config.qkd_protocol ?? null,
                    combiner: config.combiner ?? null },
          metrics: { ...metrics },
          explanation: artifacts.explanation ?? null,
        });
        const msg = new TextEncoder().encode(JSON.stringify(signedPayload));
        // noble 0.6.x: sign(message, secretKey), verify(signature, message, publicKey).
        const sig = dsa.fn.sign(msg, sk.secretKey);
        const ok = dsa.fn.verify(sig, msg, sk.publicKey);
        artifacts.result_signature = {
          algorithm: dsa.name, signature_bytes: sig.length,
          signature_hex_prefix: [...sig.slice(0, 12)].map((b) => b.toString(16).padStart(2, "0")).join(""),
          // Without the public key and the exact payload, nobody but this tab can re-check the
          // signature -- the keypair is generated per run and discarded. Both are recorded now so
          // "tamper-evident" is a claim a reader can actually test.
          public_key_hex: [...sk.publicKey].map((b) => b.toString(16).padStart(2, "0")).join(""),
          signed_payload: signedPayload,
          signed_fields: ["usecase", "config", "metrics", "explanation"],
          verified: ok,
        };
        metric(stage, "signature_verified", ok ? 1 : 0);
        metric(stage, "signature_bytes", sig.length);
        emit("log", stage, `run summary signed with ${dsa.name} (${sig.length} B, verified=${ok})`);
        break;
      }
    }
  }

  (async () => {
    for (const stage of stages) {
      await gate();
      if (state.stopped) break;
      emit("stage_start", stage, "");
      try {
        await runStage(stage);
      } catch (e) {
        emit("log", stage, `error: ${String(e)}`);
      }
      emit("stage_end", stage, "");
      await sleep(120); // let the DAG animate
    }
    h.onEnd({
      state: state.stopped ? "stopped" : "done", metrics, artifacts,
    });
  })();

  return {
    // Requested, not yet in effect: the running stage keeps going until the next `gate()`.
    pause() { state.paused = true; emit("pipeline_pause", "", "pause requested"); },
    resume() { state.paused = false; emit("pipeline_resume", "", "running"); },
    step() { state.step += 1; },
    // Same asymmetry: the flag is read between stages, so the run stops at the next boundary.
    stop() { state.stopped = true; emit("pipeline_stop_requested", "", "stop requested"); },
  };
}

// Convenience: run a use case to completion client-side and resolve with the final result.
export function runClientUseCase(
  usecase: string, config: Record<string, any> = {},
): Promise<{ state: string; metrics: Record<string, any>; artifacts: Record<string, any> }> {
  return new Promise((resolve) => {
    runClientPipeline(usecase, config, { onEvent: () => {}, onEnd: resolve });
  });
}
