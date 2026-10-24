// Client-side compute for the public demo: in-browser equivalents of the backend
// benchmark + E2E pipeline, so the demo runs entirely in the browser (PQC via @noble,
// QKD/QBER via quantumSim, private inference via onnxruntime-web, PSI via WebCrypto HMAC).
// The security-score and quantum-advantage formulas mirror src/qsh/benchmark/{security,
// quantum_advantage}.py. Neither is RENDERED: the 0-100 composite was retired in Round 22 and the
// quantum-advantage score on 2026-10-23 (see `quantumAdvantage`).
import { ml_dsa65, ml_dsa87 } from "@noble/post-quantum/ml-dsa.js";
import { ml_kem768, ml_kem1024 } from "@noble/post-quantum/ml-kem.js";
import { x25519 } from "@noble/curves/ed25519.js";
import { hkdf } from "@noble/hashes/hkdf.js";
import { sha256 } from "@noble/hashes/sha2.js";
import {
  benchCfg, costPerOp, dataCfg, demoFeatureDim, genericArchitecture, idsCfg, medicalCnn, qkdCfg, qrngCfg,
  demoSample, federatedCfg, fheCfg, qsmpcCfg, quantumAdvantageCfg, securityScoringCfg, studentModel, tierArchitectures, webCfg,
  workloadMultipliers,
} from "./config";
import { encryptedConv2d, encryptedLinear, encryptedLinearBatch, encryptedStudentInference, warmFHE,
  type CnnStudentWeights } from "./clientFHE";
import { clientTrainScoreIDS } from "./clientIDS";
import { clientQFL } from "./clientQFL";
import { secureArgmax, warmMPC } from "./clientMPC";
import { optimizeQKD } from "./clientOptiQKD";
import { generateQuantumBits } from "./clientQRNG";
import { bb84, channelLossDb, e91, FIBER, FREE_SPACE, h2, intrinsicQber, keyRateWithClock,
  qsdcLink, qsdcSecrecyCapacity,
  qcnnExpectation,
  qcnnInferTrained, qotpQcnnInfer, rng, taggedFraction,
  type QcnnWeights } from "./quantumSim";
import { type Provenance } from "../components/ProvenanceBadge";
import {
  browserModels, DEFAULT_ONNX_BACKEND, evaluatePublishedModel, type OnnxBackend, onnxStatus,
  runPublishedModel,
} from "./onnxInference";
import { realInput, realInputCorpus, realSatelliteAnomaly, realFederatedSet,
  type RealInput } from "./realData";
import { hybridCombine, hybridCombinerInfo, kemNameFor } from "./hybridCombiner";
import { isClientDemo } from "./demoMode";

const r1 = (x: number) => Math.round(x * 10) / 10;
const r2 = (x: number) => Math.round(x * 100) / 100;
/**
 * The operator's source repetition rate for one channel, in Hz, or the shipped default.
 *
 * THE CONTROL EXISTS BECAUSE THE PANEL ALREADY SAID IT DID. `SdnControlPlanePanel` told the visitor,
 * in a tooltip and again in prose, that the rate "is a control in the Parameters panel" and that
 * every figure in those columns "scales exactly" with it. No such control was ever rendered - the
 * rate came from baked config. This project has found six INERT controls, rendered and read by
 * nothing; that was the inverse, and worse, because a visitor could not even go and look.
 *
 * The control is denominated in MHz and the key SAYS SO, because 100 MHz is what the panel prints
 * and 100000000 is what a bare Hz spinner would demand. One conversion, in one place, guarded by a
 * name that carries its unit - which is the whole lesson of the round this sits in.
 *
 * Fibre and free space keep SEPARATE keys, exactly as `qkd_distance_km` and `sat_distance_km` do.
 * Their defaults are two config entries rather than one because their provenance differs: the free
 * space figure is the anchor experiment's own configuration, the fibre one is that clock carried
 * over. Collapsing them into a single control would re-create the defect the distance split exists
 * to prevent, where the satellite page silently evaluated a fibre page's parameter.
 *
 * IT TAKES THE VALUE, NOT THE KEY NAME, and that is not a stylistic choice. The first version had
 * the signature `(config, key, fallbackHz)` and read `config?.[key]` inside, so the two call sites
 * passed the string "qkd_source_rate_mhz". That is a real read at runtime and an INVISIBLE one to
 * static analysis: the inert-control gate added in this round scans for `config.<key>`-shaped
 * expressions and reported both brand-new controls as unread. The gate was right about what it could
 * see. Passing `config?.qkd_source_rate_mhz` at the call site keeps the conversion in one place and
 * keeps the read where a reader - and a checker - can find it.
 */
const sourceRateHz = (mhzFromControl: unknown, fallbackHz: number): number => {
  const mhz = Number(mhzFromControl);
  return Number.isFinite(mhz) && mhz > 0 ? mhz * 1e6 : fallbackHz;
};
const argmax = (a: number[]) => a.reduce((bi, v, i, arr) => (v > arr[bi] ? i : bi), 0);
/** Gap between the top two scores - what a top-1 decision actually had to lose. `Infinity` for a
 *  single-class row, so a degenerate model never reads as "decided by a tie". */
const decisionMargin = (row: number[]) => {
  const s = [...row].sort((x, y) => y - x);
  return s.length > 1 ? s[0] - s[1] : Infinity;
};
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
  /** "standard" = checkable against a published document; "measured" = produced by this run;
   *  "simulated" = computed by this run's analytic quantum-channel model. No QKD device exists in this
   *  project, so every QKD row is "simulated" - calling a channel model's output "measured" is the
   *  claim the Research PoC note on the Overview page disowns. */
  kind: "standard" | "measured" | "simulated";
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
 * Every entry below is checkable against a published standard, measured in this run, or computed by
 * this run's simulated quantum channel, and each says which. No entry is a weight. The NIST category leads because it is the one figure here
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
  const skr = typeof metrics.skr_per_pulse === "number" ? metrics.skr_per_pulse : null;
  // ONE protocol test, at FUNCTION scope. It used to be declared inside the QKD branch below, under a
  // comment saying "A shared constant is what stops the next row from being the next omission" - and
  // the QBER row sits OUTSIDE that branch, so it could not read the constant and did omit the test:
  // an E91 run's QBER was captioned "this run's two-intensity decoy-state bound ... aborts above 0.11".
  // E91 here has no decoy states, and its rate reaches zero at the point computed below, not at 0.11.
  const isE91 = String(config?.qkd_protocol ?? qkdCfg().protocol) === "e91";
  const fEC = qkdCfg().error_correction_eff;
  // Where the E91 model's device-dependent rate 1 - h(e) - fEC h(e) reaches zero (0.0981 at 1.16).
  const e91ZeroKey = (() => { let lo = 0, hi = 0.5;
    for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (1 - (1 + fEC) * h2(m) > 0) lo = m; else hi = m; }
    return Math.round(lo * 1e4) / 1e4; })();
  // And where a DEVICE-INDEPENDENT analysis of the same noise model stops certifying key, at the SAME
  // error-correction efficiency (Acin et al., PRL 98, 230501, 2007): r = 1 - fEC h(Q) -
  // h((1 + sqrt((S/2)^2 - 1))/2) with S = 2 sqrt(2) (1 - 2Q). This caption used to quote S > 2.424 /
  // Q < 0.0715 - the fEC = 1 figures - beside a device-dependent 0.0981 computed at fEC = 1.16, so the
  // two thresholds it compared came from different error-correction assumptions.
  const chshAt = (q: number) => 2 * Math.SQRT2 * (1 - 2 * q);
  const e91DiZeroKey = (() => { let lo = 0, hi = (1 - Math.SQRT1_2) / 2;   // S = 2 at the upper end
    const r = (q: number) => 1 - fEC * h2(q) - h2((1 + Math.sqrt(Math.max(0, (chshAt(q) / 2) ** 2 - 1))) / 2);
    for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (r(m) > 0) lo = m; else hi = m; }
    return Math.round(lo * 1e4) / 1e4; })();
  const sDi = Math.round(chshAt(e91DiZeroKey) * 1000) / 1000;
  const sDd = Math.round(chshAt(e91ZeroKey) * 1000) / 1000;
  // The channel this run used, and the eavesdropper fraction on it: both the decoy row (BB84) and the
  // E91 caption read them, so they are built once. See the decoy row for why the parameter reads have
  // to follow the use case.
  const sat = config?.usecase === "satellite_comms";
  const chan = sat ? FREE_SPACE : FIBER;
  const eveFrac = Number((sat ? config?.sat_eve : config?.qkd_eve) ?? 0);
  const runChannel = { ...chan,
    distanceKm: Number((sat ? config?.sat_distance_km : config?.qkd_distance_km) ?? chan.distanceKm),
    // No misalignment control is rendered on the satellite page, so the free-space channel's own
    // 0.025 is the honest value there. Reading `qkd_misalignment` unconditionally would have
    // substituted the fibre page's 0.015 - always, since that key is never absent.
    misalignment: Number((sat ? undefined : config?.qkd_misalignment) ?? chan.misalignment),
    eveFraction: eveFrac };

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
    // ONE protocol test, read by every row below it. It was written inline on the key-material row
    // and nowhere else, so when the secret-key-rate row was added three lines later it simply did
    // not ask - and told E91 users their rate came from a decoy-state estimate. A shared constant
    // is what stops the next row from being the next omission. (Now declared at function scope:
    // declared here, it did not reach the QBER row outside this branch.)
    // The warning is the point. A QKD profile that produced no key is the case the composite scored
    // 100 on, because nothing checked whether the channel actually delivered.
    facts.push({
      label: "QKD key material",
      value: keyBits == null ? "not reported" : `${keyBits} secure bits`,
      kind: "simulated",
      source: isE91
        ? "E91 / CHSH run, device-dependent secure fraction on the model error rate"
        : "BB84 decoy-state run, two-intensity secure fraction",
      // THE BAND, NOT JUST THE ENDPOINT. This tested `keyBits === 0` while the behaviour it warns
      // about switches at 256 - the threshold `stages.py` and `:1233` below both use to decide
      // whether QKD is folded into the session at all. So a run producing 1..255 bits showed a
      // positive count, fired no warning, and silently fell back to PQC-only. Round 45's 4x
      // correction moved the entry to that band from 93 km to 63 km: three clicks of a control
      // whose step is 5 km.
      warn: keyBits === 0
        ? "the QKD profile is selected but this channel produced NO secure key: the session rests on "
          + "the PQC hybrid alone"
        // The last sentence used to read "The bits are real; there are not enough of them." The
        // browser's bb84() returns a COUNT and never key bytes (checklist SC2), so there were no bits.
        : keyBits != null && keyBits > 0 && keyBits < 256
        ? `this channel produced ${keyBits} secure bits, below the 256 a session key needs, so no `
          + "QKD key is derived and the session rests on the PQC hybrid alone. The count comes from "
          + "the simulated channel model."
        : undefined,
    });
    if (skr != null) {
      // PROTOCOL-AWARE, like the row immediately above and unlike this row until Round 45. It was
      // emitted unconditionally as "GLLP bound over the decoy-state estimate", so the E91 page
      // attributed its rate to decoy states - a countermeasure for the multi-photon pulses of an
      // attenuated laser, which entanglement-based QKD does not have and cannot use. One of a
      // sibling pair three lines apart was made protocol-aware and the other was not.
      facts.push({ label: "Secret-key rate", value: `${skr} per pulse`, kind: "simulated",
        source: isE91
          ? "device-dependent secure fraction 1 - h(e) - f h(e) on the MODEL error rate, not the "
            + "sampled QBER (Shor-Preskill form; entangled source: Ma, Fung & Lo, Phys. Rev. A 76, "
            + "012307, 2007); no decoy states. CHSH S is a consistency check, not a certificate"
          : "two-intensity decoy-state bound, Ma, Qi, Zhao & Lo, Phys. Rev. A 72, 012326 (2005)" });
    }
    // BB84 ONLY. An E91 run showed this row too, warning that "the decoy states are what make this
    // link work at all" - on a protocol that uses none. The tagged fraction is a property of an
    // attenuated laser's multi-photon pulses; an entangled source's security analysis needs no decoy
    // states (Ma, Fung & Lo, Phys. Rev. A 76, 012307, 2007). Found on the deployed demo, 2026-10-24,
    // after a gate written the same day had asserted the row's presence on E91 runs.
    if (!isE91) {
      // What the decoy states are actually buying on THIS channel.
      //
      // ROUND 45 CORRECTED THIS COMMENT AND THE MODEL UNDER IT. It used to say "the runtime model is
      // asymptotic decoy-state, which already assumes the single-photon contribution has been
      // bounded". The runtime model was neither: it computed 1 - h2(q) - f h2(q) on the observed QBER
      // and credited every sifted bit as single-photon, which is stronger than any decoy bound rather
      // than an approximation of one. `bb84()` now runs the two-intensity estimator, so the sentence
      // describes the code for the first time. Delta >= 1 still means no detection could be certified
      // single-photon and the same hardware would yield nothing without decoys - and on this link it
      // is 7.26, so the counterfactual really is zero rather than merely smaller.
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
      // because it moved. (`runChannel` above now carries those reads for every row that needs them.)
      const delta = taggedFraction(mu, runChannel);
      facts.push({
        label: "Decoy-state necessity",
        // Always print the number. ">= 1" was a needless loss of information: on the default fibre
        // link Delta is 7.26, and "7.26" says the decoy states are carrying the entire link, where
        // ">= 1" only says they are carrying some of it.
        value: `tagged fraction ${delta.toFixed(2)} at mu ${mu}`,
        kind: "simulated",
        source: "GLLP tagged fraction over this channel; Ma PRA 72 012326, GLLP QIC 4 325",
        warn: delta >= 1
          ? "without decoy states this link yields NO key. The worst-case bound on Eve-known "
            + "(multi-photon) detections exceeds the total detection probability, so not one "
            + "detection could be certified single-photon and the whole sifted block is discarded. "
            + "The decoy states are what make this link work at all."
          : undefined,
      });
    }
  }

  if (qber != null) {
    // THE ZERO-KEY POINT IS NOT A CONSTANT, and naming one was naming a model this project no
    // longer runs. This read "GLLP zero-key point 0.098106" - the NON-DECOY figure, removed from
    // both the Python and the browser paths in Rounds 45-46 when the decoy-state bound replaced it.
    // The decoy bound's zero-key QBER depends on the channel (eta, mu, nu), so there is no single
    // number to substitute; quoting one would repeat the same mistake with a fresher digit.
    //
    // What is reportable is the mechanism: the rate this run computed, and the post-processing
    // threshold the protocol actually aborts on. The fallback, for a run that reported no rate, is
    // now that same configured threshold rather than a retired model's constant.
    const dead = skr != null ? skr <= 0 : qber >= s.qber_secure_threshold;
    facts.push({
      label: "QBER",
      value: `${qber}${dead ? " - no secure key at this error rate" : ""}`,
      kind: "simulated",
      source: skr != null && isE91
        ? (eveFrac > 0
            // A WERNER STATE ONLY WITHOUT EVE. Intercept-resend on a fraction f of pairs leaves a
            // MIXTURE of (1 - f) noisy Werner pairs and f measured-and-resent ones; its correlation
            // tensor is V'(1 - f/2, -(1 - f), 1 - f/2), not V_eff(1, -1, 1), and it is separable once
            // V'(3 - 2f) <= 1 (Horodecki criterion for Bell-diagonal states). In the plane the
            // protocol measures, its QBER and CHSH value equal a Werner state's at V_eff = V'(1 - f/2),
            // which is why S = 2 sqrt(2) (1 - 2Q) still holds - the state label was what was wrong.
            ? `E91 with an eavesdropper on ${eveFrac} of the pairs: a sampled draw around the model error `
              + "rate of a mixture of Werner pairs and intercept-resend pairs - not a Werner state, and "
              + `separable once V'(3 - 2f) <= 1 with V' = ${Math.round(qkdCfg().e91_visibility
                * (1 - 2 * intrinsicQber({ ...runChannel, eveFraction: 0 })) * 1e4) / 1e4} here. Its `
              + "QBER and CHSH value in the measured plane still obey S = 2 sqrt(2) (1 - 2Q). "
            : "E91: a sampled draw around the Werner-model error rate (1 - V)/2, with S = 2 sqrt(2) V. ")
          + `The device-dependent rate reaches zero at Q = ${e91ZeroKey} (S = ${sDd}, f = ${fEC}); a `
          + `device-independent analysis at the same f would need S > ${sDi}, i.e. Q < ${e91DiZeroKey} `
          + "(Acin et al., Phys. Rev. Lett. 98, 230501, 2007), so between those two error rates this "
          + "model reports key a device-independent analysis would not certify. The S > 2 test never "
          + `decides the key here: the rate is already zero below S = ${sDd}`
        : skr != null
        ? "this run's two-intensity decoy-state bound (Ma, Qi, Zhao & Lo, Phys. Rev. A 72, 012326); "
          + `post-processing aborts above ${s.qber_secure_threshold}`
        : `no rate reported by this run; post-processing threshold ${s.qber_secure_threshold}`,
      warn: dead
        ? (isE91 ? "the device-dependent E91 rate yields no secure key at this error rate"
                 : "the decoy-state bound yields no secure key on this channel")
        : undefined,
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
  // Mirrors `benchmark/security.py`: 0.11 is the Shor-Preskill BB84 bound, and a channel can be
  // dead below it - which is why the guard is on the KEY rather than on the QBER. The figure that
  // used to appear here, "this project's GLLP model yields zero secure key from 0.098106", belonged
  // to the non-decoy model that Rounds 45-46 removed from both languages; the decoy bound's
  // zero-key point is channel-dependent and there is no constant to put in its place. Stated in
  // the past tense because the reasoning is what survives, not the number.
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
//  Quantum advantage (mirror of qsh/benchmark/quantum_advantage.py) - RETIRED FROM DISPLAY
// --------------------------------------------------------------------------- //
// Kept as the mirror of the Python module, and called from NOWHERE. Do not wire it back to a panel,
// a log line, a diagram label or a prompt: `tests/test_the_retired_comparisons_stay_retired.py` fails
// if anything outside this definition calls it.
//
// Why it was retired, so the reason survives the code. It printed "Quantum path is ~6146x more
// communication-compact and 5794x faster; dominant advantage" and "10/10", and its three inputs did not
// measure the same thing on both sides:
//   communication  a CKKS ciphertext BYTE ratio divided by a QOTP QUBIT ratio of 1.0 that leaves out
//                  the 2n-bit classical key (and classical FHE reaches rate 1 as well: Gentry-Halevi,
//                  TCC 2019; Brakerski-Doettling-Garg-Malavolta, TCC 2019)
//   latency        a real CKKS + garbled-circuit inference divided by an 8-qubit state-vector simulation
//                  of an UNTRAINED circuit on random input - and it moved 5794x -> 8163x -> 9610x
//                  between three runs, because it is a ratio of two noisy wall-clocks
//   security       a 1.5x multiplier for a label the run does not achieve (the pad is seeded)
// It is the same retirement, for the same reason, as the 0-100 composite in Round 22.
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
   *  not estimated. Each rotation triggers a key-switch, and key-switching dominates every path
   *  measured here - 57% of a small dense layer, 70% of this convolution, 86% of the student's
   *  dense layers - at a per-rotation cost of 3.10-4.39 ms across every stage measured. */
  conv?: { rotations: number; multiplies: number; patchesPerCiphertext: number;
           outH: number; outW: number; channels: number; convMs: number; maxAbsError: number;
           rotateMs: number;
           /** The CONVOLUTION BLOCK's own time. On the published-weights path `convMs` is the whole
            *  student - conv plus its dense layers - while `rotations`/`rotateMs` are the conv
            *  block's, so a rotation share must use THIS as its denominator. Round 38 published
            *  10% from the mismatched pair. Equal to `convMs` on the synthetic-kernel path, which
            *  runs the convolution and nothing else. */
           convOnlyMs: number;
           /** The STUDENT's final-logit error, where a full student ran. Distinct from
            *  `maxAbsError`, which is the convolution block's. */
           studentLogitMaxAbsError?: number;
           /** The student's DENSE layers, on the published-weights path. Reported so the whole
            *  569 ms is accounted for rather than leaving 488 ms unexplained beside a conv figure. */
           denseMs?: number; denseRotations?: number; denseRotateMs?: number;
           /** True when the PUBLISHED student's weights drove the computation, false when only its
            *  shape did. The two must never be reported as the same thing. */
           publishedWeights: boolean; totalRotations: number;
           studentParams?: number; studentPrediction?: number };
  /** Rotations and multiply-plain for the DENSE layer, so the two workloads compare like for like. */
  /** `rotateMs` is the measured time inside `rotateVector`, kept separate from `fheMs` because the
   *  page scales `fheMs` by a cost model and a scaled total cannot yield a per-rotation cost. */
  dense?: { rotations: number; multiplies: number; rotateMs: number };
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
  /** Whether the ENCRYPTED evaluation used the page's bundled model or the fallback demo pattern.
   *  Reported because the two are indistinguishable in the output otherwise, and the page spent
   *  several rounds encrypting one model while reporting and explaining another. */
  encryptedRealModel: boolean;
  /** The rows the batched agreement figure was built from, where a corpus was supplied. */
  batch?: { rows: number; rowsOffered: number; rowsPerCiphertext: number; fheMs: number;
            maxAbsError: number; mismatches: number;
            /** Largest PLAINTEXT decision margin among the rows whose top-1 flipped. A mismatch is
             *  only evidence against the encrypted tier if the plaintext model had a real preference
             *  to lose; see the comment at the computation. */
            flipMarginMax: number;
            /** Rows whose plaintext top-2 margin is EXACTLY zero - the model does not decide them,
             *  so their encrypted outcome is a coin toss and they carry no verification signal.
             *  Counted on every run, because a run where they all happen to agree looks perfect. */
            tieRows: number;
            /** `rows - tieRows`: the rows a top-1 comparison can actually say something about. */
            decidable: number;
            /** Flips among the decidable rows. This is the one that would indicate a real problem. */
            flippedDecidable: number;
            points: { enc: number; plain: number; agree: boolean }[] } | null;
}

let _fheWarmed = false;
/** Whether the one-time CKKS/MPC warm-up has already run in this tab. */
const fheReady = () => _fheWarmed && _mpcWarmed;
let _mpcWarmed = false;
/**
 * Wall-clock of the ONE keygen this tab pays, or null before it has happened.
 *
 * Measured because raising the ring degree is a trade and this is the side of it a visitor feels.
 * SEAL key generation - public key plus the Galois keys for every rotation step - runs synchronously
 * inside WASM on the main thread at first Run, and `clientFHE` records that it "made the first click
 * feel like a hang". Its cost grows with the ring, so a ring change that is reported only through
 * the slot count it buys is reported with the price left off.
 *
 * Module-level rather than per-run: only the first run in a tab pays it, and attributing it to that
 * run's inference latency would overstate the inference by the whole keygen.
 */
let _fheKeygenMs: number | null = null;

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
                                      usecase?: string,
                                      realModel?: { W: number[][]; b: number[] } | null,
                                      corpus?: number[][] | null,
): Promise<RealHybridRun | null> {
  try {
    if (!_fheWarmed) {
      const tk = performance.now();
      await warmFHE();
      _fheKeygenMs = performance.now() - tk;
      _fheWarmed = true;
    }     // one-time CKKS keygen
    if (!_mpcWarmed) { await warmMPC(); _mpcWarmed = true; }     // one-time summon compile + emp-wasm init
    const n = features.length;
    // ENCRYPT THE MODEL THE PAGE EXPLAINS.
    //
    // This function did not take a model at all. It built `w1 = ((i % 3) - 1) * 0.15` and encrypted
    // that, while `privateInference` evaluated the page's REAL bundled model to produce the class
    // shown, and the SHAP panel explained the same real model under the label "real bundled model".
    // Three surfaces on one row, two of them describing a model the third never touched - so every
    // `fhe_*` number on finance and genomics measured a weight vector that appears nowhere else on
    // the page. `privateInference`'s own comment says the real model "is what the SHAP attribution
    // and the CKKS evaluation both consume", which is the invariant this violated three lines away.
    //
    // The demo pattern stays only where no model ships, and the run says which it used rather than
    // letting the two look alike.
    const usingRealModel = Array.isArray(realModel?.W) && realModel.W.length > 0
      && realModel.W.every((row) => Array.isArray(row) && row.length >= n);
    const W = usingRealModel
      ? realModel!.W.map((row) => row.slice(0, n))
      : [Array(n).fill(0), Array.from({ length: n }, (_, i) => ((i % 3) - 1) * 0.15)];
    const b = usingRealModel
      ? W.map((_, j) => Number(realModel!.b?.[j] ?? 0))
      : [0, -0.05];
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
                 channels: wts.conv.out_ch, convMs: r.fheMs,
                 // The CONVOLUTION BLOCK's error, from the `...conv` spread. This was
                 // `r.logitMaxAbsError` - the whole student's final-logit error, after pooling and
                 // both dense layers - published under a convolution-scoped name, on the published
                 // path only, while the synthetic path put the conv block's error in the same key.
                 // One key, two meanings, decided by whether a fetch succeeded; and the comment at
                 // the emit said the two errors were reported separately "so a regression in one
                 // cannot hide behind the other", which is precisely what it allowed.
                 maxAbsError: r.maxAbsError, studentLogitMaxAbsError: r.logitMaxAbsError,
                 rotateMs: r.rotateMs, convOnlyMs: r.convOnlyMs,
                 denseMs: r.denseMs, denseRotations: r.denseRotations, denseRotateMs: r.denseRotateMs,
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
                 rotateMs: cres.rotateMs, convOnlyMs: cres.fheMs,
                 publishedWeights: false, totalRotations: cres.rotations };
      }
    }
    // The same model over a whole corpus, where the caller supplied one. Costs no extra homomorphic
    // work - `encryptedLinearBatch` packs the rows at the reduction's own stride - and it is what
    // turns "the encrypted number agreed on this one row" into a figure a reader can check.
    let batch: RealHybridRun["batch"] = null;
    if (usingRealModel && corpus?.length) {
      const bt = await encryptedLinearBatch(W, b, corpus.map((r) => r.slice(0, n)));
      const encCls = bt.logits.map(argmax), plainCls = bt.plainLogits.map(argmax);
      // A top-1 mismatch is NOT automatically a precision failure, and reporting the count alone
      // says it is. Genomics is the case that forced this: its model is the published PGS weights
      // as `W = [-w/2, +w/2]` with zero bias, so the two logits are exact negatives and the
      // decision margin IS the polygenic score. A sample whose score sits at zero is a genuine tie
      // and flips under any perturbation whatsoever - 3 of its 30 rows did, at a deviation of
      // 7.2e-10. Printing "3 of 30 mismatches" beside that leads a reader to exactly the wrong
      // conclusion about the encrypted tier.
      //
      // So measure what the plaintext model actually had to lose: the gap between its top two
      // scores at the rows that flipped. If that gap is inside what the CKKS deviation can bridge,
      // the disagreement is the plaintext model being indifferent, not the encryption being wrong.
      const flipped = encCls.map((_, i) => i).filter((i) => encCls[i] !== plainCls[i]);
      const flipMarginMax = flipped.length
        ? Math.max(...flipped.map((i) => decisionMargin(bt.plainLogits[i]))) : 0;
      // WHY THERE IS NO TIE/NOT-TIE BOOLEAN HERE, after two attempts at one.
      //
      // The first asked `flipMarginMax <= 2 * maxAbsError`. That is a theorem, not a test: if `a`
      // is the plaintext argmax and `c` the encrypted one, margin <= plain[a] - plain[c] =
      // (enc[a] - enc[c]) + (d_c - d_a) <= 0 + 2E, because enc[c] >= enc[a] by definition of the
      // encrypted argmax and E bounds every deviation. A flip MEANS the noise exceeded the margin.
      //
      // The second divided that margin by the plaintext score range and compared against 1e-6,
      // which looked independent and is not: the same bound gives rel <= 2E/range, and with
      // E ~ 1e-9 against a range of order 1 that is ~1e-9 - three orders below the threshold, on
      // every run this system can produce. It would take a 500x precision collapse to reach the
      // other branch, and that case is already reported by `encrypted_batch_precision_bits` and by
      // the maxAbsError >= 1 failure path. Tuning the threshold would only move the impossibility.
      //
      // Worse, the experiment that "proved" the second version reachable forced `flipMarginRel` to
      // a literal and watched the caption change. That tested the rendering branch, not the
      // predicate - the wrong thing, convincingly.
      //
      // So the boolean is gone rather than re-tuned. What survives is measured and does vary:
      // `flipMarginMax`, the deviation beside it, and `tieRows` below - the rows the plaintext
      // model scores identically, which is 7 of 30 on genomics and 0 of 512 on telecom. That count
      // is what actually separates a meaningful agreement rate from a coin toss, and unlike a
      // tie/not-tie verdict it is a property of the corpus rather than a restatement of a theorem.
      const tieRows = bt.plainLogits.filter((row) => decisionMargin(row) === 0).length;
      const decidable = bt.rows - tieRows;
      // Flips among the rows the plaintext model actually decides. THIS is the number that would
      // indicate a real problem; the headline rate over all rows is the published convention and is
      // kept beside it.
      const flippedDecidable = flipped.filter((i) => decisionMargin(bt.plainLogits[i]) !== 0).length;
      batch = {
        rows: bt.rows, rowsOffered: bt.rowsOffered, rowsPerCiphertext: bt.rowsPerCiphertext,
        fheMs: bt.fheMs, maxAbsError: bt.maxAbsError,
        flipMarginMax, tieRows, decidable, flippedDecidable,
        mismatches: flipped.length,
        points: bt.logits.map((row, i) => ({
          enc: row[encCls[i]], plain: bt.plainLogits[i][encCls[i]],
          agree: encCls[i] === plainCls[i],
        })),
      };
    }
    return {
      prediction: mpc.out[0], fheMs: fhe.fheMs + (conv?.convMs ?? 0), mpcMs: mpc.mpcMs,
      commBytes: mpc.commBytes, ctBytes: fhe.ctBytes, featDim: n,
      maxAbsError: fhe.maxAbsError, plainLogits: fhe.plainLogits, logits: fhe.logits, conv,
      dense: { rotations: fhe.rotations, multiplies: fhe.multiplies, rotateMs: fhe.rotateMs },
      encryptedRealModel: usingRealModel, batch,
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
    // THE LATENT TWIN of the `fhe_max_abs_error` defect two thousand lines below: this sets the
    // config constant UNCONDITIONALLY, on the measured branch as well as the fallback. It reaches no
    // component today, which is the only reason it is not a live mislabel - and "nothing renders it"
    // is a property of the current UI, not of the value. Named for what it is.
    accuracy: null, max_abs_error_representative: cost.demo_max_abs_error, ciphertext_plaintext_ratio: ctpt,
    fhe_linear_ms: r2(fhe), mpc_nonlinear_ms: r2(mpc), mpc_comm_kb: r2(comm / 1024),
    t_gate_count: null, qubits: null,
    // "garbled-circuit 2PC", not "secret-sharing MPC". That label is right for the Python row (MPyC
    // Shamir across hosts) and was copied to this one, where the non-linear layer is an emp-wasm
    // authenticated-garbling 2PC whose INPUTS are additively shared - between notional parties in
    // one tab (clientMPC.ts). The benchmark table showed the Python mechanism on the browser row.
    security_model: "computational (RLWE) + garbled-circuit 2PC",
    security_score: securityScore({
      securityModel: "computational (RLWE / FIPS-203 family)", kem: sec?.kem,
      qkdKeyBits: sec?.qkdBits,
      qber: typeof sec?.qber === "number" ? sec.qber : null,
    }).total,
    device,
  };
}

// QSMPC-FHE row — GENUINE state-vector QOTP + Clifford+T QCNN inference (quantumSim.qotpQcnnInfer):
// `latency_ms` is the REAL-measured wall-clock of the simulation and `t_gate_count`/`qubits` are COUNTED
// from the actual circuit. A state-vector SIMULATION, not quantum hardware.
//
// ct/pt IS WITHHELD (null) FOR THIS ROW, and the comment here used to say the opposite twice over. It
// said ct/pt = 1.0 "is the QOTP perfect-hiding property" and that "Rate-1 QFHE's ~1.3 is the classical-
// key-included literature figure". Neither holds. A QOTP ciphertext is n qubits PLUS a 2n-bit classical
// key (Ambainis, Mosca, Tapp, de Wolf, FOCS 2000; Boykin and Roychowdhury, PRA 67, 042317, 2003), so
// "1.0" counts only the half that does not grow. In the Rate-1 QFHE paper's own units (qubits plus bits,
// Chardouvelis, Doettling, Malavolta, TCC 2021, Def. 6.1) an 8-qubit KEM-wrapped transport is ~1135x and
// tends to ~3x, not 1, as n grows. And that paper reports NO concrete figure - "~1.3" is not in it.
// Printing 1.0 beside the CKKS row's byte ratio is the comparison the overview card made, so the
// quantum row shows a dash and the table footnote says why.
const QSMPC_SECURITY_MODEL = "information-theoretic as a scheme (QOTP); not this run - the pad is seeded";
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
    ciphertext_plaintext_ratio: null,                 // withheld: not comparable in bytes, see the header
    t_gate_count: last.tGates, qubits: last.qubits,   // counted from the actual gate list
    // ONE constant, read twice. The row label and the scorer input were two literals, and the scorer
    // parses the label for "information-theoretic" - so rewording one copy would have moved a score
    // with every gate green (`tests/test_security_model_prose_is_not_a_silent_score_input.py`).
    security_model: QSMPC_SECURITY_MODEL,
    security_score: securityScore({
      securityModel: QSMPC_SECURITY_MODEL, kem: sec?.kem,
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
  // `advantage` is null: the quantum-advantage score is retired from display (see `quantumAdvantage`).
  return { workload, model_size: modelSize, results: [classical, quantum], advantage: null };
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
    advantage: null,                    // retired from display; see `quantumAdvantage`
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

  // THE HYBRID, not just the PQ half.
  //
  // This read `const sessionKey = enc.sharedSecret.slice(0, 32)` - the raw ML-KEM shared secret,
  // truncated. No X25519, no KDF, no transcript, no domain separator. Not merely SP 800-227's
  // expression (16) `KDF(K1, K2)`, which that document says does not preserve IND-CCA; this was K1
  // alone. And the function claims to mirror `src/qsh/benchmark/alice_bob_e2e.py`, whose Phase 3
  // calls `establish_initiator`/`establish_responder` - the hybrid combiner, then `_mix_qkd`.
  //
  // Found by a completeness pass AFTER four other combiner sites had been fixed, which is the
  // lesson worth keeping: "all the sites" was four because four were the ones anyone had listed.
  const bobX = x25519.utils.randomSecretKey();
  const bobXPub = x25519.getPublicKey(bobX);          // ek2, Bob's static X25519 public
  const aliceX = x25519.utils.randomSecretKey();
  const aliceXPub = x25519.getPublicKey(aliceX);      // c2, Alice's ephemeral public
  const xAlice = x25519.getSharedSecret(aliceX, bobXPub);
  const xBob = x25519.getSharedSecret(bobX, aliceXPub);
  const xAgree = xAlice.length === xBob.length && xAlice.every((b, i) => b === xBob[i]);
  // The shared combiner - same module the handshake and the Validations demo use, byte-compatible
  // with `src/qsh/pqc/hybrid_kem.py`. This block used to build its own info string, which is how
  // the browser came to disagree with Python and with its own two siblings.
  const sessionKey = hybridCombine(
    enc.sharedSecret, xAlice,
    { c1: enc.cipherText, c2: aliceXPub, ek1: bob.publicKey, ek2: bobXPub }, "ML-KEM-768", 32);

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
    session_key_agreement: agree && xAgree, kem: "ML-KEM-768+X25519",
    // `qkd_mixed` reported `secureKeyBits >= 256` - a PRECONDITION for mixing, stated as though the
    // mix had happened, in a function that mixed nothing. The browser's `bb84()` returns bit COUNTS
    // and no key material by design (checklist SC2), so there is nothing here to fold in and the
    // honest value is false with the reason beside it. The agent panel, which DOES mix, mixes a
    // stated fixed constant and says so - that is the shape this had to match.
    qber: r2(qkd.qber), qkd_secure_key_bits: qkd.secureKeyBits,
    qkd_mixed: false,
    qkd_mix_note: qkd.secureKeyBits >= 256
      ? "the BB84 run cleared 256 secure bits, but this mirror derives no key material from it - "
        + "bb84() returns bit counts, so there is nothing to fold into the session key"
      // Was "no secure key bits, so nothing to mix even in principle" for ANY value below 256,
      // so a run with 94 bits printed `qkd_secure_key_bits: 94` beside prose denying they existed.
      : qkd.secureKeyBits > 0
      ? `${qkd.secureKeyBits} secure bits, below the 256 a session key needs - not enough to mix, `
        + "which is a different statement from having none"
      : "no secure key bits, so nothing to mix even in principle",
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

/** Load the trained circuit, preferring a shipped variant at the REQUESTED width.
 *
 *  Until Round 40 this took only the use case, so the "QCNN qubits" control could not reach the
 *  trained branch at all: it fed the benchmark row and the untrained fallback while this returned a
 *  fixed 8-wire circuit whatever the visitor selected. A control that silently ignores you is the
 *  inert-control defect this file's checklist opens with, and Round 20 found five of them.
 *
 *  A width only exists if someone trained it. `qcnn_weights_q<N>.json` is tried first and the
 *  default file is the fallback, so a requested width that was never trained degrades to the shipped
 *  circuit rather than to nothing - and `qcnn_qubits` then reports the width that actually RAN,
 *  which is what lets the page tell the truth about a request it could not honour. */
async function loadQcnnWeights(usecase: string, qubits?: number): Promise<QcnnWeights | null> {
  const key = `${usecase}@${qubits ?? "default"}`;
  if (key in _qcnnCache) return _qcnnCache[key];
  const dir = `${import.meta.env.BASE_URL}data/${dataDirFor(usecase)}`;
  const paths = qubits != null
    ? [`${dir}/qcnn_weights_q${qubits}.json`, `${dir}/qcnn_weights.json`]
    : [`${dir}/qcnn_weights.json`];
  for (const path of paths) {
    try {
      const r = await fetch(path);
      // `r.ok` is NOT the test. This is an SPA: the host serves index.html with HTTP 200 for any path
      // that does not exist, so `r.ok` is true for a file that was never deployed. The old guard only
      // appeared to work because `r.json()` then threw on HTML and the catch swallowed it - a check
      // that reads as handling the case while the catch does the work. Content-type is the
      // discriminator, which is the same rule the verification checklist applies to `curl`.
      const json = (r.headers.get("content-type") ?? "").includes("application/json");
      if (r.ok && json) { _qcnnCache[key] = (await r.json()) as QcnnWeights; return _qcnnCache[key]; }
    } catch { /* try the next candidate */ }
  }
  _qcnnCache[key] = null;
  return null;
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
  // Keys already written, so a second write with a DIFFERENT value is visible rather than silent.
  //
  // `metrics[name] = value` is last-write-wins over a flat map, and two call sites in one run can
  // reach the same key with no syntactic relationship between them. That is not hypothetical:
  // `encrypted_agreement` was emitted by the batched figure (a rate over 30 rows) and then again by
  // the single-row path (`mism === 0 ? 1 : 0`, which over one row can only be 0 or 1). The second
  // won. On finance both read 1 and nothing showed; on genomics the page printed
  // `encrypted_agreement 1` beside `encrypted_top1_mismatches_batch 3` of 30 rows - two rows of the
  // same table contradicting each other, with the wrong one on top.
  //
  // A silent overwrite is the failure mode, so this does not throw: on a demo page an exception
  // would replace a wrong number with a blank page. It records the collision as a metric of its own
  // and warns, which makes it reachable from the browser checklist. Rewriting a key with the SAME
  // value is fine and common - several stages re-report a constant.
  const metricWrites = new Map<string, any>();
  const metric = (stage: string, name: string, value: any) => {
    if (metricWrites.has(name) && metricWrites.get(name) !== value) {
      const clash = `${name}: ${metricWrites.get(name)} -> ${value}`;
      console.warn("[metric] overwritten with a different value:", clash);
      metrics.metric_overwritten = clash;
      emit("metric", stage, "", { name: "metric_overwritten", value: clash });
    }
    metricWrites.set(name, value);
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
  async function runEncryptedTelecomStudent(
    stage: string, raw: number[] | undefined, testSample?: { X: number[][]; y: number[] },
  ) {
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

      if (!_fheWarmed) {
      const tk = performance.now();
      await warmFHE();
      _fheKeygenMs = performance.now() - tk;
      _fheWarmed = true;
    }
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
      metric(stage, "encrypted_max_abs_error", maxAbs);
      metric(stage, "encrypted_precision_bits", bits);
      metric(stage, "encrypted_argmax_location", "client, plaintext");
      emit("log", stage, `encrypted tier (REAL CKKS, ${w.n_parameters} params, `
        + `${w.W.length} dot products): class "${w.classes?.[encCls] ?? encCls}" in ${r2(fhe.fheMs)} ms. `
        + `Max abs error ${maxAbs.toExponential(3)}`
        + (bits !== null ? ` (about ${bits} bits of precision, MAX not mean)` : "")
        + (precisionFailed
          ? ` -- PRECISION FAILURE: the error is at or above 1, so there are no bits of precision to `
            + `report. The decrypted result is not a usable approximation of the plaintext one.`
          : "") + ".");

      // THE SAME STUDENT OVER THE WHOLE HELD-OUT SPLIT, not the one row above.
      //
      // `encrypted_top1_mismatches` and `encrypted_scores_compared` used to be emitted here from the
      // single row: the mismatch count could only be 0 or 1, and "scores compared" counted CLASSES.
      // A single agreement is not evidence that encrypted inference reproduces plaintext. The batch
      // costs no extra homomorphic work - `encryptedLinearBatch` packs the rows at the reduction's
      // own stride, so one multiply-plain and log2(width) rotations per class serve every row.
      if (testSample?.X?.length) {
        const rowsZ = testSample.X.map((row) =>
          row.map((v, i) => (v - (mean[i] ?? 0)) / (scale[i] || 1)));
        const batch = await encryptedLinearBatch(w.W, w.b, rowsZ);
        const encCls2 = batch.logits.map(argmax);
        const plainCls2 = batch.plainLogits.map(argmax);
        const mism = encCls2.reduce((a, c, i) => a + (c === plainCls2[i] ? 0 : 1), 0);
        // Same margin test as the finance/genomics figure, so a mismatch means the same thing on
        // every page that draws this chart. An 8-class student can tie too, and a count reported
        // without the margin reads as a precision failure on whichever page happens to hit one.
        const flipped2 = encCls2.map((_, i) => i).filter((i) => encCls2[i] !== plainCls2[i]);
        const flipMarginMax2 = flipped2.length
          ? Math.max(...flipped2.map((i) => decisionMargin(batch.plainLogits[i]))) : 0;
        // No tie/not-tie verdict here either - see the note at the finance/genomics twin.
        // The undecidable-row disclosure, on THIS path too.
        //
        // It was added to the finance/genomics/medical batch and not here, which made the fix for a
        // conditional disclosure itself conditional - on which of the two code paths a reader
        // happened to be looking at. Telecom's 8-class softmax over z-scored features is far less
        // likely to produce exact ties than genomics' [-w/2, +w/2] model, but "unlikely" is not
        // something a reader can see, and the count is what lets them stop guessing.
        const tieRows2 = batch.plainLogits.filter((row) => decisionMargin(row) === 0).length;
        const decidable2 = batch.rows - tieRows2;
        const flippedDecidable2 =
          flipped2.filter((i) => decisionMargin(batch.plainLogits[i]) !== 0).length;
        const bits2 = batch.maxAbsError > 0 && batch.maxAbsError < 1
          ? Math.floor(-Math.log2(batch.maxAbsError)) : null;
        metric(stage, "encrypted_top1_mismatches", mism);
        metric(stage, "encrypted_scores_compared", batch.rows * w.W.length);
        metric(stage, "encrypted_rows_scored", batch.rows);
        // The denominator, and the gap. `encrypted_agreement` is over `rows_scored`, not over the
        // whole split, and those differ whenever the split outgrows one ciphertext.
        metric(stage, "encrypted_rows_available", batch.rowsOffered);
        metric(stage, "encrypted_rows_dropped_by_cap", batch.rowsOffered - batch.rows);
        metric(stage, "encrypted_rows_per_ciphertext", batch.rowsPerCiphertext);
        metric(stage, "encrypted_batch_ms", r2(batch.fheMs));
        // Per-rotation cost on this page too, so the comparable unit exists for every use case that
        // runs an encrypted tier rather than only for the ones with a `private_inference` stage.
        metric(stage, "fhe_batch_rotations", batch.rotations);
        if (batch.rotations > 0) {
          metric(stage, "fhe_ms_per_rotation",
            Math.round((batch.rotateMs / batch.rotations) * 1000) / 1000);
        }
        metric(stage, "encrypted_batch_max_abs_error", batch.maxAbsError);
        metric(stage, "encrypted_batch_precision_bits", bits2);
        // The agreement figure follows Concrete-ML's "Similarity"; the mismatch COUNT follows
        // arXiv:2605.22237 Table IX, which reports it as an integer expected to be zero. Both
        // conventions are already used for the single-row path; this is the same pair over N rows.
        metric(stage, "encrypted_agreement", r2(1 - mism / Math.max(1, batch.rows)));
        metric(stage, "encrypted_agreement_rows", batch.rows);
        metric(stage, "encrypted_undecidable_rows", tieRows2);
        metric(stage, "encrypted_decidable_rows", decidable2);
        metric(stage, "encrypted_agreement_over_decidable",
          decidable2 > 0 ? r2(1 - flippedDecidable2 / decidable2) : null);
        if (mism > 0) {
          metric(stage, "encrypted_flip_margin_max", flipMarginMax2);
        }
        artifacts.encrypted_agreement_plot = {
          kind: "encrypted-agreement",
          rowsOffered: batch.rowsOffered,
          points: batch.logits.map((row, i) => ({
            enc: row[encCls2[i]], plain: batch.plainLogits[i][encCls2[i]],
            agree: encCls2[i] === plainCls2[i],
          })),
          rows: batch.rows, classes: w.W.length,
          // Over all rows x classes. `points` carries the winning class only, so the caption cannot
          // derive this from what it plots - and printing the plotted maximum beside a class-score
          // count is how the finance caption came to understate the deviation by 37%.
          maxAbsError: batch.maxAbsError,
          flipMarginMax: flipMarginMax2,
          undecidableRows: tieRows2, decidableRows: decidable2,
          agreementOverDecidable: decidable2 > 0 ? r2(1 - flippedDecidable2 / decidable2) : null,
        };
        emit("log", stage, `encrypted tier over the HELD-OUT SPLIT: ${batch.rows} of `
          + `${batch.rowsOffered} rows x ${w.W.length} classes = ${batch.rows * w.W.length} scores in `
          + `one ciphertext (${batch.rowsPerCiphertext} rows fit at this feature width), `
          + `${r2(batch.fheMs)} ms total`
          + (batch.rowsOffered > batch.rows
            ? `. ${batch.rowsOffered - batch.rows} rows did NOT fit and were not scored, so the `
              + `agreement below is over ${batch.rows}, not over the whole split`
            : "") + `. `
          + `Top-1 mismatches vs plaintext: ${mism} of ${batch.rows}. Max abs error `
          + `${batch.maxAbsError.toExponential(3)}`
          + (bits2 !== null ? ` (about ${bits2} bits, MAX not mean)` : "")
          // The undecidable count on the LOG too. It reached this path's metrics and the chart
          // caption and not this sentence, which is the third surface and the third time in one
          // round that a disclosure landed on some of them. Printed even at zero, because "0 of 512"
          // is the fact that makes this page's 1.00 a different claim from genomics' 1.00.
          + `. ${tieRows2} of the ${batch.rows} rows are scored identically across classes by the `
          + `plaintext model itself; over the ${decidable2} it does decide, agreement is `
          + (decidable2 > 0 ? r2(1 - flippedDecidable2 / decidable2) : "not computable")
          + (flipped2.length > 0
            ? `. Widest plaintext margin among the ${flipped2.length} flipped row(s): `
              + `${flipMarginMax2.toExponential(3)}`
            : "") + ".");
      }
      emit("log", stage, `the ARGMAX is NOT encrypted: all ${w.W.length} class scores are decrypted `
        + `and the maximum taken in the clear, because CKKS has no native comparison. That is the `
        + `standard construction and it reveals nothing here, since in one tab the client is the user.`);
    } catch (e) {
      // No invented numbers on failure. A skipped encrypted tier is reported as skipped.
      emit("log", stage, `encrypted tier skipped: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  // WASM unless the visitor selected WebGPU (`onnx_backend`, sent only when chosen). Until Round 60
  // an adapter merely existing selected WebGPU, and on the machine measured that was the slow path.
  const onnxBackendChoice: OnnxBackend = config.onnx_backend === "webgpu" ? "webgpu" : DEFAULT_ONNX_BACKEND;
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
      rowIndex: Math.floor(u() * 4096), modelKey: chosen, backend: onnxBackendChoice,
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
      metric(stage, "published_model_backend_requested", onnxBackendChoice);
      // Says which of the two facts applies when they differ: WebGPU selected, no usable adapter.
      const fellBack = onnxBackendChoice === "webgpu" && p.backend !== "onnx-webgpu"
        ? " (WebGPU was selected; this browser has no usable adapter)" : "";
      emit("log", stage,
        `published model ${p.modelKey} ran here on ${p.backend}${fellBack}: row ${p.rowIndex} of its own `
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
        const v = await evaluatePublishedModel(usecase, 0.05, key, onnxBackendChoice);
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
        // SP 800-227 s4.6.3: transcript is c1 || c2 || ek1 || ek2. It was ek1 || ek2 || c2 until
        // Round 38 - the ML-KEM ciphertext was missing, which is the input that delivers the
        // IND-CCA-preserving property; the standard says the encapsulation keys are NOT needed for
        // it. See src/qsh/pqc/hybrid_kem.py::encapsulate for the full note. v2 because the derived
        // key changes and deriving a different key under an unchanged label is what domain
        // separation exists to prevent.
        // The parameter set is IN the separator, not assumed. SP 800-227 s4.6.3 requires domain_sep to
        // identify the composite scheme including the "choice of parameter set", and this was the fixed
        // literal "ML-KEM-768" while `kemImpl()` genuinely switches to ML-KEM-1024 on the category-5 path -
        // The shared combiner. This block built its own info string until Round 39, which is how
        // the browser came to derive a different session key from the backend: the output-length
        // tag was added to Python alone, and nothing could see it because no browser test touched
        // the combiner. `check:combiner` now reproduces committed Python vectors through the real
        // module and asserts the separator literal lives in exactly one file.
        const kemLabel = kemNameFor(kemSel);
        const bind = (pq: Uint8Array, cl: Uint8Array) =>
          hybridCombine(pq, cl,
            { c1: e.cipherText, c2: aliceXPub, ek1: bob.publicKey, ek2: bobXPub }, kemLabel, 32);
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
          // The info string is DECODED FROM THE COMBINER, not typed here. It used to be typed,
          // and it printed a separator the run had not used - it omitted the length tag the
          // backend applies. A run log that states what the run did has to read it.
          + `ephemeral public key, combined with HKDF-SHA256(pq_ss || x25519_ss), info="${new TextDecoder().decode(hybridCombinerInfo(kemLabel, 32)).replace(/\|$/, "")}" then transcript c1||c2||ek1||ek2, per NIST SP 800-227 s4.6.3 — byte-identical to the backend, asserted by check:combiner`);
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
        // RENAMED from `skr`. ETSI GS QKD 015's `skr` is bit/s and this number is bits per PULSE;
        // leaving one name on two dimensions inside the same signed payload is the defect this
        // round exists to remove, one namespace over.
        // 1e6, matching `api.ts`. The pair split in Round 45: that file moved to six decimals with a
        // comment saying "the control-plane panel shows 0.002355", and these two sites stayed at
        // 1e5, so the panel showed 0.00235 and the comment described the other surface.
        metric(stage, "skr_per_pulse", Math.round(res.secretKeyRate * 1e6) / 1e6);
        metric(stage, "secure_key_bits", res.secureKeyBits);
        // ACTUALLY REPORTED NOW. `secretKeyRateNonDecoy` was added in Round 45 with a JSDoc
        // saying "Reported beside `secretKeyRate`" and was read by nothing but its own test -
        // an inert FIELD shipped in the same commit as the gate against inert CONTROLS. The
        // counterfactual is the point of computing a decoy bound: at mu=0.5 it is 0, because
        // the tagged fraction is 7.26 and no detection can be certified single-photon.
        if (res.secretKeyRateNonDecoy != null) {
          metric(stage, "skr_per_pulse_no_decoy",
                 Math.round(res.secretKeyRateNonDecoy * 1e6) / 1e6);
        }
        // THE FOUR TOGETHER, on both QKD stages. The satellite twin below emitted no rate at all
        // while its own comment claimed "both report the same quantity" - a comment naming a
        // boundary and describing one side of it.
        {
          const clk = keyRateWithClock(res.secretKeyRate, pulses,
                                       sourceRateHz(config?.qkd_source_rate_mhz,
                                                    Number(qkdCfg().source_rate_hz)));
          if (clk.skrBitPerSec != null) metric(stage, "skr_bit_s", Math.round(clk.skrBitPerSec));
          if (clk.channelSeconds != null) metric(stage, "channel_time_s", clk.channelSeconds);
          metric(stage, "source_rate_hz", clk.sourceRateHz);
          metric(stage, "qkd_pulses", pulses);
        }
        if (eRes) metric(stage, "chsh_s", r2(eRes.chshS));
        // The MODEL error rate e0 = (1 - V)/2, to four places, beside the sampled `qber`. S is computed
        // from the same V, so S = 2 sqrt(2)(1 - 2 e0) holds EXACTLY - a self-consistency of the Werner
        // model, not an independent validation - and only approximately from the displayed `qber`,
        // which is a shot-noise draw rounded to two places: at 1 km with no misalignment it shows
        // 0.02 while e0 is 0.0150, so 2 sqrt(2)(1 - 2 x 0.02) = 2.72 against the S of 2.74 shown.
        if (eRes) metric(stage, "e91_model_qber", Math.round(eRes.e0 * 1e4) / 1e4);
        emit("log", stage, eRes
          // "simulated" in the line itself: /orchestration shows these figures with no other label in
          // a full deployment (the banner that says so renders only in the public demo).
          ? `E91 (simulated, entanglement / CHSH S=${r2(eRes.chshS)}): QBER=${res.qber.toFixed(4)}, ${res.secureKeyBits} secure bits`
          : `BB84 (simulated, decoy-state): QBER=${res.qber.toFixed(4)}, ${res.secureKeyBits} secure bits`);
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
        // 1e6, matching `api.ts`. The pair split in Round 45: that file moved to six decimals with a
        // comment saying "the control-plane panel shows 0.002355", and these two sites stayed at
        // 1e5, so the panel showed 0.00235 and the comment described the other surface.
        metric(stage, "skr_per_pulse", Math.round(res.secretKeyRate * 1e6) / 1e6);
        metric(stage, "secure_key_bits", res.secureKeyBits);
        // ACTUALLY REPORTED NOW. `secretKeyRateNonDecoy` was added in Round 45 with a JSDoc
        // saying "Reported beside `secretKeyRate`" and was read by nothing but its own test -
        // an inert FIELD shipped in the same commit as the gate against inert CONTROLS. The
        // counterfactual is the point of computing a decoy bound: at mu=0.5 it is 0, because
        // the tagged fraction is 7.26 and no detection can be certified single-photon.
        if (res.secretKeyRateNonDecoy != null) {
          metric(stage, "skr_per_pulse_no_decoy",
                 Math.round(res.secretKeyRateNonDecoy * 1e6) / 1e6);
        }
        // The same four as the fibre stage, from the FREE-SPACE clock key. The default value is the
        // same 100 MHz, but the provenance is not: on this page it is the anchor's own measured
        // configuration, and on fibre it is that clock carried over to different apparatus. Two keys
        // because one could carry only one of those two sentences.
        {
          const clk = keyRateWithClock(res.secretKeyRate, satPulses,
                                       sourceRateHz(config?.sat_source_rate_mhz,
                                                    Number(qkdCfg().free_space_source_rate_hz)));
          if (clk.skrBitPerSec != null) metric(stage, "skr_bit_s", Math.round(clk.skrBitPerSec));
          if (clk.channelSeconds != null) metric(stage, "channel_time_s", clk.channelSeconds);
          metric(stage, "source_rate_hz", clk.sourceRateHz);
          metric(stage, "qkd_pulses", satPulses);
        }
        if (satERes) metric(stage, "chsh_s", r2(satERes.chshS));
        // The satellite twin of the fibre stage's `e91_model_qber` - same reason, same precision.
        if (satERes) metric(stage, "e91_model_qber", Math.round(satERes.e0 * 1e4) / 1e4);
        (config as any)._satQber = res.qber;
        emit("log", stage, satERes
          ? `free-space E91 (simulated, CHSH S=${r2(satERes.chshS)}): ${res.channelLossDb}dB loss, QBER=${res.qber.toFixed(4)}, ${res.secureKeyBits} bits`
          : `free-space BB84 (simulated): ${res.channelLossDb}dB loss, QBER=${res.qber.toFixed(4)}, ${res.secureKeyBits} bits`);
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
        // A LABELLED BLOCK, NOT `break`. `break` inside a `switch` case exits the whole case, so
        // selecting the PQC-only profile on satellite did not merely skip QSDC - it skipped the
        // anomaly verdict, the entire SHAP panel and the published-model row that follow, and
        // `usePipelineRun` offers `pqc-only` on every use case. The page lost four panels and said
        // nothing about why.
        //
        // That contradicts the comment on the very block it was skipping, which argues at length
        // that the step "scores a window of the BUNDLED corpus, not the telemetry the channel just
        // delivered, so it has no dependency on whether QSDC aborted". Round 31 moved that block out
        // of `if (!abort)` for exactly this reason and left it inside the PQC-only return - the same
        // fix, one of two call sites, which is this project's most-repeated shape.
        qsdc: {
        if (config.qkd === "pqc-only") {
          emit("log", stage, "QSDC direct telemetry skipped (PQC-only profile): QSDC carries the "
            + "message on the quantum channel itself, so with no quantum channel there is nothing "
            + "to send it over. The payload travels under the PQC session key instead. The telemetry "
            + "anomaly verdict and its explanation still run below: they score the bundled corpus, "
            + "not the delivered payload.");
          break qsdc;
        }
        metric(stage, "inference_provenance", "simulated");
        metric(stage, "inference_backend", "analytic free-space channel model (quantumSim)");
        // Same channel the satellite_qkd stage above built, distance included. Reading only sat_eve
        // here meant the downlink range moved the QKD QBER but not the QSDC error rate, so one
        // control produced two inconsistent pictures of the same link.
        // One shared evaluation (`qsdcLink`), so the tooltips that quote C_s read the same function.
        const link = qsdcLink(satChannel(config));
        const e = link.e;
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
        // `g` IS Q_Eve/Q_Bob AND IT INCLUDES DETECTOR EFFICIENCY. Qi, Long et al. state it twice,
        // including numerically: "the loss of the back channel, includes the efficiency of the
        // detector and channel loss, is about 4.1 dB, so that g = 2.57" - and 10^(4.1/10) = 2.5704
        // reproduces that exactly, so the FORM was right and what went into L was not. Omitting
        // eta_d makes g too small, which OVERSTATES secrecy capacity: an optimistic-security error.
        //
        // Fixed in BOTH languages in one change. `src/qsh/qkd/qsdc.py` carries the identical
        // expression, and this project has recorded nine rounds of a fix reaching one language -
        // most recently E91, where the copy that was fixed was the one no visitor executes.
        //
        // Downlink, eta_d 0.40: g 280.43 -> 701.06, a factor 2.50x. On fibre at eta_d 0.25 it is
        // 4.00x. One number would be wrong for one of the two paths.
        const lossDb = link.lossDb;
        const g = link.g;
        // THE SECOND ENTROPY TAKES e_x + e_z, NOT A POOLED RATE. Eq. (8) is
        // `Cs = Q_Bob * [1 - h(e) - g*h(e_x + e_z)]`, with e_x and e_z the per-basis error rates from
        // the check. Their own numbers settle it: the paper states I(A:E) = g*Q_Bob*h(e_x+e_z) =
        // 9.1e-4 at g=2.57, Q_Bob=0.3%, e_x=e_z=0.8%, and only h(0.016) = 9.1248e-4 reproduces that -
        // h(0.006) gives 4.08e-4, off by 2.2x. Pooling understates Eve by ~2.24x and OVERSTATES Cs by
        // 26%. Doubling is sound here only because nothing in this channel model distinguishes the
        // bases, so the two rates share an expectation; a basis-asymmetric model would need both.
        // Fixed in BOTH languages in one change - `src/qsh/qkd/qsdc.py` carries the same expression.
        const secrecy = link.cs;
        const abort = secrecy <= 0;
        // WHY the abort fired matters more than THAT it fired, and reporting both as
        // `qsdc_eavesdrop_detected` was wrong. Re-evaluate the same link with Eve removed: if it
        // still cannot close, the limit is the LINK BUDGET, and the page was accusing a clean
        // channel of carrying an eavesdropper on every single run.
        //
        // At the DEFAULT budget it still cannot close at any range: 5 dB atmospheric + 3 dB pointing
        // plus the detector's 4.0 dB give g = 15.8 even at zero range, and C_s is -3.70 at 0.1 km
        // with no eavesdropper. Only with the fixed losses below about 0.65 dB in total does a short
        // link close (at 0 + 0 dB it closes out to about 129 km, still well short of a LEO slant
        // range). That is not a bug in this code, it is the finding: QSDC does not close over a LEO
        // free-space downlink under the published criterion. Qi et al. demonstrated it at 4.1 dB on
        // FIBRE; the default downlink here is 24.5 dB. (This comment said "at every setting this
        // demo offers" until 2026-10-24 - written before the Atmospheric and Pointing controls
        // existed, with a g that omitted detector efficiency.)
        const secrecyNoEve = (() => {
          const e0 = qsdcLink(satChannel(config, 0)).e;
          // THE SECOND CALL SITE, AND IT KEPT THE POOLED ARGUMENT WHEN THE FIRST WAS FIXED.
          // Caught the same hour, inside the SAME FILE, by a commit whose own comment claimed the
          // change had been applied in both languages. The twin was three hundred lines away and
          // a `switch` arm apart. Both sites now call one exported function, which is the only
          // form of this fix that a gate can check.
          return qsdcSecrecyCapacity(e0, g);
        })();
        const lossLimited = secrecyNoEve <= 0;
        metric(stage, "qsdc_error_rate", r2(e));
        // SIGNED. `Math.max(0, secrecy)` stood here, so a link missing by 51 bit/carrier and one
        // missing by 0.001 both reported "0" - and 0 also reads as marginal-but-not-negative, the
        // one thing it is not. At 1 km with Eve at 1.0 (default budget) the value is -15.6045 by the
        // current formula (-5.2033 when this was written, before detector efficiency entered g) and
        // the page displayed 0. How badly a link fails is the quantity that distinguishes "widen the
        // aperture" from "QSDC does not close over this class of link".
        metric(stage, "qsdc_secrecy_capacity", r2(secrecy));
        metric(stage, "qsdc_eve_gain_g", r2(g));
        metric(stage, "qsdc_link_budget_limited", lossLimited ? 1 : 0);
        // Reserved for what the name says: an abort this link would NOT have suffered on its own.
        metric(stage, "qsdc_eavesdrop_detected", abort && !lossLimited ? 1 : 0);
        // THIS VERDICT IS ASYMPTOTIC, AND SAYING SO IS THE PARITY WITH PYTHON.
        //
        // `src/qsh/qkd/qsdc.py` samples a finite check subset and decides on a Serfling upper bound
        // at eps = 1e-9, because a handful of check carriers cannot certify a low error rate: it
        // used to deliver 118 times in 200 on a link whose true QBER was six times its own
        // tolerance, purely because a zero-error sample is common at 29 carriers. This path has no
        // sampled subset - it evaluates the analytic channel error directly - so it reports the
        // INFINITE-BLOCK limit, which is a different and more permissive quantity.
        //
        // The two do not contradict each other and the page must not imply a certificate it did not
        // compute. Stating the regime is the honest form: an asymptotic C_s > 0 is a necessary
        // condition for delivery, never a sufficient one, and the block length needed to certify it
        // is large - 32 KB on a 2 km fibre link, 4 MB on this downlink at its most favourable
        // setting, against the 74 bytes of telemetry the demo sends.
        metric(stage, "qsdc_regime", "asymptotic (infinite block; no finite-size certificate)");
        artifacts.telemetry_delivered = !abort;
        emit("log", stage, !abort
          ? `QSDC (simulated) delivered telemetry (secrecy ${secrecy.toFixed(4)} bit/carrier, g=${g.toFixed(1)})`
          : lossLimited
            ? `QSDC (simulated) NOT AVAILABLE on this link — secrecy capacity ${secrecy.toFixed(4)} bit/carrier `
              + `at ${lossDb.toFixed(1)} dB including the detector (g=${g.toFixed(1)}), and it is still `
              + `${secrecyNoEve.toFixed(4)} with NO eavesdropper. The limit is the link budget, not `
              // The criterion as computed: C_s = 1 - h(e) - g h(e_x + e_z), with e_x + e_z = 2e here.
              // This line quoted "C_s > 0 requires h(QBER) < 1/(1+g)", the POOLED form that the
              // per-basis correction replaced - a necessary condition, not the one this code applies.
              + `an attack: C_s = 1 - h(e) - g h(2e) = 1 - ${h2(e).toFixed(4)} - `
              + `${(g * h2(Math.min(0.5, 2 * e))).toFixed(4)} at e = ${e.toFixed(4)}. `
              + `Telemetry falls back to the PQC session key.`
            : `QSDC (simulated) ABORTED: secrecy capacity ${secrecy.toFixed(4)} <= 0 at error ${e.toFixed(4)} `
              + `with g=${g.toFixed(1)} (${lossDb.toFixed(1)} dB including the detector) — this link closes at `
              + `${secrecyNoEve.toFixed(4)} without Eve, so the loss is attributable — telemetry withheld`);
        }   // end qsdc: everything below scores the BUNDLED corpus and runs on both profiles
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
        // No all-zero fallback. `?? new Uint8Array(32)` silently substituted 32 zero bytes when the
        // handshake had not run, and the stage then reported "keyed-hash PSI" over a keyed hash
        // whose key is public knowledge - a security property the run does not provide, stated as
        // if it did. The federated path below already refuses rather than degrades, for exactly this
        // reason, and its comment says why: "the claim is written once and the fallback is taken
        // forever."
        //
        // `pqc_handshake` precedes this stage in every flow that reaches it, so the throw is
        // unreachable today. That is the argument FOR making it a throw rather than against it: this
        // round moved key material between stages, and if the ordering changes again this fails
        // loudly instead of quietly weakening the PSI to an unkeyed one.
        const key = (config as any)._sessionKey as Uint8Array | undefined;
        if (!key || key.length === 0) {
          throw new Error(
            "qsmpc_compute needs the PQC session key: keyed-hash PSI has no unkeyed mode",
          );
        }
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
        // The same held-out rows with untrained weights. Without it "accuracy 0.74" cannot be read
        // as evidence that the federated rounds did anything, and the round series cannot supply
        // that either once the problem converges inside round one.
        metric(stage, "federated_baseline_accuracy", q.baselineAcc);
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
        // Whether the QCNN branch below replaced the surfaced class. The SHAP flag is a
        // CONCLUSION about what produced that class, so it is computed once at the end of this
        // case rather than assumed at the start and then contradicted.
        let qcnnSurfacedClass = false;
        // clientIDS genuinely trains an 8-class softmax in this tab, so the timing and the F1 are
        // measured. The *data* is synthetic, which the run logs separately — measured computation
        // over synthetic input is not the same claim as measured performance on real traffic.
        // `measured-js`, NOT `measured-wasm`. `clientIDS.ts` imports only `idsCfg` and `rng`; its
        // training loop is `Math.exp`/`Math.max` arithmetic and `rng` is a JS LCG. No wasm module is
        // instantiated anywhere on this path.
        //
        // The twin was corrected for this precise reason and carries the reason in its own comment -
        // "Claiming WASM here made every pipeline run on every page render 'MEASURED (local WASM)'
        // without a wasm module ever being instantiated" - and `ProvenanceBadge.tsx` repeats it:
        // "The claim was wrong in the direction that flatters." That fix reached one of the two
        // sites. Telecom has no `private_inference` stage, so nothing downstream overwrote it, and
        // the badge rendered "MEASURED (local WASM) / Computed in this browser tab on the
        // WebAssembly runtime" for a plain-JS loop.
        //
        // Downgrading this does NOT downgrade the published ONNX model, which gets its own badge and
        // genuinely does run on the wasm runtime - the same convention the `private_inference` twin
        // follows when it publishes `measured-js` and then goes on to run node-seal CKKS.
        metric(stage, "inference_provenance", "measured-js");
        metric(stage, "inference_backend", "in-browser softmax training (clientIDS), synthetic telemetry");
        // genuine in-browser IDS: softmax logistic trained + scored on the Al-Kuwari-2026 attack signatures.
        // Training volume per class = config.n_per_class (Parameters panel), so it genuinely shifts acc/F1.
        const ids = clientTrainScoreIDS(seed, Number(config.n_per_class ?? idsCfg().n_per_class));
        metric(stage, "ids_macro_f1", r2(ids.macroF1));
        // The denominator of the headline. `ids_macro_f1` averages over the classes PRESENT in the
        // held-out split, and the sentence beside it used to say "over 8 classes" as a literal - so
        // a run whose split happened to omit a class printed a claim about a denominator it had not
        // used, with nothing on the page able to contradict it.
        metric(stage, "ids_classes_scored", ids.classesScored);
        metric(stage, "ids_worst_class_f1", r2(ids.worstClassF1));
        metric(stage, "ids_binary_f1", r2(ids.binaryF1));
        metric(stage, "ids_accuracy", r2(ids.accuracy));
        // Which of the three leads, as a row rather than only as a log line. They render as three
        // numbers of identical shape, and the two that read highest are the two that collapse the
        // seven attack types into "attack", so a reader with no marker takes the flattering pair as
        // the result. The log below says this; the table did not.
        metric(stage, "ids_metric_roles",
          `headline: ids_macro_f1 (per-class mean over ${ids.classesScored} classes) and `
          + "ids_worst_class_f1; companions: ids_binary_f1 and ids_accuracy, both of which merge "
          + "the seven attack types into one class and so read high while a single class fails");
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
          // Set ONCE, at the end of this case, from what actually surfaced - see the assignment
          // after the QCNN branch. Written here first as an unconditional `true`, which was correct
          // for the softmax and for the encrypted tier and wrong the moment the QCNN branch below
          // replaced the surfaced class without touching it.
        }
        emit("log", stage, `QML-IDS (client, softmax logistic on Al-Kuwari-2026 signatures): `
          + `macro-F1=${r2(ids.macroF1)} over the ${ids.classesScored} classes present in the `
          + `held-out split`
          + (ids.classesScored < ids.classesDefined
            ? ` (of ${ids.classesDefined} defined - this split is missing `
              + `${ids.classesDefined - ids.classesScored}, and an absent class is not scored 0)`
            : "")
          + `, worst class ${r2(ids.worstClassF1)} `
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
        await runEncryptedTelecomStudent(stage, ids.explain?.xRaw, ids.testSample);

        // The 8-class QCNN, where the visitor selected it. This lives here rather than in
        // `private_inference` because telecom has no such stage - and putting the weights on disk and
        // registering the use case WITHOUT this made the QCNN option appear in the Parameters panel
        // and do nothing. An inert control is the defect class this file's own checklist opens with
        // (Round 20 audited use-case pages and found five), and the only thing that surfaced it was
        // running the deployed page: every local gate was green.
        if (String(config.inference || "ckks") === "qcnn" && ids.explain?.xRaw) {
          const w = await loadQcnnWeights(usecase, qubitsOf(config));
          if (w) {
            // Timed here TOO, and that "too" is the point. The latency was added to the
            // `private_inference` site alone, which telecom does not have - so a page reporting
            // `primary_inference_backend = qcnn-trained` carried no `qcnn_inference_ms`, and rows
            // PF9/PF10 asserting it were false for exactly one page. Found by running the deployed
            // telecom page, not by any gate: tsc, the suite and all thirteen local gates were green.
            // Two call sites for one scheme, one of them changed, is the defect this round is named
            // for - shipped again inside the round that names it.
            const t0 = performance.now();
            const qc = qcnnInferTrained(ids.explain.xRaw.slice(0, w.n_qubits), w);
            const qcnnMs = performance.now() - t0;
            artifacts.primary_inference = { backend: "qcnn-trained", class: qc.cls,
                                            qubits: qc.qubits, expectation: r2(qc.expectation) };
            metric(stage, "primary_inference_backend", "qcnn-trained");
            metric(stage, "primary_inference_class", w.class_names?.[qc.cls] ?? qc.cls);
            metric(stage, "qcnn_classes_available", w.readout?.n_classes ?? 2);
            metric(stage, "qcnn_inference_ms", r2(qcnnMs));
            metric(stage, "qcnn_qubits", qc.qubits);
            // The class no longer comes from the softmax, so the SHAP bars are no longer an
            // attribution of it. Recorded rather than assigned here: the flag is set once, below,
            // from this variable. With the flag set true earlier and never revised, the page
            // rendered two paragraphs at once - "these bars explain the same model that produced
            // the class reported above" and "the class this page surfaces came from the trained
            // QCNN ... a different decision" - both visible, contradicting each other.
            qcnnSurfacedClass = true;
            emit("log", stage, `QCNN (trained, ${w.readout?.n_classes ?? 2}-class basis readout on `
              + `${w.readout?.readout_qubits ?? 0} of ${w.n_qubits} qubits): class `
              + `"${w.class_names?.[qc.cls] ?? qc.cls}". Class index is ${w.readout?.class_index}, `
              + `${w.readout?.bit_order} - reading those bits in another order permutes the class map `
              + `while leaving every aggregate metric identical, which is why the convention ships in `
              + `the weights.`);
            // The verdict is COMPUTED, not asserted. This read `and it LOSES:` as a literal, guarded
            // only on the field being PRESENT - no comparison was ever performed, so a retrained
            // artifact that put the circuit ahead would have gone on printing that it lost. The
            // numbers were right beside the sentence the whole time.
            const mine = w.test_macro_f1, theirs = w.classical_lr_macro_f1;
            if (typeof mine === "number" && typeof theirs === "number") {
              const behind = mine < theirs;
              emit("log", stage, `and it ${behind ? "LOSES" : "WINS"}: macro F1 ${mine} against a `
                + `classical logistic regression at ${theirs} on the same split, with `
                + `${w.n_classes_predicted ?? "?"} of ${w.readout?.n_classes ?? 2} classes predicted. `
                + `Reported because it is the measured result, not because it flatters`
                + (behind ? "." : " - and if this ever reads WINS, check the split before believing it."));
            } else if (typeof mine === "number" || w.per_class_f1) {
              // One number without the other cannot support a verdict, so it does not state one.
              emit("log", stage, `held-out macro F1 ${mine ?? "n/a"}; no classical comparator ships in `
                + `this artifact, so no verdict is stated here.`);
            }
          } else {
            emit("log", stage, "QCNN selected but no trained weights are published for this use case");
          }
        }
        // ONE assignment, after everything that can change what surfaced. Telecom's value is a
        // property of the RUN, not of the page: the in-tab softmax decides unless the visitor
        // selected QCNN, in which case the circuit does and these bars explain something else.
        artifacts.explanation_is_of_reported_class = !qcnnSurfacedClass;
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
        // the configured ring degree) runs synchronously inside WASM on the main thread, and it grows
        // with that degree - `clientFHE` records that this
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
        // The model the page REPORTS and EXPLAINS, handed to the path that encrypts it, plus the
        // corpus in the same feature space so the agreement figure covers more than one row.
        const corpus = realIn?.model?.W ? await realInputCorpus(usecase) : null;
        const real = await realClientHybrid(feats, u, usecase === "medical_fl", usecase,
                                            realIn?.model ?? null, corpus);
        if (real) {
          // A real encrypted convolution is not scaled by anything: its latency IS the measurement.
          const bscale = real.conv
            ? 1
            : benchScale(usecase, config.model_size ?? "medium", config.workload ?? "linear");
          artifacts.fhe_prediction = real.prediction;
          // WHICH model the encryption evaluated. Until Round 38 this was always the demo pattern
          // while the class and the SHAP bars came from the bundled model, so every `fhe_*` figure
          // on finance and genomics described a weight vector that appeared nowhere else on the page.
          metric(stage, "fhe_model", real.encryptedRealModel
            ? "bundled real model - the same one the class and the SHAP bars come from"
            : "demo pattern (no bundled model ships for this use case)");
          if (real.batch) {
            const bt = real.batch;
            const bits = bt.maxAbsError > 0 && bt.maxAbsError < 1
              ? Math.floor(-Math.log2(bt.maxAbsError)) : null;
            metric(stage, "encrypted_rows_scored", bt.rows);
            metric(stage, "encrypted_rows_available", bt.rowsOffered);
            metric(stage, "encrypted_rows_dropped_by_cap", bt.rowsOffered - bt.rows);
            metric(stage, "encrypted_rows_per_ciphertext", bt.rowsPerCiphertext);
            metric(stage, "encrypted_batch_ms", r2(bt.fheMs));
            metric(stage, "encrypted_batch_max_abs_error", bt.maxAbsError);
            metric(stage, "encrypted_batch_precision_bits", bits);
            metric(stage, "encrypted_agreement", r2(1 - bt.mismatches / Math.max(1, bt.rows)));
            metric(stage, "encrypted_agreement_rows", bt.rows);   // the denominator, never implicit
            // The part of that denominator that carries no signal, and the rate over the rest.
            // Emitted on EVERY run, including the ones where no tie happened to flip - see the
            // note at the computation for why a conditional disclosure is the wrong shape here.
            metric(stage, "encrypted_undecidable_rows", bt.tieRows);
            metric(stage, "encrypted_decidable_rows", bt.decidable);
            metric(stage, "encrypted_agreement_over_decidable",
              bt.decidable > 0 ? r2(1 - bt.flippedDecidable / bt.decidable) : null);
            metric(stage, "encrypted_top1_mismatches_batch", bt.mismatches);
            if (bt.mismatches > 0) {
              metric(stage, "encrypted_flip_margin_max", bt.flipMarginMax);
              // The basis of the verdict, so it is checkable rather than taken on trust.
            }
            artifacts.encrypted_agreement_plot = {
              kind: "encrypted-agreement", points: bt.points, rows: bt.rows,
              rowsOffered: bt.rowsOffered, classes: real.logits.length,
              maxAbsError: bt.maxAbsError,   // all rows x classes; `points` is winning-class only
              flipMarginMax: bt.flipMarginMax,
              undecidableRows: bt.tieRows, decidableRows: bt.decidable,
              agreementOverDecidable: bt.decidable > 0
                ? r2(1 - bt.flippedDecidable / bt.decidable) : null,
            };
            emit("log", stage, `encrypted tier over the page's own corpus: ${bt.rows} of `
              + `${bt.rowsOffered} rows in one ciphertext (${bt.rowsPerCiphertext} fit at this `
              + `feature width), ${r2(bt.fheMs)} ms. Top-1 mismatches vs plaintext: `
              + `${bt.mismatches} of ${bt.rows}. Max abs error ${bt.maxAbsError.toExponential(3)}`
              + (bits !== null ? ` (about ${bits} bits, MAX not mean)` : "")
              // Stated, not adjudicated. A flip always happens at a margin below twice the
              // deviation - that is a theorem, not a measurement - so a verdict computed from
              // those two numbers says nothing. The count of rows the model scores IDENTICALLY is
              // the number that varies and that separates one page from another.
              + `. ${bt.tieRows} of the ${bt.rows} rows are scored identically across classes by `
              + `the plaintext model itself, so their outcome is decided by tie-breaking in the `
              + `clear and by encryption noise here; over the ${bt.decidable} rows it does decide, `
              + `agreement is `
              + (bt.decidable > 0 ? r2(1 - bt.flippedDecidable / bt.decidable) : "not computable")
              + (bt.mismatches > 0
                ? `. Widest plaintext margin among the ${bt.mismatches} flipped row(s): `
                  + `${bt.flipMarginMax.toExponential(3)}`
                : "")
              + (bt.rowsOffered > bt.rows
                ? `. ${bt.rowsOffered - bt.rows} rows did not fit and were not scored, so the `
                  + `agreement is over ${bt.rows}` : "") + ".");
          }
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
            // ONLY when no batch ran, and this is a correctness fix, not a tidy-up. This line and
            // the batch's `encrypted_agreement` a few dozen lines above share a key and are emitted
            // into the same stage, so this one - computed over a SINGLE row, where it can only ever
            // be 0 or 1 - silently overwrote the rate measured over the whole corpus. Finance hid it
            // (0 mismatches, both read 1); genomics exposed it, printing `encrypted_agreement 1`
            // directly beside `encrypted_top1_mismatches_batch 3` out of 30 rows, which is 0.9.
            // The single-row scalar being wrong is precisely why the batch figure was added, so
            // leaving it in place under the same name let the thing it replaced win.
            //
            // AND IT CARRIES ITS OWN DENOMINATOR AND TIE COUNT, like every other agreement rate.
            // A parity test written after the fourth mis-scoped disclosure this round found this
            // third emission site immediately: it published a rate with no `_rows`, no undecidable
            // count and no decidable rate, which is the same omission the other two paths had just
            // been fixed for - and three review passes and the author had all walked past it.
            // Over one row the tie count is 0 or 1 and the rate is 0 or 1; stating them is what
            // stops a page without a corpus looking like a page that measured something.
            if (!real.batch) {
              const single = decisionMargin(real.plainLogits) === 0 ? 1 : 0;
              metric(stage, "encrypted_agreement", mism === 0 ? 1 : 0);
              metric(stage, "encrypted_agreement_rows", 1);
              metric(stage, "encrypted_undecidable_rows", single);
              metric(stage, "encrypted_decidable_rows", 1 - single);
              metric(stage, "encrypted_agreement_over_decidable",
                single === 0 ? (mism === 0 ? 1 : 0) : null);
            }
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
            // COST PER HOMOMORPHIC ROTATION, from the measured rotation time and the counted
            // rotations - both raw, neither scaled.
            //
            // This is the axis on which this project's encrypted tier can be compared with a
            // published one at all. Total latency cannot be: this evaluates tens to a few thousand
            // parameters on WASM CPU in a browser tab, and the 2026 figures it would sit beside are
            // ResNets and transformers on A100s. Model size drops out of a per-rotation figure, and
            // a rotation is a key-switch, which is the operation those papers report and the one
            // that dominates CKKS cost.
            //
            // It is NOT `fhe_linear_ms / fhe_dense_rotations`, which is the division the metrics
            // table invites and which is wrong: `fhe_linear_ms` below is the measurement multiplied
            // by `benchScale(...)` for the selected model size, while the rotation count is raw.
            // On genomics that division reads 158 ms per rotation against a measured ~3 ms.
            metric(stage, "fhe_rotate_measured_ms", Math.round(real.dense.rotateMs * 1000) / 1000);
            if (real.dense.rotations > 0) {
              metric(stage, "fhe_ms_per_rotation",
                Math.round((real.dense.rotateMs / real.dense.rotations) * 1000) / 1000);
            }
          }
          // The one-time keygen, on the run that paid it. Reported so the ring degree's price is
          // visible beside the slot count it buys; a later run in the same tab does not re-pay it,
          // and the metric is absent there rather than repeating a stale figure as if it had.
          metric(stage, "fhe_ring_degree", fheCfg().poly_modulus_degree);
          if (_fheKeygenMs !== null) {
            metric(stage, "fhe_keygen_ms", r2(_fheKeygenMs));
            _fheKeygenMs = null;
          }
          // The raw measurement and the multiplier applied to it, so the projection below is
          // decomposable rather than a number of unstated provenance sitting beside raw counts.
          metric(stage, "fhe_linear_measured_ms", r2(real.fheMs));
          metric(stage, "fhe_bench_scale", r2(bscale));
          metric(stage, "fhe_linear_ms", r2(real.fheMs * bscale));
          // THE MEASURED END-TO-END ENCRYPTED LATENCY, which is the only figure that can be read
          // against an operational requirement.
          //
          // `fhe_linear_ms` + `mpc_nonlinear_ms` are both scaled by `bscale`, so the panel's "total"
          // is a projection for the selected model size, not a time anything took. And
          // `client_compute_ms` beside it is the PLAINTEXT reference call - sub-millisecond for a
          // linear model - so neither of the two obvious candidates answers "how long did the
          // private inference actually take". This does: encrypted linear plus the convolution
          // where there is one, plus the secret-shared non-linear stage, all as measured.
          //
          // The window it is reported against is the one these verticals actually work in: fraud
          // scoring, clinical decision support and compliance checks are interactive-batch, seconds
          // rather than milliseconds. Stated as a range with the verdict derived, so a page that
          // leaves the window says so instead of leaving a reader to compare two numbers.
          {
            const measuredMs = real.fheMs + real.mpcMs;
            metric(stage, "encrypted_tier_measured_ms", r2(measuredMs));
            const withinWindow = measuredMs <= 10_000;
            metric(stage, "encrypted_tier_within_operational_window", withinWindow ? 1 : 0);
            emit("log", stage, `measured end-to-end encrypted inference: ${r2(measuredMs)} ms `
              + `(CKKS linear${real.conv ? " + convolution" : ""} ${r2(real.fheMs)} ms, `
              + `secret-shared non-linear ${r2(real.mpcMs)} ms), on WASM CPU in this tab. `
              + (withinWindow
                ? "Inside the 1-10 s interactive-batch window these verticals work in."
                : "OUTSIDE the 1-10 s interactive-batch window - this would not meet the "
                  + "requirement it is being demonstrated for.")
              + " Note this is NOT the panel's `total`, which scales the same measurement by "
              + `${r2(bscale)}x for the selected model size.`);
          }
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
            // TWO SCOPES, NAMED, because conflating them published a wrong figure.
            //
            // On the published-weights path `convMs` is the WHOLE student - the convolution plus
            // its two dense layers - while `rotations` and `rotateMs` describe the convolution
            // block alone. `fhe_conv_ms` has therefore always been the student's total under a
            // name that says convolution, and Round 38 divided one by the other and published
            // "rotations are 10% of the convolution, im2col dominates". That share was computed
            // against a denominator containing 136 rotations the numerator excludes.
            //
            // The rename is the fix: the total is `fhe_encrypted_student_ms`, and `fhe_conv_ms` is
            // the convolution. On the synthetic-kernel path they are equal, because there the
            // convolution IS the whole computation.
            metric(stage, "fhe_encrypted_student_ms", r2(real.conv.convMs));
            metric(stage, "fhe_conv_ms", r2(real.conv.convOnlyMs));
            metric(stage, "fhe_conv_rotate_ms", Math.round(real.conv.rotateMs * 1000) / 1000);
            if (real.conv.rotations > 0) {
              // Both scoped to the convolution block.
              metric(stage, "fhe_conv_rotation_share",
                r2(real.conv.rotateMs / Math.max(1e-9, real.conv.convOnlyMs)));
              metric(stage, "fhe_conv_ms_per_rotation",
                Math.round((real.conv.rotateMs / real.conv.rotations) * 1000) / 1000);
            }
            // The student's dense layers, so every millisecond of the total is attributed. Without
            // them the page showed an 81 ms convolution inside a 569 ms student and said nothing
            // about the other 488.
            if (real.conv.denseRotations) {
              metric(stage, "fhe_student_dense_ms", r2(real.conv.denseMs ?? 0));
              metric(stage, "fhe_student_dense_rotations", real.conv.denseRotations);
              metric(stage, "fhe_student_dense_rotate_ms",
                Math.round((real.conv.denseRotateMs ?? 0) * 1000) / 1000);
              metric(stage, "fhe_student_dense_rotation_share",
                r2((real.conv.denseRotateMs ?? 0) / Math.max(1e-9, real.conv.denseMs ?? 0)));
            }
            // The convolution's own approximation error against a plaintext reference computed on
            // the same inputs. Reported separately from the dense layer's so a regression in one
            // cannot hide behind the other.
            metric(stage, "fhe_conv_max_abs_error", real.conv.maxAbsError);
            // The student's END-TO-END error, separately, where a student ran. Both are reported so
            // a regression in the convolution and one in the dense stack cannot land in one number,
            // which is what the comment above claimed and what the single key prevented.
            if (real.conv.studentLogitMaxAbsError != null) {
              metric(stage, "fhe_student_logit_max_abs_error", real.conv.studentLogitMaxAbsError);
            }
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
              + `${real.conv.multiplies} multiply-plain in ${r2(real.conv.convOnlyMs)} ms`
              // EVERY TERM SCOPED, because this sentence carried the withdrawn attribution after
              // the metrics above it were fixed. It read "<conv rotations> ... in
              // <the WHOLE student's ms>" - the same numerator/denominator mismatch that produced
              // the 10% figure - and then labelled `real.fheMs - real.conv.convMs`, which is the
              // page's own 8-feature bundled linear demo at ~23 ms, as "dense layer" while the
              // student's actual dense stack cost ~490 ms. A visitor reading the event log got the
              // retracted story back on the same page whose table had just been corrected.
              + (real.conv.denseRotations
                ? `; the student's ${real.conv.denseRotations} dense-layer rotations take `
                  + `${r2(real.conv.denseMs ?? 0)} ms more`
                : "")
              + `. This page's own bundled linear model adds `
              + `${r2(real.fheMs - real.conv.convMs)} ms, MPC ${r2(real.mpcMs)} ms. `
              + `Nothing here is scaled by a cost model.`);
          } else {
            emit("log", stage, `private inference (REAL in-browser FHE, ${config.model_size ?? "medium"}/${config.workload ?? "linear"}): `
              + `class=${real.prediction}; node-seal CKKS ${r2(real.fheMs * bscale)} ms + garbled-circuit 2PC ${r2(real.mpcMs * bscale)} ms `
              + `(measured per-op x${r1(bscale)} arch scale; real unit ${r2(real.fheMs)}/${r2(real.mpcMs)} ms, ct=${real.ctBytes}B), `
              // "plaintext reference", NOT "ONNX". `clientMs` times `privateInference`, which has had
              // no ONNX tier since Round 17 and says so in its own comments; its body is two plain-JS
              // dot products. The genuine ONNX latency (`published_inference_ms`) lands in the SAME
              // stage log, so this line carried two ONNX-labelled numbers differing by orders of
              // magnitude. The sibling emit fifteen lines below already says "client compute".
              + `plaintext reference ${clientMs} ms`);
          }
        } else {
          // honest last resort: node-seal genuinely unavailable here (e.g. no WASM) — representative figure, logged
          artifacts.fhe_prediction = inf.prediction;
          const split = mpcFheRow(usecase, config.workload ?? "linear", config.model_size ?? "medium", null, secFromConfig(config, metrics));
          // A CONFIG CONSTANT MUST NOT BE PUBLISHED UNDER THE MEASURED METRIC'S KEY.
          //
          // This emitted `costPerOp().demo_max_abs_error` - the literal 4.6e-07 from
          // `config.defaults.json` - under the SAME key the measured path fills from a real
          // decryption, which records 6.20e-13 and 1.93e-9. The only thing distinguishing them was
          // the log line beside it, and `downloadMetricsCsv` exports {metric, value} pairs with no
          // log at all, so the CSV a reviewer opens shows a config default indistinguishable from a
          // measurement. `docs/threat-model.md` describes this key as the computed CKKS
          // approximation error, "about 1e-9".
          //
          // Note the direction: 4.6e-07 is three to four orders WORSE than measured, so this
          // understated precision rather than flattering it. That does not make it a measurement.
          //
          // Renamed and given an explicit tier status, following the `published_model_status:
          // "failed"` convention this file already uses for the other unavailable-backend case.
          metric(stage, "encrypted_tier", "unavailable: node-seal did not load in this browser");
          metric(stage, "fhe_max_abs_error_representative", costPerOp().demo_max_abs_error);
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
          const w = await loadQcnnWeights(usecase, qubitsOf(config));
          if (w) {
            // MEASURED, on the same footing as the CKKS tier's `fhe_ms_per_rotation`. The QCNN was
            // reported by accuracy alone while every encrypted figure carried a latency, so the two
            // inference paths on one page could not be compared on cost at all. State-vector
            // simulation is 2^n work in the qubit count, which is the number that makes the figure
            // interpretable, so it travels with it - as `PF2` requires of the encrypted tier.
            // THE SPACE THE ARTIFACT DECLARES, not whatever `feats` happens to be in. `feats` is
            // `realIn.values`, which is StandardScaler output because the CKKS tier and the SHAP
            // panel are fitted there. A QCNN artifact declaring `input_space: "raw"` standardises
            // its input again, so handing it `feats` charged the transform twice and flattened the
            // wire: measured at tanh sd 3.3e-05 on `amount`, with 101 of 400 rows - 25.25% - given
            // a different class than the same rows correctly fed. (17.25% appeared here for one
            // round; that figure is a different comparison - two encoders over one array, rather
            // than one encoder over two - and describes a fix that was not the one made.)
            //
            // Falls back to `feats` only when the artifact does NOT declare `raw`, so a future
            // artifact in a third space cannot silently pick up the wrong array; the pairing is
            // asserted in tests/test_the_encoder_receives_the_space_the_artifact_declares.py.
            // BY NAME, with no default branch. `w.input_space` names the units the circuit was
            // trained on; the sample ships its rows keyed by the space each is in. A boolean
            // `=== "raw" ? a : b` would send an artifact retrained into a third space to whichever
            // array the else-branch happens to name - which is exactly how this shipped.
            const qcnnFeats = realIn?.bySpace?.[w.input_space] ?? (realIn ? undefined : feats);
            const t0 = performance.now();
            const qc = qcnnInferTrained((qcnnFeats ?? feats).slice(0, w.n_qubits), w);
            const qcnnMs = performance.now() - t0;
            artifacts.primary_inference = { backend: "qcnn-trained", class: qc.cls,
                                            qubits: qc.qubits, expectation: r2(qc.expectation) };
            metric(stage, "primary_inference_class", qc.cls);
            metric(stage, "primary_inference_backend", "qcnn-trained");
            metric(stage, "qcnn_expectation_z0", r2(qc.expectation));
            metric(stage, "qcnn_inference_ms", r2(qcnnMs));
            metric(stage, "qcnn_qubits", qc.qubits);
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
            // COMPUTED, like its twin in `qml_ids`. "the classical model is ahead" was unconditional
            // text over two numbers read from a file, so a retrained artifact that reversed the
            // ordering would have kept asserting it.
            const ahead = typeof mine === "number" && typeof theirs === "number"
              ? (theirs > mine ? "the classical model is ahead" : "the circuit is ahead here")
              : "no comparator ships in this artifact, so no verdict is stated";
            // `w.theta.length`, NOT `layers x n_qubits`. That product is the RETIRED
            // hardware-efficient ansatz's parameter count - one angle per wire per layer - and it
            // survived on the visitor-facing log line after Round 46 replaced the circuit with a
            // shared-filter QCNN. It rendered "3x8 trained angles" = 24 for artifacts carrying 77,
            // and "6x8" = 48 for telecom's 57: the only architecture description on the page,
            // describing a circuit the page does not run. The count is now read from the artifact,
            // so it cannot disagree with the weights it is printed beside.
            emit("log", stage, `inference backend = QCNN (QHE, TRAINED): class=${qc.cls} `
              + `(<Z0>=${qc.expectation.toFixed(4)}, ${w.theta.length} shared-filter angles over `
              + `${w.n_qubits} wires, ${w.layers} conv rounds per stage, `
              + `state-vector sim). Held-out ${label} ${(mine ?? 0).toFixed(4)} against a baseline of `
              + `${(base ?? 0).toFixed(4)} and logistic regression at ${(theirs ?? 0).toFixed(4)} on `
              + `the same split - ${ahead}, and this reports what was measured rather than what would `
              + `flatter. Test rows were never used to pick the checkpoint.`);
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
        // NO ADVANTAGE SCORE IS COMPUTED OR LOGGED HERE ANY MORE. This stage emitted
        // `quantum_advantage_score` and a log line ending "quantum advantage 10/10 (dominant)", on
        // /orchestration and on every inference page, and the agent prompt then quoted it back as run
        // context. Its inputs were a CKKS byte ratio divided by a QOTP qubit ratio and a CKKS+MPC
        // inference latency divided by an 8-qubit state-vector simulation latency - neither pair
        // measures the same thing. Retired on 2026-10-23, as the 0-100 composite was in Round 22.
        metric(stage, "qsmpc_t_gates", qrow.t_gate_count);
        emit("log", stage, `QSMPC-FHE comparison (QOTP+QCNN, ${qrow.t_gate_count} T-gates): `
          + "gate count, qubit count and simulation latency are measured; no advantage score is derived");
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
          // `fhe`/`mpc` record what RAN, not what the config carries. In the client-only demo the
          // selects are replaced by a badge reading "node-seal CKKS + secret-sharing MPC · real,
          // in-browser", while the state still held the server-side defaults `tenseal-cpu` and
          // `mpyc-shamir` - nobody chose them and neither executed. `config.fhe` and `config.mpc`
          // are read NOWHERE else in this file, so the only consumer was this payload: a
          // tamper-evident record naming two backends that did not run, ten lines under a badge
          // saying so. A signed record of the wrong thing is worse than an unsigned one, because
          // the signature is what invites a reader to trust it.
          config: { kem: config.kem ?? null, qkd: config.qkd ?? null, sig: config.sig ?? null,
                    fhe: isClientDemo() ? "node-seal-ckks" : (config.fhe ?? null),
                    mpc: isClientDemo() ? "secret-sharing-mpc-wasm" : (config.mpc ?? null),
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
