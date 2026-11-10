// Client-side compute for the public demo: in-browser equivalents of the backend
// benchmark + E2E pipeline, so the demo runs entirely in the browser (PQC via @noble,
// QKD/QBER via quantumSim, private inference via onnxruntime-web, PSI via RFC 9497's OPRF in oprfPsi.ts).
// The retired quantum-advantage formula mirrors src/qsh/benchmark/quantum_advantage.py and is rendered
// nowhere (retired 2026-10-23, see `quantumAdvantage`). The 0-100 security composite was retired in
// Round 22, and its module was removed in Round 86.
import { ml_dsa65, ml_dsa87 } from "@noble/post-quantum/ml-dsa.js";
import { ml_kem768, ml_kem1024 } from "@noble/post-quantum/ml-kem.js";
import { x25519 } from "@noble/curves/ed25519.js";
import { hkdf } from "@noble/hashes/hkdf.js";
import { sha256 } from "@noble/hashes/sha2.js";
import {
  benchCfg, costPerOp, dataCfg, demoFeatureDim, genericArchitecture, idsCfg, lendingCfg, medicalCnn, qkdCfg, qrngCfg,
  demoSample, federatedCfg, fheCfg, multiPartyPsiCfg, qsmpcCfg, quantumAdvantageCfg, studentModel, studentParamCount,
  tierArchitectures, webCfg, workloadMultipliers, modelsFor } from "./config";
import { CKKS_FLOOD_BITS, encryptedConv2d, encryptedHeldOutAgreement, encryptedLinear, encryptedLinearBatch,
  encryptedStudentInference, warmFHE, type CnnStudentWeights } from "./clientFHE";
import type { OcclusionMap } from "./cnnSaliency";
import type { HybridCnnResult } from "./clientHybridCnn";
import type { QcnnCkksResult } from "./clientQcnnCkks";
import { clientTrainScoreIDS } from "./clientIDS";
import { clientQFL } from "./clientQFL";
import { secureArgmax, warmMPC } from "./clientMPC";
import { randomBelow } from "./csprng";
import { optimizeQKD } from "./clientOptiQKD";
import { generateQuantumBits } from "./clientQRNG";
import { outOfView, satBlock, satChannel } from "./satLink";
import { blockCounts, blocksForPositiveKey, DEVIATION_LABELS, fibreBlockPulses, FINITE_KEY_BOUNDS, finiteKeySourceOf,
  keyLength, scaleCounts, type Deviation } from "./finiteKey";
import { stikeRates, stikeSend } from "./stike";
import { bb84, channelLossDb, e91, FIBER, fibreIntensities, h2, inView, intrinsicQber,
  keyRateWithClock, linkElevationDeg, qsdcLink, qsdcSecrecyCapacity, sourceOf, sourceRateHz, type ChannelParams,
  qcnnExpectation,
  qcnnInferTrained, qotpQcnnInfer, rng, taggedFraction,
  type QcnnWeights } from "./quantumSim";
import { type Provenance } from "../components/ProvenanceBadge";
import {
  browserModels, DEFAULT_ONNX_BACKEND, evaluatePublishedModel, type OnnxBackend, onnxStatus,
  runPublishedModel,
} from "./onnxInference";
import { heldOutCorpus, realInput, realInputCorpus, realSatelliteAnomaly, realFederatedSet,
  type HeldOutCorpus, type RealInput, realAlleleCounts } from "./realData";
import { hybridCombine, hybridCombinerInfo, kemNameFor } from "./hybridCombiner";
import { isClientDemo } from "./demoMode";
import { oprfPSI, psiLearnedSummary } from "./oprfPsi";
import type { MultiPsiResult } from "./clientMultiPsi";
import { asJson, checkPinned, HubFileRefused } from "./pinnedFetch";
import { loadQcnnHeldOut, rescoreQcnn, type QcnnRescore } from "./qcnnRescore";
import type { MlpWeights } from "./clientHybridMlp";
import type { LendingMarketReport } from "./lending/report";
import type { KeyholderSetup } from "./lending/privateCheck";
import type { TaskInit } from "./lending/tasks";

const r1 = (x: number) => Math.round(x * 10) / 10;
const r2 = (x: number) => Math.round(x * 100) / 100;
// Rates of agreement, to four places (Round 86, review PI-12): at two, anything from 1 - 40/8192 up
// printed as 1.00, which reads as perfect agreement where up to 40 rows disagreed.
const r4 = (x: number) => Math.round(x * 10000) / 10000;
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
const argmax = (a: number[]) => a.reduce((bi, v, i, arr) => (v > arr[bi] ? i : bi), 0);
/** Gap between the top two scores - what a top-1 decision actually had to lose. `Infinity` for a
 *  single-class row, so a degenerate model never reads as "decided by a tie". */
const decisionMargin = (row: number[]) => {
  const s = [...row].sort((x, y) => y - x);
  return s.length > 1 ? s[0] - s[1] : Infinity;
};
/**
 * What makes a trained QCNN classically simulable, in the artifact's own words (`circuit.simulability`,
 * Round 84), with the reference added where the artifact does not carry it. The two QCNN log lines
 * were literals - "two-qubit blocks on disjoint pairs" on telecom, "a tree of two-qubit blocks" on
 * finance and medical - and telecom's SU(4) circuit on every neighbouring pair is neither. Every
 * shipped artifact carries the field since finance's Round 84 retrain; the default below is for one
 * that does not, which this file reads as the tree.
 */
const qcnnSimulability = (w: QcnnWeights) => {
  const said = w.circuit?.simulability ?? "a tree of two-qubit blocks, so classically simulable by construction";
  return /Bermejo/.test(said) ? said : `${said} - Bermejo et al., PRX Quantum 7, 020304, 2026`;
};
const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));
// small prime utilities for the QSMS secure-sum field Z_p (config.prime; 0 = auto-select a prime above the
// largest total the inputs can reach, `qsmsField`)
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

/** The largest stand-in a party draws when a use case has no counts to pool: `floor(u() * 100)`. */
export const QSMS_STAND_IN_MAX = 99;

/**
 * The secure sum's field: the requested prime when one is set, else the smallest prime above the
 * largest total the inputs CAN reach (Round 84). Twin of `secure_summation.py::field_for_total`.
 *
 * It took `rawTotal + 1`, the ANSWER: the prime then pinned the total to the gap below it (23 to
 * 28 for GF(29) on genomics) and `qsms_matches_plain_total` was true by construction. The bound
 * is public - two effect alleles per person, or the stand-ins' range per party - and the sum is not.
 */
export function qsmsField(maxTotal: number, askedPrime: number): number {
  return askedPrime > 1 ? nextPrime(Math.floor(askedPrime)) : nextPrime(Math.floor(maxTotal) + 1);
}

// --------------------------------------------------------------------------- //
//  KEM security categories and the posture table's inputs
// --------------------------------------------------------------------------- //
const KEM_CATEGORY: Record<string, number> = {
  "ml-kem-512": 1, "ml-kem-768": 3, "ml-kem-1024": 5,
  "ml-kem-768+x25519": 3, "ml-kem-1024+x25519": 5,
};
export function kemCategory(kem?: string | null): number {
  return kem ? (KEM_CATEGORY[kem.toLowerCase()] ?? 3) : 3;
}

// Crypto-agility selectors threaded from the run config (KEM / signature) — so picking ml-kem-1024
// (NIST cat-5) vs ml-kem-768 (cat-3) genuinely runs the larger KEM AND moves the posture facts' category, and
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
// What the benchmark rows' security posture is built from: the run's chosen KEM (cat-3/cat-5) and whether a
// QKD key was mixed in (config.qkd === "pqc-only" gives none). Both are crypto-agility selections.
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
/** The configured qubit range (`benchmark.quantum_qubits_min`/`_max`), applied to any requested width. */
export const clampQubits = (n: number): number =>
  Math.min(benchCfg().quantum_qubits_max, Math.max(benchCfg().quantum_qubits_min, Math.round(n)));

const qubitsOf = (config: any) => {
  const n = Number(config?.qubits ?? benchCfg().quantum_qubits);
  if (!Number.isFinite(n)) return Number(benchCfg().quantum_qubits);
  return clampQubits(n);
};

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
  const abortAt = qkdCfg().pp_threshold;   // the QKD stage's own abort threshold
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
  const eveFrac = Number((sat ? config?.sat_eve : config?.qkd_eve) ?? 0);
  // The satellite's is the run's own channel, profile and all (Round 77), through the one builder
  // both satellite stages use.
  const runChannel: ChannelParams = sat ? satChannel(config ?? {}) : { ...FIBER,
    distanceKm: Number(config?.qkd_distance_km ?? FIBER.distanceKm),
    // No misalignment control is rendered on the satellite page, so the free-space channel's own
    // floor is the honest value there. Reading `qkd_misalignment` unconditionally would have
    // substituted the fibre page's - always, since that key is never absent.
    misalignment: Number(config?.qkd_misalignment ?? FIBER.misalignment),
    eveFraction: eveFrac,
    // The run's intensity pair (Round 84), as the stage ran it: the facts quote its signal intensity.
    ...fibreIntensities(config?.qkd_intensities) };
  // Which decoy bound this run's rate is, by the source it modelled: Micius sends a vacuum state.
  const boundName = sourceOf(runChannel).vacuum
    ? "vacuum + weak decoy-state bound (Ma, Qi, Zhao & Lo, Phys. Rev. A 72, 012326, 2005, Eqs. 34 and 37)"
    : "two-intensity decoy-state bound (Ma, Qi, Zhao & Lo, Phys. Rev. A 72, 012326, 2005, Eq. 41)";

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
    // EVERY DECOY KEY IS A BLOCK'S FINITE KEY: the satellite's since Round 79, the fibre's since
    // Round 80. `qkd_finite_key_bits` is present exactly when the run computed one, and the row then
    // names the block and the bound rather than an asymptotic fraction.
    const finite = metrics.qkd_finite_key_bits != null;
    const blockKind = String(metrics.qkd_block ?? "");
    const blockSecs = Number(metrics.qkd_block_s ?? 0);           // 4 significant figures, as emitted
    const blockPulses = Number(metrics.qkd_block_pulses ?? 0).toExponential(2);
    const blockWhat = sat
      ? blockKind === "seconds"
        ? `${Number(metrics.qkd_block_copies)} s held at this range`
        : `${blockKind === "passes" ? `${Number(metrics.qkd_block_copies)} passes pooled, each` : "one pass"} `
          + `peaking at ${Number(metrics.sat_pass_max_elevation_deg)} deg elevation, ${blockSecs} s of link time`
      : blockKind === "pa_bits"
        ? `a privacy-amplification block of ${Number(config?.qkd_block_pa_bits ?? qkdCfg().finite_key_block_pa_bits).toExponential(0)} `
          + `key-basis detections: ${blockPulses} pulses, ${blockSecs} s at the assumed clock`
        : blockKind === "seconds"
        ? `${blockSecs} s of link time at the assumed clock, ${blockPulses} pulses`
        : `this run's ${blockPulses} pulses, ${blockSecs} s at the assumed clock`;
    facts.push({
      label: "QKD key material",
      value: keyBits == null ? "not reported" : `${keyBits.toLocaleString("en-US")} secure bits`,
      kind: "simulated",
      source: isE91
        ? "E91 / CHSH run, device-dependent secure fraction on the model error rate"
        : finite
        ? `composable finite key over ${blockWhat}: ${metrics.qkd_finite_key_bound}, eps_sec `
          + `${Number(metrics.qkd_eps_sec).toExponential(0)}`
          // Which statistics (Round 82): a key is one analysis's number, and the page says which.
          + (metrics.qkd_fk_deviation ? `, ${DEVIATION_LABELS[metrics.qkd_fk_deviation as Deviation] ?? metrics.qkd_fk_deviation}` : "")
          + "; "
          + (sat ? "this run's own fraction of a second at the range measured the QBER and decides the abort"
                 : "this run's own pulses measured the QBER and decide the abort")
          // What the fibre's bound assumes that Rusca et al.'s does not (Round 84): `finiteKeySourceOf`
          // takes the bases at even odds, as Micius's passive beam splitter chose them, where Rusca et
          // al. optimise the bias; and the channel's receiver loss is the detectors' 3 dB, where their
          // Table I distances carry 6 (`QkdCharts.tsx`, RUSCA_TABLE_I_RECEIVER_DB).
          + (sat ? "" : "; bases at even odds (pZ = 0.5), where Rusca et al. optimise the bias, and the "
                      + `detectors' ${(-10 * Math.log10(Number(qkdCfg().detector_efficiency))).toFixed(1)} dB `
                      + "as the only receiver loss, where their Table I distances carry 6 dB")
        : `BB84 decoy-state run, ${sourceOf(runChannel).vacuum ? "vacuum + weak decoy" : "two-intensity"} secure fraction`,
      // THE BAND, NOT JUST THE ENDPOINT. This tested `keyBits === 0` while the behaviour it warns
      // about switches at 256 - the threshold `stages.py` and `:1233` below both use to decide
      // whether QKD is folded into the session at all. So a run producing 1..255 bits showed a
      // positive count, fired no warning, and silently fell back to PQC-only. Round 45's 4x
      // correction moved the entry to that band from 93 km to 63 km: three clicks of a control
      // whose step is 5 km.
      warn: keyBits === 0 && finite && metrics.qkd_block_units_for_key != null && Number(metrics.qkd_block_units_for_key) > 1
        ? `this block carries NO composable key: the statistics of ${blockWhat} are too few for the `
          + `bound. ${!sat ? `A block ${Number(metrics.qkd_block_units_for_key)} times larger`
            : blockKind === "seconds" ? `${Number(metrics.qkd_block_units_for_key)} s at this range`
            : `${Number(metrics.qkd_block_units_for_key)} such passes pooled`} `
          + "would carry one. The session rests on the PQC hybrid alone"
        : keyBits === 0
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
      // The detectors' dead time is in the rate since Round 76 wherever the link has one (the fibre
      // set's 100 ns; none is stated for the downlink), so the source says so.
      const deadS = Number(sat ? qkdCfg().free_space_detector_dead_time_s : qkdCfg().detector_dead_time_s);
      const deadNote = deadS > 0
        ? `; gains corrected for the detectors' ${Math.round(deadS * 1e9)} ns dead time (Rusca et al., `
          + "Appl. Phys. Lett. 112, 171104, 2018, Eq. B3)"
        : "";
      facts.push({ label: "Secret-key rate", value: `${Number(skr.toPrecision(4))} per pulse sent`, kind: "simulated",
        source: (isE91
          ? "device-dependent secure fraction 1 - h(e) - f h(e) on the MODEL error rate, not the "
            + "sampled QBER (Shor-Preskill form; entangled source: Ma, Fung & Lo, Phys. Rev. A 76, "
            + "012307, 2007); no decoy states. CHSH S is a consistency check, not a certificate"
          : boundName) + deadNote });
    }
    // BB84 ONLY. An E91 run showed this row too, warning that "the decoy states are what make this
    // link work at all" - on a protocol that uses none. The tagged fraction is a property of an
    // attenuated laser's multi-photon pulses; an entangled source's security analysis needs no decoy
    // states (Ma, Fung & Lo, Phys. Rev. A 76, 012307, 2007). Found on the deployed demo, 2026-10-24,
    // after a gate written the same day had asserted the row's presence on E91 runs.
    // Not out of view either (Round 77): with no link there is no detection to tag.
    if (!isE91 && inView(runChannel)) {
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
      // The run's own signal intensity: Micius's 0.8 on the satellite since Round 77.
      const mu = sourceOf(runChannel).mu;
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
    const dead = skr != null ? skr <= 0 : qber >= abortAt;
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
        ? `sampled from this run's pulses (the key rate uses the ${boundName}); post-processing aborts above ${abortAt}`
        : `no rate reported by this run; post-processing threshold ${abortAt}`,
      warn: dead
        ? (isE91 ? "the device-dependent E91 rate yields no secure key at this error rate"
                 : "the decoy-state bound yields no secure key on this channel")
        : undefined,
    });
  }
  return facts;
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
 *   1. the bundled real linear model shipped with the demo      -> MEASURED (plain JS arithmetic)
 *   2. placeholder weights                                      -> SIMULATED
 *
 * There used to be a third above these - the published ONNX model - and it was unreachable. See the
 * note in the body: it ran on an 8-element vector no published model accepts.
 */
export async function privateInference(
  usecase: string, features: number[], realModel?: { W: number[][]; b: number[] } | null,
): Promise<InferenceResult> {
  // The bundled model's own width, never fewer than 8 (Round 84): genomics' model is the full score
  // over every variant, and a fixed slice(0, 8) scored it on its first eight.
  const width = Math.max(8, realModel?.W?.[0]?.length ?? 0);
  const f = features.slice(0, width);
  while (f.length < width) f.push(0);

  // The published-ONNX tier used to be attempted HERE, on this 8-element vector, and it could never
  // succeed: no published model takes [1,8]. It takes [1,32], [1,1,28,28], [1,250,6], [1,19] or
  // [1,7]. So every run fetched a model - 11 MB and 1975 ms of the 2966 ms this function reported as
  // `client_compute_ms` - created a session, failed on input rank, and fell through to the line
  // below. A measured number that is two thirds download for an inference that cannot happen.
  //
  // The published model now runs in `runPublishedForStage`, on a row of its own corpus at its own
  // input shape, and reports its own latency. This function is what it always actually was: the
  // page's explainable path over its real features (8 on the tabular pages, every score variant on
  // genomics), which is what the SHAP attribution and the CKKS evaluation both consume.
  if (realModel?.W?.length) {
    const logits = realModel.W.map((row, j) =>
      row.reduce((a, wij, i) => a + wij * (f[i] ?? 0), 0) + (realModel.b?.[j] ?? 0));
    return {
      logits, prediction: argmax(logits), backend: "js-linear (bundled real model)",
      // Not measured-wasm: this loop is plain JS numbers. Claiming WASM here made every pipeline run
      // on every page render "MEASURED (local WASM)" without a wasm module ever being instantiated.
      provenance: "measured-js", modelKey: null,
      note: "bundled linear model, evaluated in plain JS arithmetic",
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
  // The four columns the finance model reads (registry.PAYSIM_MODEL_FEATURES; no balance-derived one).
  finance_aml: ["amount", "cross_border", "night_tx", "new_payee"],
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
const FHE_MS_PER_DOT = () => costPerOp().fhe_ms_per_dot;   // one CKKS encrypted linear dot per output neuron
const MPC_MS_PER_CMP = () => costPerOp().mpc_ms_per_cmp;   // one secret-sharing secure comparison (ReLU/argmax)
const tierOps = (dims: number[]) => dims.slice(1).reduce((a, x) => a + x, 0);  // output neurons = CKKS dots
// Secure comparisons: one per hidden ReLU output, and k - 1 for the argmax over the k logits. Counted
// as one per output neuron until Round 75, when the charge per comparison became the MPyC tier's
// measured 15,120 B and the extra comparison stopped being a rounding error.
const tierCmps = (dims: number[]) => Math.max(0, tierOps(dims) - 1);

/** The tier a representative row is costed at. The model-size selector left in Round 18 and nothing sends one; named so the read is not mistaken for a control. */
const BENCH_MODEL_SIZE = "medium";
function archDims(usecase: string | undefined, modelSize: string, workload: string, nIn?: number) {
  // GENOMICS PRICES THE SCORE IT RUNS (Round 84). `tier_architectures` gave it a "genomic risk MLP",
  // 8 -> 64 -> 2 and 706 parameters, that nothing trains and the registry calls not a neural network,
  // and `benchScale` multiplied the measured dot product by 33 to price it. What runs is one weighted
  // sum over the score's variants (class 0's logit is the constant 0) and a two-way argmax, at every
  // size, so the row prices that at scale 1. The width is the measured run's where one exists.
  if (usecase === "genomics") {
    const n = nIn && nIn > 0 ? Math.floor(nIn) : null;
    return {
      dims: [n ?? demoFeatureDim(), 2],
      label: `polygenic score: ${n ? `a ${n}-variant` : "one"} weighted sum under CKKS, then the high-risk `
        + "cut (not a neural network; every model size runs this)",
      params: n,
    };
  }
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
    const label = `image-classifier CNN (MedMNIST-scale, ${(mc.channels as number[]).length}-block) · ${modelSize}`;
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
  const studentParams = studentParamCount(student) ?? 0;
  if (studentParams > 0) {
    const inDim = demoFeatureDim();
    // Two layers whose parameter count matches the student's, so tierOps tracks a real model's
    // width instead of an invented one. The student is linear for three of the four use cases and
    // one small conv block for medical; either way its parameter count is what was measured.
    const width = Math.max(2, Math.round((studentParams - 2) / Math.max(1, inDim + 1)));
    const dims = [inDim, width, 2];
    const params = studentParams;
    // The label names the published student whose size this cost row takes. It is NOT always the
    // model CKKS evaluated on the page: finance's CKKS run scores the bundled four-feature PaySim model,
    // and the row's `fhe_model` metric says which ran.
    // Round 86: it read "CKKS-linear fraud student (33 params, sizing this row; ...)", which named the
    // AMLworld student beside timings that are the PaySim surrogate's. It now leads with what the row is.
    return { dims, label: `cost row sized by ${student?.label ?? "the encrypted student"} (${params} params); its CKKS timing is of the model named by fhe_model`, params };
  }
  const a = usecase ? tierArchitectures()[usecase] : undefined;
  const base = a ? (a.hidden[modelSize] ?? [16]) : (genericArchitecture()[modelSize] ?? [16]);
  const hidden = base.map((h) => Math.max(2, Math.round(h * mult)));
  const dims = [demoFeatureDim(), ...hidden, 2];
  const label = (a ? a.label : "MLP") + ` · ${modelSize}${wtag} (a cost model only: latency projected from the measured run, not the model the page evaluates)`;
  const params = dims.slice(0, -1).reduce((s, inp, i) => s + inp * dims[i + 1] + dims[i + 1], 0);
  return { dims, label, params };
}

// The demo's REAL in-browser run measures a small (n->2) model; this scale = (selected architecture's op count)
// / (that measured model's op count) lets both the measured classical row AND the quantum-sim row genuinely
// track lightweight/medium/large × linear/cnn/transformer, while keeping the real per-op measurement as the
// unit cost. lightweight+linear === the measured model -> scale 1.0 (reports the raw measurement).
export function benchScale(usecase: string | undefined, modelSize: string, workload: string): number {
  const ops = tierOps(archDims(usecase, modelSize, workload).dims);
  const baseOps = Math.max(1, tierOps(archDims(usecase, "lightweight", "linear").dims));
  return ops / baseOps;
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
            *  10% from the mismatched pair. */
           convOnlyMs: number;
           /** The STUDENT's final-logit error, where a full student ran. Distinct from
            *  `maxAbsError`, which is the convolution block's. */
           studentLogitMaxAbsError?: number;
           /** The student's DENSE layers, on the published-weights path. Reported so the whole
            *  569 ms is accounted for rather than leaving 488 ms unexplained beside a conv figure. */
           denseMs?: number; denseRotations?: number; denseRotateMs?: number;
           /** True when the PUBLISHED student's weights drove the computation. Always true since
            *  Round 84: without them no convolution runs (`RealHybridRun.studentNotRun`), where a
            *  stated kernel pattern used to stand in and set this false. */
           publishedWeights: boolean; totalRotations: number;
           studentParams?: number; studentPrediction?: number };
  /** Rotations and multiply-plain for the DENSE layer, so the two workloads compare like for like. */
  /** `rotateMs` is the measured time inside `rotateVector`, kept separate from `fheMs` because the
   *  page scales `fheMs` by a cost model and a scaled total cannot yield a per-rotation cost. */
  dense?: { rotations: number; multiplies: number; rotateMs: number };
}

/** Which half of the real in-browser hybrid failed, and why. Returned in place of `null` because the
 *  two halves fail for different reasons and the run must say which. Until Round 68 every failure,
 *  the garbled-circuit protocol's included, was reported as "node-seal did not load". An MPC failure
 *  can be the protocol ABORTING - a MAC or commitment check failing, which is what malicious security
 *  looks like from the honest party's side - and that must not read as a missing library. */
export interface HybridFailure {
  failed: "fhe" | "mpc";
  reason: string;
}

export interface RealHybridRun extends MeasuredHybrid {
  prediction: number;
  /** The index the same argmax gives in the clear on the same quantised logits (`clearOut` in
   *  clientMPC.ts): the self-test the caller reports as `mpc_matches_clear`. */
  mpcClearIndex: number;
  /** The argmax of the float logits the circuit was given (Round 86, review PI-4): `mpcClearIndex`
   *  shares the circuit's quantisation, so only this can show a near-tie that the quantisation decided. */
  mpcFloatIndex: number;
  maxAbsError: number;
  /** The decrypted logits and the plaintext reference the CKKS path was measured against.
   *  Carried so every use case can report the same encrypted-verification figures telecom got in
   *  Round 33 - a top-1 mismatch COUNT and bits of precision - instead of only a raw error magnitude
   *  whose scale a reader has no way to judge. */
  logits: number[];
  plainLogits: number[];
  /** Why the medical student's encrypted inference did not run, when it was asked for and did not:
   *  null otherwise. Since Round 84 no kernel pattern stands in for its weights. */
  studentNotRun?: string | null;
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
            /** Ciphertexts the rows filled (Round 84: every row is scored, so more than one where
             *  the corpus outgrows a ciphertext), and the encrypted work per ciphertext. */
            ciphertexts: number; msPerCiphertext: number | null;
            /** Smallest non-zero plaintext margin, and how many deviations can meet at a flip: 1 when
             *  only one class score is encrypted (an all-zero class row is bias-only and exact). */
            smallestMargin: number | null; flipBoundFactor: 1 | 2;
            /** Median |encrypted - plaintext| over the class scores that went through CKKS. */
            medianAbsError: number | null; noisyScores: number;
            /** "logit" for a two-class model: both axes carry class 1 minus class 0. */
            scoreKind: "logit" | "winning-score";
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
// REAL node-seal CKKS (clientFHE) for the linear layer + a REAL Polytune authenticated-garbling 2PC
// (clientMPC) for the non-linear argmax, all measured. Returns null only if the real crypto is genuinely
// unavailable here (e.g. a browser without WASM) — the caller then reports a clearly-labelled
// representative figure as an honest last resort (logged with the cause, not silently masked).
/** Fetch a published student's readable weights from its own pinned HF revision.
 *
 *  Returns null rather than throwing when the model publishes none: three of the four students are
 *  sklearn-linear and ship a different JSON shape. A null means the encrypted student does not run,
 *  and the run says so; no kernel pattern stands in for the weights (Round 84).
 */
/** The medical CNN student under MPC, as the page shows it. */
export interface MpcCnnReport {
  result: Omit<HybridCnnResult, never>; ranIn: string; imageIndex: number; imageCount: number; imageLabel: number; classes: Record<string, string> | null; explanation?: OcclusionMap;
}

/** The published student's held-out images, at the pin its weights come from. */
const _studentSampleCache = new Map<string, { samples: number[][]; labels: number[]; classes: Record<string, string> | null } | null>();
async function publishedStudentSample(usecase: string): Promise<{ samples: number[][]; labels: number[]; classes: Record<string, string> | null } | null> {
  if (_studentSampleCache.has(usecase)) return _studentSampleCache.get(usecase) ?? null;
  let out: { samples: number[][]; labels: number[]; classes: Record<string, string> | null } | null = null;
  try {
    const spec = studentModel(usecase);
    if (spec?.hf_repo_id && spec.hf_revision) {
      const res = await fetch(`https://huggingface.co/${spec.hf_repo_id}/resolve/${spec.hf_revision}/eval_sample.json`, { mode: "cors", signal: AbortSignal.timeout(30_000) });
      if (res.ok) {
        const bytes = await res.arrayBuffer();
        await checkPinned(bytes, spec.file_sha256?.["eval_sample.json"], `${spec.key} eval_sample.json`);
        const j = asJson<{ samples?: number[][]; labels?: number[]; classes?: Record<string, string> | null }>(bytes);
        if (Array.isArray(j?.samples) && Array.isArray(j?.labels)) out = { samples: j.samples, labels: j.labels, classes: j.classes ?? null };
      }
    }
  } catch (e) {
    if (e instanceof HubFileRefused) console.error(e.message);
    out = null;
  }
  _studentSampleCache.set(usecase, out);
  return out;
}

/** The PSI among three to five banks under MPC, as one run reports it (Round 86): the card reads this. */
export interface MultiPsiReport {
  result: MultiPsiResult;
  ranIn: string;
}

/** The accounts each bank brings, from a generator keyed on the run's seed and separate from the run's own, so that ticking this changes no other stage's draw. */
async function runMultiPartyPsi(stage: string, parties: number, seed: number, emit: (t: "log", s: string, m: string) => void,
                                metric: (s: string, k: string, v: number | string) => void, artifacts: Record<string, unknown>): Promise<void> {
  const cfg = multiPartyPsiCfg();
  if (!(parties >= 3 && parties <= cfg.max_parties)) throw new Error(`the PSI under MPC takes 3 to ${cfg.max_parties} banks, got ${parties}`);
  const u = rng(seed + 0x6d70);
  const name = () => Math.floor(u() * 1e9);
  const shared = Array.from({ length: cfg.shared }, () => `acct-${name()}`);
  const lists = Array.from({ length: parties }, (_, p) => [
    ...shared, ...Array.from({ length: cfg.set_size - cfg.shared }, () => `bank${p}-${name()}`),
  ]);
  const { multiPartyPsiOffThread } = await import("./clientMultiPsi");
  emit("log", stage, `PSI among ${parties} banks under MPC: ${cfg.set_size} accounts each, one garbled circuit under Polytune, in a Web Worker.`);
  try {
    const r = await multiPartyPsiOffThread(lists);
    artifacts.multi_party_psi = { result: r, ranIn: r.ranIn } satisfies MultiPsiReport;
    metric(stage, "mpsi_ran", 1);
    metric(stage, "mpsi_parties", r.parties);
    metric(stage, "mpsi_set_size", r.setSize);
    metric(stage, "mpsi_intersection", r.intersection);
    metric(stage, "mpsi_matches_clear", r.matchesClear ? 1 : 0);
    metric(stage, "mpsi_and_gates", r.andGates);
    metric(stage, "mpsi_mb", r2(r.bytes / 1e6));
    metric(stage, "mpsi_ms", r2(r.mpcMs));
    emit("log", stage, `PSI among ${r.parties} banks: ${r.intersection} accounts held by all of them, ${r.matchesClear ? "the same set as the intersection in the clear" : "DIFFERENT from the intersection in the clear"}; `
      + `${r.andGates} AND gates, ${r2(r.bytes / 1e6)} MB, ${r2(r.mpcMs / 1000)} s. The banks run in this tab (notional parties), and the protocol's oblivious transfers rest on a discrete-log assumption, so it is not post-quantum.`);
  } catch (e) {
    metric(stage, "mpsi_ran", 0);
    emit("log", stage, `The PSI among ${parties} banks did not finish: ${(e as Error).message}`);
  }
}

/** What the run reports for the trained QCNN with its input under CKKS: the panel reads this (Round 86). */
export interface QcnnCkksReport {
  usecase: string; qubits: number; result: QcnnCkksResult; classNames: string[] | null;
  /** Recorded at training time on the artifact's held-out split, as the weights file carries them; the comparison the literature asks of a QML model. */
  headline: { label: string; qcnn: number | null; classical: number | null; surrogate: number | null; surrogateFeatures: number | null; baseline: number | null };
  circuit: string;
}

/**
 * The trained QCNN re-scored in this tab over the rows of the trainer's own test split (Round 86). Both QCNN sites call this one
 * function and print `rescoreSentence`, so the two cannot report it differently. Until Round 86 they printed the weights file's figure
 * and said it was "not re-scored in this tab"; `scripts/data/build_qcnn_heldout.py` now ships the rows, written only after the
 * trainer's circuit reproduces every recorded metric from them. Null when the rows cannot be read: the sites then say so.
 */
async function rescoreQcnnInTab(stage: string, usecase: string, w: QcnnWeights,
                                metric: (s: string, k: string, v: number | string) => void,
                                artifacts: Record<string, unknown>): Promise<QcnnRescore | null> {
  try {
    const r = await rescoreTrainedQcnn(usecase, w);
    if (!r) return null;
    artifacts.qcnn_rescore = r;
    metric(stage, "qcnn_rescored_rows", r.rows);
    metric(stage, "qcnn_rescored_headline", Math.round(r.headline * 1e6) / 1e6);
    metric(stage, "qcnn_rescore_matches_record", r.matchesRecord ? 1 : 0);
    metric(stage, "qcnn_rescore_ms", r2(r.ms));
    return r;
  } catch {
    return null;
  }
}

/** The re-score the circuit card's button runs: the same rows and function as the pipeline's QCNN sites. Null when the rows cannot be read. */
export async function rescoreTrainedQcnn(usecase: string, w: QcnnWeights): Promise<QcnnRescore | null> {
  const held = await loadQcnnHeldOut(dataDirFor(usecase));
  return held ? rescoreQcnn(w, held) : null;
}

/** What both sites say about the circuit's held-out figure: re-scored here and equal to the record, re-scored and different, or not re-scored. */
function rescoreSentence(r: QcnnRescore | null, recorded: number | null | undefined): string {
  if (!r) return "recorded at training time and shipped in the weights file; the held-out rows could not be read, so this tab did not re-score it";
  return `re-scored in this tab by this circuit over the ${r.rows} rows of the trainer's test split: ${r.headline.toFixed(4)}, `
    + (r.matchesRecord ? "the value the weights file records" : `which DIFFERS from the ${recorded ?? "missing value"} the weights file records`);
}

/**
 * The trained QCNN's class with its input under CKKS, surfaced in place of the plaintext circuit's. Both QCNN sites call this one function, so the two cannot report it differently.
 * Returns false if it fails: the site then reports the plaintext class, and the log says that it is not encrypted.
 */
async function surfaceQcnnUnderCkks(stage: string, usecase: string, w: QcnnWeights, row: number[], emit: (t: "log", s: string, m: string) => void,
                                    metric: (s: string, k: string, v: number | string) => void, artifacts: Record<string, unknown>): Promise<boolean> {
  try {
    const { qcnnInferCkks } = await import("./clientQcnnCkks");
    const r = await qcnnInferCkks(w, row);
    const head = w.headline_metric, macro = head === "macro_f1", f1 = head === "minority_class_f1";
    artifacts.qcnn_ckks = {
      usecase, qubits: w.n_qubits, result: r, classNames: w.class_names ?? null, circuit: w.circuit?.name ?? "tree",
      headline: {
        label: macro ? "macro F1" : f1 ? "minority-class F1" : "accuracy",
        qcnn: (macro ? w.test_macro_f1 : f1 ? w.test_minority_f1 : w.test_accuracy) ?? null,
        classical: (macro ? w.classical_lr_macro_f1 : f1 ? w.classical_lr_minority_f1 : w.classical_lr_accuracy) ?? null,
        surrogate: (macro ? w.classical_surrogate?.test_macro_f1 : f1 ? w.classical_surrogate?.test_minority_f1 : w.classical_surrogate?.test_accuracy) ?? null,
        surrogateFeatures: w.classical_surrogate?.n_features ?? null,
        baseline: (macro ? w.macro_f1_floors?.uniform_random : f1 ? w.all_positive_f1 : w.majority_baseline) ?? null,
      },
    } satisfies QcnnCkksReport;
    artifacts.primary_inference = { backend: "qcnn-ckks", class: r.cls, qubits: w.n_qubits };
    metric(stage, "primary_inference_backend", "qcnn-ckks");
    metric(stage, "primary_inference_class", w.class_names?.[r.cls] ?? r.cls);
    metric(stage, "qcnn_ckks_ran", 1);
    metric(stage, "qcnn_ckks_terms", r.nTerms);
    metric(stage, "qcnn_ckks_ciphertexts_up", r.nCiphertexts);
    metric(stage, "qcnn_ckks_rotations", r.rotations);
    metric(stage, "qcnn_ckks_up_kb", r2(r.upBytes / 1024));
    metric(stage, "qcnn_ckks_down_kb", r2(r.downBytes / 1024));
    metric(stage, "qcnn_ckks_ms", r2(r.totalMs));
    metric(stage, "qcnn_ckks_max_error", r.maxError);
    metric(stage, "qcnn_ckks_matches_clear", r.matchesClear ? 1 : 0);
    emit("log", stage, `QCNN under CKKS: class ${w.class_names?.[r.cls] ?? r.cls}${r.matchesClear ? ", as the same circuit gives on the plaintext features" : `, and the same circuit gives ${r.clearClass} on the plaintext features`}. `
      + `The readout is a linear form over ${r.nTerms} products of sines and cosines of the encoded angles, so the features were encrypted as ${r.nCiphertexts} ciphertext${r.nCiphertexts > 1 ? "s" : ""} and the server took one inner product per readout `
      + `(${r.rotations} rotations, one multiplicative level). The decrypted readout differs from the clear circuit's by at most ${r.maxError.toExponential(1)}. ${r2(r.totalMs)} ms. Both parties are in this tab, so the separation is notional, and only that allows the comparison.`);
    return true;
  } catch (e) {
    metric(stage, "qcnn_ckks_ran", 0);
    emit("log", stage, `The QCNN under CKKS did not finish: ${(e as Error).message}. The class reported here is the plaintext circuit's and is not encrypted.`);
    return false;
  }
}

/** What a published MLP's run under MPC reports (Round 86): the card reads this. */
export interface MpcMlpReport {
  key: string; nFeatures: number; result: HybridCnnResult; ranIn: string; rowIndex: number; rowCount: number; rowLabel: number; expectedLogits: number[]; classes: Record<string, string>;
}

/** One held-out row through a published MLP under CKKS and garbled circuits, off the page's thread, reported whichever way it ends. */
async function runMlpUnderMpc(stage: string, seed: number, emit: (t: "log", s: string, m: string) => void,
                              metric: (s: string, k: string, v: number | string) => void, artifacts: Record<string, unknown>): Promise<void> {
  const key = "finance_aml.mlp_large";
  const spec = modelsFor("finance_aml").find((m) => m.key === key);
  let weights: MlpWeights | null = null;
  try {
    if (!spec?.hf_repo_id || !spec.hf_revision) throw new Error("it is not pinned to a revision");
    const res = await fetch(`https://huggingface.co/${spec.hf_repo_id}/resolve/${spec.hf_revision}/weights.json`, { mode: "cors", signal: AbortSignal.timeout(30_000) });
    if (!res.ok) throw new Error(`the Hub answered HTTP ${res.status}`);
    const bytes = await res.arrayBuffer();
    await checkPinned(bytes, spec.file_sha256?.["weights.json"], `${key} weights.json`);
    weights = asJson<MlpWeights>(bytes);
  } catch (e) {
    metric(stage, "mpc_mlp_ran", 0);
    emit("log", stage, `The MLP under MPC did not run: ${key} could not be used (${(e as Error).message}).`);
    return;
  }
  const n = weights.fixtures.rows.length, index = ((seed % n) + n) % n;
  const { hybridMlpInferenceOffThread } = await import("./clientHybridMlp");
  emit("log", stage, `The MLP under MPC: ${key}, held-out row ${index} of ${n}, in a Web Worker. CKKS for the linear layers, garbled circuits for the ReLUs and the argmax.`);
  try {
    const r = await hybridMlpInferenceOffThread(weights, weights.fixtures.rows[index]);
    const expected = weights.fixtures.logits[index];
    artifacts.mpc_mlp = { key, nFeatures: weights.n_features, result: r, ranIn: r.ranIn, rowIndex: index, rowCount: n, rowLabel: weights.fixtures.labels[index], expectedLogits: expected, classes: weights.classes } satisfies MpcMlpReport;
    metric(stage, "mpc_mlp_ran", 1);
    metric(stage, "mpc_mlp_model", key);
    metric(stage, "mpc_mlp_class", r.prediction);
    metric(stage, "mpc_mlp_matches_clear", r.matchesClear ? 1 : 0);
    metric(stage, "mpc_mlp_relu_count", r.reluCount);
    metric(stage, "mpc_mlp_and_gates", r.andGates);
    metric(stage, "mpc_mlp_gc_mb", r2(r.gcBytes / 1e6));
    metric(stage, "mpc_mlp_total_ms", r2(r.totalMs));
    metric(stage, "mpc_mlp_statistical_distance_log2", r2(r.statisticalDistanceLog2));
    metric(stage, "mpc_mlp_logit_max_abs_error", r.logitMaxAbsError);
    emit("log", stage, `MLP under MPC: class ${r.prediction} from the 2PC argmax${r.matchesClear ? ", as the same network gives in the clear" : `, and the same network gives ${r.clearPrediction} in the clear`}. `
      + `${r.reluCount} ReLUs in garbled circuits, ${r2(r.gcBytes / 1e6)} MB, ${r2(r.totalMs / 1000)} s in ${r.ranIn}. Both parties are in this tab, so the separation is notional, and only that allows the comparison.`);
  } catch (e) {
    metric(stage, "mpc_mlp_ran", 0);
    emit("log", stage, `The MLP under MPC did not finish: ${(e as Error).message}`);
  }
}

/** One held-out image through the student under CKKS and garbled circuits, off the page's thread, reported whichever way it ends. */
async function runStudentUnderMpc(stage: string, usecase: string, seed: number, emit: (t: "log", s: string, m: string) => void,
                                  metric: (s: string, k: string, v: number | string) => void, artifacts: Record<string, unknown>): Promise<void> {
  const [w, sample] = await Promise.all([publishedStudentWeights(usecase), publishedStudentSample(usecase)]);
  if (!w || !sample) {
    metric(stage, "mpc_cnn_ran", 0);
    emit("log", stage, "The CNN student under MPC did not run: its weights or its held-out images could not be fetched from the pinned Hugging Face revision.");
    return;
  }
  const n = sample.samples.length, index = ((seed % n) + n) % n;
  const size = Math.round(Math.sqrt(sample.samples[index].length));
  const image = Array.from({ length: size }, (_, y) => sample.samples[index].slice(y * size, (y + 1) * size));
  const { hybridStudentInferenceOffThread } = await import("./clientHybridCnn");
  emit("log", stage, `The CNN student under MPC: held-out image ${index} of ${n}, in a Web Worker. CKKS for the linear layers, garbled circuits for the ReLUs and the argmax, about 20 s.`);
  try {
    let lastQuarter = 0;
    const r = await hybridStudentInferenceOffThread(w, image, { onProgress: (p) => {
      if (p.phase !== "relu" || p.total < 100) return;
      const quarter = Math.floor((4 * p.done) / p.total);
      if (quarter > lastQuarter) { lastQuarter = quarter; emit("log", stage, `garbled circuits: ${Math.min(p.done, p.total)} of the convolution's ${p.total} ReLUs`); }
    } });
    const { occlusionMap } = await import("./cnnSaliency");
    artifacts.mpc_cnn = { result: r, ranIn: r.ranIn, imageIndex: index, imageCount: n, imageLabel: sample.labels[index], classes: sample.classes, explanation: occlusionMap(w, image) } satisfies MpcCnnReport;
    metric(stage, "mpc_cnn_ran", 1);
    metric(stage, "mpc_cnn_class", r.prediction);
    metric(stage, "mpc_cnn_matches_clear", r.matchesClear ? 1 : 0);
    metric(stage, "mpc_cnn_relu_count", r.reluCount);
    metric(stage, "mpc_cnn_and_gates", r.andGates);
    metric(stage, "mpc_cnn_gc_mb", r2(r.gcBytes / 1e6));
    metric(stage, "mpc_cnn_gc_ms", r2(r.gcMs));
    metric(stage, "mpc_cnn_ckks_ms", r2(r.ckksMs));
    metric(stage, "mpc_cnn_total_ms", r2(r.totalMs));
    metric(stage, "mpc_cnn_statistical_distance_log2", r2(r.statisticalDistanceLog2));
    metric(stage, "mpc_cnn_logit_max_abs_error", r.logitMaxAbsError);
    emit("log", stage, `CNN under MPC: class ${r.prediction} from the 2PC argmax${r.matchesClear ? ", as the same network gives in the clear" : `, and the same network gives ${r.clearPrediction} in the clear: treat this result as unverified`}. `
      + `${r.reluCount} ReLUs in garbled circuits, ${r.andGates} AND gates, ${r2(r.gcBytes / 1e6)} MB between the two notional parties, ${r2(r.totalMs / 1000)} s in ${r.ranIn}. `
      + `Statistical distance of everything masked, at most 2^${r2(r.statisticalDistanceLog2)}; the masks are bounded reals, so this is weaker than masking over the plaintext ring.`);
  } catch (e) {
    metric(stage, "mpc_cnn_ran", 0);
    emit("log", stage, `The CNN student under MPC failed and no figure from it is shown: ${(e as Error)?.message ?? e}`);
  }
}

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
        const bytes = await res.arrayBuffer();
        await checkPinned(bytes, spec.file_sha256?.[art.path], `${spec.key} ${art.path}`);
        const j = asJson<CnnStudentWeights & { kind?: string }>(bytes);
        // Only the structural CNN encoding is evaluable here; a linear student's JSON is a
        // different shape and must not be coerced into this one.
        if (j && j.kind === "cnn-1conv-2dense") out = j as CnnStudentWeights;
      }
    }
  } catch (e) {
    if (e instanceof HubFileRefused) console.error(e.message);
    out = null;                       // offline, blocked or refused: the student does not run, and says so
  }
  _studentWeightCache.set(usecase, out);
  return out;
}

export async function realClientHybrid(features: number[],
                                      convolve = false,
                                      usecase?: string,
                                      realModel?: { W: number[][]; b: number[] } | null,
                                      corpus?: number[][] | null,
): Promise<RealHybridRun | HybridFailure> {
  let half: HybridFailure["failed"] = "fhe";       // which half is running, for the failure report
  try {
    if (!_fheWarmed) {
      const tk = performance.now();
      await warmFHE();
      _fheKeygenMs = performance.now() - tk;
      _fheWarmed = true;
    }     // one-time CKKS keygen
    if (!_mpcWarmed) { half = "mpc"; await warmMPC(); _mpcWarmed = true; half = "fhe"; }  // one-time Polytune wasm load + init
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
    // The REAL garbled-circuit 2PC argmax below, measured, runs over logits the client has ALREADY
    // DECRYPTED. Both parties are in this tab and the key holder holds the cleartext logits, so the
    // protocol hides nothing here; it runs to measure the protocol, not to keep the logits from anyone.
    half = "mpc";
    const mpc = await secureArgmax(fhe.logits);
    half = "fhe";

    // The CNN workload now runs a REAL encrypted convolution instead of multiplying this dot
    // product's latency by a cost model. Shape comes from the trained student (`medical_cnn`),
    // not from `tier_architectures`: 14x14 in, 3x3 kernels, the student's own channel count.
    let conv: MeasuredHybrid["conv"];
    let studentNotRun: string | null = null;
    if (convolve) {
      const mc = medicalCnn();
      const size = Number(mc.img_size);
      const img = Array.from({ length: size }, (_, y) =>
        Array.from({ length: size }, (_, x) => features[(y * size + x) % features.length] ?? 0));
      // The PUBLISHED student's own weights when they can be fetched. Running the real network is
      // what turns this from a cost model of the right shape into the model's actual layer-by-layer
      // CKKS inference (decrypted between layers), which makes its encrypted-against-plaintext
      // agreement checkable in a browser; crypto_fidelity (student against teacher) stays offline.
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
        // NO WEIGHTS, NO STUDENT (Round 84). This branch ran the student's SHAPE with kernels
        // ((ky * 3 + kx + k) % 5 - 2) * 0.1 and zero biases, the conv layer only, while its log said
        // "none were invented" and the page promised the run would not substitute any: a fixed
        // kernel pattern IS invented weights. Now the encrypted student does not run without its
        // published weights, the run says so, and the CKKS figures are the bundled regression's.
        studentNotRun = "its published weights could not be fetched from the pinned Hugging Face "
          + "revision (offline, blocked, or not the student's JSON shape)";
      }
    }
    // The same model over a whole corpus, where the caller supplied one. `encryptedLinearBatch` packs
    // the rows at the reduction's own stride, and it is what turns "the encrypted number agreed on
    // this one row" into a figure a reader can check.
    //
    // EVERY ROW (Round 84). The default cap is one ciphertext, which at genomics' full width - every
    // score variant, a 512-slot block - holds 8 of its 30 individuals; `corpus.length` is a bound the
    // row count can never exceed, so the batch fills as many ciphertexts as the rows need and says so.
    let batch: RealHybridRun["batch"] = null;
    if (usingRealModel && corpus?.length) {
      const bt = await encryptedLinearBatch(W, b, corpus.map((r) => r.slice(0, n)),
                                            { maxCiphertexts: corpus.length });
      const encCls = bt.logits.map(argmax), plainCls = bt.plainLogits.map(argmax);
      // A top-1 mismatch is NOT automatically a precision failure, and reporting the count alone
      // says it is. Genomics' former sign test is the case that forced this: its model was the
      // published PGS weights as `W = [-w/2, +w/2]` with zero bias over 8 variants, so the two
      // logits were exact negatives and the margin was a partial score. A sample whose score sat at
      // zero was a genuine tie and flipped under any perturbation whatsoever - 3 of its 30 rows did,
      // at a deviation of 7.2e-10. Printing "3 of 30 mismatches" beside that leads a reader to
      // exactly the wrong conclusion about the encrypted tier.
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
      // model scores identically, which was 7 of 30 on genomics' sign test and is 0 of 30 on its full
      // score. That count
      // is what actually separates a meaningful agreement rate from a coin toss, and unlike a
      // tie/not-tie verdict it is a property of the corpus rather than a restatement of a theorem.
      const tieRows = bt.plainLogits.filter((row) => decisionMargin(row) === 0).length;
      const decidable = bt.rows - tieRows;
      // Flips among the rows the plaintext model actually decides. THIS is the number that would
      // indicate a real problem; the headline rate over all rows is the published convention and is
      // kept beside it.
      const flippedDecidable = flipped.filter((i) => decisionMargin(bt.plainLogits[i]) !== 0).length;
      let smallestMargin: number | null = null;
      for (const row of bt.plainLogits) {
        const mg = decisionMargin(row);
        if (mg > 0 && Number.isFinite(mg) && (smallestMargin == null || mg < smallestMargin)) smallestMargin = mg;
      }
      // The scores that went through CKKS: an all-zero class row is the bias alone, exact both ways.
      const noisy = W.map((row, j) => (row.some((v) => v !== 0) ? j : -1)).filter((j) => j >= 0);
      const devs: number[] = [];
      bt.logits.forEach((row, r) => noisy.forEach((j) => devs.push(Math.abs(row[j] - bt.plainLogits[r][j]))));
      devs.sort((x, y) => x - y);
      const medianAbsError = devs.length
        ? (devs.length % 2 ? devs[(devs.length - 1) / 2] : (devs[devs.length / 2 - 1] + devs[devs.length / 2]) / 2)
        : null;
      // A two-class model plots its decision logit, class 1 minus class 0, so every row sits somewhere
      // informative. The winning class's score put each row the full genomics score calls class 0 -
      // the bias-only zero row - at exactly (0, 0).
      const binary = W.length === 2;
      batch = {
        rows: bt.rows, rowsOffered: bt.rowsOffered, rowsPerCiphertext: bt.rowsPerCiphertext,
        fheMs: bt.fheMs, maxAbsError: bt.maxAbsError,
        flipMarginMax, tieRows, decidable, flippedDecidable,
        mismatches: flipped.length,
        ciphertexts: bt.ciphertexts, msPerCiphertext: bt.msPerCiphertext,
        smallestMargin, flipBoundFactor: noisy.length <= 1 ? 1 : 2,
        medianAbsError, noisyScores: devs.length, scoreKind: binary ? "logit" : "winning-score",
        points: bt.logits.map((row, i) => ({
          enc: binary ? row[1] - row[0] : row[encCls[i]],
          plain: binary ? bt.plainLogits[i][1] - bt.plainLogits[i][0] : bt.plainLogits[i][encCls[i]],
          agree: encCls[i] === plainCls[i],
        })),
      };
    }
    return {
      prediction: mpc.out[0], mpcClearIndex: mpc.clearOut[0], mpcFloatIndex: argmax(fhe.logits),
      fheMs: fhe.fheMs + (conv?.convMs ?? 0), mpcMs: mpc.mpcMs,
      commBytes: mpc.commBytes, ctBytes: fhe.ctBytes, featDim: n,
      maxAbsError: fhe.maxAbsError, plainLogits: fhe.plainLogits, logits: fhe.logits, conv,
      dense: { rotations: fhe.rotations, multiplies: fhe.multiplies, rotateMs: fhe.rotateMs },
      encryptedRealModel: usingRealModel, batch, studentNotRun,
    };
  } catch (e) {
    // Surface the cause (no-fallback policy: a fixable defect must not hide behind the representative
    // path), and say WHICH half failed, so the run reports a protocol abort as one.
    console.warn(`[realClientHybrid] the real in-browser ${half === "mpc" ? "garbled-circuit MPC" : "CKKS FHE"} failed:`, e);
    return { failed: half, reason: e instanceof Error ? e.message : String(e) };
  }
}

// MPC-FHE row — uses REAL measured client-side FHE/MPC when ``measured`` is supplied, else a representative
// cost model. Read by assembleBenchmark (the standalone clientBenchmark that also used it was removed in Round 86).
function mpcFheRow(usecase: string | undefined, workload: string, modelSize: string, measured?: MeasuredHybrid | null,
  // `qber` belongs here even though this row is the CLASSICAL path. Both rows run under ONE transport
  // configuration, so both must be scored under it. This parameter used to omit it, so this row's
  // eavesdropper term was structurally zero while its counterpart threaded the value through - and
  // the 68-vs-96 gap on screen was that asymmetry, not a property of the cryptography.
  sec?: { kem?: string; qkdBits?: number | null; qber?: number | null }) {
  const { dims, label, params } = archDims(usecase, modelSize, workload, measured?.featDim);
  const cost = costPerOp();
  // ctpt is nullable: the measured branch withholds it when the serialized ciphertext size could
  // not be read, rather than reporting an impossible 0 or borrowing the cost model's literature
  // figure into a row labelled measured.
  let fhe: number, mpc: number, comm: number, backend: string, ctpt: number | null,
    projected: boolean, device: string;
  if (measured) {
    // The real run measures a small (n->2) model; scale that MEASURED per-op cost by the SELECTED architecture's
    // op count so the latency genuinely tracks lightweight/medium/large × linear/cnn/transformer (the unit cost
    // is real node-seal CKKS / Polytune; only the op count is the chosen architecture's).
    // When a real encrypted convolution ran, its latency is MEASURED and nothing is scaled: the
    // whole point of running one is to stop multiplying a dot product by a cost model. The scaling
    // path below survives only for the linear workload, where the measured model really is an
    // (n -> 2) layer standing in for a wider one.
    const scale = measured.conv ? 1 : benchScale(usecase, modelSize, workload);
    fhe = measured.fheMs * scale; mpc = measured.mpcMs * scale; comm = measured.commBytes * scale;
    // Genomics runs the same score at every size, so nothing of its "large" is projected.
    projected = !measured.conv && modelSize === "large" && usecase !== "genomics";
    backend = measured.conv
      ? `node-seal CKKS + Polytune 2PC (measured: real ${measured.conv.channels}-channel encrypted `
        + `convolution, ${measured.conv.rotations} rotations over `
        + `${measured.conv.patchesPerCiphertext} packed patches)`
      : "node-seal CKKS + Polytune 2PC (measured per-op, scaled to the selected architecture)";
    // null propagates. Previously a failed size read became 0 and this printed "ct/pt 0" beside
    // rows carrying real ratios - an impossible value in a MEASURED column. The cost-model
    // default is NOT substituted here either: this branch is the measured one, and quietly
    // borrowing a literature figure would relabel an estimate as a measurement.
    ctpt = measured.ctBytes == null || measured.featDim <= 0
      ? null
      : r2(measured.ctBytes / (measured.featDim * 8));
    device = "cpu";                                   // node-seal CKKS is WASM-CPU (no commercial-OK WebGPU FHE found, searched 2026-11-03)
  } else {
    // The COST-MODEL branch: nothing here was measured. It used to name whatever execution provider
    // `onnxBackend()` happened to report, which was never this row's - and is now flatly wrong,
    // because the only session on the page belongs to the published model running on a different
    // corpus. A modelled row does not get to borrow a measured row's backend.
    const ops = tierOps(dims), cmps = tierCmps(dims);
    fhe = ops * cost.fhe_ms_per_dot; mpc = cmps * cost.mpc_ms_per_cmp; comm = cmps * cost.mpc_comm_bytes_per_cmp;
    backend = "ckks + mpyc-shamir (cost model, not measured)";
    ctpt = cost.demo_ct_pt_ratio; projected = modelSize === "large" && usecase !== "genomics";
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
    // Round 86 (fact-check C4): on the medical page `fhe` is TWO models, the encrypted CNN student plus the bundled
    // Wisconsin regression, and the 2PC is the regression's argmax alone. The student's own share, so the panel can
    // split the figure instead of printing 'total ... / inference' over two inferences.
    fhe_student_ms: measured?.conv ? r2(measured.conv.convMs) : null,
    t_gate_count: null, qubits: null,
    // "garbled-circuit 2PC", not "secret-sharing MPC". That label is right for the Python row (MPyC
    // Shamir across hosts) and was copied to this one, where the non-linear layer is a Polytune
    // authenticated-garbling 2PC whose INPUTS are additively shared - between notional parties in
    // one tab (clientMPC.ts). The benchmark table showed the Python mechanism on the browser row.
    // "not post-quantum" since Round 86: the RLWE half is (N = 8192 with a 200-bit modulus is inside
    // the HE Standard's 128-bit quantum bound), but Polytune's OT extension (KOS) runs over
    // Chou-Orlandi base OTs on ristretto255 (`ot_core/mod.rs` at the pinned commit), which rest on
    // discrete log. The row named the 2PC with no such note while the PSI row carried one.
    security_model: "computational (RLWE) + garbled-circuit 2PC (discrete-log base OT, not post-quantum)",
    device,
  };
}

// QSMPC-FHE row - a state-vector simulation of an UNTRAINED QOTP + Clifford+T comparison circuit
// (quantumSim.qotpQcnnInfer) on random input. It is not a QCNN and not an inference: there are no
// trained weights, and the pad is applied and never undone, so the output follows the key, not the
// input. `latency_ms` is the REAL-measured wall-clock of the simulation and `t_gate_count`/`qubits`
// are COUNTED from the actual circuit. A state-vector SIMULATION, not quantum hardware.
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
  const nq = clampQubits(nQubits);
  qotpQcnnInfer(Array.from({ length: nq }, () => u() * 2 - 1), 0, nq);   // warm-up (drop the JIT spike)
  let last = qotpQcnnInfer(Array.from({ length: nq }, () => u() * 2 - 1), 1, nq);
  let total = 0;
  for (let i = 0; i < samples; i++) {
    last = qotpQcnnInfer(Array.from({ length: nq }, () => u() * 2 - 1), i + 2, nq);
    total += last.latencyMs;
  }
  // measured per-run state-vector sim (an nq-qubit circuit, 8 by default), scaled to the selected
  // architecture so the quantum row tracks lightweight/medium/large × linear/cnn/transformer like the classical row.
  const latency = r2((total / samples) * scale);
  return {
    path: "QSMPC-FHE", backend: "qotp + clifford+t (state-vector sim, not quantum HW)", workload, model_size: modelSize,
    // Names what this row IS, because the model column read "—" beside two rows that name real
    // models, which invites the reader to assume the blank is an oversight rather than the point.
    // There are no learned parameters here: qotpQcnnInfer applies a fixed H/X/Z/T/RY structure whose
    // angles come from its input, and the input is random.
    model_arch: `${nq}-qubit QOTP + Clifford+T comparison circuit (untrained, predicts nothing)`,
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
    // ONE constant. The row label and the retired security score's input were two literals, and the scorer
    // parsed the label for "information-theoretic", so rewording one copy moved a score with every gate green;
    // the score itself was removed in Round 86, and the advantage function still parses the label.
    security_model: QSMPC_SECURITY_MODEL,
    device: "cpu-sim", simulated: true,
  };
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
    input_shape: m.published_input_shape ?? null,
    revision: m.published_model_revision ?? null,
  };
}

export function assembleBenchmark(
  usecase: string, config: any,
  opts: { metrics?: any; artifacts?: any; stageMs?: Record<string, number> },
) {
  const modelSize = BENCH_MODEL_SIZE;
  const workload = config?.workload ?? "linear";
  // Pass the run's metrics: this is the call site whose score the benchmark panel renders, and
  // without them the QBER and secure-key-bits terms fall back to constants.
  const sec = secFromConfig(config, opts?.metrics);
  const u = rng(7);
  const q = qsmpcFheRow(workload, modelSize, u, benchCfg().quantum_samples, benchScale(usecase, modelSize, workload), qubitsOf(config), sec);
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
//  In-browser E2E pipeline with Pause / Resume / Step / Stop
// --------------------------------------------------------------------------- //
// ---- Federated learning ("Quantum Federated Learning", QFL, by name) - client-side mirror (demo) --
// Nothing quantum runs in it: see clientQFL.ts.

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
  // No quantum channel and no model: the market is simulated and the check is a margin's sign (Round 86).
  defi_lending: ["pqc_handshake", "lending_market", "private_liquidation", "result", "sign_result"],
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
export async function loadQcnnWeights(usecase: string, qubits?: number): Promise<QcnnWeights | null> {
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
  // Federated learning (QFL by name) runs on the finance and medical pages: usePipelineRun sets `qfl` for those
  // two, and this injects a federated_training stage right before the inference/compute step on finance,
  // whose flow lacks one (medical_fl already has the stage in its flow).
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
  /** The private-inference figure for a page with held-out rows (Round 84, PI7-PI9).
   *
   * A PUBLISHED model on rows it was never fitted on, every row: finance's AMLworld student on its
   * 8,192 test-window rows (64 ciphertexts), medical's WDBC regression on its 171 held-out rows,
   * telecom's student on its own 600-row held-out quarter. The figure used to call finance's and
   * medical's FITTING rows held-out, on finance for a PaySim surrogate the page does not name. */
  async function emitHeldOutAgreement(stage: string, held: HeldOutCorpus) {
    const ho = await encryptedHeldOutAgreement(held);
    const bt = ho.batch;
    metric(stage, "encrypted_model", held.modelKey ?? held.modelLabel);
    metric(stage, "encrypted_rows_held_out", true);
    metric(stage, "encrypted_corpus", held.corpus);
    metric(stage, "encrypted_rows_scored", bt.rows);
    metric(stage, "encrypted_rows_available", bt.rowsOffered);
    metric(stage, "encrypted_rows_dropped_by_cap", bt.rowsOffered - bt.rows);
    metric(stage, "encrypted_rows_per_ciphertext", bt.rowsPerCiphertext);
    metric(stage, "encrypted_ciphertexts", bt.ciphertexts);
    metric(stage, "encrypted_batch_ms", r2(bt.fheMs));
    metric(stage, "encrypted_ms_per_ciphertext", bt.msPerCiphertext != null ? r2(bt.msPerCiphertext) : null);
    metric(stage, "fhe_batch_rotations", bt.rotations);
    // ITS OWN KEY, not `fhe_ms_per_rotation`. On finance and medical the single-row dense layer
    // below reports that key from its own timed rotations, and the two measurements differ by a
    // few tenths of a millisecond, so sharing one name tripped `metric_overwritten` on every run.
    // Telecom, which has no dense figure, also reports this one under the shared key at its call.
    if (bt.rotations > 0) {
      metric(stage, "fhe_batch_ms_per_rotation", Math.round((bt.rotateMs / bt.rotations) * 1000) / 1000);
    }
    metric(stage, "encrypted_batch_max_abs_error", bt.maxAbsError);
    metric(stage, "encrypted_batch_precision_bits", ho.precisionBits);
    metric(stage, "encrypted_batch_median_abs_error", ho.medianAbsError);
    metric(stage, "encrypted_batch_median_precision_bits", ho.medianPrecisionBits);
    metric(stage, "encrypted_smallest_plaintext_margin", ho.smallestMargin);
    metric(stage, "encrypted_flip_possible", !ho.flipImpossible);
    metric(stage, "encrypted_top1_mismatches_batch", ho.mismatches);
    // The scores CKKS evaluated: a one-logit student's zero class-0 row is its bias, never encrypted, so
    // rows x classes double-counted finance (Round 86).
    metric(stage, "encrypted_scores_compared_batch", ho.plot.noisyScores ?? bt.rows * held.model.W.length);
    // The rate and its three companions, together and unconditionally (see the parity test).
    metric(stage, "encrypted_agreement", r4(1 - ho.mismatches / Math.max(1, bt.rows)));
    metric(stage, "encrypted_agreement_rows", bt.rows);
    metric(stage, "encrypted_undecidable_rows", ho.tieRows);
    metric(stage, "encrypted_decidable_rows", ho.decidable);
    metric(stage, "encrypted_agreement_over_decidable",
      ho.decidable > 0 ? r4(1 - ho.flippedDecidable / ho.decidable) : null);
    if (ho.mismatches > 0) metric(stage, "encrypted_flip_margin_max", ho.flipMarginMax);
    if (ho.encryptedAccuracy != null) metric(stage, "encrypted_accuracy_vs_labels", ho.encryptedAccuracy);
    // Its baseline, beside it: what always calling the most common label scores on the same rows.
    if (ho.majorityBaseline != null) metric(stage, "encrypted_accuracy_majority_baseline", r4(ho.majorityBaseline));
    artifacts.encrypted_agreement_plot = ho.plot;
    emit("log", stage, `encrypted tier over ${bt.rows} HELD-OUT rows of ${held.corpus}, none used to `
      + `fit ${held.modelKey ?? held.modelLabel}: ${bt.ciphertexts} ciphertext(s) of up to `
      + `${bt.rowsPerCiphertext} rows, ${r2(bt.fheMs)} ms`
      + (bt.msPerCiphertext != null ? ` (${r2(bt.msPerCiphertext)} ms each)` : "")
      + `. Top-1 mismatches vs plaintext: ${ho.mismatches} of ${bt.rows}. Max abs error `
      + `${bt.maxAbsError.toExponential(3)}`
      + (ho.precisionBits !== null ? ` (about ${ho.precisionBits} bits, MAX not mean)` : "")
      + (ho.medianAbsError != null ? `, median ${ho.medianAbsError.toExponential(3)}` : "")
      + `. ${ho.tieRows} of the ${bt.rows} rows are scored identically across classes by the `
      + `plaintext model itself; over the ${ho.decidable} it does decide, agreement is `
      + (ho.decidable > 0 ? r2(1 - ho.flippedDecidable / ho.decidable) : "not computable")
      + (ho.smallestMargin != null
        ? `. Smallest plaintext margin ${ho.smallestMargin.toExponential(3)}: `
          + (ho.flipImpossible ? "no decided row could have flipped" : "a flip was possible on the closest rows")
        : "")
      + (ho.mismatches > 0
        ? `. Widest plaintext margin among the ${ho.mismatches} flipped row(s): `
          + `${ho.flipMarginMax.toExponential(3)}` : "")
      + `. Rows: ${held.licence.name}${held.licence.url ? ` (${held.licence.url})` : ""}.`);
    return ho;
  }

  /** A published student over its whole held-out split under real CKKS, on a page with no private-inference stage. */
  async function scoreHeldOutStudent(stage: string, usecase: string, payload: string) {
    if (!_fheWarmed) {
      const tk = performance.now();
      await warmFHE();
      _fheKeygenMs = performance.now() - tk;
      _fheWarmed = true;
    }
    const held = await heldOutCorpus(usecase);
    if (!held) {
      emit("log", stage, `encrypted tier: the student's held-out rows (${payload}) could not be read, so no agreement figure is drawn`);
      return;
    }
    const bt = (await emitHeldOutAgreement(stage, held)).batch;
    // The per-rotation cost under the key every encrypted page reports (PF1): these pages' only timed rotations
    // are the batch's.
    if (bt.rotations > 0) {
      metric(stage, "fhe_ms_per_rotation", Math.round((bt.rotateMs / bt.rotations) * 1000) / 1000);
    }
  }

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
      // `!(x < 1)`, not `x >= 1` (Round 86, review PI-3): NaN fails every comparison, so `>= 1` let it
      // through as a pass.
      const precisionFailed = !(maxAbs < 1);

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
      // (`scoreHeldOutStudent`, below, which the satellite page shares.)
      //
      // `encrypted_top1_mismatches` and `encrypted_scores_compared` used to be emitted here from the
      // single row: the mismatch count could only be 0 or 1, and "scores compared" counted CLASSES.
      // A single agreement is not evidence that encrypted inference reproduces plaintext. The batch
      // packs the rows at the reduction's own stride, so one multiply-plain and log2(width)
      // rotations per class serve every row of a ciphertext.
      // THE STUDENT'S OWN HELD-OUT ROWS (Round 84), not the rows the tab's JS mirror of the
      // generator drew: the 600 of generate(n_per_class=300, seed=42) the trainer held out, which the
      // student was evaluated on and never fitted on. Every row is scored (two ciphertexts at width
      // 8), so the 88-row cap disclosure no longer applies here.
      await scoreHeldOutStudent(stage, "telecom_ids", "data/telecom/student_heldout.json");
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
      const fileSha256 = onnxStatus().sha256;
      if (fileSha256) metric(stage, "published_model_sha256", fileSha256);
      // Says which of the two facts applies when they differ: WebGPU selected, no usable adapter.
      const fellBack = onnxBackendChoice === "webgpu" && p.backend !== "onnx-webgpu"
        ? " (WebGPU was selected; this browser has no usable adapter)" : "";
      emit("log", stage,
        `published model ${p.modelKey} ran here on ${p.backend}${fellBack}: row ${p.rowIndex} of its `
        + `published evaluation sample (${p.split ?? "split not declared"}), input `
        + `${p.inputShape.join("x")}, ${r2(p.latencyMs)} ms. ${outcome}`
        + `. Weights from ${p.repoId} at ${p.revision.slice(0, 12)}${fileSha256 ? `, SHA-256 ${fileSha256.slice(0, 12)}... as recorded for that revision` : ""}, ${p.provenance}`);
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
      // Two different facts, so two values: "none" when no model is registered for this use case,
      // "withheld" when one is registered and held back until a republish re-pins it. One value,
      // "unavailable", used to cover both, and it read as "exists but unreachable" for the first.
      if (pub.withheld.length) {
        metric(stage, "published_model_status", "withheld");
        emit("log", stage, `published model ${pub.withheld.join(", ")} withheld: its Hub pin is stale `
          + `(retrained since it was last published), so it is not run until a republish re-pins it`);
      } else {
        metric(stage, "published_model_status", "none");
        emit("log", stage, `no published model for this use case: ${pub.reason}`);
      }
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
            // Four places: at two, 0.9976 printed as 1.
            metric(stage, "verified_score", r4(v.score));
            metric(stage, "verified_expected", r4(v.expected));
            metric(stage, "verified_n", v.n);
            metric(stage, "verification", v.matchesExpected ? "PASS" : "FAIL");
            // Carried as an artifact so the page can DRAW what was measured. A single number is the
            // right thing to gate on and the wrong thing to reason from: "minority-class F1 0.51"
            // does not say whether the classes separate or the threshold sat somewhere lucky.
            artifacts.verification = { modelKey: key, ...v };
          }
          const delta = Math.abs(v.score - v.expected);
          const tag = isSelected ? "" : " (not the selected model)";
          // The split is the registry's declaration for this model, not a word the log supplies:
          // "held-out" was printed here for every model while two samples were drawn from the
          // whole corpus, training rows included (Round 71).
          const where = `${v.n} rows of its published evaluation sample (${v.split ?? "split not declared"})`
            + (v.decisionThreshold != null
              ? `, classified at its decision threshold ${v.decisionThreshold.toFixed(4)}` : "");
          emit("log", stage, v.matchesExpected
            ? `verification PASS ${key}${tag}: re-scored all ${where} in this `
              + `tab and got ${v.metricName} ${r2(v.score)} against the ${r2(v.expected)} recorded with `
              + `that sample (|delta| ${delta.toExponential(2)}): the published graph's own figure, `
              + `reproduced here rather than asserted. A model card's headline computed on another form `
              + `of the model (for example before ONNX conversion) can differ from it.`
            : `verification FAIL ${key}${tag}: re-scoring ${where} gave ${v.metricName} `
              + `${r2(v.score)} but the recorded figure is ${r2(v.expected)} (|delta| `
              + `${delta.toExponential(2)}, tolerance 0.05). Treat every number on this page for `
              + `${key} as unverified until this is explained.`);
          // The forecaster student (Round 86): what encryption did to its forecast and to the events it gives.
          const sd = v.studentDetail;
          if (sd?.ckks) {
            const c = sd.ckks, e = c.encrypted.anomalies;
            emit("log", stage, `${key}: its forecast under CKKS, all ${c.rows} rows in ${c.ciphertexts} `
              + `ciphertexts (${r2(c.fheMs / 1000)} s${sd.ranIn === "a Web Worker" ? ", in a Web Worker beside the page" : ""}): `
              + `max |decrypted - plaintext| ${c.maxAbsError.toExponential(2)}; `
              + `the decrypted forecast, thresholded in this tab, changes ${c.flagsChanged} of ${sd.plain.flags.length} `
              + `flags and scores ${r2(e.f05)} (tp ${e.tp} fp ${e.fp} fn ${e.fn}) against ${r2(sd.plain.anomalies.f05)} `
              + `from the plaintext forecast, whose flags match the published digest: ${sd.flagsDigestMatches ? "yes" : "NO"}.`);
          }
          // A model that cannot beat a trivial predictor is worth saying out loud even when it
          // reproduces its own figure exactly - reproducing a bad number is still reproducing it.
          if (!v.beatsBaseline && v.baseline != null) {
            // A forecaster's mean over a few hundred windows can be carried by two large misses (Round 86), so the
            // robust reading goes on the same line rather than letting the mean alone stand for the model.
            const fd = v.forecastDetail;
            emit("log", stage, `note: this model does not beat the trivial baseline on that sample `
              + `(${r2(v.score)} against ${r2(v.baseline)}). Reproducing a figure and being useful `
              + `are separate questions.`
              + (fd ? ` Beyond the mean: its median absolute error is ${r2(fd.medianAe)} against `
                + `${r2(fd.persistenceMedianAe)} for no change, and its own mean error is the lower on `
                + `${fd.windowsBetter} of ${fd.windows} windows, so a few large misses carry the mean.` : ""));
          }
        } else if (v && v.error) {
          if (isSelected) metric(stage, "verification", "ERROR");
          emit("log", stage, `verification could not run for ${key}: ${v.error}`);
        } else {
          // Distinct from a failure: some models publish no evaluation sample, and saying nothing
          // would let "not checked" read as "checked and fine".
          if (isSelected) metric(stage, "verification", "no-sample");
          emit("log", stage, `no published evaluation sample for ${key}`
            + (v?.noSampleReason ? `, by design: ${v.noSampleReason}` : "")
            + `, so its recorded metric is NOT reproduced in this tab.`);
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
        // The credit, and where the licence link and the changes are (RD23, Round 86): an attribution or share-alike licence
        // binds wherever the rows are served, and this line is what the page prints about them.
        if (ds) emit("log", stage, `dataset (bundled demo sample): ${ds.provenance}`
          + (ds.attribution ? ` Credit: ${ds.attribution}` : "")
          + (ds.file ? ` Licence link and changes made: the \`licence_url\` and \`modifications\` fields of ${ds.file}.` : ""));
        else emit("log", stage, "synthetic dataset prepared (client-side; no real sample is bundled for this use case's pipeline)");
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

        // Classical half: X25519 ECDH, then HKDF-SHA256 over both secrets with the transcript in
        // its info. A KDF rather than a bare XOR, so neither half can be cancelled out by an
        // attacker who controls the other. (This said SHA3-256 until Round 75; nothing here uses it.)
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
          + `ephemeral public key, combined with HKDF-SHA256(pq_ss || x25519_ss), info="${new TextDecoder().decode(hybridCombinerInfo(kemLabel, 32)).replace(/\|$/, "")}" then transcript c1||c2||ek1||ek2: a NIST SP 800-227 s4.6.2 two-step combiner over the s4.6.3 transcript — byte-identical to the backend, asserted by check:combiner`);
        // Quantum entropy source (simulated state-vector measurement, not hardware) — measured health + throughput.
        const qr = generateQuantumBits(Number(config.qrng_bits ?? qrngCfg().bits),
          { source: "sim", postProcessing: String(config.qrng_post ?? qrngCfg().post_processing),
            epsilonLog2: Number(config.qrng_eps_log2 ?? qrngCfg().security_epsilon_log2), seed });
        metric(stage, "qrng_health_passed", qr.healthPassed ? 1 : 0);
        metric(stage, "qrng_throughput_kbps", r1(qr.throughputBitsPerSec / 1000));
        metric(stage, "qrng_min_entropy_per_bit", r2(qr.minEntropyPerBit));
        // The ASSESSED entropy the cutoffs come from, beside the estimate of this run's stream: two
        // different numbers since Round 75, when the cutoffs stopped being computed from the stream
        // they test (SP 800-90B 4.4 fixes H for the source).
        metric(stage, "qrng_assessed_min_entropy_per_bit", r2(qr.assessedMinEntropyPerBit));
        metric(stage, "qrng_bits_delivered", qr.nBits);
        metric(stage, "qrng_extraction_rate", r2(qr.extractionRate));
        metric(stage, "qrng_raw_bits", qr.rawBits);
        metric(stage, "qrng_extractor_yield", r2(qr.extractorYield));
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
        // "Simulated" is generous: `measurePlus` draws Bernoulli(0.5) from the run-seeded mulberry32
        // stream, and no stage downstream reads these bits; only the health and throughput metrics use
        // them. SP 800-22 here is 3 of that suite's 15 tests (monobit, runs, block frequency).
        // Round 86: the counts say which stream each figure is computed on (the health tests and the estimate
        // read the raw draws, not the delivered bits), and the closing sentence says what the estimate is not.
        emit("log", stage, `QRNG (simulated: Bernoulli(0.5) draws from the run-seeded mulberry32 generator, standing in for |+> measurements; no key uses these bits): ${qr.nBits} bits delivered from ${qr.rawBits} raw draws (the extractor's yield ${r2(qr.extractorYield)}); health on the raw draws `
          + `${qr.healthConclusive ? (qr.healthPassed ? "PASS" : "FAIL") : "INCONCLUSIVE"} per SP 800-90B 4.4 `
          + `(repetition-count ${qr.repetitionCountLongest}/${qr.repetitionCountCutoff}, `
          + `adaptive-proportion ${qr.adaptiveProportionMax}/${qr.adaptiveProportionCutoff}, cutoffs at the `
          + `assessed ${qr.assessedMinEntropyPerBit} bit/sample; the raw draws' most-common-value estimate `
          + `${r2(qr.minEntropyPerBit)}${qr.estimateBelowAssessment ? ", BELOW the assessment" : ""}); `
          + `SP 800-22 subset (monobit, runs and block frequency: 3 of its 15 tests) reported, not gating: ${qr.sp80022Passed ? "pass" : "fail"} (monobit p=${qr.monobitP.toFixed(3)}), `
          + `${r1(qr.throughputBitsPerSec / 1000)} kbit/s, the speed of the JS generator standing in for the source. `
          + `An indication, not an SP 800-90B assessment: that needs at least 1,000,000 raw samples (section 3.1.1), `
          + `and the most-common-value estimate stands alone only for an IID source that has passed the section 5 `
          + `tests, which do not run here.`);
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
          eveFraction: Number(config.qkd_eve ?? qkdCfg().eve_intercept_fraction),
          // The run's clock, the same value the bit/s below is computed from: with a dead time the
          // per-pulse rate depends on it (Round 76), so the two cannot come from different places.
          sourceRateHz: sourceRateHz(config?.qkd_source_rate_mhz, Number(qkdCfg().source_rate_hz)),
          // The run's intensity pair (Round 84, the user's choice). On the channel, so the block, the
          // model, the bound and OptiQKD's baseline all read it through `sourceOf`. Twin of the
          // Python stage's `fibre_source(qkd_intensities)`.
          ...fibreIntensities(config.qkd_intensities) };
        // config.qkd_protocol genuinely selects which physical QKD model runs (BB84 decoy-state vs E91/CHSH),
        // so switching it visibly moves QBER / SKR / secure-key-bits — independently of the OptiQKD toggle.
        const protocol = String(config.qkd_protocol ?? qkdCfg().protocol);
        const pulses = Number(config.qkd_pulses ?? qkdCfg().pulses);
        // THE KEY'S BLOCK (Round 80, the user's choice): the run's controls, else the configured fibre
        // block, resolved once and handed to the model so the key and the metrics describe one block.
        // Twin of `QKDKeyAgreementStage`.
        // The statistics (Round 82), one control for the fibre and the satellite alike.
        const fkSrc = finiteKeySourceOf(chan, Number(config.qkd_eps_sec ?? qkdCfg().finite_key_eps_sec), undefined,
                                        String(config.fk_deviation ?? qkdCfg().finite_key_deviation) as Deviation);
        const blockKind = String(config.qkd_block ?? qkdCfg().finite_key_block);
        // One block definition, also read by OptiQKD's comparison below.
        const blockSpec = { kind: blockKind, runPulses: pulses,
          seconds: Number(config.qkd_block_seconds ?? qkdCfg().finite_key_block_seconds),
          paBits: Number(config.qkd_block_pa_bits ?? qkdCfg().finite_key_block_pa_bits) };
        const blockPulses = fibreBlockPulses(chan, fkSrc, blockSpec);
        const eRes = protocol === "e91" ? e91(pulses, chan, seed, qkdCfg().e91_visibility) : null;   // typed with chshS
        const res = eRes ?? bb84(pulses, chan, seed, undefined, { blockPulses, epsSec: fkSrc.epsSec,
                                                                   deviation: fkSrc.deviation });
        // The eavesdropper fraction that actually ran. The telecom diagram labelled its QKD arrow
        // "(w/ eavesdropper)" unconditionally while `qkd_eve` defaults to 0.0, so a default run drew
        // an attack it did not simulate. Publishing the value lets the label read the run.
        metric(stage, "qkd_eve_fraction", chan.eveFraction);
        // Which decoy states the rate rests on, so a panel names the bound that RAN (Round 78).
        metric(stage, "qkd_decoy_states", eRes ? "none (entangled source)"
          : sourceOf(chan).vacuum ? "vacuum + one decoy" : "one decoy");
        metric(stage, "qber", r2(res.qber));
        // RENAMED from `skr`. ETSI GS QKD 015's `skr` is bit/s and this number is bits per PULSE;
        // leaving one name on two dimensions inside the same signed payload is the defect this
        // round exists to remove, one namespace over.
        // 1e9, matching `api.ts`. The pair split in Round 45: that file moved to six decimals with a
        // comment saying "the control-plane panel shows 0.002355", and these two sites stayed at
        // 1e5, so the panel showed 0.00235 and the comment described the other surface. Nine since
        // Round 75: six held the satellite's 1.723e-5 to two figures, printed as 1.700e-5.
        metric(stage, "skr_per_pulse", Math.round(res.secretKeyRate * 1e9) / 1e9);
        // The block the key is the composable finite key of (Round 80). E91 has no decoy bound and no
        // block: its count stays the asymptotic one, and the facts row says so.
        // The key block's link time, kept for `channel_time_s` below: one number at both sites.
        const fibreBlockS = eRes ? null : Number((blockPulses / Number(chan.sourceRateHz)).toPrecision(4));
        if (!eRes) {
          metric(stage, "qkd_block", blockKind);
          metric(stage, "qkd_block_pulses", Number(blockPulses.toPrecision(6)));
          metric(stage, "qkd_block_s", fibreBlockS);
          metric(stage, "qkd_finite_key_bits", res.secureKeyBits);
          metric(stage, "qkd_finite_key_bound", fkSrc.vacuum ? FINITE_KEY_BOUNDS.vacuum : FINITE_KEY_BOUNDS.oneDecoy);
          metric(stage, "qkd_eps_sec", fkSrc.epsSec);
          metric(stage, "qkd_fk_deviation", fkSrc.deviation);
          metric(stage, "qkd_intensities", String(config.qkd_intensities ?? qkdCfg().intensity_preset));
          metric(stage, "qkd_mu_signal", sourceOf(chan).mu);
          metric(stage, "qkd_mu_decoy", sourceOf(chan).nu);
          // How many of this block, pooled, a key needs: 1 when it carries one.
          metric(stage, "qkd_block_units_for_key",
                 blocksForPositiveKey(...blockCounts([[blockPulses, chan]], fkSrc), fkSrc));
        }
        metric(stage, "secure_key_bits", res.secureKeyBits);
        // ACTUALLY REPORTED NOW. `secretKeyRateNonDecoy` was added in Round 45 with a JSDoc
        // saying "Reported beside `secretKeyRate`" and was read by nothing but its own test -
        // an inert FIELD shipped in the same commit as the gate against inert CONTROLS. The
        // counterfactual is the point of computing a decoy bound: at mu=0.5 it is 0, because
        // the tagged fraction is 7.26 and no detection can be certified single-photon.
        if (res.secretKeyRateNonDecoy != null) {
          metric(stage, "skr_per_pulse_no_decoy",
                 Math.round(res.secretKeyRateNonDecoy * 1e9) / 1e9);
        }
        // THE CLOCK METRICS TOGETHER, on both QKD stages. The satellite twin below emitted no rate at
        // all while its own comment claimed "both report the same quantity" - a comment naming a
        // boundary and describing one side of it.
        //
        // TWO CHANNEL TIMES, NAMED APART (Round 84). `channel_time_s` was the sampled run's pulses
        // over the clock, 0.02 s, while the control-plane panel's "channel time" was the key block's,
        // 5.3 s: one name, two quantities, one page. The name now carries the KEY's time - the block's
        // when there is one, since the finite key and the budget are the block's - which is the
        // figure the panel shows; the sampled run's own time is `run_channel_time_s`. E91 has no
        // block, so its key and its run share one time.
        {
          const clk = keyRateWithClock(res.secretKeyRate, pulses,
                                       sourceRateHz(config?.qkd_source_rate_mhz,
                                                    Number(qkdCfg().source_rate_hz)));
          // `skr_asymptotic_bit_s` since Round 86 (was `skr_bit_s`, which read as the delivered rate): the
          // asymptotic decoy bound times the clock. The finite key over its own block time is beside it, and
          // is the lower figure; both come from the model's expected counts, not from this run's pulses.
          if (clk.skrBitPerSec != null) metric(stage, "skr_asymptotic_bit_s", Math.round(clk.skrBitPerSec));
          const keySeconds = fibreBlockS ?? clk.channelSeconds;
          if (keySeconds != null) metric(stage, "channel_time_s", keySeconds);
          if (keySeconds != null && keySeconds > 0 && typeof res.secureKeyBits === "number" && fibreBlockS != null) {
            metric(stage, "skr_finite_bit_s", Math.round(res.secureKeyBits / keySeconds));
          }
          if (clk.channelSeconds != null) metric(stage, "run_channel_time_s", clk.channelSeconds);
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
          : `BB84 (simulated, decoy-state): QBER=${res.qber.toFixed(4)} sampled from this run's pulses, `
            + `${res.secureKeyBits} secure bits: the composable finite key of a ${Number(blockPulses).toExponential(2)}-pulse block `
            + `(${Number((blockPulses / Number(chan.sourceRateHz)).toPrecision(4))} s at the assumed clock), the block the Key block setting names`);
        if (config.qkd_optimize) {   // AI-QKD parameter optimization (OptiQKD, WS1): a search, not applied
          const opt = optimizeQKD(chan, protocol,
            Number(config.qkd_opt_samples ?? qkdCfg().optimize_n_samples));
          if (opt.improvementPct !== null) metric(stage, "qkd_skr_improvement_pct", r1(opt.improvementPct));
          // BOTH ends of the ratio. The names keep "surrogate" from when clientOptiQKD scored its own
          // transcription of the decoy bound beside a different `bb84()`. It no longer does: it calls
          // the same `decoySecureFraction`, dead-time factor and signal share as `bb84Rate`, so its
          // default IS this run's `skr_per_pulse` (0.0014267105881521113 both, on the default fibre)
          // and the tuned figure is the same asymptotic bound at the tuned parameters. The log said
          // the two came from "a different model" until Round 84.
          metric(stage, "qkd_surrogate_skr_per_pulse_default", Math.round(opt.baselineSkr * 1e6) / 1e6);
          metric(stage, "qkd_surrogate_skr_per_pulse_tuned", Math.round(opt.bestSkr * 1e6) / 1e6);
          metric(stage, "qkd_convergence_evals", opt.convergenceEvals);
          // Only quote the decoy intensities for protocols whose model reads them. The E91/CHSH
          // model ignores muSignal/muDecoy entirely, so printing "tuned" values there described a
          // search that never happened.
          const tuned = protocol === "e91"
            ? `basis p=${r2(opt.bestParams.basisProb)}`
            : `basis p=${r2(opt.bestParams.basisProb)}, mu_signal=${r2(opt.bestParams.muSignal)}, `
              + `mu_decoy=${r2(opt.bestParams.muDecoy)}`;
          // WHAT THE SEARCH DOES NOT DO (Round 84), stated from this run's own numbers. Nothing it
          // finds is applied: the key above was computed at the run's intensity pair. Its basis bias
          // has no counterpart in that key, which takes the bases at even odds (`finiteKeySourceOf`,
          // pZ = 0.5); the rate is linear in the sift ratio, so the bias's share is exact. And at
          // its tuned intensities this run's block can carry LESS composable key per pulse: on the
          // default fibre 4,349,454 bits in 4.75e9 pulses against 5,214,327 in 5.33e9, 6.4% less,
          // while the asymptotic bound reads +60.8%, a factor of 1.41 of its 1.61 being the bias.
          const sift = (pz: number) => pz * pz + (1 - pz) * (1 - pz);
          const biasFactor = sift(opt.bestParams.basisProb) / sift(Number(qkdCfg().basis_prob));
          let tunedKey: { bits: number; pulses: number; changePct: number | null } | null = null;
          if (!eRes && !opt.baselineIsDead) {
            const tunedChan = { ...chan, muSignal: opt.bestParams.muSignal, muDecoy: opt.bestParams.muDecoy };
            const tSrc = finiteKeySourceOf(tunedChan, fkSrc.epsSec, undefined, fkSrc.deviation);
            const tBlock = fibreBlockPulses(tunedChan, tSrc, blockSpec);
            const bits = bb84(pulses, tunedChan, seed, undefined, { blockPulses: tBlock, epsSec: fkSrc.epsSec,
                                                                   deviation: fkSrc.deviation }).secureKeyBits;
            const runPerPulse = res.secureKeyBits / blockPulses, tunedPerPulse = bits / tBlock;
            tunedKey = { bits, pulses: tBlock,
                         changePct: runPerPulse > 0 && Number.isFinite(tunedPerPulse)
                           ? 100 * (tunedPerPulse / runPerPulse - 1) : null };
            metric(stage, "qkd_tuned_finite_key_bits", bits);
            metric(stage, "qkd_tuned_finite_key_block_pulses", Number(tBlock.toPrecision(6)));
            if (tunedKey.changePct != null) metric(stage, "qkd_tuned_finite_key_change_pct", r1(tunedKey.changePct));
          }
          const factor = opt.baselineSkr > 0 ? opt.bestSkr / opt.baselineSkr : null;
          emit("log", stage, opt.baselineIsDead
            ? `OptiQKD (coordinate ascent): baseline SKR is 0 - no secure key to improve, so no `
              + `percentage is reported (${tuned}; ${opt.nEvals} evals - simulated)`
            : `OptiQKD (coordinate ascent, ${opt.nEvals} evals, converged in ${opt.convergenceEvals}): `
              + (protocol === "e91"
                ? "on the device-dependent secure fraction this run's E91 rate comes from, "
                : "on the asymptotic decoy-state bound this run's skr_per_pulse comes from, ")
              + `${opt.baselineSkr.toExponential(2)} -> ${opt.bestSkr.toExponential(2)} per pulse sent, `
              + `+${r1(opt.improvementPct as number)}% at ${tuned}. It is a search: nothing it finds is `
              + `applied to this run. `
              + (factor != null
                ? `A factor of ${r2(biasFactor)} of its x${r2(factor)} is the basis bias, which this run's `
                  + `key does not model: it takes the bases at even odds (pZ = 0.5). `
                : "")
              + (tunedKey
                ? `At the tuned intensities, with the bases at even odds, this run's block would carry `
                  + `${tunedKey.bits.toLocaleString("en-US")} composable bits in ${tunedKey.pulses.toExponential(2)} `
                  + `pulses against ${res.secureKeyBits.toLocaleString("en-US")} in ${blockPulses.toExponential(2)}`
                  + (tunedKey.changePct != null
                    ? `: ${r1(Math.abs(tunedKey.changePct))}% ${tunedKey.changePct < 0 ? "less" : "more"} key per pulse sent. `
                    : ". ")
                : "")
              + `The gain is over this run's parameters (its intensity pair, symmetric bases), not over a `
              + `tuned static link, so it is not comparable with published gains such as arXiv:2603.04192v2's +14-25%.`);
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
        // The pass geometry (Round 77): the elevation this range implies, and no run at all when the
        // satellite is out of view - a sentence, not an infinite loss.
        {
          const elev = linkElevationDeg(satChan);
          if (elev !== null) metric(stage, "sat_elevation_deg", r2(elev));
          const why = outOfView(satChan);
          metric(stage, "sat_in_view", why === null ? 1 : 0);
          if (why !== null) {
            metric(stage, "skr_per_pulse", 0);
            metric(stage, "secure_key_bits", 0);
            metric(stage, "qkd_eve_fraction", satChan.eveFraction);
            emit("log", stage, `free-space downlink: ${why}`);
            break;
          }
        }
        const satERes = satProtocol === "e91" ? e91(satPulses, satChan, seed, qkdCfg().e91_visibility) : null;
        const res = satERes ?? bb84(satPulses, satChan, seed);
        // Same key the fibre stage uses: one QKD stage runs per use case, and both report the same
        // quantity. The diagram's eavesdropper tag reads it rather than asserting one.
        metric(stage, "qkd_eve_fraction", satChan.eveFraction);
        metric(stage, "qkd_decoy_states", satERes ? "none (entangled source)"
          : sourceOf(satChan).vacuum ? "vacuum + one decoy" : "one decoy");
        metric(stage, "qber", r2(res.qber));
        metric(stage, "channel_loss_db", res.channelLossDb);
        // 1e9, matching `api.ts`. The pair split in Round 45: that file moved to six decimals with a
        // comment saying "the control-plane panel shows 0.002355", and these two sites stayed at
        // 1e5, so the panel showed 0.00235 and the comment described the other surface. Nine since
        // Round 75: six held the satellite's 1.723e-5 to two figures, printed as 1.700e-5.
        metric(stage, "skr_per_pulse", Math.round(res.secretKeyRate * 1e9) / 1e9);
        // THE KEY IS THE BLOCK'S COMPOSABLE FINITE KEY (Round 79, the user's choice). The run above
        // samples a fifth of a second at this range, which measures the QBER and decides the abort;
        // the key is the finite-key bound on the counts of a block of link time - one pass peaking at
        // this range by default (`satBlock`). Twin of `SatelliteQKDStage._block_key`. An entangled
        // source has no decoy bound here, so E91 keeps the asymptotic count and says so.
        let satKeyBits = res.secureKeyBits;
        let satBlockS: number | null = null;      // the key block's link time, for `channel_time_s`
        let satBlockNote = "(asymptotic: no finite-key bound is modelled for an entangled source)";
        if (!satERes) {
          const block = satBlock(config, satChan);
          // The satellite's own statistics (the #78 fix): Chernoff by default, not the fibre's union of two.
          const fkSrc = finiteKeySourceOf(satChan, Number(config.sat_eps_sec ?? qkdCfg().free_space_eps_sec), undefined,
                                          String(config.sat_fk_deviation ?? qkdCfg().free_space_finite_key_deviation) as Deviation);
          const [uz, ux] = blockCounts(block.unit, fkSrc);
          const fk = keyLength(scaleCounts(uz, block.copies), scaleCounts(ux, block.copies), fkSrc);
          const fkBits = Number.isFinite(fk.keyBits) && fk.keyBits > 0 ? Math.trunc(fk.keyBits) : 0;
          const blockS = block.copies * block.unitSeconds;
          metric(stage, "qkd_block", block.kind);
          metric(stage, "qkd_block_copies", block.copies);
          satBlockS = Number(blockS.toPrecision(4));
          metric(stage, "qkd_block_s", satBlockS);   // 4 significant figures at every site
          metric(stage, "qkd_block_pulses",
                 Number((block.copies * block.unit.reduce((a, [n]) => a + n, 0)).toPrecision(6)));
          if (block.maxElevationDeg !== null) metric(stage, "sat_pass_max_elevation_deg", r2(block.maxElevationDeg));
          metric(stage, "qkd_finite_key_bits", fkBits);
          metric(stage, "qkd_finite_key_bound", fkSrc.vacuum ? FINITE_KEY_BOUNDS.vacuum : FINITE_KEY_BOUNDS.oneDecoy);
          metric(stage, "qkd_eps_sec", fkSrc.epsSec);
          metric(stage, "qkd_fk_deviation", fkSrc.deviation);
          metric(stage, "qkd_block_units_for_key", blocksForPositiveKey(uz, ux, fkSrc));
          satKeyBits = res.aborted ? 0 : fkBits;
          const what = block.kind === "pass" ? "one pass"
            : block.kind === "passes" ? `${block.copies} passes pooled` : `${block.copies} s at this range`;
          satBlockNote = `(finite key over ${what}, ${Number(blockS.toPrecision(4))} s of link time, eps_sec ${fkSrc.epsSec}, `
            + `${DEVIATION_LABELS[fkSrc.deviation]})`;
        }
        metric(stage, "secure_key_bits", satKeyBits);
        // ACTUALLY REPORTED NOW. `secretKeyRateNonDecoy` was added in Round 45 with a JSDoc
        // saying "Reported beside `secretKeyRate`" and was read by nothing but its own test -
        // an inert FIELD shipped in the same commit as the gate against inert CONTROLS. The
        // counterfactual is the point of computing a decoy bound: at mu=0.5 it is 0, because
        // the tagged fraction is 7.26 and no detection can be certified single-photon.
        if (res.secretKeyRateNonDecoy != null) {
          metric(stage, "skr_per_pulse_no_decoy",
                 Math.round(res.secretKeyRateNonDecoy * 1e9) / 1e9);
        }
        // The same four as the fibre stage, from the FREE-SPACE clock key. The default value is the
        // same 100 MHz, but the provenance is not: on this page it is the anchor's own measured
        // configuration, and on fibre it is that clock carried over to different apparatus. Two keys
        // because one could carry only one of those two sentences.
        {
          const clk = keyRateWithClock(res.secretKeyRate, satPulses,
                                       sourceRateHz(config?.sat_source_rate_mhz,
                                                    Number(qkdCfg().free_space_source_rate_hz)));
          if (clk.skrBitPerSec != null) metric(stage, "skr_asymptotic_bit_s", Math.round(clk.skrBitPerSec));
          // The key's time and the sampled run's, named apart as on the fibre stage (Round 84).
          const keySeconds = satBlockS ?? clk.channelSeconds;
          if (keySeconds != null) metric(stage, "channel_time_s", keySeconds);
          if (clk.channelSeconds != null) metric(stage, "run_channel_time_s", clk.channelSeconds);
          metric(stage, "source_rate_hz", clk.sourceRateHz);
          metric(stage, "qkd_pulses", satPulses);
        }
        if (satERes) metric(stage, "chsh_s", r2(satERes.chshS));
        // The satellite twin of the fibre stage's `e91_model_qber` - same reason, same precision.
        if (satERes) metric(stage, "e91_model_qber", Math.round(satERes.e0 * 1e4) / 1e4);
        (config as any)._satQber = res.qber;
        emit("log", stage, satERes
          ? `free-space E91 (simulated, CHSH S=${r2(satERes.chshS)}): ${res.channelLossDb}dB loss, QBER=${res.qber.toFixed(4)}, ${satKeyBits} bits ${satBlockNote}`
          : `free-space BB84 (simulated): ${res.channelLossDb}dB loss, QBER=${res.qber.toFixed(4)}, ${satKeyBits} bits ${satBlockNote}`);
        // The fibre stage returns before its optimise block under the PQC-only profile; this one
        // had no such guard, so the same checkbox did work here and nothing there.
        if (config.qkd_optimize && String(config.qkd ?? "") !== "pqc-only") {
          const opt = optimizeQKD(satChan, satProtocol,
            Number(config.qkd_opt_samples ?? qkdCfg().optimize_n_samples));
          if (opt.improvementPct !== null) metric(stage, "qkd_skr_improvement_pct", r1(opt.improvementPct));
          metric(stage, "qkd_surrogate_skr_per_pulse_default", Math.round(opt.baselineSkr * 1e6) / 1e6);
          metric(stage, "qkd_surrogate_skr_per_pulse_tuned", Math.round(opt.bestSkr * 1e6) / 1e6);
          metric(stage, "qkd_convergence_evals", opt.convergenceEvals);
          const tunedSat = satProtocol === "e91"
            ? `basis p=${r2(opt.bestParams.basisProb)}`
            : `basis p=${r2(opt.bestParams.basisProb)}, mu_signal=${r2(opt.bestParams.muSignal)}, `
              + `mu_decoy=${r2(opt.bestParams.muDecoy)}`;
          // The fibre stage's twin (Round 84): the same bound the run's rate comes from, and a search
          // whose result is reported and not applied. "surrogate SKR" dated from a separate model.
          emit("log", stage, opt.baselineIsDead
            ? `OptiQKD (free-space): baseline SKR is 0 - no secure key to improve, so no percentage `
              + `is reported (${tunedSat}; ${opt.nEvals} evals - simulated)`
            : `OptiQKD (free-space, coordinate ascent): on the asymptotic bound this run's `
              + `skr_per_pulse comes from, ${opt.baselineSkr.toExponential(2)} -> `
              + `${opt.bestSkr.toExponential(2)}, +${r1(opt.improvementPct as number)}% vs default `
              + `(${tunedSat}; ${opt.nEvals} evals, converged in ${opt.convergenceEvals} - simulated). `
              + `It is a search: nothing it finds is applied to this run's key. `
              + `The gain is over this demo's default parameters, not over a tuned static link, so it `
              + `is not comparable with published gains such as arXiv:2603.04192v2's +14-25%.`);
        }
        break;
      }
      case "qsdc_telemetry": {
        // The QSDC and free-space QKD figures come from an analytic channel model. There is no
        // browser API for quantum communication and there cannot be one, so this is SIMULATED and
        // the page says so rather than presenting a modelled number as a measurement.
        // The same guard the free-space QKD stage above carries. Without it, selecting pqc-only on
        // satellite logged "free-space QKD disabled (PQC-only profile)" and then, one stage later,
        // emitted qsdc_error_rate, qsdc_eq8_bound and qsdc_eavesdrop_detected - delivering
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
        // The badge this feeds names the QSDC channel model. The page's telemetry anomaly verdict
        // further down comes from a plaintext linear model over bundled OPS-SAT-AD segments, not from
        // this model and not under FHE or MPC.
        metric(stage, "inference_backend", "analytic free-space channel model (quantumSim); the anomaly verdict is a plaintext linear model");
        {
          // Out of view: there is no quantum channel to carry the telemetry (Round 77).
          const why = outOfView(satChannel(config));
          if (why !== null) {
            metric(stage, "qsdc_link_budget_limited", 1);
            metric(stage, "qsdc_eavesdrop_detected", 0);
            artifacts.telemetry_delivered = false;
            emit("log", stage, `QSDC (simulated) NOT AVAILABLE: ${why}. Telemetry falls back to the PQC session key.`);
            break qsdc;
          }
        }
        // Same channel the satellite_qkd stage above built, distance included. Reading only sat_eve
        // here meant the downlink range moved the QKD QBER but not the QSDC error rate, so one
        // control produced two inconsistent pictures of the same link.
        // One shared evaluation (`qsdcLink`), so the tooltips that quote C_s read the same function.
        const link = qsdcLink(satChannel(config));
        const e = link.e;
        // QSDC aborts on SECRECY CAPACITY, not on a fixed QBER. Qi et al.'s preprint, arXiv:1810.11806v1,
        // protocol step (2): "If the secrecy capacity Cs is zero, they conclude the channel is insecure
        // and terminate the process." The published version, Light Sci. Appl. 8, 22 (2019), stops when
        // the error rate exceeds the LDPC code's correcting capability, and states that a secure code
        // exists "if the secrecy capacity is non-zero"; this aborts when it is not.
        //
        //     C_s = 1 - h(e) - g*h(e_x + e_z),   g = Q_Eve / Q_Bob = 10^(loss_dB / 10)   (Qi Eq. 8)
        //
        // The scalar 0.11 used here before is the g = 1 (LOSS-FREE) root of the pooled form, which
        // is why it coincided exactly with the Shor-Preskill BB84 bound. Real links are lossy and g
        // grows with loss, so the true threshold falls fast: 0.0757 at 0 dB, 0.0286 at Qi 2019's
        // g = 2.57, and about 6.5e-5 on this page's 800 km downlink (g = 532.66, from 27.26 dB: the
        // channel's 32.21 less the station's 7.96 dB of optics, plus the detector's 3.01) - so 0.11 was
        // wrong by three orders of magnitude, and wrong in the PERMISSIVE direction: it delivered
        // telemetry over channels whose secrecy capacity was already zero.
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
        // The factor is 1/eta_d, so it follows each channel's detector: downlink at eta_d 0.5, g 266.33
        // -> 532.66, 2.00x, and fibre at 0.5 the same 2.00x today (the table in `qsdc.py`).
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
        // At the DEFAULT budget it still cannot close at any range: the air's 1.87 dB at the zenith
        // (the profile's 5.5 dB at 1,200 km, scaled by air mass) + 3 dB pointing plus the detector's
        // 3.01 dB give g = 6.14 even at zero range, and C_s is -0.024 at 0.1 km with no eavesdropper.
        // Only with those two terms below about 4.8 dB in total does a short link close (at 0 + 0 dB
        // it closes out to about 173 km), still well short of a LEO slant range. That is not a bug in
        // this code, it is the finding: QSDC does not close over a LEO free-space downlink under the
        // published criterion. Qi et al. ran it over 1.5 km of fibre with a 4.1 dB back channel
        // (g = 2.57); the default downlink's g here is 532.66 (27.26 dB, the channel's 32.21 less the
        // station's optics plus the detector; C_s -80.82), and 152.99 at the zenith, where C_s is
        // -22.46.
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
        metric(stage, "qsdc_eq8_bound", r2(secrecy));
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
        // WHEN TWO-WAY QSDC CANNOT CLOSE, ONE-WAY QUASI-QSDC (Round 86, the owner's choice). Pan et al.'s STIKE:
        // the message is one-time padded from a pre-shared pool and loss costs rate rather than secrecy. See `stike.ts`.
        const stikeChannel = satChannel(config);
        const stikePoolBits = Math.round(Number(config.sat_stike_pool_mbit ?? qkdCfg().stike_pool_bits / 1e6) * 1e6);
        const quasi = abort && stikePoolBits > 0
          ? stikeSend(webCfg().demo_sat_telemetry_bytes, stikeRates(stikeChannel), {
            pSignal: Number(sourceOf(stikeChannel).pSignal), sourceRateHz: Number(stikeChannel.sourceRateHz),
            poolBits: stikePoolBits, votesPerBit: Number(config.sat_stike_votes ?? qkdCfg().stike_votes_per_bit),
            seed: Number(config.seed ?? 0),
          })
          : null;
        if (quasi) {
          metric(stage, "stike_status", quasi.status);
          metric(stage, "stike_delivered", quasi.status === "delivered" ? 1 : 0);
          metric(stage, "stike_secrecy_per_signal", Number(quasi.rates.secrecy.toPrecision(4)));
          metric(stage, "stike_reliable_per_signal", Number(quasi.rates.reliable.toPrecision(4)));
          metric(stage, "stike_qber", Number(quasi.rates.qber.toFixed(5)));
          metric(stage, "stike_signal_pulses", quasi.signalPulses);
          metric(stage, "stike_seconds", Number(quasi.seconds.toFixed(3)));
          metric(stage, "stike_key_in_flight_bits", quasi.keyInFlightBits);
          metric(stage, "stike_key_consumed_bits", quasi.keyConsumedBits);
          metric(stage, "stike_key_regenerated_bits", quasi.keyRegeneratedBits);
          metric(stage, "stike_pool_after_bits", quasi.poolAfterBits);
          artifacts.telemetry_quasi_qsdc = quasi.status === "delivered";
        }
        // What became of the telemetry, for the verdict below: a run that sent it one way must not also say it fell back.
        const quasiSent = quasi?.status === "delivered";
        const afterAbort = quasiSent ? "One-way quasi-QSDC carries the telemetry instead, as the next line says." : "Telemetry falls back to the PQC session key.";
        const quasiLog = quasi == null ? null
          : quasi.status === "delivered"
            ? `Quasi-QSDC (simulated, one-way STIKE, Pan et al. 2025) delivered the ${webCfg().demo_sat_telemetry_bytes}-byte telemetry where `
              + `two-way QSDC cannot close: ${quasi.signalPulses.toLocaleString("en-US")} signal pulses (${quasi.seconds.toFixed(2)} s), QBER `
              + `${(100 * quasi.rates.qber).toFixed(2)}%, Eq. 3 bound ${quasi.rates.secrecy.toExponential(2)} bit per signal pulse. Key pool: `
              + `${quasi.keyInFlightBits.toLocaleString("en-US")} bits reserved, ${quasi.keyConsumedBits.toLocaleString("en-US")} spent on the pulses Bob `
              + `retained, ${quasi.keyRegeneratedBits.toLocaleString("en-US")} regenerated at the asymptotic rate. It rests on that pre-shared key: this `
              + `is quasi-QSDC, and not the keyless direct communication Eq. 8 bounds.`
            : quasi.status === "pool_too_small"
              ? `Quasi-QSDC (STIKE) not attempted: a frame reserves a key as long as its spread codeword, `
                + `${quasi.signalPulses.toLocaleString("en-US")} bits here, and the pool holds ${quasi.poolBeforeBits.toLocaleString("en-US")}. Telemetry falls back to the PQC session key.`
              : quasi.status === "no_secure_rate"
                ? `Quasi-QSDC (STIKE) refused: no positive key rate at QBER ${(100 * quasi.rates.qber).toFixed(2)}% (Eq. 3 of Pan et al.). Telemetry falls back to the PQC session key.`
                : `Quasi-QSDC (STIKE) could not decode: ${quasi.wrongBits} bits wrong and ${quasi.erasedBits} with no retained pulse. Telemetry falls back to the PQC session key.`;
        emit("log", stage, !abort
          ? `QSDC (simulated) delivered telemetry (Qi et al.'s Eq. 8 bound per received carrier, C_s/Q_Bob = ${secrecy.toFixed(4)} bit, g=${g.toFixed(1)})`
          : lossLimited
            ? `QSDC (simulated) NOT AVAILABLE on this link - Qi et al.'s Eq. 8 bound per received carrier, C_s/Q_Bob, evaluates to ${secrecy.toFixed(4)} bit (below zero: it certifies no secrecy capacity) `
              + `at ${lossDb.toFixed(1)} dB from Eve to Bob's detector, the channel less the station optics `
              + `plus the detector (g=${g.toFixed(1)}), and it is still `
              + `${secrecyNoEve.toFixed(4)} with NO eavesdropper. The limit is the link budget, not `
              // The criterion as computed: C_s = 1 - h(e) - g h(e_x + e_z), with e_x + e_z = 2e here.
              // This line quoted "C_s > 0 requires h(QBER) < 1/(1+g)", the POOLED form that the
              // per-basis correction replaced - a necessary condition, not the one this code applies.
              + `an attack: C_s/Q_Bob = 1 - h(e) - g h(2e) = 1 - ${h2(e).toFixed(4)} - `
              + `${(g * h2(Math.min(0.5, 2 * e))).toFixed(4)} at the channel model's error rate e = ${e.toFixed(4)} `
              + `(the QKD stage's QBER is a sampled draw around it). This models the downlink alone: Qi et al.'s DL04 `
              + `protocol is two-way, and its return path would only add loss. ${afterAbort}`
            : `QSDC (simulated) ABORTED: Qi et al.'s Eq. 8 bound per received carrier, C_s/Q_Bob, is ${secrecy.toFixed(4)} <= 0 at error ${e.toFixed(4)} `
              + `with g=${g.toFixed(1)} (${lossDb.toFixed(1)} dB: the channel less the station optics, plus `
              + `the detector) — this link closes at `
              + `${secrecyNoEve.toFixed(4)} without Eve, so the loss is attributable — ${quasiSent ? "one-way quasi-QSDC carries the telemetry instead, as the next line says" : "telemetry withheld"}`);
        if (quasiLog) emit("log", stage, quasiLog);
        }   // end qsdc: everything below scores the BUNDLED corpus and runs on both profiles
        {
          // REAL OPS-SAT-AD telemetry segments (CC BY 4.0) + a demo linear anomaly regression, scored
          // in-browser IN PLAINTEXT - a JS dot product, with no FHE or MPC (the CKKS version of this
          // regression runs only in the Python deployment; the page's CKKS check scores the published
          // student, below). The satellite flow has no private_inference stage, so this is
          // where the anomaly verdict is surfaced. The SMAP/MSL windows this read until Round 84 had
          // no data licence from any provider and are no longer served.
          //
          // OUTSIDE the delivery branch, for exactly the reason the published-model block below was
          // moved out: this step scores a segment of the BUNDLED corpus, not the telemetry the channel
          // just delivered, so it has no dependency on whether QSDC aborted.
          //
          // Gating it was not a harmless over-caution, because two-way QSDC essentially never delivers here.
          // `C_s > 0` needs `1 - h(e) > g h(2e)`, and `g` scales with link loss: secrecy capacity is
          // -80.82 bit/carrier at the default 800 km (g = 532.66) and still -22.46 at the zenith,
          // 499 km (g = 152.99), with no eavesdropper. So the anomaly verdict, the model's provenance
          // line and the whole SHAP panel were unreachable through the UI - a page advertising a model
          // whose output no user action could produce. The abort itself is correct physics and is
          // unchanged; only the false dependency is gone. The full deployment's CKKS twin
          // (`TelemetryAnomalyStage`) was decoupled the same way in Round 84.
          const anom = await realSatelliteAnomaly(u);
          if (anom) {
            artifacts.telemetry_anomaly = anom.isAnomaly;
            metric(stage, "telemetry_anomaly_score", anom.anomalyScore);
            // The verdict says which model and which row produced it: the segment is a held-out row
            // with a known label, and the regression is a demo model, not the published one.
            emit("log", stage, `telemetry anomaly detection, in plaintext (${anom.provenance}): `
              + `${anom.isAnomaly ? "ANOMALY" : "nominal"} (score ${anom.anomalyScore})`
              + (anom.trueLabel != null ? `; the segment is labelled ${anom.trueLabel ? "anomalous" : "nominal"}` : "")
              + `. Model: ${anom.modelNote}`);
            // Satellite had NO attribution in the browser, while the Python pipeline has computed
            // one since Round 18 that nothing rendered. The model here is genuinely linear - a 2xN
            // margin over one OPS-SAT segment's 19 features - so exact linear SHAP applies unchanged
            // and no new method is introduced. Reference point is the payload's E[x] over the segments
            // the regression was fitted on, never the held-out rows being explained.
            artifacts.explanation = linearShap(
              anom.model.W, anom.model.b, anom.window, anom.featureNames, 5, anom.baseline);
            artifacts.explanation_provenance = "demo OPS-SAT-AD regression (not the published model)";
            artifacts.explanation_baseline = anom.baselineNote;
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
        // The published student under CKKS (Round 86), verification rather than a decision, as on telecom: the page's
        // anomaly verdict above is a different, plaintext regression. Outside the delivery branch for the same
        // reason: it scores the benchmark's test partition, not the telemetry this run delivered.
        try {
          await scoreHeldOutStudent(stage, "satellite_comms", "data/satellite/student_heldout.json");
          emit("log", stage, "the DECISION is not encrypted: each segment's score is decrypted and its sign read in "
            + "the clear, because CKKS has no native comparison.");
        } catch (e) {
          emit("log", stage, `encrypted tier skipped: ${e instanceof Error ? e.message : String(e)}`);
        }
        break;
      }
      case "qsmpc_compute": {
        // genomics defaults to QSMS secure summation; others to the OPRF PSI (matches the backend).
        // Only the PSI reads the session key, so its guard is in the PSI branch below.
        const op = config.qsmpc_op || (usecase === "genomics" ? "sum" : "psi");
        if (op === "sum") {
          // QSMS additive secret-sharing over a prime field Z_p (config.prime; 0/auto = the smallest prime
          // above the LARGEST total the inputs can reach, `qsmsField` - Round 84; it was the smallest
          // prime above the actual total, which made the match below true by construction and let p
          // narrow the total to the gap beneath it). Each party masks its secret; the masks cancel
          // mod p on reconstruction. A requested p that cannot hold the largest total is refused below.
          // In this tab one step sums both shares of every party, so it shows the arithmetic, not a
          // separation between share holders.
          // Party count from config, so the browser and the backend describe the same computation
          // instead of the browser inventing 5 while the backend ran n_parties.
          //
          // Round 80: on genomics the secrets are the cohorts' real allele counts. They were
          // `floor(u() * 100)` per party - stand-ins - while the page and its diagram said the cohorts
          // pool allele counts. Now each of the backend's `genomics_biobanks` cohorts contributes its
          // effect-allele count at the score's first variant over the bundled 1000 Genomes slice, split
          // as the backend splits it. Another use case set to "Secure sum" has no counts to pool and
          // keeps the stand-ins, and its log and diagram say so.
          const streamParties = Math.max(2, Number(qsmpcCfg().n_parties ?? 5));
          const drawn = Array.from({ length: streamParties }, () => Math.floor(u() * (QSMS_STAND_IN_MAX + 1)));
          const alleles = usecase === "genomics" ? await realAlleleCounts(Number(dataCfg().genomics_biobanks ?? 4)) : null;
          if (usecase === "genomics" && !alleles) {
            throw new Error("the genomics secure sum pools the cohorts' allele counts, and the bundled 1000 Genomes "
              + "slice did not load; no stand-in is pooled under that name");
          }
          const secrets = alleles ? alleles.counts : drawn;
          const nParties = secrets.length;
          const rawTotal = secrets.reduce((a, b) => a + b, 0);
          // Snapped, not trusted. The field is labelled "QSMS prime" and used to accept any
          // integer, so entering 100 shared additively over Z_100 -- which is not a field, while
          // the log still printed "over Z_100" as though it were.
          const askedPrime = Number(config.prime);
          // The largest total these inputs can reach, which is public: each person carries at most
          // two copies of the effect allele, and a stand-in is at most QSMS_STAND_IN_MAX per party.
          const maxTotal = alleles ? 2 * alleles.individuals : QSMS_STAND_IN_MAX * nParties;
          const p = qsmsField(maxTotal, askedPrime);
          // A FIELD THAT CANNOT HOLD THE TOTAL IS REFUSED (Round 84), as `genomics.build` refuses it in
          // Python. A requested p at or below the largest total these inputs can reach returned that
          // total reduced mod p under the "pooled count" name - GF(11) turns 42 alleles into 9. The
          // stream advances as a completed run's does, so the row private_inference encrypts is unmoved.
          if (p <= maxTotal) {
            for (let k = 0, later = nParties + Math.max(0, streamParties - nParties); k < later; k++) u();
            throw new Error(`QSMS field ${p} cannot hold the pooled ${alleles
              ? `allele count, which can reach ${maxTotal} for ${alleles.individuals} people`
              : `stand-in total, which can reach ${maxTotal} for ${nParties} parties`}: the sum would `
              + `wrap mod the field. Leave the prime at 0 (auto) or set one above ${maxTotal}.`);
          }
          // Say so when the request was not honoured. A clamp that reports nothing is how a run ends
          // up computing over a different field than the panel above it displays.
          if (askedPrime > MAX_FIELD_PRIME || (askedPrime > 1 && !Number.isFinite(askedPrime))) {
            emit("log", stage, `requested field modulus ${Number.isFinite(askedPrime) ? askedPrime : "infinite"} `
              + `exceeds the ${MAX_FIELD_PRIME} ceiling and was clamped: past 2^53 the modular sum `
              + `stops being exact, and trial division at that size does not return`);
          }
          // Each mask is uniform over Z_p from the CSPRNG (csprng.ts). Until Round 66 it was
          // floor(u() * p) from the run's seeded generator: anyone with the seed on the page could
          // recompute it, and scaling a 32-bit output to [0, p) is biased unless p divides 2^32,
          // which no odd prime does. The generator is still advanced once per secret, as it was when
          // it drew the masks, so a seed selects the same inputs downstream as before - the row
          // private_inference encrypts among them.
          const shares = secrets.map((s) => {
            u();
            const mask = randomBelow(p);
            return [((s - mask) % p + p) % p, mask];
          });
          // Four cohorts advance the stream four times above; the seven stand-in draws used to advance
          // it seven. The difference is taken here, so the row private_inference encrypts is unchanged.
          for (let k = nParties; k < streamParties; k++) u();
          const total = shares.reduce((acc, [s0, s1]) => (acc + s0 + s1) % p, 0) % p;
          metric(stage, "secure_sum", total);
          metric(stage, "qsms_prime", p);
          metric(stage, "qsms_parties", nParties);
          // The bound the field was chosen from, so a reader can check p > it without the inputs.
          metric(stage, "qsms_max_total", maxTotal);
          // What was pooled, and whether the shares reconstruct the plain total. They do because p
          // exceeds every total the inputs can reach (a field that does not is refused above), which
          // is a property of the protocol and the bound, not of having read the answer.
          metric(stage, "qsms_inputs", alleles ? "allele_counts" : "stand_ins");
          metric(stage, "qsms_matches_plain_total", total === rawTotal ? 1 : 0);
          if (alleles) {
            metric(stage, "pooled_allele_count", total);
            metric(stage, "qsms_variant", alleles.variant);
          }
          // Named for what the BROWSER does, and the backend for what IT does. This tab adds two
          // additive shares per party mod p in one step. The Python backend runs Sutradhar's Shamir +
          // d-level QFT construction (qsh/qsmpc/secure_summation.py), and it is not "the quantum
          // version of this": its genomics field is the next prime above twice ITS cohort (197 for
          // the committed 97-person slice), where d**t passes its statevector bound, so it samples the
          // protocol's measurement outcomes analytically and simulates no GHZ state either.
          emit("log", stage, `secure summation = ${total} over Z_${p} (${alleles
              // The pooled count only: printing each cohort's count would publish exactly what the
              // secure sum exists to keep private.
              ? `${nParties} cohorts' effect-allele counts at ${alleles.variant}, over the ${alleles.individuals} `
                + `bundled 1000 Genomes individuals; ${total === rawTotal
                  ? `equal to the plain total, as it must be: p exceeds ${maxTotal}, the most ${alleles.individuals} people can carry`
                  : "DIFFERENT from the plain total, so the shares did not reconstruct it"}`
              : `${nParties} parties' stand-in values; this use case has no allele counts to pool`}; `
            + `two additive shares mod p per party, both summed in this one step, so this shows the `
            + `arithmetic and separates no share holders)`
            + (alleles ? `. The Python backend runs Sutradhar's Shamir + d-level QFT construction over `
              + `its own cohort, with the field the next prime above twice that cohort; there d**t `
              + `passes its statevector bound, so it samples the protocol's measurement outcomes `
              + `analytically and simulates no GHZ state` : ""));
        } else {
          // THE OPRF PSI (oprfPsi.ts, RFC 9497 mode 0x00 over ristretto255): Bank A blinds its IDs,
          // Bank B evaluates them under a key fresh to this run and sends its own outputs shuffled,
          // and Bank A intersects. What keeps the lists apart is the OPRF: Bank A cannot compute Bank
          // B's PRF without Bank B, and Bank B sees only uniform blinded elements. The PQC session key
          // seals the three messages, which keeps the transcript from a third party, including one
          // that records it now and attacks ristretto255 later; it hides nothing between the banks.
          // There is no unsealed mode, so no key means no run: `?? new Uint8Array(32)` once put 32
          // zero bytes, public knowledge, in its place here while the stage still reported a PSI.
          // `pqc_handshake` precedes this stage in every flow that reaches it, so this throw is
          // unreachable today and fails loudly if the ordering ever changes.
          const key = (config as any)._sessionKey as Uint8Array | undefined;
          if (!key || key.length === 0) {
            throw new Error("qsmpc_compute needs the PQC session key: the OPRF PSI's messages are sealed "
              + "under it, and there is no unsealed mode");
          }
          // The set sizes scale with the transaction volume (config.aml_transactions): more transactions,
          // larger sets, more OPRF evaluations (about 1 ms each in node) and a larger intersection. The
          // cap bounds the work for responsiveness and is stated in the Parameters panel. The IDs come
          // from the run's seeded generator, as before; the key, the blinds and Bank B's shuffle come
          // from the CSPRNG and take no draw from it.
          const nTx = Number(config.aml_transactions);
          const txv = Number.isFinite(nTx) && nTx > 0 ? nTx : dataCfg().aml_transactions;
          const perParty = usecase === "finance_aml" ? Math.max(20, Math.min(2000, Math.round(txv / 20))) : 40;
          const nShared = Math.max(4, Math.round(perParty * 0.3));
          const shared = Array.from({ length: nShared }, () => `acct-${Math.floor(u() * 1e6)}`);
          const a = [...shared, ...Array.from({ length: perParty }, () => `a-${Math.floor(u() * 1e6)}`)];
          const b = [...shared, ...Array.from({ length: perParty }, () => `b-${Math.floor(u() * 1e6)}`)];
          const psi = await oprfPSI(a, b, key);
          // The parties by the names the canvas gives them: the banks on finance, Party A and Party B
          // wherever else the PSI runs (the generic demo, or another use case set to PSI).
          const [pa, pb] = usecase === "finance_aml" ? ["Bank A", "Bank B"] : ["Party A", "Party B"];
          artifacts.psi_intersection = psi.intersection;
          metric(stage, "psi_intersection_size", psi.intersection_size);
          metric(stage, "psi_set_size_a", psi.size_a);
          metric(stage, "psi_set_size_b", psi.size_b);
          metric(stage, "psi_blind_evaluations", psi.blind_evaluations);
          metric(stage, "psi_protocol", psi.protocol);
          emit("log", stage, `PSI complete: ${psi.intersection_size} shared accounts over ${psi.size_a}x${psi.size_b} `
            + `by the ${psi.protocol}. ${psiLearnedSummary(psi, pa, pb)}. Its messages are sealed under the PQC `
            + `session key; ${pa} and ${pb} run in this tab (notional parties), and the OPRF rests on a `
            + "discrete-log assumption, so it is not post-quantum");
          if (usecase === "finance_aml" && Number(config.multi_party_psi) >= 3) {
            await runMultiPartyPsi(stage, Number(config.multi_party_psi), seed, emit, metric, artifacts);
          }
        }
        break;
      }
      case "federated_training": {
        // REAL client-side federated learning ("Quantum Federated Learning" by name): FedAvg of a
        // logistic head over the closed-form `qcnnExpectation` features - no circuit is simulated -
        // across simulated clients, round accuracy genuinely computed in-browser. The full
        // PennyLane HybridQNN (server side, direct API call only) trains a real variational circuit; this does not.
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
        // No session key: the clients agree each pair's mask seed among themselves by X25519 at the
        // start of the run (clientQFL.ts), and the aggregator receives only their public keys and
        // the masked updates. The PQC session key, which the aggregator side also holds, seeds no mask.
        const q = await clientQFL(seed, { data: qflData });
        metric(stage, "federated_accuracy", q.finalAcc);
        metric(stage, "qfl_rounds", q.rounds);
        metric(stage, "qfl_clients", q.clients);
        metric(stage, "qfl_mask_pairs", q.pairs);
        metric(stage, "federated_eval_rows", q.evalRows);
        // The same held-out rows with untrained weights. Without it "accuracy 0.74" cannot be read
        // as evidence that the federated rounds did anything, and the round series cannot supply
        // that either once the problem converges inside round one.
        metric(stage, "federated_baseline_accuracy", q.baselineAcc);
        // Round 86: finance's 400 PaySim rows are a 50/50 draw (the sample's own `evaluation.note`), so an
        // accuracy over them says nothing at PaySim's fraud rate; the label and the log say so when it holds.
        const posShare = qflData?.Y?.length ? qflData.Y.reduce((a: number, v: number) => a + (v ? 1 : 0), 0) / qflData.Y.length : null;
        const balanced = posShare != null && Math.abs(posShare - 0.5) < 1e-9;
        metric(stage, "federated_rows_class_balanced", balanced);
        // What the aggregation ACTUALLY was, published so the diagram can render it instead of
        // carrying its own claim. See the QFLResult comment for why this replaced `secureAgg`.
        metric(stage, "qfl_aggregation", q.aggregation);
        metric(stage, "federated_accuracy_source",
          qflData ? `bundled ${usecase} sample, ${qflData.X.length} rows`
                  : "synthetic 2-cluster toy (no bundled sample for this use case)");
        artifacts.qfl = q;
        // Two decimals, as `clientQFL` rounds them: three printed a third digit that is always 0.
        emit("log", stage, `Federated learning (client; named "Quantum Federated Learning", but the feature map `
          + `is a closed-form function, not a circuit): ${q.clients} clients x ${q.rounds} `
          + `rounds (classical FedAvg; ${q.aggregation}) over `
          + `${qflData ? `the bundled ${usecase} sample (${qflData.X.length} rows)` : "a synthetic toy"}`
          // Round 86 (fact-check C9): the "hospitals" are shards dealt round-robin from one shuffled pool
          // (clientQFL.ts), so every client's data share one distribution - the easy case for FedAvg.
          + `, dealt round-robin to the clients from one shuffled pool, so every client's rows share one `
          + `distribution (IID), which real hospitals' do not: on highly skewed non-IID shards, Zhao et al. `
          + `(arXiv:1806.00582) measured accuracy losses of up to 55% for neural networks`
          + ` - held-out accuracy ${q.baselineAcc.toFixed(2)} untrained (every row class 0), `
          + `${q.roundAcc[0].toFixed(2)} after round 1, ${q.finalAcc.toFixed(2)} after round ${q.rounds}`
          + (balanced ? "; the rows are a class-balanced draw, half of them positive, so no accuracy here "
            + "estimates performance at the corpus's own rate" : ""));
        if (q.pairs > 0) {
          emit("log", stage, `Secure aggregation: each of the ${q.clients} clients drew an X25519 key pair for this run `
            + `and each of the ${q.pairs} pairs derived its mask seed from their shared secret by HKDF-SHA256 `
            + `(Bonawitz et al., CCS 2017), in ${q.keyAgreementMs.toFixed(1)} ms; the aggregator received only the `
            + "public keys and the masked updates, and the PQC session key seeds no mask. All clients run in this tab, "
            + "which holds every private key; there is no dropout recovery, no malicious security and no "
            + "authentication of the public keys.");
        }
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
        // genuine in-browser IDS: softmax logistic trained + scored on this project's synthetic attack
        // signatures for the 8 classes of Al-Kuwari et al. 2026 (the classes are the paper's; the 7
        // features and their values are this project's own, see clientIDS.ts).
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
        // numbers of identical shape, and the two that read highest are binary F1, which folds the
        // seven attack types into "attack", and accuracy, which counts exact classes but averages a
        // failing class away - so a reader with no marker takes the flattering pair as the result.
        // The log below says this; the table did not.
        metric(stage, "ids_metric_roles",
          `headline: ids_macro_f1 (per-class mean over ${ids.classesScored} classes) and `
          + "ids_worst_class_f1; companions: ids_binary_f1, which merges the seven attack types into "
          + "one class, and ids_accuracy, an 8-class accuracy that averages a failing class away - so "
          + "both read high while a single class fails");
        metric(stage, "ids_params", ids.nParams);
        // clientIDS clamps this to >=20 and falls back on a non-finite input, so the requested value
        // is not always the value used. Report what the run actually trained on.
        // `ids_generated_per_class` since Round 86 (was `ids_train_per_class`): the rows drawn per class, of
        // which the held-out fraction is scored, not trained on; the rows trained on are reported beside it.
        metric(stage, "ids_generated_per_class", ids.nPerClassUsed);
        metric(stage, "ids_train_rows", ids.nPerClassUsed * ids.classesDefined - ids.nTest);
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
          // Not one-vs-rest (Round 86): `explainSample` sets the predicted class's weights against the MEAN of
          // the other seven classes' weights, so the bars attribute that contrast, and the class is named.
          artifacts.explanation_provenance =
            `in-browser softmax IDS, its predicted class ${ids.explain.className} against the mean of the other `
            + `${ids.classesDefined - 1} classes`;
          artifacts.explanation_baseline = "zero vector (features are z-scored, so E[x] = 0)";
          // Set ONCE, at the end of this case, from what actually surfaced - see the assignment
          // after the QCNN branch. Written here first as an unconditional `true`, which was correct
          // for the softmax and for the encrypted tier and wrong the moment the QCNN branch below
          // replaced the surfaced class without touching it.
        }
        // "QML-IDS (client, softmax ..." until Round 86 named a classical model after the workstream.
        emit("log", stage, `IDS (client, classical softmax logistic regression, the QML-IDS workstream's baseline, on this project's synthetic `
          + `signatures for the 8 classes of Al-Kuwari et al. 2026): `
          + `macro-F1=${r2(ids.macroF1)} over the ${ids.classesScored} classes present in the `
          + `held-out split`
          + (ids.classesScored < ids.classesDefined
            ? ` (of ${ids.classesDefined} defined - this split is missing `
              + `${ids.classesDefined - ids.classesScored}, and an absent class is not scored 0)`
            : "")
          + `, worst class ${r2(ids.worstClassF1)} `
          + `- measured on ${ids.nTest} held-out samples of the ${ids.nPerClassUsed * ids.classesDefined} `
          + `generated (${ids.nPerClassUsed} per class before the split)`);
        emit("log", stage, `companions (NOT the headline): binary attack-vs-normal F1=${r2(ids.binaryF1)}, `
          + `aggregate accuracy=${r2(ids.accuracy)}. Binary F1 folds the seven attack types into one class, `
          + `and accuracy counts exact classes but averages a failing class away, which is why both read `
          + `high while a single class can be failing - see scripts/train/telecom_ids.py`);
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
        const qcnnAsked = String(config.inference || "ckks");
        if ((qcnnAsked === "qcnn" || qcnnAsked === "qcnn-ckks") && ids.explain?.xRaw) {
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
            if (!(qcnnAsked === "qcnn-ckks" && await surfaceQcnnUnderCkks(stage, usecase, w, ids.explain.xRaw.slice(0, w.n_qubits), emit, metric, artifacts))) {
              metric(stage, "primary_inference_backend", "qcnn-trained");
              metric(stage, "primary_inference_class", w.class_names?.[qc.cls] ?? qc.cls);
            }
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
              + `the weights. ${qcnnAsked === "qcnn-ckks"
                ? "This run of the circuit on the plaintext features is the self-check of the CKKS evaluation reported above, which decides the class. "
                : `The circuit runs UNENCRYPTED, simulated exactly on a state vector in this tab; ${qcnnSimulability(w)}.`}`);
            // The verdict is COMPUTED, not asserted. This read `and it LOSES:` as a literal, guarded
            // only on the field being PRESENT - no comparison was ever performed, so a retrained
            // artifact that put the circuit ahead would have gone on printing that it lost. The
            // numbers were right beside the sentence the whole time.
            // THREE-WAY (Round 84): a tie printed WINS, because only `mine < theirs` was tested.
            // RE-SCORED IN THIS TAB since Round 86: the circuit's figure is computed here over the test split's rows and the
            // verdict uses it; logistic regression's is still read from the weights file, and the line says which is which.
            const rescore = await rescoreQcnnInTab(stage, usecase, w, metric, artifacts);
            const mine = rescore ? Math.round(rescore.headline * 1e6) / 1e6 : w.test_macro_f1, theirs = w.classical_lr_macro_f1;
            if (typeof mine === "number" && typeof theirs === "number") {
              const behind = mine < theirs, level = mine === theirs;
              emit("log", stage, `and it ${behind ? "LOSES" : level ? "TIES" : "WINS"}: macro F1 ${mine} `
                + `against a classical logistic regression at ${theirs} on the same held-out split, with `
                + `${w.n_classes_predicted ?? "?"} of ${w.readout?.n_classes ?? 2} classes predicted. The circuit's figure was `
                + `${rescoreSentence(rescore, w.test_macro_f1)}; logistic regression's was recorded at training time. `
                + `Reported because it is the result, not because it flatters`
                + (behind || level ? "." : " - and if this ever reads WINS, check the split before believing it."));
            } else if (typeof mine === "number" || w.per_class_f1) {
              // One number without the other cannot support a verdict, so it does not state one.
              emit("log", stage, `held-out macro F1 ${mine ?? "n/a"}, ${rescoreSentence(rescore, w.test_macro_f1)}; no `
                + `classical comparator ships in this artifact, so no verdict is stated here.`);
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
        // Use REAL data + a REAL model where one is shipped (medical = Wisconsin breast-cancer + a real
        // logistic regression; genomics = PGS Catalog PRS weights over 1000 Genomes genotypes, see
        // realData.ts for their terms; finance = the PaySim simulator sample, synthetic rows under
        // CC BY-SA 4.0, with its four-feature surrogate) - matching the full deployment. The other use
        // cases ship no real sample for this stage: synthetic features, logged as such.
        const realIn: RealInput | null = await realInput(usecase, u);
        const feats = realIn ? realIn.values : Array.from({ length: 8 }, () => u() * 2 - 1);
        emit("log", stage, realIn
          ? `data: ${realIn.provenance}`
          : "data: synthetic features (no real sample is bundled for this use case's pipeline)");
        if (realIn?.metrics) for (const [k, v] of Object.entries(realIn.metrics)) metric(stage, k, v);
        const t = performance.now();
        // The bundled real linear model where one ships, else placeholder weights (the published ONNX
        // model runs separately, below). inf.provenance records which, and the UI shows it.
        const inf = await privateInference(usecase, feats, realIn?.model ?? null);
        // Whether the class CKKS surfaces comes from the model the SHAP bars below explain (Round 84):
        // set where the CKKS class is, read where `explanation_is_of_reported_class` is.
        let ckksClassIsTheExplainedModels = false;
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
        // "made the first click feel like a hang"; the Polytune wasm load follows. The tab
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
        // THE FIGURE'S ROWS ARE HELD OUT (Round 84). Finance and medical have a held-out corpus and
        // are scored by `emitHeldOutAgreement` below; only a page without one (genomics, whose 30
        // reference-panel individuals are external rather than a split) still feeds rows to the batch
        // inside `realClientHybrid`. Passing both would encrypt medical's rows twice.
        const held = await heldOutCorpus(usecase);
        const corpus = !held && realIn?.model?.W ? await realInputCorpus(usecase) : null;
        const real = await realClientHybrid(feats, usecase === "medical_fl", usecase,
                                            realIn?.model ?? null, corpus);
        if (!("failed" in real)) {
          // A real encrypted convolution is not scaled by anything: its latency IS the measurement.
          const bscale = real.conv
            ? 1
            : benchScale(usecase, BENCH_MODEL_SIZE, config.workload ?? "linear");
          artifacts.fhe_prediction = real.prediction;
          // The garbled-circuit argmax over CKKS logits of `realIn.model` when the encryption evaluated
          // it; over the demo pattern otherwise, which the bars do not explain.
          ckksClassIsTheExplainedModels = real.encryptedRealModel;
          // WHICH model the encryption evaluated. Until Round 38 this was always the demo pattern
          // while the class and the SHAP bars came from the bundled model, so every `fhe_*` figure
          // on finance and genomics described a weight vector that appeared nowhere else on the page.
          metric(stage, "fhe_model", real.encryptedRealModel
            ? "bundled real model - the same one the reported class and its explanation come from"
            : "demo pattern (no bundled model ships for this use case)");
          if (real.batch) {
            const bt = real.batch;
            const bits = bt.maxAbsError > 0 && bt.maxAbsError < 1
              ? Math.floor(-Math.log2(bt.maxAbsError)) : null;
            metric(stage, "encrypted_rows_scored", bt.rows);
            metric(stage, "encrypted_rows_available", bt.rowsOffered);
            metric(stage, "encrypted_rows_dropped_by_cap", bt.rowsOffered - bt.rows);
            metric(stage, "encrypted_rows_per_ciphertext", bt.rowsPerCiphertext);
            metric(stage, "encrypted_ciphertexts", bt.ciphertexts);
            metric(stage, "encrypted_batch_ms", r2(bt.fheMs));
            metric(stage, "encrypted_batch_max_abs_error", bt.maxAbsError);
            metric(stage, "encrypted_batch_precision_bits", bits);
            metric(stage, "encrypted_smallest_plaintext_margin", bt.smallestMargin);
            metric(stage, "encrypted_agreement", r4(1 - bt.mismatches / Math.max(1, bt.rows)));
            metric(stage, "encrypted_agreement_rows", bt.rows);   // the denominator, never implicit
            // The part of that denominator that carries no signal, and the rate over the rest.
            // Emitted on EVERY run, including the ones where no tie happened to flip - see the
            // note at the computation for why a conditional disclosure is the wrong shape here.
            metric(stage, "encrypted_undecidable_rows", bt.tieRows);
            metric(stage, "encrypted_decidable_rows", bt.decidable);
            metric(stage, "encrypted_agreement_over_decidable",
              bt.decidable > 0 ? r4(1 - bt.flippedDecidable / bt.decidable) : null);
            metric(stage, "encrypted_top1_mismatches_batch", bt.mismatches);
            if (bt.mismatches > 0) {
              metric(stage, "encrypted_flip_margin_max", bt.flipMarginMax);
              // The basis of the verdict, so it is checkable rather than taken on trust.
            }
            artifacts.encrypted_agreement_plot = {
              kind: "encrypted-agreement", points: bt.points, rows: bt.rows,
              rowsOffered: bt.rowsOffered, classes: real.logits.length,
              maxAbsError: bt.maxAbsError,   // all rows x classes; `points` is one score per row
              flipMarginMax: bt.flipMarginMax,
              undecidableRows: bt.tieRows, decidableRows: bt.decidable,
              agreementOverDecidable: bt.decidable > 0
                ? r2(1 - bt.flippedDecidable / bt.decidable) : null,
              ciphertexts: bt.ciphertexts, rowsPerCiphertext: bt.rowsPerCiphertext, fheMs: bt.fheMs,
              msPerCiphertext: bt.msPerCiphertext, smallestMargin: bt.smallestMargin,
              flipBoundFactor: bt.flipBoundFactor, medianAbsError: bt.medianAbsError,
              noisyScores: bt.noisyScores, scoreKind: bt.scoreKind,
              rowsAre: usecase === "genomics"
                ? "external reference-panel individuals (1000 Genomes genotypes; PGS000072 was not "
                  + "fitted on them, and there is no split), whose decision logit on both axes is the "
                  + "full score minus the cut"
                : undefined,
            };
            emit("log", stage, `encrypted tier over the page's own corpus: ${bt.rows} of `
              + `${bt.rowsOffered} rows in ${bt.ciphertexts} ciphertext(s) of up to `
              + `${bt.rowsPerCiphertext} at this feature width, ${r2(bt.fheMs)} ms. Top-1 mismatches vs plaintext: `
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
          // THE FULL SCORE, SAID IN FULL (Round 84): what is encrypted, what stays in plaintext, what
          // is decrypted, and how the flag is decided, with the score and the flag against the clear.
          // The model is W = [0, w], b = [0, -cut] (`realInput`), so slot 0 of the summed ciphertext
          // is the score and the cut is subtracted after decryption.
          if (usecase === "genomics" && realIn?.model) {
            const bCut = realIn.model.b[1] ?? 0;
            const encScore = real.logits[1] - bCut, plainScore = real.plainLogits[1] - bCut;
            const scoreErr = Math.abs(real.logits[1] - real.plainLogits[1]);
            const plainFlag = argmax(real.plainLogits);
            const slots = Math.floor(Number(fheCfg().poly_modulus_degree) / 2);
            const rot = real.dense?.rotations ?? 0, muls = real.dense?.multiplies ?? 0;
            metric(stage, "prs_encrypted_score", Number(encScore.toFixed(6)));
            metric(stage, "prs_score_abs_error", scoreErr);
            metric(stage, "prs_high_risk_flag", real.prediction);
            metric(stage, "prs_high_risk_flag_plaintext", plainFlag);
            metric(stage, "prs_flags_agree", real.prediction === plainFlag ? 1 : 0);
            const bt = real.batch;
            // A row is high risk when its decision logit, score minus cut, is above 0: the argmax's rule.
            const flaggedEnc = bt ? bt.points.filter((q) => q.enc > 0).length : null;
            const flaggedPlain = bt ? bt.points.filter((q) => q.plain > 0).length : null;
            if (bt) {
              metric(stage, "prs_panel_flagged_encrypted", flaggedEnc);
              metric(stage, "prs_panel_flagged_plaintext", flaggedPlain);
            }
            emit("log", stage, `full polygenic score under CKKS. Encrypted: this person's ${real.featDim} `
              + `effect-allele dosages, together in one CKKS ciphertext (${real.featDim} of its ${slots} `
              + `slots, the rest zero). In plaintext: the ${real.featDim} effect weights, applied by `
              + `multiply-plain (${muls} multiply-plain in all, the output mask's included; class 0's `
              + `all-zero row adds none); ${rot} rotate-and-add steps`
              + `${rot > 0 ? ` (1 to ${2 ** (rot - 1)})` : ""} `
              + `then sum the products into slot 0. Decrypted: the result ciphertext, read at slot 0, `
              + `which holds the score ${encScore.toFixed(6)} (in the clear ${plainScore.toFixed(6)}, `
              + `|difference| ${scoreErr.toExponential(3)}); every other slot is zeroed by the output mask and `
              + `flooded (+/-2^${CKKS_FLOOD_BITS}) before decryption, so it shows none of the partial sums. The cut ${(-bCut).toFixed(4)} is the `
              + `${realIn.metrics?.prs_high_risk_percentile}th percentile of the plaintext score over the `
              + `${realIn.metrics?.prs_cut_population} bundled 1000 Genomes individuals, this one included `
              + `(linear interpolation between order statistics). Score minus cut, beside a class-0 logit `
              + `of 0, goes to the garbled-circuit argmax: high risk = ${real.prediction} (plaintext: `
              + `${plainFlag}).`
              + (bt ? ` Over all ${bt.rows} bundled individuals, in ${bt.ciphertexts} ciphertext(s): the `
                + `flag agrees on ${bt.rows - bt.mismatches} of ${bt.rows} (${flaggedEnc} above the cut `
                + `encrypted, ${flaggedPlain} in plaintext), max |decrypted - plaintext score| `
                + `${bt.maxAbsError.toExponential(3)}.` : ""));
          }
          const ho = held ? await emitHeldOutAgreement(stage, held) : null;
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
            // Not when the held-out figure ran either: its `encrypted_agreement` is over every
            // held-out row, and this one-row 0-or-1 would overwrite it under the same key.
            if (!real.batch && !ho) {
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
              + (!(real.maxAbsError < 1)
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
            // On genomics that division read 158 ms per rotation against a measured ~3 ms while its
            // scale was 33 (1 since Round 84). `fhe_linear_ms` is a projection by contract, so the
            // measured rotation time is the one to divide.
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
          // The bound it is reported against is an ASSUMPTION, not a cited requirement: an upper end of
          // 10 s for interactive use. The code tests only `<= 10 s`, so a 0.13 s run is below any
          // "1-10 s window", not inside it, and the log says "within 10 s" rather than naming one.
          {
            const measuredMs = real.fheMs + real.mpcMs;
            metric(stage, "encrypted_tier_measured_ms", r2(measuredMs));
            const withinWindow = measuredMs <= 10_000;
            metric(stage, "encrypted_tier_within_operational_window", withinWindow ? 1 : 0);
            // Round 86: the CKKS time is stated on its own, because the 2PC half runs over scores the client has
            // already decrypted and so measures a protocol cost, not a protection (see `secureArgmax` above).
            emit("log", stage, `measured encrypted-tier latency, all parts together: ${r2(measuredMs)} ms `
              + `(CKKS linear${real.conv ? " + convolution" : ""} ${r2(real.fheMs)} ms, the encrypted part on its own; `
              + `garbled-circuit 2PC argmax ${r2(real.mpcMs)} ms over the scores the client has already decrypted, `
              + `so that part is the protocol's cost and hides nothing here), on WASM CPU in this tab. `
              + (withinWindow
                ? "Within 10 s, the upper end assumed here for interactive use (an assumption, not a cited requirement)."
                : "Over 10 s, the upper end assumed here for interactive use (an assumption, not a cited "
                  + "requirement), so too slow for that use.")
              // The scaled `total` note only where a scale applies: the model-size and workload selectors are
              // gone, so at x1 it described a panel and a choice this page no longer has (Round 86).
              + (bscale !== 1
                ? " Note this is NOT the benchmark's `total`, which scales the same measurement by "
                  + `${r2(bscale)}x for the configured model size.`
                : ""));
          }
          metric(stage, "mpc_nonlinear_ms", r2(real.mpcMs * bscale));
          metric(stage, "mpc_comm_kb", r2((real.commBytes * bscale) / 1024));
          // The measured halves of those two, under their own names, as `fhe_linear_measured_ms` already is (PF3, Round 86).
          if (bscale !== 1) {
            metric(stage, "mpc_nonlinear_measured_ms", r2(real.mpcMs));
            metric(stage, "mpc_comm_measured_kb", r2(real.commBytes / 1024));
          }
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
            // the convolution.
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
            // Which weights drove this. Only the published ones can since Round 84: without them the
            // student does not run (the branch below), where a stated kernel pattern used to.
            metric(stage, "fhe_weights", "published student (HF, pinned revision)");
            metric(stage, "fhe_total_rotations", real.conv.totalRotations);
            if (real.conv.studentParams != null) metric(stage, "fhe_student_params", real.conv.studentParams);
            if (real.conv.studentPrediction != null) metric(stage, "fhe_student_class", real.conv.studentPrediction);
            // What the CNN was given, beside its class (Round 86, review PI-10): this page's tabular row
            // tiled to fill the image, with the student's pixel preprocessing not applied.
            if (real.conv.studentPrediction != null) {
              metric(stage, "fhe_student_input", "this page's tabular row tiled to fill the image; not an image");
            }
            emit("log", stage, `private inference (real in-browser CKKS, leveled with no bootstrapping, the PUBLISHED student's own `
              + `weights fetched from its pinned HF revision, ${real.conv.studentParams} params): `
              // WHICH class: `real.prediction` is the garbled-circuit argmax over the page's bundled
              // linear model (the WDBC logistic regression), not the CNN's; the CNN's own argmax is
              // taken in the clear (`fhe_student_class`). And what that argmax receives: logits the
              // client has already decrypted (Round 84).
              + `reported class=${real.prediction}, from the garbled-circuit argmax over this page's bundled `
              + `linear model's logits, which the client has already decrypted, so in this tab it hides `
              + `nothing from the key holder; not from the CNN. CKKS evaluates each of the student's `
              + `layers with its bias added under encryption and every slot but the layer's outputs `
              + `masked and flooded, so the client decrypts each layer's outputs and nothing else; it `
              + `decrypts after every layer, applies ReLU and average pooling in plaintext, and takes `
              + `the CNN's own argmax in the clear. Its input is this page's tabular row tiled to fill `
              + `the image, not an image, so the CNN's class is not a diagnosis. `
              + `${real.conv.channels}x3x3 kernels over a `
              + `${real.conv.outH}x${real.conv.outW} output, ${real.conv.patchesPerCiphertext} im2col `
              + `patches packed into one ciphertext, ${real.conv.rotations} rotations and `
              + `${real.conv.multiplies} multiply-plain (kernels and output masks) in `
              + `${r2(real.conv.convOnlyMs)} ms`
              // EVERY TERM SCOPED, because this sentence carried the withdrawn attribution after
              // the metrics above it were fixed. It read "<conv rotations> ... in
              // <the WHOLE student's ms>" - the same numerator/denominator mismatch that produced
              // the 10% figure - and then labelled `real.fheMs - real.conv.convMs`, which is the
              // page's own 8-feature bundled linear demo at ~23 ms, as "dense layer" while the
              // student's actual dense stack cost ~490 ms. A visitor reading the event log got the
              // retracted story back on the same page whose table had just been corrected.
              + (real.conv.denseRotations
                ? `; the student's ${real.conv.denseRotations} dense-layer rotations, each layer's rows `
                  + `packed into one ciphertext, take ${r2(real.conv.denseMs ?? 0)} ms more`
                : "")
              + `. This page's own bundled linear model adds `
              + `${r2(real.fheMs - real.conv.convMs)} ms, MPC ${r2(real.mpcMs)} ms. `
              + `Nothing here is scaled by a cost model.`);
          } else {
            // The medical student was asked for and did not run: say so, and what the figures are.
            if (real.studentNotRun) {
              metric(stage, "fhe_weights", "not fetched: the encrypted student did not run");
              emit("log", stage, `the published medical student's encrypted inference did NOT run: `
                + `${real.studentNotRun}. No kernel pattern stands in for its weights, so the CKKS and `
                + `MPC figures below are the bundled Wisconsin regression's alone.`);
            }
            // Round 86: "REAL in-browser FHE, medium/linear" and "x1 arch scale" named selectors that were removed
            // and called leveled CKKS fully homomorphic; the scale is printed only where it is not 1.
            emit("log", stage, `private inference (real in-browser CKKS, leveled with no bootstrapping): `
              + `class=${real.prediction}; node-seal CKKS ${r2(real.fheMs * bscale)} ms + garbled-circuit 2PC ${r2(real.mpcMs * bscale)} ms `
              + (bscale !== 1
                ? `(the cost-model scale x${r1(bscale)}; measured ${r2(real.fheMs)}/${r2(real.mpcMs)} ms, ct=${real.ctBytes}B), `
                : `(measured; ct=${real.ctBytes}B), `)
              // "plaintext reference", NOT "ONNX". `clientMs` times `privateInference`, which has had
              // no ONNX tier since Round 17 and says so in its own comments; its body is two plain-JS
              // dot products. The genuine ONNX latency (`published_inference_ms`) lands in the SAME
              // stage log, so this line carried two ONNX-labelled numbers differing by orders of
              // magnitude. The sibling emit fifteen lines below already says "client compute".
              + `plaintext reference ${clientMs} ms`);
          }
          // The garbled circuit's answer, checked against the same argmax in the clear. Only this tab
          // can run that check, because it holds both parties' shares; across two hosts nobody could.
          // A disagreement is a broken integration, and it is reported as one.
          const mpcAgrees = real.prediction === real.mpcClearIndex;
          metric(stage, "mpc_matches_clear", mpcAgrees ? 1 : 0);
          // Against the FLOAT argmax too (Round 86, review PI-4): the self-test above shares the
          // circuit's quantisation, so a near-tie the 2^-20 step decided would pass it unseen.
          const floatAgrees = real.prediction === real.mpcFloatIndex;
          metric(stage, "mpc_matches_float_argmax", floatAgrees ? 1 : 0);
          if (!floatAgrees) emit("log", stage, `The circuit's class ${real.prediction} differs from the float argmax `
            + `${real.mpcFloatIndex}: two logits within the quantisation step tied, and the tie went to the `
            + `lower index.`);
          emit("log", stage, mpcAgrees
            ? `garbled-circuit argmax checked in the clear: index ${real.prediction} both ways`
            : `garbled-circuit argmax DISAGREES with the same argmax in the clear: ${real.prediction} `
              + `against ${real.mpcClearIndex}. Treat this run's MPC result as unverified.`);
          if (usecase === "medical_fl" && config.mpc_cnn) await runStudentUnderMpc(stage, usecase, seed, emit, metric, artifacts);
          if (usecase === "finance_aml" && config.mpc_mlp) await runMlpUnderMpc(stage, seed, emit, metric, artifacts);
        } else {
          // Honest last resort, and it names the half that failed: node-seal not loading (no WASM,
          // say), or the garbled-circuit protocol not completing, which includes it ABORTING on a
          // failed check. Until Round 68 both were reported as the first.
          artifacts.fhe_prediction = inf.prediction;
          // `privateInference` scored the same weights the bars explain: `realIn.model`, or the
          // placeholder pair `linearShap` is given when no model ships.
          ckksClassIsTheExplainedModels = true;
          const representative = mpcFheRow(usecase, config.workload ?? "linear", BENCH_MODEL_SIZE, null, secFromConfig(config, metrics));
          const cause = real.failed === "mpc"
            ? `the garbled-circuit MPC did not run to completion (${real.reason})`
            : `the in-browser CKKS computation failed (${real.reason})`;
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
          metric(stage, "encrypted_tier", `failed: ${cause}`);
          metric(stage, "fhe_max_abs_error_representative", costPerOp().demo_max_abs_error);
          // The same rule for the split: the measured branch fills `fhe_linear_ms`, `mpc_nonlinear_ms`
          // and `mpc_comm_kb` from a real run, and until Round 68 this branch filled the same three
          // keys from the cost model.
          metric(stage, "fhe_linear_ms_representative", representative.fhe_linear_ms);
          metric(stage, "mpc_nonlinear_ms_representative", representative.mpc_nonlinear_ms);
          metric(stage, "mpc_comm_kb_representative", representative.mpc_comm_kb);
          emit("log", stage, `private inference: class=${inf.prediction}; ${cause} - representative split `
            + `${representative.fhe_linear_ms}/${representative.mpc_nonlinear_ms} ms, client compute ${clientMs} ms`);
        }

        // Inference backend selection (config.inference): "qcnn" routes the reported class through the
        // TRAINED QCNN, which runs UNENCRYPTED on the plaintext features (no QOTP, no homomorphic step);
        // "ckks"/default keeps the CKKS(FHE)+MPC path. The CKKS run happens either way - this selects
        // which one is surfaced as the primary prediction (genuinely changes it).
        const inferBackend = String(config.inference || "ckks");
        if (inferBackend === "qcnn" || inferBackend === "qcnn-ckks") {
          const nq = qubitsOf(config);
          // TRAINED variational circuit where weights exist for this use case; the untrained
          // Clifford+T path is not a predictor. Measured: over 24 vectors x 8 seeds its class flipped
          // for 100% of inputs, mean |<Z0>| 0.076, correlation with mean(feature) -0.070. It stays as
          // the QOTP pad-structure + T-count demonstration, which is what it can honestly support.
          const w = await loadQcnnWeights(usecase, qubitsOf(config));
          // THE SPACE THE ARTIFACT DECLARES, not whatever `feats` happens to be in. `feats` is
          // `realIn.values`, which is StandardScaler output because the CKKS tier and the SHAP
          // panel are fitted there. A QCNN artifact declaring `input_space: "raw"` standardises
          // its input again, so handing it `feats` charged the transform twice and flattened the
          // wire: measured at tanh sd 3.3e-05 on `amount`, with 101 of 400 rows - 25.25% - given
          // a different class than the same rows correctly fed. (17.25% appeared here for one
          // round; that figure is a different comparison - two encoders over one array, rather
          // than one encoder over two - and describes a fix that was not the one made.)
          //
          // BY NAME, with no default branch. `w.input_space` names the units the circuit was
          // trained on; the sample ships its rows keyed by the space each is in. A boolean
          // `=== "raw" ? a : b` would send an artifact retrained into a third space to whichever
          // array the else-branch happens to name - which is exactly how this shipped. `feats` is
          // used only when no real sample exists at all; a sample that lacks the declared space is
          // REFUSED below rather than fed (Round 84: the `?? feats` that followed this lookup fed
          // z-scores to medical's raw-input model while its bundle shipped no raw rows). The pairing
          // is asserted in tests/test_the_encoder_receives_the_space_the_artifact_declares.py.
          const qcnnFeats = w ? (realIn?.bySpace?.[w.input_space] ?? (realIn ? undefined : feats)) : undefined;
          // AS WIDE AS THE ARTIFACT READS, or refused (Round 84). The row goes in by position and is
          // sliced to the wire count, so a bundled row of another width - the eight-column finance
          // block beside the four-feature retrain - would hand the circuit its first columns, which
          // are not the ones its `feature_names` lists.
          const qcnnWidth = w ? (w.n_features ?? w.n_qubits) : 0;
          const qcnnMisfit = !!(w && qcnnFeats && realIn && qcnnFeats.length !== qcnnWidth);
          if (w && (!qcnnFeats || qcnnMisfit)) {
            const cls = artifacts.fhe_prediction ?? inf.prediction;
            artifacts.primary_inference = { backend: "ckks", class: cls };
            metric(stage, "primary_inference_class", cls);
            metric(stage, "primary_inference_backend", "ckks");
            emit("log", stage, qcnnMisfit
              ? `QCNN selected, but the bundled sample's ${w.input_space} row has ${qcnnFeats!.length} `
                + `values and the trained QCNN reads ${qcnnWidth}; not running it. Reporting the CKKS class.`
              : `QCNN selected, but the bundled sample ships no rows in the space the `
                + `trained QCNN declares (${w.input_space}); not running it. Reporting the CKKS class.`);
          } else if (w && qcnnFeats) {
            // MEASURED, on the same footing as the CKKS tier's `fhe_ms_per_rotation`. The QCNN was
            // reported by accuracy alone while every encrypted figure carried a latency, so the two
            // inference paths on one page could not be compared on cost at all. State-vector
            // simulation is 2^n work in the qubit count, which is the number that makes the figure
            // interpretable, so it travels with it - as `PF2` requires of the encrypted tier.
            const t0 = performance.now();
            const qc = qcnnInferTrained(qcnnFeats.slice(0, w.n_qubits), w);
            const qcnnMs = performance.now() - t0;
            artifacts.primary_inference = { backend: "qcnn-trained", class: qc.cls,
                                            qubits: qc.qubits, expectation: r2(qc.expectation) };
            if (!(inferBackend === "qcnn-ckks" && await surfaceQcnnUnderCkks(stage, usecase, w, qcnnFeats.slice(0, w.n_qubits), emit, metric, artifacts))) {
              metric(stage, "primary_inference_class", qc.cls);
              metric(stage, "primary_inference_backend", "qcnn-trained");
            }
            metric(stage, "qcnn_expectation_z0", r2(qc.expectation));
            metric(stage, "qcnn_inference_ms", r2(qcnnMs));
            metric(stage, "qcnn_qubits", qc.qubits);
            // Report the metric the CORPUS warrants, not always accuracy. At finance's 5% fraud
            // rate accuracy reads 0.955 against a 0.948 majority baseline and says almost nothing,
            // while minority-class F1 is 0.229 against an all-positive 0.098. The trainer decides
            // which is the headline and ships it in `headline_metric`.
            // Three regimes, matching `headline_metric` in the artifact. The multiclass arm reads
            // the uniform-random floor rather than the constant-predictor one: it is the higher of
            // the two the trainer ships, so the comparison is against the harder baseline.
            const head = w.headline_metric;
            const macroLed = head === "macro_f1", f1Led = head === "minority_class_f1";
            const recordedMine = macroLed ? w.test_macro_f1 : f1Led ? w.test_minority_f1 : w.test_accuracy;
            // RE-SCORED IN THIS TAB since Round 86, by the same function as `qml_ids`; the baseline and logistic
            // regression are still read from the weights file.
            const rescore = await rescoreQcnnInTab(stage, usecase, w, metric, artifacts);
            const mine = rescore ? Math.round(rescore.headline * 1e6) / 1e6 : recordedMine;
            const theirs = macroLed ? w.classical_lr_macro_f1
                         : f1Led ? w.classical_lr_minority_f1 : w.classical_lr_accuracy;
            const base = macroLed ? w.macro_f1_floors?.uniform_random
                       : f1Led ? w.all_positive_f1 : w.majority_baseline;
            const label = macroLed ? "macro F1" : f1Led ? "minority-class F1" : "accuracy";
            // COMPUTED, like its twin in `qml_ids`. "the classical model is ahead" was unconditional
            // text over two numbers read from a file, so a retrained artifact that reversed the
            // ordering would have kept asserting it.
            // THREE-WAY (Round 84): medical's circuit TIES logistic regression at the shipped seed
            // (0.953488 each, the same class on all 86 held-out rows), which the two-way form
            // printed as "the circuit is ahead here". LR's five-seed lead is on the model card.
            const ahead = typeof mine === "number" && typeof theirs === "number"
              ? (theirs > mine ? "the classical model is ahead"
                : theirs === mine ? "the circuit and logistic regression are level on this split"
                : "the circuit is ahead here")
              : "no comparator ships in this artifact, so no verdict is stated";
            // `w.theta.length`, NOT `layers x n_qubits`. That product is the RETIRED
            // hardware-efficient ansatz's parameter count - one angle per wire per layer - and it
            // survived on the visitor-facing log line after Round 46 replaced the circuit with a
            // shared-filter QCNN. It rendered "3x8 trained angles" = 24 for artifacts carrying 77,
            // and "6x8" = 48 for telecom's 57: the only architecture description on the page,
            // describing a circuit the page does not run. The count is now read from the artifact,
            // so it cannot disagree with the weights it is printed beside.
            emit("log", stage, `inference backend = QCNN (TRAINED, ${inferBackend === "qcnn-ckks" ? "this run on the plaintext features is the self-check of the CKKS evaluation reported above" : "NOT encrypted"}): class=${qc.cls} `
              + `(<Z0>=${qc.expectation.toFixed(4)}, ${w.theta.length} trained angles - shared conv filters `
              + `plus pooling and final rotations - over ${w.n_qubits} wires, ${w.layers} conv rounds per stage, `
              + `simulated exactly on a state vector; ${qcnnSimulability(w)}). `
              // Round 84 named which figures were recorded rather than measured; since Round 86 the circuit's is
              // measured here and the sentence says so, and the baseline's and logistic regression's still come from the file.
              + `Held-out ${label} ${(mine ?? 0).toFixed(4)} against a baseline of `
              + `${(base ?? 0).toFixed(4)} and logistic regression at ${(theirs ?? 0).toFixed(4)} on `
              + `the same split - ${ahead}. The circuit's figure was ${rescoreSentence(rescore, recordedMine)}; the `
              + `baseline and logistic regression were recorded at training time. Test rows were never used to pick the checkpoint.`);
          } else {
            // No trained weights for this use case: refuse to surface a class rather than pass off
            // the untrained circuit's coin flip as a prediction.
            const qc = qotpQcnnInfer(feats.slice(0, nq), seed, nq);
            artifacts.primary_inference = { backend: "ckks", class: artifacts.fhe_prediction ?? inf.prediction };
            metric(stage, "primary_inference_class", artifacts.fhe_prediction ?? inf.prediction);
            metric(stage, "primary_inference_backend", "ckks");
            emit("log", stage, `QCNN selected but no trained weights ship for ${usecase}: the `
              + `untrained Clifford+T circuit is a pad-structure and T-count demonstration `
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
        // What each bar's feature is, when the payload says (PP14): the bars printed bare AML-style names for PaySim columns.
        artifacts.explanation_feature_definitions = realIn?.definitions ?? null;
        // FROM THE BACKEND THAT SURFACED THE CLASS (Round 84). Round 83 left this `false` on every
        // path, so on finance, medical and genomics, where CKKS evaluates the same bundled model
        // explained here and reports its argmax, the panel called the class "a different
        // computation". It is false when the trained QCNN surfaced the class, or when the encryption
        // evaluated the demo pattern rather than the explained model.
        artifacts.explanation_is_of_reported_class =
          artifacts.primary_inference?.backend === "ckks" && ckksClassIsTheExplainedModels;
        metric(stage, "explanation_model", artifacts.explanation_provenance);
        metric(stage, "explanation_top_contribution", artifacts.explanation.top_features[0]?.contribution ?? 0);
        {
          // With its sign since Round 86: the name alone read as a reason FOR the reported class even where
          // the contribution pushes the other way (finance's new_payee, -2.08, toward legitimate).
          const top = artifacts.explanation.top_features[0];
          const c = Number(top?.contribution ?? 0);
          emit("log", stage, `XAI: top driver = ${top?.feature} (contribution ${c >= 0 ? "+" : ""}${r2(c)}, `
            + `${c >= 0 ? "toward class 1" : "toward class 0"})`);
        }
        break;
      }
      case "lending_market": {
        const { simulate, withRunParams } = await import("./lending/model");
        const { runLendingTasks } = await import("./lending/pool");
        const { summariseSweep } = await import("./lending/report");
        const base = lendingCfg();
        const cfg = withRunParams(base, config);
        const started = performance.now();
        const run = simulate(cfg, seed);
        const simulateMs = performance.now() - started;
        const repaid = run.events.reduce((sum, e) => sum + e.repaid, 0);
        const bobEvents = run.events.filter((e) => e.borrower === 0).length;
        metric(stage, "lending_hours", run.hours);
        metric(stage, "lending_borrowers", run.borrowers);
        metric(stage, "lending_liquidations", run.events.length);
        metric(stage, "lending_repaid", Math.round(repaid));
        metric(stage, "lending_bad_debt", Math.round(run.badDebt));
        metric(stage, "lending_peak_utilisation", Math.max(...run.utilisation));
        metric(stage, "lending_peak_borrow_rate", Math.max(...run.borrowRate));
        metric(stage, "lending_bob_min_health", Math.min(...run.bobHealth));
        metric(stage, "lending_bob_liquidations", bobEvents);
        metric(stage, "lending_books_gap", run.maxAccountingGap);
        emit("log", stage, `market simulated in this tab: ${run.hours} hours, ${run.borrowers} borrowers, generated prices with one stated fall `
          + `of ${(cfg.shock_fraction * 100).toFixed(0)}% at hour ${cfg.shock_hour}; ${run.events.length} liquidations, ${bobEvents} of them Bob's. No chain is involved.`);
        // The sweep moves only volatility, from the configured one, so this run's multiplier does not move the curve's axis.
        const sweepCfg = { ...cfg, annual_vol: base.annual_vol };
        const seeds = Array.from({ length: cfg.sweep_seeds }, (_, i) => seed + i);
        artifacts.lending = { cfg, run, simulateMs } satisfies LendingMarketReport;
        try {
          const pool = await runLendingTasks(cfg.sweep_multipliers.map((multiplier) => ({ kind: "sweep" as const, cfg: sweepCfg, multiplier, seeds })), {}, cfg.max_workers);
          artifacts.lending_sweep = summariseSweep(pool);
          metric(stage, "lending_sweep_paths", cfg.sweep_multipliers.length * cfg.sweep_seeds);
          metric(stage, "lending_sweep_workers", pool.workers);
          emit("log", stage, `volatility sweep: ${cfg.sweep_multipliers.length} levels of ${cfg.sweep_seeds} price paths, in ${pool.ranIn}, ${pool.wallMs.toFixed(0)} ms.`);
        } catch (e) {
          // The month above is intact, so the failure is the sweep's alone and the private check still runs.
          emit("log", stage, `the volatility sweep did not finish: ${String(e)}`);
        }
        break;
      }
      case "private_liquidation": {
        const market = artifacts.lending as LendingMarketReport | undefined;
        if (!market) throw new Error("the private check needs the market's checkpoints, and the market stage produced none");
        const sessionKey = (config as any)._sessionKey as Uint8Array | undefined;
        if (!sessionKey || sessionKey.length === 0) throw new Error("the private check is sealed under the PQC session key, and there is no unsealed mode");
        const { runLendingTasks } = await import("./lending/pool");
        const { summariseChecks } = await import("./lending/report");
        const inputs = market.run.checkpoints;
        const sig = String(config.sig ?? "ml-dsa-65");
        const custody = market.cfg.custody;
        let init: TaskInit = { sessionKey, sig, cfg: market.cfg, custody, kem: kemName(config) };
        let setup: KeyholderSetup | undefined;
        if (custody === "keyholder") {
          // Carol's keys, Bob's one submission and the pool-Carol handshake, once, on one worker, before any check.
          const first = await runLendingTasks([{ kind: "setup", input: inputs[0] }], init, 1);
          const result = first.results[0];
          if (result.kind !== "setup") throw new Error("the key holder's setup returned something else");
          setup = result.setup;
          init = { ...init, keyholder: setup };
          emit("log", stage, `Carol generated the CKKS keys; Bob encrypted his balances once under her public key and signed them (${(setup.submission.length / 1e6).toFixed(2)} MB, sealed to the pool), and takes no further part.`);
        }
        emit("log", stage, `checking Bob's position at ${inputs.length} hours: the margin under CKKS, its sign in a garbled circuit, off the page's thread. `
          + (custody === "keyholder" ? "Carol holds the key, and prices and thresholds are public." : "Bob holds the key, and the pool's thresholds are its own."));
        const pool = await runLendingTasks(inputs.map((input) => ({ kind: "check" as const, input })), init, market.cfg.max_workers);
        const report = summariseChecks(pool, sigImpl(sig).name, market.cfg.margin_mask_bits, custody, setup);
        artifacts.lending_private = report;
        metric(stage, "lending_custody", custody);
        metric(stage, "lending_bob_submissions", report.bobSubmissions);
        if (report.bitSignaturesVerified != null) metric(stage, "lending_bit_signatures_verified", report.bitSignaturesVerified ? 1 : 0);
        if (report.setup) metric(stage, "lending_link_agreed", report.setup.linkAgreed ? 1 : 0);
        metric(stage, "lending_checks", report.outcomes.length);
        metric(stage, "lending_checks_agree", report.allAgree ? 1 : 0);
        metric(stage, "lending_flagged", report.flagged);
        metric(stage, "lending_margin_max_error", report.maxAbsError * 1000);
        metric(stage, "lending_statistical_distance_log2", report.statisticalDistanceLog2);
        metric(stage, "lending_ckks_mb", +(report.ckksBytes / 1e6).toFixed(2));
        metric(stage, "lending_key_mb", +(report.keyBytes / 1e6).toFixed(2));
        metric(stage, "lending_gc_mb", +(report.gcBytes / 1e6).toFixed(2));
        metric(stage, "lending_and_gates", report.andGates);
        metric(stage, "lending_check_ms", Math.round(report.pool.wallMs));
        metric(stage, "lending_workers", report.pool.workers);
        metric(stage, "lending_speedup", +report.pool.speedUp.toFixed(2));
        emit("log", stage, `${report.outcomes.length} checks in ${report.pool.ranIn}, ${report.pool.wallMs.toFixed(0)} ms: ${report.flagged} could be liquidated, `
          + `and ${report.allAgree ? "every" : "NOT every"} bit matches the check in the clear (a self-test only this tab can make).`);
        break;
      }
      case "quantum_comparison": {
        // run the QSMPC-FHE comparison circuit - an UNTRAINED QOTP + Clifford+T circuit on random
        // input, not a QCNN readout and not an inference - for the benchmark row; no quantum
        // advantage is derived from it. Mirrors the backend QuantumComparisonStage.
        const ms = BENCH_MODEL_SIZE, wl = config.workload ?? "linear";
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
        // `archDims` takes the `studentParams > 0` branch and ignores `modelSize` - and it was 33 for
        // genomics, which fell through to a tier-architecture MLP until Round 84 priced its weighted
        // sum instead. So the diagram bar read 10/10 on the genomics page while the benchmark panel two
        // cards down read about 9.1 from the same run. One quantity, two values, both on screen.
        const qrow = qsmpcFheRow(wl, ms, rng(7), benchCfg().quantum_samples, benchScale(usecase, ms, wl), qubitsOf(config), sec);
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
        emit("log", stage, `QSMPC-FHE comparison (untrained QOTP + Clifford+T circuit on random input, `
          + `${qrow.t_gate_count} T-gates on ${qrow.qubits} qubits, counted from the circuit): it predicts `
          + "nothing, the pad is never undone, and no advantage score is derived");
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
        // record the UI advertises. Canonicalised with sorted keys so the
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
          // `fhe`/`mpc` record the in-browser BACKENDS THIS RUN EXECUTED (Round 84), not what the
          // config carries: CKKS when a CKKS evaluation reported (`fhe_model` from the encrypted
          // hybrid, `encrypted_rows_scored` from a batch), the garbled-circuit argmax when it was
          // checked against the clear one (`mpc_matches_clear`), else null. Until Round 84 both were
          // named on every use case, including telecom and satellite, which run a CKKS check of a student
          // and no MPC. In the client-only demo the
          // selects are replaced by a badge reading "node-seal CKKS + garbled-circuit 2PC", while the
          // state still held the server-side defaults `tenseal-cpu` and
          // `mpyc-shamir` - nobody chose them and neither executed. `config.fhe` and `config.mpc`
          // are read NOWHERE else in this file, so the only consumer was this payload: a
          // tamper-evident record naming two backends that did not run, ten lines under a badge
          // saying so. A signed record of the wrong thing is worse than an unsigned one, because
          // the signature is what invites a reader to trust it.
          config: { kem: config.kem ?? null, qkd: config.qkd ?? null, sig: config.sig ?? null,
                    fhe: isClientDemo()
                      ? (metrics.fhe_model != null || metrics.encrypted_rows_scored != null ? "node-seal-ckks" : null)
                      : (config.fhe ?? null),
                    mpc: isClientDemo()
                      ? (metrics.mpc_matches_clear != null ? "polytune-garbled-circuit-2pc-wasm" : null)
                      : (config.mpc ?? null),
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
          // signature -- the keypair is generated per run and discarded. Both are recorded now so a
          // reader can re-check it. A per-run key shows the payload is unchanged since signing, not who signed it.
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
