// Per-use-case TWO-PARTY (Alice↔Bob) MPC-FHE secret-sharing sequence diagram — driven LIVE by the
// running pipeline. Each party sits on its own LAN; they collaborate over a WAN without a trusted
// third party and without sharing raw data. A step lights up when its stageId is the pipeline's
// active stage and is marked done once the stage completes — so Start/Stop/Restart/Pause/Step on the
// page drive this diagram as a simulator. The MPC-FHE private inference is shown as its two halves:
// FHE-linear (a CKKS ciphertext crosses the WAN) + MPC-non-linear (secret-shares exchanged for
// ReLU/argmax). A final full-width "QSMPC-FHE comparison" bar shows the quantum-secure path that the
// benchmark evaluates against (the `quantum_comparison` stage), so the advantage/security numbers are
// clearly produced by an actual run. Grounded in references/ (Zhang QOT npj 2026, Li&Deng QHE-FL,
// Sutradhar QSMS, EncFormer linear-FHE/non-linear-MPC split). PoC runs LAN/local.

type Kind = "local-a" | "local-b" | "local-wide" | "q-arrow" | "c-arrow" | "c-bidir"
  | "fhe-arrow" | "mpc-bidir" | "compare";
type M = Record<string, any>;
/** The crypto-agility selections for the run being drawn.
 *
 *  `qkd` and `qkdProtocol` are different things, and conflating them is what kept the E91 selection
 *  invisible. `qkd` is the crypto-agility FAMILY and only ever holds `bb84-decoy` or `pqc-only` - it
 *  answers "is there a quantum channel at all". `qkdProtocol` is the Parameters-panel choice
 *  (`bb84` | `e91`) and is what `clientCompute` actually branches on. The diagram was handed only the
 *  family, so `qkdName()`'s `e91` branch below was unreachable: the run emitted `chsh_s`, which only
 *  E91 produces, while the arrow above it still read "BB84 decoy-state". */
export interface DiagramConfig {
  kem?: string; qkd?: string; sig?: string;
  qkdProtocol?: string;
  /** `psi` | `sum`, or absent for the use case's default. The QSMPC step reads this. */
  qsmpcOp?: string;
  /** Whether the run tuned the channel with OptiQKD. The checkbox genuinely changes the run -
   *  it emits a measured SKR improvement and a convergence count - and the diagram used to show
   *  nothing at all, so ticking it appeared to do nothing to the architecture it describes. */
  optiqkd?: boolean;
  /** `linear` | `cnn`. On medical this selects a real encrypted CONVOLUTION over a packed
   *  ciphertext instead of a dot product, which is a different computation crossing the same
   *  arrow - so the arrow has to say which one it carried. */
  workload?: string;
  /** `` | `ckks` | `qcnn`. Which inference backend the run surfaces as primary. */
  inference?: string;
}
interface Step {
  stageId: string;
  /** Fixed, or derived from the config when the operation itself changes the channel's nature. */
  kind: Kind | ((c: DiagramConfig) => Kind);
  label: (m: M, c: DiagramConfig) => string;
  /** Drawn only when this returns false. Used to drop the quantum arrow under a PQC-only profile. */
  hidden?: (c: DiagramConfig) => boolean;
  /**
   * Did the step SUCCEED, as distinct from having finished? The completion tick means only that the
   * stage ran without throwing, but it renders immediately after a measured value, so a reader reads
   * it as a verdict on that value. With an eavesdropper at 0.35 the BB84 row drew
   * "QBER 0.1 ✓" while the same page's posture table reported 0 secure bits - the diagram and the
   * table disagreed about one run. Return false and the step is marked as completed-but-failed.
   */
  ok?: (m: M, c: DiagramConfig) => boolean;
}
interface Spec { a: { name: string; sub: string }; b: { name: string; sub: string }; steps: Step[] }

// The protocol names used to be hardcoded here, so selecting ml-kem-1024 still drew "ML-KEM-768",
// selecting e91 still drew "BB84", and selecting pqc-only still drew a quantum arrow with an
// em-dash QBER. The diagram is the most-looked-at part of the page; it must show what actually ran.
const KEM_LABEL: Record<string, string> = {
  "ml-kem-768+x25519": "ML-KEM-768 + X25519",
  "ml-kem-1024+x25519": "ML-KEM-1024 + X25519",
};
const kemName = (c: DiagramConfig) =>
  KEM_LABEL[String(c.kem ?? "")] ?? (c.kem ? String(c.kem).toUpperCase() : "hybrid KEM");
const sigName = (c: DiagramConfig) => (c.sig ? String(c.sig).toUpperCase() : "ML-DSA");
// The protocol wins over the family. `clientCompute` branches on `qkd_protocol`, so that is what the
// visitor actually ran; the family only decides whether a quantum channel exists at all.
const qkdName = (c: DiagramConfig) => {
  const p = String(c.qkdProtocol ?? "");
  if (p === "e91") return "E91 (entanglement / CHSH)";
  if (p === "bb84") return "BB84 decoy-state";
  const q = String(c.qkd ?? "");
  if (q.startsWith("bb84")) return q.includes("decoy") ? "BB84 decoy-state" : "BB84";
  return q || "QKD";
};
const pqcOnly = (c: DiagramConfig) => String(c.qkd ?? "") === "pqc-only";

/** The OptiQKD suffix for a QKD arrow: the measured improvement once a run has produced one,
 *  otherwise just that tuning is enabled. Reads the run's metrics rather than restating the
 *  checkbox, so the arrow cannot claim an optimisation that has not happened yet. */
/** The eavesdropper the run actually simulated, or nothing when there was none.
 *
 * Before the run there is no fraction to report, so this stays empty rather than asserting either
 * way - the arrow is dim at that point and the QBER beside it is an em dash. */
const eveTag = (m: M) => {
  // One key for both channels: only one QKD stage runs per use case, and the fibre and free-space
  // stages report the same quantity - the intercept-resend fraction that was simulated.
  //
  // Not named `f`: that is the metric formatter declared below, and shadowing it inside a label
  // helper is a trap for whoever next wants to format something here. `f` is used for the value so
  // a step-produced 0.30000000000000004 renders as 0.3.
  const eve = m["qkd_eve_fraction"];
  return eve == null || Number(eve) <= 0 ? "" : ` (w/ eavesdropper ${f(m, "qkd_eve_fraction")})`;
};

const optiTag = (m: M, c: DiagramConfig) => {
  if (!c.optiqkd) return "";
  const pct = m["qkd_skr_improvement_pct"];
  const evals = m["qkd_convergence_evals"];
  if (pct == null) return " · OptiQKD on";
  return ` · OptiQKD ${Number(pct) >= 0 ? "+" : ""}${pct}% surrogate SKR`
    + (evals == null ? "" : ` in ${evals} evals`);
};

// The QSMPC step. `clientCompute:896` resolves `config.qsmpc_op || (genomics ? "sum" : "psi")`, so
// the diagram has to resolve it the same way or it describes a computation that did not happen.
//
// The KIND was `q-arrow` for a secure sum, on the grounds that "a secure sum runs over a d-qudit GHZ
// state". That is true of the BACKEND and false of the browser, which is the only thing this diagram
// draws: `clientCompute` computes additive shares mod p and its own log line calls that "the classical
// equivalent of the backend's Shamir + d-level QFT over a GHZ state, which needs qudits no browser can
// prepare". `q-arrow` is styled in `--quantum` with the same dash and arrowhead as the QKD rows, and
// the legend labels that exact style "quantum (BB84 decoy-state)" - so the genomics page drew a
// classical modular sum as a quantum channel, and so did finance whenever Secure sum was selected.
// One line wide, two pages deep.
//
// `mpc-bidir` is the honest style: additive secret-sharing IS secret-share traffic, and the legend
// already names it. PSI stays classical-bidirectional - it is a WebCrypto keyed hash.
const qsmpcOpOf = (c: DiagramConfig, usecaseDefault: "psi" | "sum") =>
  (String(c.qsmpcOp ?? "") === "psi" || String(c.qsmpcOp ?? "") === "sum")
    ? (String(c.qsmpcOp) as "psi" | "sum")
    : usecaseDefault;
const qsmpcKind = (dflt: "psi" | "sum") => (c: DiagramConfig): Kind =>
  qsmpcOpOf(c, dflt) === "sum" ? "mpc-bidir" : "c-bidir";
const qsmpcLabel = (dflt: "psi" | "sum") => (m: M, c: DiagramConfig) =>
  qsmpcOpOf(c, dflt) === "sum"
    ? `secure sum over GF(${f(m, "qsms_prime")}) · pooled = ${f(m, "secure_sum")}`
    : `keyed-hash PSI · shared flagged = ${f(m, "psi_intersection_size")}`;

const f = (m: M, k: string, d = "—") => {
  const v = m[k];
  if (v == null) return d;
  if (typeof v === "number")
    return Math.abs(v) >= 1e-3 || v === 0 ? String(Number(v.toFixed(5))) : v.toExponential(2);
  return String(v);
};

/**
 * A key-agreement step succeeded only if it yielded key material. The QBER alone cannot say so: the
 * GLLP zero-key point measured in this repository is 0.098106, BELOW the textbook BB84 bound of
 * 0.11, so a QBER inside [0.098, 0.11) looks acceptable against the textbook figure while the
 * decoy-state estimate returns nothing. Ask the run what it produced instead of re-deriving it.
 */
const qkdOk = (m: M) => {
  const bits = m["secure_key_bits"];
  return typeof bits === "number" ? bits > 0 : true;   // no measurement yet -> do not claim failure
};

// The MPC-FHE private inference, drawn as its two halves (the 2026 hybrid: linear→FHE, non-linear→MPC).
// Both map to the backend `private_inference` stage, so they light up together when it runs and the
// labels fill from the live `fhe_linear_ms` / `mpc_nonlinear_ms` / `mpc_comm_kb` metrics.
// The FHE arrow names the computation it carried, not just "CKKS ct". Workload=cnn packs im2col
// patches into one ciphertext and reduces them by rotate-and-add; workload=linear is a packed dot
// product. Those are different operations over the same wire, and the run counts the rotations for
// each, so the arrow reports the count rather than leaving the selector invisible here.
const HYBRID_INFERENCE: Step[] = [
  { stageId: "private_inference", kind: "fhe-arrow",
    label: (m, c) => {
      // Total across every CKKS layer once the full student runs encrypted, not just
      // the conv block: 152 for this student against 3 for a lone dot product.
      const rot = m["fhe_total_rotations"] ?? m["fhe_conv_rotations"] ?? m["fhe_dense_rotations"];
      const shape = m["fhe_conv_output_shape"];
      // The medical page evaluates the published student end to end under CKKS; every other use
      // case runs its linear student as a packed dot product. Name whichever actually happened.
      const kindTxt = shape ? `encrypted CNN ${shape}` : "CKKS dot product";
      return `FHE-linear · ${kindTxt} · ${f(m, "fhe_linear_ms")} ms`
        + (rot == null ? "" : ` · ${rot} rotations`);
    } },
  { stageId: "private_inference", kind: "mpc-bidir",
    label: (m) => `MPC-non-linear · shares · ${f(m, "mpc_nonlinear_ms")} ms · ${f(m, "mpc_comm_kb")} KB` },
];

// The quantum-secure ALTERNATIVE the benchmark scores against (the `quantum_comparison` stage runs a
// real QOTP+QCNN private inference). Rendered as a full-width bar so it's clear QSMPC-FHE is executed
// and the quantum-advantage / security-score are produced by an actual run, not asserted.
// `inference` picks which backend the run SURFACES as primary (clientCompute reads it at the
// private_inference stage). Both are computed either way, so this bar says which one was surfaced
// rather than implying the selector did nothing.
const QSMPC_COMPARE: Step = {
  stageId: "quantum_comparison", kind: "compare",
  label: (m, c) => {
    // Read the OBSERVED backend, never the control. `config.inference` says what was asked for;
    // `primary_inference_backend` says what ran. Those differ whenever QCNN is selected on a use
    // case that ships no trained weights, and keying this label off the control printed "QCNN
    // surfaced as primary" for a class CKKS had produced. The comment below used to name that exact
    // defect class while the code above it committed the error.
    // "" and "ckks" are the SAME run - clientCompute does `config.inference || "ckks"`, so there is
    // no per-use-case default behind the empty value; the label says so rather than implying one.
    const ran = m["primary_inference_backend"];
    const primary = ran === "qcnn-trained" ? "trained QCNN surfaced as primary"
      : ran == null ? "inference not run"
      : String(c.inference ?? "") === "qcnn" ? "CKKS surfaced as primary (QCNN selected, no trained weights ship)"
      : String(c.inference ?? "") === "ckks" ? "CKKS surfaced as primary"
      : "CKKS surfaced as primary (default)";
    const cls = m["primary_inference_class"];
    return `QSMPC-FHE comparison · QOTP + QCNN (T-gates) · ${primary}`
      + (cls == null ? "" : ` · class ${cls}`)
      + ` · advantage ${f(m, "quantum_advantage_score")}/10`;
  },
};

/**
 * The published model, run in this browser, drawn on whichever stage carries it.
 *
 * Local to party A because it is not a protocol exchange: nothing crosses the wire for it. It is
 * here because the diagram is the most-looked-at part of the page and it was the one place a visitor
 * could not tell that the pinned model on Hugging Face had anything to do with the run. The Models
 * panel further down would execute it; the animation above never mentioned it.
 *
 * Four states, and they must not collapse into each other: not reached yet, ran, failed, and none
 * registered. The last two are the ones worth distinguishing - a use case with no published model is
 * a documented decision, while a published model that failed to run is a defect.
 */
const publishedStep = (stageId: string): Step => ({
  stageId,
  // Full width, not a party-sized box: this label is about 460 units long and a `local-a` box is
  // 300, so it used to spill past both edges and lose its first 17 characters off the viewBox.
  kind: "local-wide",
  label: (m) => {
    const status = m["published_model_status"];
    if (status === "unavailable") return "published model · none registered for this use case";
    const key = m["published_model"];
    if (status === "failed") return `published model ${key} · DID NOT RUN (no substitute shown)`;
    if (key == null) return "published model · —";
    // Forecasters report absolute error against the target their sample ships; classifiers a class.
    const mae = m["published_row_mae"];
    const cls = m["published_predicted_class"];
    const outcome = mae != null ? `abs err ${f(m, "published_row_mae")}`
      : cls != null ? `class ${cls}`
      : "";
    return `published ${key} @ ${f(m, "published_model_revision")} · ${f(m, "published_model_backend")}`
      + ` · ${f(m, "published_input_shape")} · ${f(m, "published_inference_ms")} ms`
      + (outcome ? ` · ${outcome}` : "");
  },
});

/**
 * Federated training, drawn from the aggregation the run REPORTS rather than a string kept here.
 *
 * This label used to read "federated train + pairwise masking (demo keys)". `clientQFL` generated no
 * masks, no pairwise seeds and no key material - it was a size-weighted plaintext mean, and the only
 * `mask` in `clientCompute.ts` belonged to the QSMPC secret-sharing branch, a different stage. So the
 * diagram named a cryptographic property that no code provided, which is the worst thing this figure
 * can do: it is the most-looked-at part of the page and a visitor has no other way to learn what the
 * aggregation was.
 *
 * Keying it off `qfl_aggregation` - emitted by the stage that did the work - applies the same rule as
 * the inference label above ("read the OBSERVED backend, never the control"). A hand-kept string here
 * can drift from the algorithm; a measured one cannot.
 *
 * Shared by medical and finance so the two cannot disagree about one function's behaviour.
 */
const FEDERATED_STEP: Step = {
  stageId: "federated_training",
  kind: "c-bidir",
  // The accuracy carries its denominator. The bundled federated samples are 16 rows (finance) and
  // 12 (medical), so this is a 4-row and a 3-row score - "acc 0.25" alone reads as a model that is
  // worse than chance rather than as one held-out row.
  label: (m) => `federated train · ${f(m, "qfl_aggregation")} · acc `
    + `${f(m, "federated_baseline_accuracy")} untrained -> ${f(m, "federated_accuracy")} `
    + `on ${f(m, "federated_eval_rows")} held-out rows`,
};

export const USECASE_SPECS: Record<string, Spec> = {
  finance_aml: {
    a: { name: "Bank A", sub: "blacklist + fraud model · LAN" },
    b: { name: "Bank B", sub: "candidate accounts · LAN" },
    steps: [
      { stageId: "pqc_handshake", kind: "c-bidir", label: (_m, c) => `PQC handshake — ${kemName(c)}` },
      { stageId: "qkd_key_agreement", kind: "q-arrow", label: (m, c) => `${qkdName(c)} · QBER ${f(m, "qber")}${optiTag(m, c)}`, hidden: pqcOnly, ok: qkdOk },
      { stageId: "qsmpc_compute", kind: qsmpcKind("psi"), label: qsmpcLabel("psi") },
      // Finance is in FEDERATED_USECASES, so `runClientPipeline` injects `federated_training` here -
      // before the first of private_inference/qml_ids/qsdc_telemetry/quantum_comparison. This spec
      // had no such step, and `OrchestrationGraph.FLOWS` had no finance key either, so the run
      // advanced to a stage NEITHER figure contained: measured live, the stage started at 3.999 s,
      // reported accuracy, and no svg on the page contained the string "federat". That is the exact
      // "no render target" defect `usePipelineRun`'s FEDERATED_USECASES comment describes as the
      // reason the set exists - reintroduced by adding finance to the set on the data test alone.
      FEDERATED_STEP,
      ...HYBRID_INFERENCE,
      publishedStep("private_inference"),
      QSMPC_COMPARE,
      { stageId: "sign_result", kind: "c-bidir", label: (_m, c) => `${sigName(c)}-signed audit record + explanation` },
    ],
  },
  medical_fl: {
    a: { name: "Hospital", sub: "patient data (never leaves) · LAN" },
    b: { name: "Aggregator", sub: "global model · LAN" },
    steps: [
      { stageId: "pqc_handshake", kind: "c-bidir", label: (_m, c) => `PQC handshake — ${kemName(c)}` },
      { stageId: "qkd_key_agreement", kind: "q-arrow", label: (m, c) => `${qkdName(c)} · QBER ${f(m, "qber")}${optiTag(m, c)}`, hidden: pqcOnly, ok: qkdOk },
      FEDERATED_STEP,
      ...HYBRID_INFERENCE,
      publishedStep("private_inference"),
      QSMPC_COMPARE,
      { stageId: "sign_result", kind: "c-bidir", label: (_m, c) => `${sigName(c)}-signed audit record + explanation` },
    ],
  },
  genomics: {
    a: { name: "Cohort 1", sub: "allele counts · LAN" },
    b: { name: "Cohort 2", sub: "allele counts · LAN" },
    steps: [
      { stageId: "pqc_handshake", kind: "c-bidir", label: (_m, c) => `PQC handshake — ${kemName(c)}` },
      { stageId: "qkd_key_agreement", kind: "q-arrow", label: (m, c) => `${qkdName(c)} · QBER ${f(m, "qber")}${optiTag(m, c)}`, hidden: pqcOnly, ok: qkdOk },
      { stageId: "qsmpc_compute", kind: qsmpcKind("sum"), label: qsmpcLabel("sum") },
      ...HYBRID_INFERENCE,
      publishedStep("private_inference"),
      QSMPC_COMPARE,
      { stageId: "sign_result", kind: "c-bidir", label: (_m, c) => `${sigName(c)}-signed audit record` },
    ],
  },
  telecom_ids: {
    a: { name: "Carrier", sub: "QKD Tx + IDS plane · LAN" },
    b: { name: "Remote site / Eve", sub: "QKD Rx (eavesdropper)" },
    steps: [
      { stageId: "pqc_handshake", kind: "c-bidir", label: (_m, c) => `PQC handshake — ${kemName(c)}` },
      // "(w/ eavesdropper)" used to be unconditional here while `qkd_eve` defaults to 0.0, so the
      // page whose subject is intrusion detection drew an attack that had not been simulated. The
      // suffix now reads the fraction the stage REPORTS, which is the same rule the federated and
      // inference labels follow: describe the run, never the intent.
      { stageId: "qkd_key_agreement", kind: "q-arrow",
        label: (m, c) => `${qkdName(c)}${eveTag(m)} · QBER ${f(m, "qber")}${optiTag(m, c)}`,
        hidden: pqcOnly, ok: qkdOk },
      { stageId: "qml_ids", kind: "local-a", label: (m) => `QML-IDS attack detection · macro-F1 ${f(m, "ids_macro_f1")}` },
      publishedStep("qml_ids"),
      QSMPC_COMPARE,
      { stageId: "sign_result", kind: "c-bidir", label: (_m, c) => `${sigName(c)}-signed audit record` },
    ],
  },
  satellite_comms: {
    a: { name: "Ground station", sub: "telemetry uplink · LAN" },
    b: { name: "LEO satellite", sub: "free-space downlink" },
    steps: [
      { stageId: "pqc_handshake", kind: "c-bidir", label: (_m, c) => `PQC handshake — ${kemName(c)}` },
      { stageId: "satellite_qkd", kind: "q-arrow", label: (m, c) => `free-space ${qkdName(c)}${eveTag(m)} (${f(m, "channel_loss_db")} dB) · QBER ${f(m, "qber")}${optiTag(m, c)}`,
        ok: qkdOk,
        hidden: pqcOnly },
      // The abort has to reach the arrow. With Eve at 0.5 the log said "QSDC ABORTED: error 0.158 >
      // 0.11 - telemetry withheld" while this label read "QSDC direct telemetry · secrecy 0
      // bit/carrier" with a done tick beside it - which reads as delivered-but-not-secret, the
      // opposite of what happened. The inline eavesdrop check is this page's headline feature and
      // the diagram was the one surface that did not show it firing.
      //
      // AND THE FIX DID NOT FIX IT. Keying the abort off `qsdc_eavesdrop_detected` moved it onto a
      // metric that is `abort && !lossLimited`, and `lossLimited` is 1 at every setting this demo
      // offered: the link budget floors loss at 5 dB atmospheric + 3 dB pointing, so g >= 6.31 and
      // C_s > 0 needs h(e) < 1/(1+g) = 0.137 while the misalignment floor of 0.025 alone gives
      // h(0.025) = 0.169. Measured on the deployed site at 1 km with Eve at 1.0 - minimum range,
      // maximum eavesdropper - the log read "QSDC NOT AVAILABLE on this link ... Telemetry falls
      // back to the PQC session key" while this label read "QSDC direct telemetry · secrecy 0
      // bit/carrier ✓". The ABORTED branch was unreachable, so the symptom above survived its own
      // repair, and the sign clamp on `qsdc_secrecy_capacity` rendered -5.2033 as "0".
      //
      // Three states now, because there are three: delivered; withheld because the LINK cannot
      // carry it; withheld because of an attack this link would not have suffered on its own.
      // `ok` marks the two withheld states completed-but-failed, so the tick stops reading as a
      // verdict on a delivery that did not happen.
      { stageId: "qsdc_telemetry",
        kind: "q-arrow",
        ok: (m) => Number(m["qsdc_link_budget_limited"]) !== 1
          && Number(m["qsdc_eavesdrop_detected"]) !== 1,
        label: (m) => (Number(m["qsdc_eavesdrop_detected"]) === 1
          // "over threshold" was accurate while the abort compared a scalar QBER. It no longer
          // does: the criterion is secrecy capacity Cs <= 0, and Cs depends on channel loss via
          // g = Q_Eve/Q_Bob, so an error rate WELL under the old 0.11 can still abort. Naming
          // the real quantity keeps the diagram and the metrics table telling one story.
          ? `QSDC ABORTED · eavesdropper · C_s ${f(m, "qsdc_secrecy_capacity")} at error `
            + `${f(m, "qsdc_error_rate")} (g=${f(m, "qsdc_eve_gain_g")}) · telemetry withheld`
          : Number(m["qsdc_link_budget_limited"]) === 1
            ? `QSDC NOT AVAILABLE · link-budget limited · C_s ${f(m, "qsdc_secrecy_capacity")} `
              + `bit/carrier at ${f(m, "channel_loss_db")} dB (g=${f(m, "qsdc_eve_gain_g")}) · `
              + `telemetry withheld, falls back to the PQC session key`
            : `QSDC direct telemetry · secrecy ${f(m, "qsdc_secrecy_capacity")} bit/carrier`) },
      publishedStep("qsdc_telemetry"),
      QSMPC_COMPARE,
      { stageId: "sign_result", kind: "c-bidir", label: (_m, c) => `${sigName(c)}-signed audit record` },
    ],
  },
};

export function hasTwoPartySpec(usecase: string): boolean {
  return usecase in USECASE_SPECS;
}

const ALICE_X = 172;
const BOB_X = 612;
const MID = (ALICE_X + BOB_X) / 2;
const ROW0 = 100;
const ROW_H = 58;

export default function TwoPartyDiagram({
  usecase, activeStage, doneStages, metrics, config = {},
}: {
  usecase: string;
  activeStage: string | null;
  doneStages: Set<string>;
  metrics: Record<string, any>;
  /** Crypto-agility selections, so the arrows name what actually ran. */
  config?: DiagramConfig;
}) {
  const spec = USECASE_SPECS[usecase];
  if (!spec) return null;
  // Drop steps the selected profile does not run, rather than drawing them with an em-dash value.
  const steps = spec.steps.filter((st) => !st.hidden?.(config));
  const idle = activeStage == null && doneStages.size === 0;
  const opacity = (s: Step) =>
    idle ? 1 : s.stageId === activeStage ? 1 : doneStages.has(s.stageId) ? 0.95 : 0.26;
  const isActive = (s: Step) => !idle && s.stageId === activeStage;
  const isDone = (s: Step) => doneStages.has(s.stageId);
  const bottomY = ROW0 + steps.length * ROW_H + 6;
  const rowY = (i: number) => ROW0 + i * ROW_H;

  return (
    <>
    {/* TEXT EQUIVALENT. The SVG below is one `role="img"`, which collapses its entire subtree for
        assistive technology - so everything this file calls the most-looked-at part of the page was
        unavailable: the QBER, the secure-key verdict, the secure-sum or PSI result, the QSDC abort
        reason, the published-model row and the advantage score. None of it appears in text anywhere
        else on the page either, so it was not duplicated elsewhere; it was simply unreachable.
        The list below is generated from the SAME `label(metrics, config)` calls the diagram draws, so
        it cannot drift from what is on screen, and it carries the per-step state that the diagram
        encodes only as opacity and a check mark. */}
    <ol className="sr-only">
      {steps.map((s, i) => (
        <li key={i}>
          {`${s.label(metrics, config)} - ${
            isActive(s) ? "in progress" : isDone(s) ? "complete" : "not yet run"}`}
        </li>
      ))}
    </ol>
    <svg viewBox={`0 0 780 ${bottomY + 78}`} width="100%" xmlns="http://www.w3.org/2000/svg"
      // minWidth so it SCROLLS inside its card rather than shrinking, which is what `.card
      // { overflow-x: auto }` already provides for every table. With `width: 100%` and no minimum the
      // SVG never overflows: at a 360 px content width the 780-unit viewBox scales to ~0.44 and the
      // 11-unit step labels render at about 4.5 px. Illegible is worse than off-screen, because
      // off-screen can be scrolled to.
      style={{ fontFamily: "inherit", minWidth: 680 }} role="img"
      aria-label={`${usecase} two-party diagram; the same content is listed in text immediately before this figure`}>
      <defs>
        <marker id="tp-c" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
          <path d="M0,0 L8,3.2 L0,6.4 Z" fill="var(--accent)" /></marker>
        <marker id="tp-cl" markerWidth="9" markerHeight="9" refX="1" refY="3.2" orient="auto">
          <path d="M8,0 L0,3.2 L8,6.4 Z" fill="var(--accent)" /></marker>
        <marker id="tp-q" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
          <path d="M0,0 L8,3.2 L0,6.4 Z" fill="var(--quantum)" /></marker>
        <marker id="tp-m" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
          <path d="M0,0 L8,3.2 L0,6.4 Z" fill="#f5a623" /></marker>
        <marker id="tp-ml" markerWidth="9" markerHeight="9" refX="1" refY="3.2" orient="auto">
          <path d="M8,0 L0,3.2 L8,6.4 Z" fill="#f5a623" /></marker>
      </defs>

      {/* WAN band between the two LANs (kept above + clear of the party boxes) */}
      <text x={MID} y={18} textAnchor="middle" fontSize={10.5} fill="#6b7689">⟷ WAN ⟷</text>

      {/* lane headers */}
      {[[spec.a, ALICE_X, "var(--accent)"], [spec.b, BOB_X, "var(--quantum)"]].map(([p, x, col]: any) => (
        <g key={p.name}>
          <rect x={x - 96} y={26} width={192} height={44} rx={9} fill="var(--panel)" stroke={col} strokeWidth={1.5} />
          <text x={x} y={48} textAnchor="middle" fontSize={16} fontWeight={700} fill={col}>{p.name}</text>
          <text x={x} y={63} textAnchor="middle" fontSize={10.5} fill="#97a3b6">{p.sub}</text>
        </g>
      ))}

      {/* lifelines */}
      {[ALICE_X, BOB_X].map((x) => (
        <line key={x} x1={x} y1={72} x2={x} y2={bottomY} stroke="#2a3650" strokeWidth={1.5} strokeDasharray="3 4" />
      ))}

      {/* protocol steps */}
      {steps.map((s, i) => {
        const y = rowY(i);
        const op = opacity(s);
        // A finished step whose own `ok` says it failed must not carry the success tick.
        const failed = isDone(s) && s.ok ? !s.ok(metrics, config) : false;
        const mark = isDone(s) ? (failed ? " ✗" : " ✓") : isActive(s) ? " ▶" : "";
        const label = s.label(metrics, config) + mark;
        // A step whose channel depends on the run resolves its kind here, so the arrow style and the
        // legend cannot disagree with the operation the label names.
        const kind: Kind = typeof s.kind === "function" ? s.kind(config) : s.kind;

        // full-width QSMPC-FHE comparison bar (the quantum-secure alternative the benchmark scores)
        if (kind === "compare") {
          return (
            <g key={i} opacity={op}>
              <rect x={64} y={y - 17} width={652} height={34} rx={8}
                fill={isActive(s) ? "#231a3a" : "var(--panel)"} stroke="var(--quantum)"
                strokeWidth={isActive(s) ? 2.4 : 1.4} strokeDasharray="6 3" />
              <text x={MID} y={y + 4} textAnchor="middle" fontSize={11} fill="var(--quantum)" fontWeight={600}>{label}</text>
            </g>
          );
        }
        // A local computation whose label is too long for a party-sized box.
        //
        // `local-a` is a fixed 300-unit box centred on a party. The published-model step names a
        // model, a 12-char revision, an execution provider, an input shape, a latency and an
        // outcome - about 460 units at this font - so it spilled ~80 units past each edge of its
        // box, and the left end ran off the viewBox entirely: the label rendered as
        // "te_comms.opssat_student @ ..." with "published satelli" clipped away. Same width and
        // position as the comparison bar, which is why that one has always fitted; the styling
        // stays local (solid stroke, accent colour) because nothing here crosses the wire.
        if (kind === "local-wide") {
          return (
            <g key={i} opacity={op}>
              <rect x={64} y={y - 16} width={652} height={32} rx={7}
                fill={isActive(s) ? "#1c2740" : "var(--panel)"} stroke="var(--accent)"
                strokeWidth={isActive(s) ? 2.4 : 1.3} />
              <text x={MID} y={y + 4} textAnchor="middle" fontSize={11} fill="#dfe7f3">{label}</text>
            </g>
          );
        }
        // local computation box at one party
        if (kind === "local-a" || kind === "local-b") {
          const cx = kind === "local-a" ? ALICE_X : BOB_X;
          const col = kind === "local-a" ? "var(--accent)" : "var(--quantum)";
          return (
            <g key={i} opacity={op}>
              <rect x={cx - 150} y={y - 16} width={300} height={32} rx={7}
                fill={isActive(s) ? "#1c2740" : "var(--panel)"} stroke={col} strokeWidth={isActive(s) ? 2.4 : 1.3} />
              <text x={cx} y={y + 4} textAnchor="middle" fontSize={11} fill="#dfe7f3">{label}</text>
            </g>
          );
        }
        // message arrow between the two parties (classical / quantum / FHE ciphertext / MPC shares)
        const quantum = kind === "q-arrow";
        const mpc = kind === "mpc-bidir";
        const bidir = kind === "c-bidir" || kind === "mpc-bidir";
        const col = quantum ? "var(--quantum)" : mpc ? "#f5a623" : "var(--accent)";  // fhe-arrow = accent
        const dash = quantum ? "5 3" : mpc ? "2 2" : undefined;
        const mEnd = quantum ? "url(#tp-q)" : mpc ? "url(#tp-m)" : "url(#tp-c)";
        const mStart = bidir ? (mpc ? "url(#tp-ml)" : "url(#tp-cl)") : undefined;
        return (
          <g key={i} opacity={op}>
            <text x={MID} y={y - 9} textAnchor="middle" fontSize={11}
              fill={failed ? "#ff8f6b" : isActive(s) ? "#ffffff" : "#c4cee0"}>{label}</text>
            <line x1={ALICE_X + 6} y1={y} x2={BOB_X - 6} y2={y} stroke={col}
              strokeWidth={isActive(s) ? 2.6 : 1.8} strokeDasharray={dash}
              markerEnd={mEnd} markerStart={mStart} />
          </g>
        );
      })}

      {/* channel legend — moved to the BOTTOM so it never overlaps the party boxes */}
      <g transform={`translate(70, ${bottomY + 18})`} fontSize={10} fill="#97a3b6">
        <line x1={0} y1={-3} x2={22} y2={-3} stroke="var(--accent)" strokeWidth={2} markerEnd="url(#tp-c)" />
        <text x={27} y={1}>classical / FHE ct</text>
        {/* The legend has to agree with the arrows: under a PQC-only profile no quantum channel is
            drawn, so advertising one in the key was the same hardcoding one level down. */}
        {!pqcOnly(config) && (<>
          <line x1={200} y1={-3} x2={222} y2={-3} stroke="var(--quantum)" strokeWidth={2} strokeDasharray="4 3" markerEnd="url(#tp-q)" />
          <text x={227} y={1}>quantum ({qkdName(config)})</text>
        </>)}
        <line x1={430} y1={-3} x2={452} y2={-3} stroke="#f5a623" strokeWidth={2} strokeDasharray="2 2" markerEnd="url(#tp-m)" markerStart="url(#tp-ml)" />
        <text x={457} y={1}>MPC secret-shares</text>
      </g>

      <text x={MID} y={bottomY + 46} textAnchor="middle" fontSize={11} fill="#97a3b6">
        no trusted third party · no raw data shared · {spec.a.name} (LAN) ↔ WAN ↔ {spec.b.name}
      </text>
      <text x={MID} y={bottomY + 62} textAnchor="middle" fontSize={10} fill="#6b7689">
        MPC-FHE hybrid: linear → FHE (CKKS ciphertext) · non-linear → MPC (secret-shares) · vs QSMPC-FHE · PoC runs LAN/local
      </text>
    </svg>
    </>
  );
}
