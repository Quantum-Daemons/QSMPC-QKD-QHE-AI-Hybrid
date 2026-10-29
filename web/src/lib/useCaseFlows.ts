// The five use cases' stages, as the architecture canvas draws them (Round 80), and since Round 81
// the generic demo's, which the E2E page's canvas draws.
//
// Moved here from `components/TwoPartyDiagram.tsx` when the owner chose an animated architecture
// canvas over the sequence diagram that file drew. The labels, the `ok` verdicts and the `hidden`
// rules are that diagram's, unchanged: each encodes a defect found and fixed in an earlier round (the
// observed inference backend rather than the control, the three QSDC states, the eavesdropper tag
// read from the run, the quantum channel dropped under a PQC-only profile), and the canvas shows
// every one of them, in its stage list and on its channels. What is new is where each step happens:
// `canvasRow` below says which component on which side runs it, and which way its exchange goes.
// Grounded in references/ (Zhang QOT npj 2026, Li and Deng QHE-FL, Sutradhar QSMS, the EncFormer
// linear-FHE / non-linear-MPC split).

import { appConfig } from "./config";

export type Kind = "local-a" | "local-b" | "local-wide" | "q-arrow" | "c-arrow" | "c-bidir"
  | "fhe-arrow" | "mpc-bidir" | "compare";
export type M = Record<string, any>;
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
export interface Step {
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
/** A party, as the canvas heads its column. */
export interface Party { name: string; sub: string }
export interface Spec {
  a: Party; b: Party; steps: Step[];
  /** The side whose input the private inference protects: it encrypts and decrypts, the other side
   *  evaluates. Finance's input is Bank B's transaction, scored by Bank A's fraud model. */
  inferenceClient?: "a" | "b";
  /** How the two sides meet, the first sentence of the canvas's caption. One sentence served all
   *  five until Round 81, and it put the satellite and its ground station "over a WAN". */
  meet: (c: DiagramConfig) => string;
}

// The protocol names used to be hardcoded here, so selecting ml-kem-1024 still drew "ML-KEM-768",
// selecting e91 still drew "BB84", and selecting pqc-only still drew a quantum arrow with an
// em-dash QBER. The diagram is the most-looked-at part of the page; it must show what actually ran.
const KEM_LABEL: Record<string, string> = {
  "ml-kem-768+x25519": "ML-KEM-768 + X25519",
  "ml-kem-1024+x25519": "ML-KEM-1024 + X25519",
};
export const kemName = (c: DiagramConfig) =>
  KEM_LABEL[String(c.kem ?? "")] ?? (c.kem ? String(c.kem).toUpperCase() : "hybrid KEM");
const sigName = (c: DiagramConfig) => (c.sig ? String(c.sig).toUpperCase() : "ML-DSA");
// The protocol wins over the family. `clientCompute` branches on `qkd_protocol`, so that is what the
// visitor actually ran; the family only decides whether a quantum channel exists at all.
export const qkdName = (c: DiagramConfig) => {
  const p = String(c.qkdProtocol ?? "");
  if (p === "e91") return "E91 (entanglement / CHSH)";
  if (p === "bb84") return "BB84 decoy-state";
  const q = String(c.qkd ?? "");
  if (q.startsWith("bb84")) return q.includes("decoy") ? "BB84 decoy-state" : "BB84";
  return q || "QKD";
};
export const pqcOnly = (c: DiagramConfig) => String(c.qkd ?? "") === "pqc-only";

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
    // What was pooled, from the run (Round 80): the cohorts' allele counts on genomics, stand-ins on
    // a use case with none. The label used to say neither, beside cohort headers saying allele counts.
    ? `secure sum of ${m["qsms_inputs"] === "allele_counts" ? "allele counts"
        : m["qsms_inputs"] === "stand_ins" ? "stand-in values" : "inputs"} over GF(${f(m, "qsms_prime")}) · pooled = ${f(m, "secure_sum")}`
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
 * GLLP zero-key point measured in this repository is the decoy-state bound's channel-dependent zero-key point, BELOW the textbook BB84 bound of
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
    // No "advantage N/10" suffix: the quantum-advantage score is retired from display and this stage
    // no longer emits `quantum_advantage_score` (see `quantumAdvantage` in clientCompute.ts).
    return `QSMPC-FHE comparison · QOTP + QCNN (T-gates) · ${primary}`
      + (cls == null ? "" : ` · class ${cls}`)
      + ` · ${f(m, "qsmpc_t_gates")} T-gates counted`;
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
  // The accuracy carries its denominator, read from the run. It was added when the bundled samples
  // were 16 and 12 rows and a score rested on 4 or 3 held-out rows; they are 400 rows each now, an
  // 80-row held-out split, and the denominator still says how much a score rests on.
  label: (m) => `federated train · ${f(m, "qfl_aggregation")} · acc `
    + `${f(m, "federated_baseline_accuracy")} untrained -> ${f(m, "federated_accuracy")} `
    + `on ${f(m, "federated_eval_rows")} held-out rows`,
};

/** The caption's first sentence for parties that compute together over a network. */
const overWan = (who: string) => () =>
  `No trusted third party and no raw data shared: ${who} meet over a WAN under the session key.`;

export const USECASE_SPECS: Record<string, Spec> = {
  finance_aml: {
    a: { name: "Bank A", sub: "blacklist + fraud model · LAN" },
    b: { name: "Bank B", sub: "candidate accounts · LAN" },
    inferenceClient: "b",
    meet: overWan("Bank A and Bank B"),
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
    // Plural since the canvas (Round 80): the federated stage runs four hospitals, and the canvas
    // draws all four; one of them is the endpoint of the two-party stages.
    a: { name: "Hospitals", sub: "patient data (never leaves) · LAN" },
    b: { name: "Aggregator", sub: "global model · LAN" },
    inferenceClient: "a",
    meet: overWan("the hospitals and the aggregator"),
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
    // "Cohort 1" and "Cohort 2" until the canvas (Round 80), while the secure sum pooled four
    // cohorts. The canvas draws the four, and the other side is where the pooled count and the PRS
    // model are: the two-party stages run between a cohort and that site.
    a: { name: "Cohorts", sub: "allele counts (never leave) · LAN" },
    b: { name: "Analysis site", sub: "pooled count, PRS model · LAN" },
    inferenceClient: "a",
    meet: overWan("the cohorts and the analysis site"),
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
    // The SECOND unconditional eavesdropper claim on this page, and it outlived the fix to the
    // first. The arrow's tag now reads the fraction the run reports, so at `qkd_eve` 0.0 - the
    // default - it correctly says nothing; this box went on calling the far end "Eve" and its link
    // "QKD Rx (eavesdropper)" on the same clean run. Found by an over-broad browser assertion that
    // scanned every svg rather than the arrow, which is the one time that is worth doing.
    //
    // Party names are static (the Spec type has no metrics), so the honest form is one that is true
    // either way: this page's subject is watching the link for intrusion, and whether anyone is on
    // it is what the run decides. The arrow says which.
    b: { name: "Remote site", sub: "QKD Rx · link monitored for intrusion" },
    // Not "no raw data shared ... over a WAN": nothing here is computed jointly. The carrier's IDS
    // reads its own link's telemetry, and what crosses to the remote site is the key and the record.
    meet: (c) => `The carrier and the remote site share a fibre link${pqcOnly(c) ? "" : " with QKD on it"}, `
      + "which the carrier watches for intrusion: the IDS classifies the link's telemetry at the carrier.",
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
    // The link is a DOWNLINK (Liao et al. 2017's Micius), satellite to ground, for the key and for
    // the telemetry QSDC carries. The sequence diagram drew both arrows ground to satellite and
    // called the ground station "telemetry uplink"; the canvas draws them the way they run.
    a: { name: "Ground station", sub: "telescope and detectors · receives the downlink" },
    b: { name: "LEO satellite", sub: "QKD source, telemetry · free-space downlink" },
    // A satellite and its ground station do not meet over a WAN, and the telemetry IS the data that
    // crosses: the caption said both until Round 81.
    meet: (c) => pqcOnly(c)
      ? "The LEO satellite sends its telemetry down to the ground station under the PQC session key."
      : "The LEO satellite and the ground station meet over a free-space downlink while the satellite is in view: "
        + "the key comes down the quantum channel, and the telemetry by QSDC where the link can carry it, "
        + "else under the PQC session key.",
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
      // Under a PQC-only profile the browser SKIPS QSDC (clientCompute's `qsdc:` block): there is no
      // quantum channel to carry the telemetry, so it travels under the PQC session key. Both figures
      // still drew a quantum arrow there, labelled with an em-dash secrecy, which is the channel the
      // run had just said did not exist (Round 80, found drawing the canvas). The kind follows the
      // profile, as the QSMPC step's does, and the label says what happened.
      { stageId: "qsdc_telemetry",
        kind: (c) => (pqcOnly(c) ? "c-arrow" : "q-arrow"),
        ok: (m) => Number(m["qsdc_link_budget_limited"]) !== 1
          && Number(m["qsdc_eavesdrop_detected"]) !== 1,
        label: (m, c) => pqcOnly(c)
          ? "telemetry under the PQC session key · QSDC skipped: a PQC-only profile has no quantum channel"
          : (Number(m["qsdc_eavesdrop_detected"]) === 1
          // "over threshold" was accurate while the abort compared a scalar QBER. It no longer
          // does: the criterion is secrecy capacity Cs <= 0, and Cs depends on channel loss via
          // g = Q_Eve/Q_Bob, so an error rate WELL under the old 0.11 can still abort. Naming
          // the real quantity keeps the diagram and the metrics table telling one story.
          ? `QSDC ABORTED · eavesdropper · C_s ${f(m, "qsdc_secrecy_capacity")} at error `
            + `${f(m, "qsdc_error_rate")} (g=${f(m, "qsdc_eve_gain_g")}) · telemetry withheld`
          : Number(m["qsdc_link_budget_limited"]) === 1
            // "at 24.5 dB (g=701.06)" paired the CHANNEL loss with a g that includes the detector's
            // 4.0 dB (10^(24.5/10) is 282, not 701), while the run log printed the same g at 28.5 dB.
            // Both now name the channel loss and say g carries the detector too. Round 58. Since
            // Round 75 g also leaves out the station's optics, as Qi et al.'s did, so "34 dB" and
            // g = 808 (29.1 dB) differ by more than the detector: the label says both.
            ? `QSDC NOT AVAILABLE · link-budget limited · C_s ${f(m, "qsdc_secrecy_capacity")} `
              + `bit/carrier at ${f(m, "channel_loss_db")} dB channel loss (g=${f(m, "qsdc_eve_gain_g")}: `
              + `the detector in, the station optics out) · telemetry withheld, falls back to the PQC session key`
            : `QSDC direct telemetry · secrecy ${f(m, "qsdc_secrecy_capacity")} bit/carrier`) },
      publishedStep("qsdc_telemetry"),
      QSMPC_COMPARE,
      { stageId: "sign_result", kind: "c-bidir", label: (_m, c) => `${sigName(c)}-signed audit record` },
    ],
  },
  // The E2E page's generic flow (Round 81), so that page's canvas has something to draw for every
  // entry in its selector. Its stages are clientCompute's `demo` flow: a keyed-hash PSI over two
  // generated account lists, then the MPC-FHE inference on synthetic features. No industry, so no
  // industry's names: the parties say what each holds.
  demo: {
    a: { name: "Party A", sub: "generic · a list, the input" },
    b: { name: "Party B", sub: "generic · a list, the model" },
    inferenceClient: "a",
    meet: overWan("Party A and Party B"),
    steps: [
      { stageId: "pqc_handshake", kind: "c-bidir", label: (_m, c) => `PQC handshake — ${kemName(c)}` },
      { stageId: "qkd_key_agreement", kind: "q-arrow", label: (m, c) => `${qkdName(c)} · QBER ${f(m, "qber")}${optiTag(m, c)}`, hidden: pqcOnly, ok: qkdOk },
      { stageId: "qsmpc_compute", kind: qsmpcKind("psi"), label: qsmpcLabel("psi") },
      ...HYBRID_INFERENCE,
      publishedStep("private_inference"),
      QSMPC_COMPARE,
      { stageId: "sign_result", kind: "c-bidir", label: (_m, c) => `${sigName(c)}-signed audit record` },
    ],
  },
};

/** Which way an exchange goes on the canvas: side A to side B, B to A, or both ways. */
export type Flow = "ab" | "ba" | "both";
/** What crosses: a classical message under the session key, photons (simulated), a CKKS
 *  ciphertext, or MPC secret shares. */
export type Channel = "classical" | "quantum" | "fhe" | "mpc" | "none";

/** Where a step happens on the architecture canvas. */
export interface CanvasRow {
  /** The component on each side that does the step, or absent where that side does nothing. */
  a?: string;
  b?: string;
  flow?: Flow;
  channel: Channel;
  /** Many parties on side A, each exchanging with side B: their noun and how many the run has. */
  many?: { noun: string; count: number };
  /** Not an exchange between the sides: the published model run in this tab, the comparison. */
  wide?: boolean;
}

/** How many parties a many-party stage has: the run's own count once it reports one, else the
 *  configured count it will run with. Never a number of this file's choosing. */
const runCount = (m: M, metric: string, configured: number): number => {
  const v = Number(m[metric]);
  return Number.isFinite(v) && v >= 2 ? v : configured;
};

/**
 * Which component on which side runs a step, and which way its exchange goes.
 *
 * Read from what the code does, stage by stage: the session key consumers are the PSI and the
 * federated masks; the CKKS client encrypts its input and decrypts the scores while the model's
 * holder evaluates the linear layer; the argmax runs as two-party MPC over additive shares between
 * notional parties in this one tab; the satellite's key and telemetry come DOWN the link.
 */
export function canvasRow(usecase: string, step: Step, kind: Kind, m: M, c: DiagramConfig = {}): CanvasRow {
  if (kind === "local-wide" || kind === "compare") return { wide: true, channel: "none" };
  const client = USECASE_SPECS[usecase]?.inferenceClient ?? "a";
  const sides = (clientSide: string, serverSide: string) =>
    (client === "a" ? { a: clientSide, b: serverSide } : { a: serverSide, b: clientSide });
  const cfg = appConfig() as any;
  switch (step.stageId) {
    case "pqc_handshake":
      return { a: "PQC: encapsulates (initiator)", b: "PQC: decapsulates (responder)", flow: "both", channel: "classical" };
    case "qkd_key_agreement":
      return { a: "QKD source: prepares the states", b: "QKD detectors: measure and sift", flow: "ab", channel: "quantum" };
    case "satellite_qkd":
      return { a: "QKD receiver: telescope and detectors", b: "QKD source on board", flow: "ba", channel: "quantum" };
    case "qsdc_telemetry":
      return pqcOnly(c)
        ? { a: "receives the telemetry", b: "sends the telemetry", flow: "ba", channel: "classical" }
        : { a: "QSDC receiver: decodes the telemetry", b: "QSDC sender: encodes the telemetry", flow: "ba", channel: "quantum" };
    case "qsmpc_compute":
      // `mpc-bidir` is the secure SUM (qsmpcKind): every party splits its value into two additive
      // shares mod p, and the shares, never a value, are summed.
      return kind === "mpc-bidir"
        ? { a: "splits its value into two shares mod p", b: "sums the shares mod p", flow: "ab", channel: "mpc",
            many: { noun: usecase === "genomics" ? "cohort" : "party",
                    count: runCount(m, "qsms_parties", usecase === "genomics" ? Number(cfg.data?.genomics_biobanks ?? 4)
                      : Number(cfg.qsmpc?.n_parties ?? 5)) } }
        : { a: "PSI: HMAC tags of its list", b: "PSI: HMAC tags of its list", flow: "both", channel: "classical" };
    case "federated_training":
      return { a: "trains locally, masks its update", b: "FedAvg: sums the masked updates", flow: "both", channel: "classical",
               many: { noun: usecase === "medical_fl" ? "hospital" : "client",
                       count: runCount(m, "qfl_clients", Number(cfg.federated?.hospitals ?? 4)) } };
    case "private_inference":
      if (kind === "fhe-arrow")
        return { ...sides("CKKS: encrypts the input, decrypts the scores", "CKKS: evaluates the linear layers"), flow: "both", channel: "fhe" };
      return { ...sides("MPC argmax: holds shares a", "MPC argmax: holds shares b"), flow: "both", channel: "mpc" };
    case "qml_ids":
      return { a: "QML-IDS: classifies the link telemetry", channel: "none" };
    case "sign_result":
      return { a: "signs the record", b: "verifies the signature", flow: "ab", channel: "classical" };
  }
  return kind === "local-a" ? { a: "runs locally", channel: "none" }
    : kind === "local-b" ? { b: "runs locally", channel: "none" }
    : { wide: true, channel: "none" };
}

/** The steps a run with this configuration draws, in order. */
export const visibleSteps = (usecase: string, c: DiagramConfig): Step[] =>
  (USECASE_SPECS[usecase]?.steps ?? []).filter((st) => !st.hidden?.(c));

/** A step's drawing kind under this configuration. */
export const kindOf = (s: Step, c: DiagramConfig): Kind => (typeof s.kind === "function" ? s.kind(c) : s.kind);

/** Whether a step reads the PQC session key. The keyed-hash PSI and the federated masks' seed are
 *  the only two readers of `_sessionKey` in clientCompute, so the canvas's key bus runs to them. */
export const consumesSessionKey = (step: Step, row: CanvasRow): boolean =>
  step.stageId === "federated_training" || (step.stageId === "qsmpc_compute" && row.channel === "classical");

/** The kinds of lane a run with this configuration draws, and whether it draws the key bus: what the
 *  canvas's legend may list. The legend listed all five on every use case until Round 81, so the
 *  telecom and satellite canvases explained CKKS, MPC and key lanes neither has. */
export function drawnChannels(usecase: string, c: DiagramConfig = {}, m: M = {}): { lanes: Channel[]; keyBus: boolean } {
  const rows = visibleSteps(usecase, c).map((s) => ({ s, row: canvasRow(usecase, s, kindOf(s, c), m, c) }));
  const drawn = new Set(rows.filter(({ row }) => !row.wide && row.channel !== "none"
    && (row.many || (row.a && row.b && row.flow))).map(({ row }) => row.channel));
  return {
    lanes: (["classical", "quantum", "fhe", "mpc"] as Channel[]).filter((ch) => drawn.has(ch)),
    keyBus: rows.some(({ s }) => s.stageId === "pqc_handshake") && rows.some(({ s, row }) => consumesSessionKey(s, row)),
  };
}

/** The canvas's caption: how the parties meet, the inference split where the run performs one, and
 *  what is simulated. One text served every use case until Round 81, and on telecom and satellite it
 *  described an MPC-FHE inference neither run performs. */
export function captionOf(usecase: string, c: DiagramConfig = {}): string {
  const spec = USECASE_SPECS[usecase];
  if (!spec) return "";
  const infers = visibleSteps(usecase, c).some((s) => s.stageId === "private_inference");
  return [
    spec.meet(c),
    infers ? "MPC-FHE hybrid: linear layers under CKKS, the non-linear argmax as MPC, scored against QSMPC-FHE." : "",
    pqcOnly(c) ? "This profile has no quantum channel, and the proof of concept runs in this tab."
      : "The quantum channel is simulated, and the proof of concept runs in this tab.",
  ].filter(Boolean).join(" ");
}
