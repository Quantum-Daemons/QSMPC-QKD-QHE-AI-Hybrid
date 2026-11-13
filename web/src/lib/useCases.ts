// Shared per-use-case metadata for the index page, the per-use-case pages, and the nav.
import { isClientDemo } from "./demoMode";

export interface UseCaseMeta {
  id: string;
  title: string;
  nav: string;          // short sidebar label
  industry: string;
  blurb: string;
  // A `market` one-liner (analyst market-size forecasts) stood here and rendered on /usecases and in
  // every use-case page header. Removed 2026-10-24: this project's public material carries no
  // market-size figures. Two of the five also attributed a bank AML and a telecom PQC migration to
  // CNSA 2.0, which binds national security systems only.
  inference: boolean;   // has a classical-vs-quantum private-inference path (full benchmark + SHAP)
  // Agentic AI · MCP (Round 10): MCP is a per-use-case cross-cutting layer, so each
  // use-case page embeds an AgenticAIPanel with an industry-specific prompt + MCP tool name.
  agenticTool: string;     // the MCP tool this industry's request is named after; the browser calls nothing
  agenticPrompt: string;   // the default in-browser-agent prompt for this industry
  clientOnly?: boolean;    // runs only in the browser build: the full deployment's pipelines have no such stages
  // Round 86 (the owner's choice of 2026-11-13): whether this page applies differential privacy, and why or why not.
  privacy: string;
}

const ALL_USECASES: UseCaseMeta[] = [
  {
    id: "finance_aml", title: "Finance — cross-bank AML / PSI", nav: "Finance — AML", industry: "finance",
    blurb: "Two banks find shared flagged accounts with an OPRF-based PSI (RFC 9497 over ristretto255), then score a transaction under CKKS. Bank A blinds its IDs and Bank B evaluates them under an OPRF key only it holds, so Bank A learns the shared accounts and the size of Bank B's list, Bank B learns only the size of Bank A's, and neither can test IDs against the other's list on its own. The PSI is semi-honest and not post-quantum, both banks run in this tab, and its messages are sealed under the PQC session key. The scored rows are PaySim simulator output, and the model reads four quantities computed from them under AML-style names: amount, cross_border (the TRANSFER or CASH_OUT type, since PaySim records no geography), night_tx and new_payee. None is built from PaySim's balance columns, which the dataset's description says must not be used for fraud detection. (The full deployment's score_transaction tool is a separate fixed rule of thumb that does read balances; nothing on this page runs it.)",
    inference: true,
    agenticTool: "score_transaction",
    // Was "... why must an AML agent ... run over a quantum-safe MCP channel rather than sharing the
    // blacklists?", which asserted the answer. The question asks what is hidden, and from whom, because
    // the OPRF PSI hides different things from each bank: Bank A learns the shared accounts and the
    // size of Bank B's list, Bank B only the size of Bank A's.
    agenticPrompt: "In one sentence, when two banks' agents match flagged accounts with an OPRF-based PSI over a quantum-safe MCP channel, what is hidden, and from whom?",
    privacy: "applied to the count of shared accounts as it would be published, per account, at epsilon 1 and delta 1e-8: one party adds exact discrete Gaussian noise to the count once. Not to the PSI's answer to the banks, which is what the PSI exists to give them; noise there would add false matches and misses.",
  },
  {
    id: "medical_fl", title: "Medical — federated learning + private inference", nav: "Medical — FL", industry: "healthcare",
    // Round 86: "then a clinician runs a CKKS-evaluated diagnosis" claimed two things the page does not do. The
    // federated model is measured and set aside (QFLResult carries no weights), and the model evaluated
    // under CKKS is a regression fitted centrally, offline, on the same 398 pooled rows; and the page itself
    // says nothing on it is a diagnosis. Since 2026-11-13 the federation is four real hospitals (clientFedHeart.ts) on
    // another corpus, so the two models no longer share rows, and the blurb says so.
    blurb: "Four real hospitals (the Cleveland Clinic, the Hungarian Institute of Cardiology, the University Hospitals of Zurich and Basel, and the VA in Long Beach: UCI Heart Disease, CC BY 4.0) train a shared model with FedAvg under pairwise masks that cancel exactly, and the page sets it beside each hospital trained alone and all four pooled, FLamby's Fed-Heart-Disease benchmark; no patient row leaves its hospital. A CKKS-evaluated classification runs beside it with a signed explanation: a demonstration, not a diagnosis. They are separate models on separate corpora: the one evaluated under CKKS is a logistic regression on Wisconsin breast-cancer rows, fitted centrally, offline, and the published models are BreastMNIST image models trained centrally (the CNN student reads a held-out image), from a dataset MedMNIST states is not intended for clinical use. Each pair of hospitals agrees its mask seed by X25519, so an aggregator that follows the protocol, receiving only the public keys and the masked updates, cannot remove a mask alone; the seeds are not post-quantum, there is no dropout recovery and no authentication of the public keys, and every hospital runs in this one tab.",
    inference: true,
    agenticTool: "query_cohort",
    // Was "... need a quantum-safe (PQC + QKD) channel under HIPAA?". HIPAA names no PQC or QKD
    // requirement, so the question asked the in-browser model to explain an obligation that does not
    // exist - and the model answers what it is asked.
    agenticPrompt: "In one sentence, what does a quantum-safe channel add when a clinician's agent queries a federated diagnostic model over MCP?",
    privacy: "applied to the federation, per patient: beside FedAvg, a record-level arm in which each hospital adds its share of exact discrete Gaussian noise inside the masks, at epsilon 8 down to 1 and delta 1e-6, in the four-hospital card, and once more inside threshold BGV on its encrypted-aggregation button. Not to the private-inference models, which are trained centrally on public data.",
  },
  {
    id: "genomics", title: "Genomics — confidential analysis", nav: "Genomics", industry: "genomics",
    // Round 84. The blurb says what runs: the full score under CKKS (not a sign test on a few
    // variants), and the browser's sum as additive shares (not "the classical equivalent" of a GHZ
    // simulation the backend does not run at its field). PGS000072's counts come from the payload.
    blurb: "Cohorts pool allele counts by secure summation over a prime field large enough for any total they can reach, then one person's full PGS000072 polygenic score, over the 183 of its 187 variants with a 1000 Genomes genotype, is evaluated under CKKS: their effect-allele dosages at every variant in one ciphertext, multiplied by the plaintext weights and summed by rotation, and the decrypted score compared with the 90th percentile of the score over the bundled 1000 Genomes individuals. The Python backend runs Sutradhar's (t,n)-threshold secure summation (Shamir shares, d-level QFT); at a field that large it samples the protocol's outcomes analytically rather than simulating the GHZ state. This tab adds two additive shares per cohort in one step, and the page labels the quantum channel SIMULATED.",
    inference: true,
    agenticTool: "pool_allele_counts",
    // Neutral (Round 84): the old question asked why the quantum secure sum "is safer", a property
    // this page does not claim, and the model answers what it is asked.
    agenticPrompt: "In one sentence, what does a secure sum of per-cohort allele counts hide, and from whom?",
    privacy: "applied to the pooled allele count each run releases, per person, at the epsilon chosen in the controls and delta 1e-5: the cohorts add exact discrete Gaussian noise inside the secure sum, and a button sums the same noisy contributions under threshold BGV. Not to the person's own score, which goes back to that person.",
  },
  {
    id: "telecom_ids", title: "Telecom / Defense — QKD + IDS", nav: "Telecom / Defense", industry: "telecom",
    blurb: "A carrier runs a simulated QKD link, with an optional eavesdropper (the Eve control), beside an intrusion detector over the eight classes of Al-Kuwari et al. 2026 (seven QKD attacks and normal traffic). In this tab the detector is a softmax trained and scored on synthetic telemetry it generates, so neither the link's QBER nor the Eve control reaches it; the published model is a random forest, and the quantum classifiers (QSVC, QNN) are trained offline.",
    inference: false,
    agenticTool: "inspect_qkd_telemetry",
    agenticPrompt: "In one sentence, how does a rising QBER in the QKD telemetry let an IDS agent detect an eavesdropper on the link?",
    privacy: "not applied: the detector is trained and scored on synthetic telemetry this tab generates, which describes no person.",
  },
  {
    id: "satellite_comms", title: "Satellite — free-space QKD + QSDC", nav: "Satellite", industry: "satellite",
    blurb: "A ground station and a LEO satellite run a simulated free-space BB84 downlink, then test whether Quantum Secure Direct Communication could carry mission telemetry over it, by an analytic secrecy-capacity test that includes the eavesdropper (no check carriers are sampled). With the default link budget two-way QSDC cannot close at any range under the published secrecy criterion, and the run says why; the telemetry then goes one way, by quasi-QSDC from a pre-shared key pool. The telemetry anomaly check runs in plaintext, and a CKKS check scores the published OPS-SAT student on the benchmark's 529 test segments against plaintext. The run's verification also puts every forecast of the published Telemanom student's stretch through CKKS, in a Web Worker beside the page.",
    inference: false,
    agenticTool: "relay_mission_telemetry",
    // Was "why relay satellite mission telemetry via QSDC ... instead of classical encryption". This
    // page's own finding is that QSDC cannot close over this LEO downlink at any range it offers, so
    // the old question asked the model to argue against the page.
    agenticPrompt: "In one sentence, what limits Quantum Secure Direct Communication over a LEO free-space downlink compared with QKD plus classical encryption?",
    privacy: "not applied: spacecraft telemetry (ESA-ADB, OPS-SAT) describes no person, and both datasets are public.",
  },
  {
    id: "defi_lending", title: "DeFi — private liquidation check", nav: "DeFi lending", industry: "defi",
    blurb: "Alice's lending pool and Bob, one of its borrowers, run through a simulated month: a two-slope interest rate, debt that grows on a public index, and liquidation below a health factor of 1, the mechanics of Aave V3's public contracts, with illustrative parameters and generated prices. At chosen hours the pool asks whether Bob can be liquidated. By default Carol, a separate key holder standing in for the threshold committees confidential-DeFi systems use, holds the CKKS key: Bob encrypts his balances once and can go offline, the pool computes his margin on that ciphertext at public prices and thresholds, and a garbled circuit between the pool and Carol gives only its sign. Two other arrangements are one setting away: a committee that runs such a threshold, where no one holds the key and any two of three members re-encrypt each masked margin to Carol on threshold BGV, and a bilateral mode, with Bob holding the key and the pool's thresholds kept from him. No chain is involved.",
    inference: false,
    agenticTool: "check_position_health",
    agenticPrompt: "In one sentence, what does a lending pool learn when a borrower's margin is computed under CKKS and its sign in a garbled circuit?",
    privacy: "not applied: the pool learns one bit, whether Bob can be liquidated, and that bit is the decision. Noise would make it wrong on purpose (randomized response at epsilon ln 3 answers wrongly one time in four), so what repeated checks reveal is limited by how often they run, not by noise.",
    clientOnly: true,
  },
];

/** The use cases this build can run: the full deployment drops the ones its pipelines do not have. */
export const USECASES: UseCaseMeta[] = ALL_USECASES.filter((u) => !u.clientOnly || isClientDemo());

export const USECASE_META: Record<string, UseCaseMeta> = Object.fromEntries(USECASES.map((u) => [u.id, u]));
