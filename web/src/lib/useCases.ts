// Shared per-use-case metadata for the index page, the per-use-case pages, and the nav.
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
  // Agentic AI · MCP (Round 10): the report frames MCP as a per-use-case cross-cutting layer, so each
  // use-case page embeds an AgenticAIPanel with an industry-specific prompt + MCP tool name.
  agenticTool: string;     // the MCP tool this industry's agent "calls" (delegate-and-verify)
  agenticPrompt: string;   // the default in-browser-agent prompt for this industry
}

export const USECASES: UseCaseMeta[] = [
  {
    id: "finance_aml", title: "Finance — cross-bank AML / PSI", nav: "Finance — AML", industry: "finance",
    blurb: "Two banks find shared flagged accounts with a keyed-hash PSI over a PQC-protected channel, then score a transaction under CKKS. No third party sees either list, but both banks hold the HMAC key, so over a small ID space either could recover the other's list; a production PSI would use an OPRF-based protocol. The scored rows are PaySim simulator output, and the model reads four quantities computed from them under AML-style names: amount, cross_border (the TRANSFER or CASH_OUT type, since PaySim records no geography), night_tx and new_payee. None is built from PaySim's balance columns, which the dataset's description says must not be used for fraud detection.",
    inference: true,
    agenticTool: "score_transaction",
    // Was "... why must an AML agent ... run over a quantum-safe MCP channel rather than sharing the
    // blacklists?". That presupposes the lists stay private, and this page says the keyed-hash PSI
    // lets either bank recover the other's list over a small ID space. The question now asks what is
    // hidden instead of asserting it.
    agenticPrompt: "In one sentence, when two banks' agents match flagged accounts with a keyed-hash PSI over a quantum-safe MCP channel, what is hidden, and from whom?",
  },
  {
    id: "medical_fl", title: "Medical — federated learning + private inference", nav: "Medical — FL", industry: "healthcare",
    blurb: "Hospitals train a shared model with FedAvg under pairwise masks that cancel exactly (no patient row leaves a hospital), then a clinician runs a CKKS-evaluated diagnosis with a signed explanation. Every mask derives from the one session key, which the aggregator also holds, so the masks hide nothing from it in this demo. The federation runs over the bundled Wisconsin breast-cancer rows; the published models are BreastMNIST image models trained centrally, and MedMNIST states that its dataset is not intended for clinical use.",
    inference: true,
    agenticTool: "query_cohort",
    // Was "... need a quantum-safe (PQC + QKD) channel under HIPAA?". HIPAA names no PQC or QKD
    // requirement, so the question asked the in-browser model to explain an obligation that does not
    // exist - and the model answers what it is asked.
    agenticPrompt: "In one sentence, what does a quantum-safe channel add when a clinician's agent queries a federated diagnostic model over MCP?",
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
  },
  {
    id: "telecom_ids", title: "Telecom / Defense — QKD + IDS", nav: "Telecom / Defense", industry: "telecom",
    blurb: "A carrier runs a simulated QKD link, with an optional eavesdropper (the Eve control), beside an intrusion detector over the eight classes of Al-Kuwari et al. 2026 (seven QKD attacks and normal traffic). In this tab the detector is a softmax trained and scored on synthetic telemetry it generates, so neither the link's QBER nor the Eve control reaches it; the published model is a random forest, and the quantum classifiers (QSVC, QNN) are trained offline.",
    inference: false,
    agenticTool: "inspect_qkd_telemetry",
    agenticPrompt: "In one sentence, how does a rising QBER in the QKD telemetry let an IDS agent detect an eavesdropper on the link?",
  },
  {
    id: "satellite_comms", title: "Satellite — free-space QKD + QSDC", nav: "Satellite", industry: "satellite",
    blurb: "A ground station and a LEO satellite run a simulated free-space BB84 downlink, then test whether Quantum Secure Direct Communication could carry mission telemetry over it, by an analytic secrecy-capacity test that includes the eavesdropper (no check carriers are sampled). With the default link budget it cannot close at any range under the published secrecy criterion, and the run says why. The telemetry anomaly check runs in plaintext; no FHE or MPC stage runs on this page.",
    inference: false,
    agenticTool: "relay_mission_telemetry",
    // Was "why relay satellite mission telemetry via QSDC ... instead of classical encryption". This
    // page's own finding is that QSDC cannot close over this LEO downlink at any range it offers, so
    // the old question asked the model to argue against the page.
    agenticPrompt: "In one sentence, what limits Quantum Secure Direct Communication over a LEO free-space downlink compared with QKD plus classical encryption?",
  },
];

export const USECASE_META: Record<string, UseCaseMeta> = Object.fromEntries(USECASES.map((u) => [u.id, u]));
