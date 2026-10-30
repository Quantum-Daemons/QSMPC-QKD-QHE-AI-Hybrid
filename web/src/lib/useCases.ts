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
    blurb: "Two banks find shared flagged accounts with a keyed-hash PSI over a PQC-protected channel, then score a transaction under CKKS. No third party sees either list, but both banks hold the HMAC key, so over a small ID space either could recover the other's list; a production PSI would use an OPRF-based protocol. The scored rows are PaySim simulator output whose AML-style feature names relabel PaySim columns (device_risk is PaySim's origin-balance-inconsistency flag).",
    inference: true,
    agenticTool: "score_transaction",
    agenticPrompt: "In one sentence, why must an AML agent that scores a transaction across two banks run over a quantum-safe MCP channel rather than sharing the blacklists?",
  },
  {
    id: "medical_fl", title: "Medical — federated learning + private inference", nav: "Medical — FL", industry: "healthcare",
    blurb: "Hospitals train a shared model with FedAvg under pairwise masks that cancel exactly (no patient row leaves a hospital), then a clinician runs a CKKS-evaluated diagnosis with a signed explanation. Every mask derives from the one session key, which the aggregator also holds, so the masks hide nothing from it in this demo. The federation runs over the bundled Wisconsin breast-cancer rows; the published models are BreastMNIST image models trained centrally.",
    inference: true,
    agenticTool: "query_cohort",
    // Was "... need a quantum-safe (PQC + QKD) channel under HIPAA?". HIPAA names no PQC or QKD
    // requirement, so the question asked the in-browser model to explain an obligation that does not
    // exist - and the model answers what it is asked.
    agenticPrompt: "In one sentence, what does a quantum-safe channel add when a clinician's agent queries a federated diagnostic model over MCP?",
  },
  {
    id: "genomics", title: "Genomics — confidential analysis", nav: "Genomics", industry: "genomics",
    blurb: "Cohorts pool allele counts by secure summation over a prime field, then run a CKKS sign test on a partial polygenic score (8 of the 313 PGS000004 variants) - no cohort's genotypes are sent in the clear. The reference backend runs the Sutradhar (t,n)-threshold quantum secure summation (d-level QFT over a t-qudit GHZ state); the in-browser demo computes the classical equivalent, and the page labels the quantum channel SIMULATED.",
    inference: true,
    agenticTool: "pool_allele_counts",
    agenticPrompt: "In one sentence, why is pooling allele counts via quantum secure summation over a quantum-safe MCP channel safer than sharing raw genotypes between biobanks?",
  },
  {
    id: "telecom_ids", title: "Telecom / Defense — QKD + QML-IDS", nav: "Telecom / Defense", industry: "telecom",
    blurb: "A carrier runs a simulated QKD link, with an optional eavesdropper (the Eve control), and a QML intrusion-detection plane that flags attack signatures in synthetic QKD telemetry.",
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
