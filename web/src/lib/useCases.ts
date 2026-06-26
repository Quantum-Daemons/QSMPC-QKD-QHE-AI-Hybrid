// Shared per-use-case metadata for the index page, the per-use-case pages, and the nav.
export interface UseCaseMeta {
  id: string;
  title: string;
  nav: string;          // short sidebar label
  industry: string;
  blurb: string;
  market: string;       // 2026 market one-liner (web-verified)
  inference: boolean;   // has a classical-vs-quantum private-inference path (full benchmark + SHAP)
  // Agentic AI · MCP (Round 10): the report frames MCP as a per-use-case cross-cutting layer, so each
  // use-case page embeds an AgenticAIPanel with an industry-specific prompt + MCP tool name.
  agenticTool: string;     // the MCP tool this industry's agent "calls" (delegate-and-verify)
  agenticPrompt: string;   // the default in-browser-agent prompt for this industry
}

export const USECASES: UseCaseMeta[] = [
  {
    id: "finance_aml", title: "Finance — cross-bank AML / PSI", nav: "Finance — AML", industry: "finance",
    blurb: "Two banks find shared flagged accounts via quantum-safe PSI, then privately score a transaction — without revealing their blacklists.",
    market: "Privacy-preserving AI $3.67B(2025)→$46B(2035), CAGR 28.8%; BFSI the top vertical (~28% share); AML/PSI driven by CNSA 2.0 (2027).",
    inference: true,
    agenticTool: "score_transaction",
    agenticPrompt: "In one sentence, why must an AML agent that scores a transaction across two banks run over a quantum-safe MCP channel rather than sharing the blacklists?",
  },
  {
    id: "medical_fl", title: "Medical — federated learning + private inference", nav: "Medical — FL", industry: "healthcare",
    blurb: "Hospitals train a shared diagnostic model with secure aggregation (no patient data leaves), then a clinician runs a private, signed-explained diagnosis.",
    market: "Healthcare/life-sciences the largest + fastest-growing vertical (~30% of a $3.67B→$46B market); FL is the #1 PET (~29%). EU AI Act Art. 13 in force Aug 2026.",
    inference: true,
    agenticTool: "query_cohort",
    agenticPrompt: "In one sentence, why does a clinician's agent querying a federated diagnostic model over MCP need a quantum-safe (PQC + QKD) channel under HIPAA?",
  },
  {
    id: "genomics", title: "Genomics — confidential analysis", nav: "Genomics", industry: "genomics",
    blurb: "Cohorts pool allele counts via quantum secure summation (d-qudit GHZ), then run a private phenotype inference — no cohort exposes raw genotypes.",
    market: "Genomic data-collaboration rides the healthcare/life-sciences segment (~30%); FHE the largest PET (~35.6% of tech mix); secure MPC ~$1.1B→$2.7B.",
    inference: true,
    agenticTool: "pool_allele_counts",
    agenticPrompt: "In one sentence, why is pooling allele counts via quantum secure summation over a quantum-safe MCP channel safer than sharing raw genotypes between biobanks?",
  },
  {
    id: "telecom_ids", title: "Telecom / Defense — QKD + QML-IDS", nav: "Telecom / Defense", industry: "telecom",
    blurb: "A carrier runs a QKD link under an eavesdropper and a QML intrusion-detection plane that flags attack signatures from the QKD telemetry.",
    market: "Quantum cryptography (incl. QKD) ~$0.8B→$4.6–18B; defense/telecom early adopters; PQC migration mandated by CNSA 2.0 (2027).",
    inference: false,
    agenticTool: "inspect_qkd_telemetry",
    agenticPrompt: "In one sentence, how does a rising QBER in the QKD telemetry let an IDS agent detect an eavesdropper on the link?",
  },
  {
    id: "satellite_comms", title: "Satellite — free-space QKD + QSDC", nav: "Satellite", industry: "satellite",
    blurb: "A ground station and a LEO satellite distribute a key over a free-space BB84 downlink, then send mission telemetry by Quantum Secure Direct Communication with an inline eavesdrop check.",
    market: "Satellite QKD scaling (Micius-class → constellations); free-space links a 2026 growth area for defense + critical-infra resilience.",
    inference: false,
    agenticTool: "relay_mission_telemetry",
    agenticPrompt: "In one sentence, why relay satellite mission telemetry via QSDC over a quantum-safe MCP channel instead of classical encryption over the free-space link?",
  },
];

export const USECASE_META: Record<string, UseCaseMeta> = Object.fromEntries(USECASES.map((u) => [u.id, u]));
