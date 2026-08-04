export default function Value() {
  return (
    <div>
      <h1 className="page-title">Introduction & Value</h1>
      <p className="page-sub">
        Why a quantum-safe orchestration layer now — the regulatory clock, the market,
        and where this architecture creates value for regulated industries.
      </p>

      <h2>The driver: Q-Day is being pulled forward</h2>
      <div className="grid cols-3">
        <div className="card"><h3>HNDL is active</h3><p>"Harvest Now, Decrypt Later": data with 30-year secrecy needs is already at risk. The hybrid (PQC+QKD) channel protects it today.</p></div>
        <div className="card"><h3>Resource estimates collapsing</h3><p>RSA-2048 break estimated under 1M noisy qubits (Gidney, arXiv:2505.15917, 2025) &mdash; about 1/20 the qubits of the 2019 estimate, though the runtime rises roughly 20&times; in exchange. Some forecasts put Q-Day at 2028–2030.</p></div>
        <div className="card"><h3>Regulation mandates migration</h3><p>NIST FIPS 203/204/205 final. NSA CNSA 2.0 requires new NSS acquisitions to be quantum-resistant from 2027 (widely reported; NSA's own pages are not machine-retrievable, so treat the exact wording as unverified). US <b>EO 14412</b> (22 Jun 2026) is the binding instrument: PQC key establishment on high-value assets by 2030 and signatures by 2031, with National Security Systems excluded. Cloudflare commits to full PQ security including authentication by 2029; Google sets 2029 as a migration timeline. <b>EU AI Act Art. 13 transparency</b> (high-risk AI) is in force <b>Aug 2026</b> — explainability is now obligatory, not optional.</p></div>
      </div>

      <h2 style={{ marginTop: 28 }}>Market context</h2>
      <table>
        <thead><tr><th>Segment</th><th>2025–26</th><th>Outlook</th><th>CAGR</th></tr></thead>
        <tbody>
          <tr><td><b>Privacy-preserving AI</b> (the umbrella TAM)</td><td><b>~$3.67B</b></td><td><b>~$46B by 2035</b></td><td><b>~28.8%</b></td></tr>
          <tr><td>Post-Quantum Cryptography</td><td>~$2.2B</td><td>$20–30B by 2033–34</td><td>~38–46%</td></tr>
          <tr><td>Quantum AI</td><td>~$0.6B</td><td>$5.0B by 2033</td><td>~35%</td></tr>
          <tr><td>Secure MPC</td><td>~$1.1B</td><td>$2.7B by 2034</td><td>~12%</td></tr>
          <tr><td>Quantum cryptography (incl. QKD)</td><td>~$0.8B</td><td>$4.6–18B by 2030–34</td><td>~18–38%</td></tr>
        </tbody>
      </table>
      <p className="muted" style={{ fontSize: 12 }}>
        Within privacy-preserving AI the analysts disagree, and the disagreement does not reconcile: market.us
        sizes the market at $3.67B(2025)→$46.11B(2035) with <b>homomorphic encryption the largest technology
        segment (~35.6%)</b> and <b>BFSI the leading vertical (27.8%)</b>, while Precedence Research sizes a
        different market at $4.25B(2025)→$39.93B(2035) with <b>federated learning largest (~29%)</b>. Those
        percentages are shares of <i>different denominators</i> and must not be added or ranked against each
        other. An earlier revision of this page presented them as one consistent picture and asserted
        &ldquo;healthcare/life-sciences (~30%) lead the verticals&rdquo;, which neither report supports.
      </p>

      <h2 style={{ marginTop: 28 }}>Where this architecture creates value</h2>
      <div className="grid cols-2">
        <div className="card"><h3>Regulatory eligibility</h3><p>PQC-first hybrid + crypto-agility + XAI audit trails position bids for CNSA 2.0 / FIPS-aligned procurement that non-compliant vendors can't enter.</p></div>
        <div className="card"><h3>Loss avoidance</h3><p>QML-IDS + QKD reduce the breach probability for long-retention data; the hybrid channel removes HNDL exposure.</p></div>
        <div className="card"><h3>New privacy-preserving services</h3><p>Cross-bank AML PSI, federated medical models, confidential genomics, and free-space <b>satellite QKD/QSDC</b> telemetry — revenue from data and links that previously couldn't be combined or trusted.</p></div>
        <div className="card"><h3>Operational savings</h3><p><b>OptiQKD</b> (AI-tuned decoy/basis parameters, with a convergence metric; the reference framework reports +14&ndash;25% in simulation) automates QKD calibration and failover; a <b>QRNG</b> entropy source (min-entropy-bounded) plus a config-gated leakage-resilient PQC+QKD combiner harden the key material; one config change migrates algorithms.</p></div>
      </div>

      <div className="note" style={{ marginTop: 24 }}>
        Benefits are framed as a research/PoC investment thesis. Concrete ROI depends on
        field validation of the simulated quantum-channel results (the M3/M4 gates in the source report).
      </div>
    </div>
  );
}
