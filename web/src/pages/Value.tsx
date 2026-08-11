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
        <div className="card"><h3>Regulation mandates migration</h3><p>NIST FIPS 203/204/205 final. NSA CNSA 2.0 requires new NSS acquisitions to be quantum-resistant from 2027 (widely reported; NSA's own pages are not machine-retrievable, so treat the exact wording as unverified). US <b>EO 14412</b> (22 Jun 2026) is the binding instrument: PQC key establishment on high-value assets by 2030 and signatures by 2031, with National Security Systems excluded. Cloudflare commits to full PQ security including authentication by 2029; Google sets 2029 as a migration timeline. <b>EU AI Act Art. 13 transparency</b> (high-risk AI) is <b>not yet in application</b>: the AI Omnibus (Regulation (EU) 2026/1744, in force 27 Jul 2026) deferred Chapter III Sections 1&ndash;3, which contain Art. 13, to <b>2 Dec 2027</b> for Annex III systems and <b>2 Aug 2028</b> for high-risk AI embedded in regulated products. What applies from 2 Aug 2026 is general applicability, enforcement and governance, plus Art. 50 transparency for chatbots and synthetic media. Signed explanations are therefore preparation for a 2027/2028 obligation, not a present-day mandate.</p></div>
      </div>

      <h2 style={{ marginTop: 28 }}>Market context</h2>
      <table>
        <thead><tr><th>Segment</th><th>2025–26</th><th>Outlook</th><th>CAGR</th></tr></thead>
        <tbody>
          <tr><td><b>Privacy-preserving AI</b> (the umbrella TAM)</td><td><b>~$3.67B</b></td><td><b>~$46B by 2035</b></td><td><b>~28.8%</b></td></tr>
          <tr><td>Post-Quantum Cryptography</td><td>~$2.2B</td><td>$20–30B by 2033–34</td><td>~38–46%</td></tr>
          <tr><td>Quantum AI</td><td>~$0.6B</td><td>$5.0B by 2033</td><td>~35%</td></tr>
          <tr><td>Secure MPC</td><td>~$1.1B</td><td>$2.7B by 2034</td><td>~12%</td></tr>
          <tr><td>Quantum cryptography (incl. QKD)</td><td>~$0.7–1.1B (2025–26)</td>
            <td>$3.7B by 2035 <span className="muted">(Roots Analysis)</span> · $4.6B by 2030 <span className="muted">(Grand View)</span> · $18.4B by 2034 <span className="muted">(Precedence)</span></td>
            <td>18.3% · 38.3% · 38.3%</td></tr>
        </tbody>
      </table>
      <p className="muted" style={{ fontSize: 12 }}>
        The quantum-cryptography row lists three forecasts side by side rather than blending them, because
        they do not share a base year, a scope or an endpoint: an earlier version of this table combined
        Roots Analysis' ~$0.8B base with Grand View's and Precedence's far higher endpoints and quoted the
        CAGR span "~18–38%", producing a range no single analyst projects. Read each triple as one house's
        view.
      </p>
      <p className="muted" style={{ fontSize: 12 }}>
        Within privacy-preserving AI the analysts disagree, and the disagreement does not reconcile: market.us
        sizes the market at $3.67B(2025)→$46.11B(2035) with <b>homomorphic encryption the largest technology
        segment (~35.6%)</b> and <b>BFSI the leading vertical (27.8%)</b>, while Precedence Research sizes a
        different market at $4.25B(2025)→$39.93B(2035) with <b>federated learning largest (~29%)</b>. Those
        percentages are shares of <i>different denominators</i> and must not be added or ranked against each
        other. An earlier revision of this page presented them as one consistent picture and asserted
        &ldquo;healthcare/life-sciences (~30%) lead the verticals&rdquo;, which neither report supports.
      </p>

      <h2 style={{ marginTop: 28 }}>What is new here, and what is not</h2>
      <p className="muted">
        This project's own verification rule is that a novelty claim has to survive an adversarial
        search for prior art. Applied to its own headline claim, the honest answer is asymmetric, and
        both halves are stated here rather than only the flattering one.
      </p>
      <div className="grid cols-2">
        <div className="card">
          <h3>Not new: MPC in a browser</h3>
          <p>
            Secure multiparty computation has been compiled to WebAssembly before.{" "}
            <b>webSPDZ</b> (IACR ePrint 2025/487) compiles MP-SPDZ to WASM and communicates over
            WebRTC or WebSockets, and it names <b>JIFF</b> and <b>MPyC-Web</b> as earlier
            browser MPC engines. The garbled-circuit half of this demo is a re-implementation of a
            known idea, not a first.
          </p>
        </div>
        <div className="card">
          <h3>Unusual: FHE <i>evaluation</i> in a browser</h3>
          <p>
            The search for a browser that <b>evaluates</b> a homomorphic circuit came back empty.
            Every artefact found encrypts in the browser and evaluates on a server - IACR ePrint
            2026/515 defines "client-side operations" as encryption and decryption while
            <i> users offload large computations to servers</i>, and Zama's own TFHE-rs WASM API
            documents that it supports key generation, encryption and decryption but{" "}
            <b>not FHE computation</b>, citing WASM threading and the 2&nbsp;GB memory ceiling.
            Here node-seal evaluates a real CKKS dot product and convolution in the tab.
          </p>
        </div>
      </div>
      <p className="muted small">
        So the claim this project makes is the <b>composition</b>, not either half: PQC key
        establishment, then QKD-derived key material, then secure multiparty computation, then
        homomorphic evaluation, then the model inference - chained end to end in one browser tab with
        no server-side compute. A search for an integrated browser system doing all of that returned
        nothing. Two honest caveats travel with it: the two MPC parties are notional, running over an
        in-memory transport in a single tab (the protocol, bytes and wall-clock are real, the party
        separation is not), and the quantum channel is simulated - there is no browser API for QKD and
        there will not be one. Client-side is also not automatically private: WWW&nbsp;2026's{" "}
        <b>WAMO</b> paper argues that browser deployment moves a model into a hostile environment,
        which is why the models here are public on Hugging Face rather than treated as secrets.
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
