export default function Value() {
  return (
    <div>
      <h1 className="page-title">Introduction & Value</h1>
      <p className="page-sub">
        Why a quantum-safe orchestration layer now — the regulatory clock, and where this
        architecture creates value for regulated industries.
      </p>

      <h2>The driver: Q-Day is being pulled forward</h2>
      <div className="grid cols-3">
        <div className="card"><h3>HNDL is active</h3><p>"Harvest Now, Decrypt Later": data with 30-year secrecy needs is already at risk. A PQC hybrid key exchange (ML-KEM + X25519) addresses it today; QKD is a simulated add-on in this demo.</p></div>
        <div className="card"><h3>Resource estimates collapsing</h3><p>RSA-2048 break estimated under 1M noisy qubits (Gidney, arXiv:2505.15917, 2025) &mdash; about 1/20 the qubits of the 2019 estimate, though the runtime rises roughly 20&times; in exchange.</p></div>
        <div className="card"><h3>Regulation mandates migration</h3><p>NIST FIPS 203/204/205 final. For national security systems only, CNSSP 15 requires CNSA 2.0 algorithms in all new acquisitions from 1 Jan 2027 and in all use by 31 Dec 2031 (NSA CNSA 2.0 FAQ v2.2, Oct 2026); NSA sets no requirement outside NSS. US <b>EO 14412</b> (22 Jun 2026) is the binding instrument for federal agencies: PQC key establishment on high-value assets and high impact systems by 31 Dec 2030 and signatures by 31 Dec 2031, with National Security Systems excluded; for contractors it directs a <i>proposed</i> FAR rule within 180 days. Cloudflare targets full PQ security including authentication by 2029; Google sets 2029 as its migration timeline. <b>EU AI Act Art. 13 transparency</b> (high-risk AI) is <b>not yet in application</b>: the AI Omnibus (Regulation (EU) 2026/1744, in force 27 Jul 2026) deferred Chapter III Sections 1&ndash;3, which contain Art. 13, to <b>2 Dec 2027</b> for Annex III systems and <b>2 Aug 2028</b> for high-risk AI embedded in regulated products. What applies from 2 Aug 2026 is general applicability, enforcement and governance, plus Art. 50 transparency for chatbots and synthetic media. Signed explanations are therefore preparation for a 2027/2028 obligation, not a present-day mandate.</p></div>
      </div>

      {/* A "Market context" table of analyst market-size forecasts stood here, with three notes on why
          the houses' figures do not reconcile. Removed 2026-10-24: this project's public material
          carries no market-size figures. */}

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
        {/* "then QKD-derived key material" until 2026-10-24: the browser's bb84() returns a count and
            never key bytes, so no QKD key material exists in the chain this sentence describes. The
            same words are in README.md and report/technical-report.tex - change all three together. */}
        So the claim this project makes is the <b>composition</b>, not either half: PQC key
        establishment, then a simulated QKD link (it yields a key length, not key bytes), then secure
        multiparty computation, then homomorphic evaluation, then the model inference - chained end to
        end in one browser tab with no server-side compute. A search for an integrated browser system
        doing all of that returned nothing. Two honest caveats travel with it: the two MPC parties are notional, running over an
        in-memory transport in a single tab (the protocol, bytes and wall-clock are real, the party
        separation is not), and the quantum channel is simulated - there is no browser API for QKD and
        there will not be one. Client-side is also not automatically private: WWW&nbsp;2026's{" "}
        <b>WAMO</b> paper argues that browser deployment moves a model into a hostile environment,
        which is why the models here are public on Hugging Face rather than treated as secrets.
      </p>

      <h2 style={{ marginTop: 28 }}>Where this architecture creates value</h2>
      <div className="grid cols-2">
        {/* Was "position bids for CNSA 2.0 / FIPS-aligned procurement that non-compliant vendors can't
            enter". The defaults are not CNSA 2.0 parameter sets and nothing here is validated, so the
            card claimed an eligibility the project does not have. */}
        <div className="card"><h3>Regulatory readiness</h3><p>Crypto-agility lets an operator select the CNSA 2.0 parameter sets ML-KEM-1024 (here inside an X25519 hybrid) and ML-DSA-87, and XAI audit trails prepare for explanation duties. The defaults are not CNSA 2.0 (ML-KEM-768, ML-DSA-65, a SHA-256 combiner), and nothing here is FIPS 140-3- or NIAP-validated.</p></div>
        <div className="card"><h3>Loss avoidance</h3><p>QML-IDS could reduce the breach probability for long-retention data, and the PQC hybrid key exchange addresses HNDL exposure of session keys; the QKD layer is simulated here.</p></div>
        <div className="card"><h3>New privacy-preserving services</h3><p>Cross-bank AML PSI, federated medical models, confidential genomics, and free-space <b>satellite QKD</b> links (simulated here; this demo finds that QSDC cannot close over a LEO downlink) — services over data that previously couldn't be combined or trusted.</p></div>
        <div className="card"><h3>Operational savings</h3><p><b>OptiQKD</b> (AI-tuned decoy/basis parameters, with a convergence metric; the reference framework reports +14&ndash;25% in simulation) automates QKD calibration and failover; a <b>QRNG</b> entropy source (min-entropy-bounded) plus a config-gated leakage-resilient PQC+QKD combiner harden the key material; one config change migrates algorithms.</p></div>
      </div>

      <div className="note" style={{ marginTop: 24 }}>
        Benefits are framed as a research/PoC investment thesis. Concrete ROI depends on
        field validation of the simulated quantum-channel results (the M3/M4 gates in the source report).
      </div>
    </div>
  );
}
