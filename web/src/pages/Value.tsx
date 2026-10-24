import Details from "../components/Details";
import KeyPoints from "../components/KeyPoints";
import RegulatoryClock from "../components/RegulatoryClock";

// LAYOUT, Round 58: the novelty sentence first, the evidence for it second, the context after.
//
// The sentence sat about 1,300 px down at 1280x800, below a 191-word regulation paragraph, although
// this page exists to say it. It now follows the page heading, VERBATIM (README.md and
// report/technical-report.tex carry the same QKD wording, and a gate holds the three together), and it
// is set as body text rather than in the small muted style it had. The regulation paragraph became the
// "regulatory clock" table below, where every date keeps its scope and firmness.
export default function Value() {
  return (
    <div>
      <h1 className="page-title">Introduction & Value</h1>
      <p className="page-sub">
        Why a quantum-safe orchestration layer now — the regulatory clock, and where this
        architecture creates value for regulated industries.
      </p>

      <h2>What is new here, and what is not</h2>
      <KeyPoints items={[
        <><b>Not new:</b> secure multiparty computation in a browser (webSPDZ, JIFF, MPyC-Web).</>,
        <><b>Unusual:</b> homomorphic <i>evaluation</i> in a browser - every artefact found encrypts in the
          browser and evaluates on a server.</>,
      ]} />
      <p className="lead">
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
      <div className="card" style={{ marginTop: 12 }}>
        <h3>Prior art, checked</h3>
        <table className="facts-table small">
          <thead><tr><th>Capability</th><th>What the search found</th><th>Verdict</th><th>In this demo</th></tr></thead>
          <tbody>
            <tr><td>MPC in a browser</td><td>webSPDZ (IACR ePrint 2025/487) compiles MP-SPDZ to WASM over WebRTC/WebSockets and names JIFF and MPyC-Web as earlier engines</td><td>not new</td><td>a garbled-circuit 2PC (emp-wasm) re-implements a known idea</td></tr>
            <tr><td>FHE encrypt / decrypt in a browser</td><td>IACR ePrint 2026/515 treats client-side operations as encryption and decryption; Zama's TFHE-rs WASM API supports key generation, encryption and decryption</td><td>common</td><td>-</td></tr>
            <tr><td>FHE <i>evaluation</i> in a browser</td><td>none found; TFHE-rs WASM documents that it does <b>not</b> support FHE computation, citing WASM threading and the 2&nbsp;GB memory ceiling</td><td>unusual</td><td>node-seal evaluates a real CKKS dot product and convolution in the tab</td></tr>
          </tbody>
        </table>
        <Details summary="How the prior-art search was run">
          <p>
            This project's own verification rule is that a novelty claim has to survive an adversarial
            search for prior art. Applied to its own headline claim, the honest answer is asymmetric, and
            both halves are stated here rather than only the flattering one.
          </p>
          <p>
            Secure multiparty computation has been compiled to WebAssembly before. <b>webSPDZ</b> (IACR
            ePrint 2025/487) compiles MP-SPDZ to WASM and communicates over WebRTC or WebSockets, and it
            names <b>JIFF</b> and <b>MPyC-Web</b> as earlier browser MPC engines. The garbled-circuit half
            of this demo is a re-implementation of a known idea, not a first.
          </p>
          <p>
            The search for a browser that <b>evaluates</b> a homomorphic circuit came back empty. Every
            artefact found encrypts in the browser and evaluates on a server - IACR ePrint 2026/515
            defines "client-side operations" as encryption and decryption while users offload large
            computations to servers, and Zama's own TFHE-rs WASM API documents that it supports key
            generation, encryption and decryption but not FHE computation.
          </p>
        </Details>
      </div>

      <h2 style={{ marginTop: 28 }}>The driver: Q-Day is being pulled forward</h2>
      <div className="grid cols-3">
        <div className="card"><h3>HNDL is active</h3><p>"Harvest Now, Decrypt Later": data with 30-year secrecy needs is already at risk. A PQC hybrid key exchange (ML-KEM + X25519) addresses it today; QKD is a simulated add-on in this demo.</p></div>
        <div className="card"><h3>Resource estimates collapsing</h3><p>RSA-2048 break estimated under 1M noisy qubits (Gidney, arXiv:2505.15917, 2025) &mdash; about 1/20 the qubits of the 2019 estimate, though the runtime rises roughly 20&times; in exchange.</p></div>
        <div className="card"><h3>Regulation sets the clock</h3>
          <KeyPoints items={[
            "NIST FIPS 203/204/205 are final (August 2024).",
            "US: EO 14412 binds federal agencies; CNSA 2.0 binds national security systems only.",
            "EU: AI Act high-risk transparency is deferred to 2027/2028; Art. 50 applies now.",
          ]} />
          <p><a href="#clock">Every date, with its scope</a></p>
        </div>
      </div>

      <h2 id="clock" style={{ marginTop: 28 }}>The regulatory clock</h2>
      <div className="card">
        <RegulatoryClock />
        <p style={{ marginTop: 10 }}>
          Signed explanations are therefore preparation for a 2027/2028 obligation, not a present-day
          mandate.
        </p>
      </div>

      <h2 style={{ marginTop: 28 }}>Where this architecture creates value</h2>
      {/* Four cards became one table with the caveat in its own column, so no lever is read without
          its limit. "Regulatory readiness" was "position bids for CNSA 2.0 / FIPS-aligned procurement
          that non-compliant vendors can't enter" until Round 58: the defaults are not CNSA 2.0
          parameter sets and nothing here is validated. */}
      <div className="card">
        <table className="facts-table small">
          <thead><tr><th>Lever</th><th>Mechanism</th><th>Limit</th></tr></thead>
          <tbody>
            <tr><td>Regulatory readiness</td><td>Crypto-agility lets an operator select the CNSA 2.0 parameter sets ML-KEM-1024 (here inside an X25519 hybrid) and ML-DSA-87; XAI audit trails prepare for explanation duties</td><td>The defaults are not CNSA 2.0 (ML-KEM-768, ML-DSA-65, a SHA-256 combiner), and nothing here is FIPS 140-3- or NIAP-validated</td></tr>
            <tr><td>Loss avoidance</td><td>QML-IDS could reduce the breach probability for long-retention data; the PQC hybrid key exchange addresses HNDL exposure of session keys</td><td>The QKD layer is simulated here</td></tr>
            <tr><td>New privacy-preserving services</td><td>Cross-bank AML PSI, federated medical models, confidential genomics, free-space <b>satellite QKD</b> links - services over data that previously couldn't be combined or trusted</td><td>Simulated here; this demo finds that QSDC cannot close over a LEO downlink</td></tr>
            <tr><td>Operational savings</td><td><b>OptiQKD</b> (AI-tuned decoy/basis parameters, with a convergence metric) automates QKD calibration and failover; a <b>QRNG</b> entropy source (min-entropy-bounded) plus a config-gated leakage-resilient PQC+QKD combiner harden the key material; one config change migrates algorithms</td><td>The reference framework reports +14&ndash;25% in simulation; nothing here is field-validated</td></tr>
          </tbody>
        </table>
      </div>

      <div className="note" style={{ marginTop: 24 }}>
        Benefits are framed as a research/PoC investment thesis. Concrete ROI depends on
        field validation of the simulated quantum-channel results (the M3/M4 gates in the source report).
      </div>
    </div>
  );
}
