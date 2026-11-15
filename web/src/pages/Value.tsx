import Details from "../components/Details";
import KeyPoints from "../components/KeyPoints";
import RegulatoryClock from "../components/RegulatoryClock";
import { DEMO_BANNER, isClientDemo } from "../lib/demoMode";

// LAYOUT, Round 58: the novelty sentence first, the evidence for it second, the context after.
//
// The sentence sat about 1,300 px down at 1280x800, below a 191-word regulation paragraph, although
// this page exists to say it. It now follows the page heading, VERBATIM (README.md carries the same
// QKD wording and a gate holds the two together; so does the technical report, kept outside this
// repository since Round 86, whose own tests hold its copy), and it is set as body text rather than in the small muted style it had. The regulation paragraph became the
// "regulatory clock" table below, where every date keeps its scope and firmness.
export default function Value() {
  return (
    <div>
      <h1 className="page-title">Introduction & Value</h1>
      <p className="page-sub">
        Why a quantum-safe orchestration layer now — the regulatory clock, and where this
        architecture creates value for regulated industries.
      </p>

      {isClientDemo() && <div className="note" style={{ marginBottom: 14 }}>{DEMO_BANNER}</div>}
      <h2>What is new here, and what is not</h2>
      <KeyPoints items={[
        <><b>Not new:</b> secure multiparty computation in a browser (webSPDZ, JIFF, MPyC-Web).</>,
        <><b>Not new either:</b> homomorphic <i>evaluation</i> in a browser - node-seal, the SEAL build this
          demo uses, published benchmarks of CKKS evaluation in three browsers (its BENCHMARK.md, at
          the 7.0.0 tag; the file left its main branch on 2026-10-31), and she-wasm (AsiaCCS 2018)
          evaluates two-level HE in WebAssembly. <b>Not found:</b> a peer-reviewed 2025-2026 report of encrypted
          neural-network inference inside a browser tab.</>,
      ]} />
      <p className="lead">
        {/* "then QKD-derived key material" until 2026-10-24: the browser's bb84() returns a count and
            never key bytes, so no QKD key material exists in the chain this sentence describes. The
            same claim is in README.md and in the technical report (kept outside this repository
            since Round 86), each in its own words - change all three together.
            "with no server-side compute" is a fact of the client-only build the public demo ships,
            so the sentence names that build: a full deployment sends every run to its server
            (POST /api/run in `api.ts`), and this page renders there too. */}
        So the claim this project makes is the <b>composition</b>, not either half: PQC key
        establishment, then a simulated QKD link (it yields a key length, not key bytes), then secure
        multiparty computation, then homomorphic evaluation, then the model inference - chained end to
        end in one browser tab with no server-side compute, in the client-only build the public demo
        ships{isClientDemo() ? "" : " (this full deployment runs its pipeline on its server instead)"}.
        A search for an integrated browser system doing all of that returned nothing. Two honest caveats travel with it: the two MPC parties are notional, running over an
        in-memory transport in a single tab (the protocol, bytes and wall-clock are real, the party
        separation is not), and the quantum channel is simulated - there is no browser API for QKD and
        there will not be one. Client-side is also not automatically private: the <b>WAMO</b> paper (Wang et al.,
        The Web Conference 2026,{" "}
        <a href="https://doi.org/10.1145/3774904.3792297" target="_blank" rel="noopener noreferrer">doi:10.1145/3774904.3792297</a>)
        argues that a model deployed in the browser exposes its code and parameters to an untrusted environment,
        which is why the models here are public on Hugging Face rather than treated as secrets.
      </p>
      <div className="card" style={{ marginTop: 12 }}>
        <h3>Prior art, checked</h3>
        <table className="facts-table small">
          <thead><tr><th>Capability</th><th>What the search found</th><th>Verdict</th><th>In this demo</th></tr></thead>
          <tbody>
            <tr><td>MPC in a browser</td><td>webSPDZ (IACR ePrint 2025/487) compiles MP-SPDZ to WASM over WebRTC/WebSockets and names JIFF and MPyC-Web as earlier engines</td><td>not new</td><td>a garbled-circuit 2PC (Polytune) re-implements a known idea</td></tr>
            <tr><td>FHE encrypt / decrypt in a browser</td><td>IACR ePrint 2026/515 treats client-side operations as encryption and decryption; Zama's TFHE-rs WASM API supports key generation, encryption and decryption</td><td>common</td><td>-</td></tr>
            <tr><td>HE <i>evaluation</i> in a browser</td><td>node-seal (Microsoft SEAL in WebAssembly) benchmarks CKKS evaluation operations such as rotation in Chrome, Firefox and Safari; she-wasm (Attrapadung et al., AsiaCCS 2018) evaluates two-level HE in WebAssembly. TFHE-rs WASM documents that it does <b>not</b> support FHE computation; it gives no reason, and separately notes slow key generation without WASM threads and keys over WASM&apos;s 2&nbsp;GB limit</td><td>not new</td><td>node-seal evaluates a real CKKS dot product and convolution in the tab</td></tr>
            <tr><td>Encrypted neural-network inference in a browser tab</td><td>no peer-reviewed 2025-2026 report found; an absence of evidence from a search that was not exhaustive</td><td>not found</td><td>on the medical page, the conv and dense layers of a small CNN student under CKKS, decrypted between layers, with ReLU and pooling in the clear; and the same student on one held-out image per run with its ReLUs and argmax in garbled circuits, so neither notional party holds an activation in the clear</td></tr>
          </tbody>
        </table>
        <Details summary="How the prior-art search was run">
          <p>
            This project's own verification rule is that a novelty claim has to survive an adversarial
            search for prior art. Applied to its own headline claim, the answer is that neither half is new
            on its own, which is why the claim is the composition.
          </p>
          <p>
            Secure multiparty computation has been compiled to WebAssembly before. <b>webSPDZ</b> (IACR
            ePrint 2025/487) compiles MP-SPDZ to WASM and communicates over WebRTC or WebSockets, and it
            names <b>JIFF</b> and <b>MPyC-Web</b> as earlier browser MPC engines. The garbled-circuit half
            of this demo is a re-implementation of a known idea, not a first.
          </p>
          <p>
            Homomorphic <b>evaluation</b> in a browser has been done before too. node-seal, the
            WebAssembly build of Microsoft SEAL this demo runs, publishes benchmarks of CKKS evaluation
            operations such as rotation in Chrome, Firefox and Safari, and she-wasm (Attrapadung et al.,
            AsiaCCS 2018) evaluates two-level homomorphic encryption in WebAssembly. Much of the
            literature still treats the browser as the encrypting end only - IACR ePrint 2026/515 defines
            "client-side operations" as encryption and decryption while users offload large computations
            to servers, and Zama's TFHE-rs WASM API supports key generation, encryption and decryption but
            not FHE computation. What the search did not find is a peer-reviewed 2025-2026 report of
            encrypted neural-network inference inside a browser tab; that is an absence of evidence, not
            a proof.
          </p>
        </Details>
      </div>

      <h2 style={{ marginTop: 28 }}>The driver: resource estimates are falling and migration targets carry dates</h2>
      <div className="grid cols-3">
        {/* A RISK, which is what the primary sources call it; "HNDL is active" asserted an observed
            campaign that no primary source documents. EO 14412 sec. 1 (signed 2026-06-22, read at
            whitehouse.gov on 2026-10-31): ongoing cyber activity "presents the risk of adversaries
            collecting United States information now, and decrypting it later once large-scale
            quantum computers are operational". */}
        <div className="card"><h3>HNDL is a present risk</h3><p>"Harvest Now, Decrypt Later": data with 30-year secrecy needs is already at risk. A PQC hybrid key exchange (ML-KEM + X25519) addresses it today; QKD is a simulated add-on in this demo.</p></div>
        {/* "roughly 20x" read the abstract's upper bound ("less than a week") as the estimate; the
            paper's own figure is 4.96 days per factoring (section 3.2), against the 2019 estimate's 8
            hours. The two 2026 preprints (both 2026-03-30; abstracts read 2026-11-06) are resource
            estimates, like Gidney's, and none of the three is a demonstration. */}
        <div className="card"><h3>Resource estimates falling</h3><p>RSA-2048 break estimated under 1M noisy qubits (Gidney, arXiv:2505.15917, 2025) &mdash; about 1/20 the qubits of the 2019 estimate, though the runtime rises from 8 hours to about 5 days (&asymp;15&times;) in exchange. Two 2026 preprints lower the counts again: about 10,000 reconfigurable atomic qubits for Shor at cryptographic scale, and a few days for P-256 with 26,000 (Cain et al., arXiv:2603.28627), and a 256-bit elliptic-curve discrete logarithm in minutes on fewer than 500,000 superconducting qubits (Babbush et al., arXiv:2603.28846). All three are estimates, not demonstrations.</p></div>
        <div className="card"><h3>Regulation sets the clock</h3>
          <KeyPoints items={[
            "NIST FIPS 203/204/205 are final (August 2024).",
            "US: EO 14412, through OMB M-26-15, binds federal high value assets and high impact systems (not national security systems); CNSA 2.0 binds national security systems only.",
            "EU: AI Act high-risk transparency is deferred to 2027/2028; Art. 50 applies from 2 Aug 2026, with a grace period to 2 Dec 2026 for Art. 50(2) systems already on the market.",
          ]} />
          <p><a href="#clock">Every date, with its scope</a></p>
        </div>
      </div>

      <h2 id="clock" style={{ marginTop: 28 }}>The regulatory clock</h2>
      <div className="card">
        <RegulatoryClock />
        {/* The GDPR half was missing. "Not a present-day mandate" read as if no explanation duty
            applied yet, and one does: the CJEU held in C-203/22 Dun & Bradstreet Austria (judgment
            of 27 February 2025; press release No 22/25, read at curia.europa.eu on 2026-10-31) that
            under Art. 15(1)(h) GDPR the person subject to an automated decision is entitled to an
            explanation of the procedure and principles actually applied. What is not mandated today,
            by either instrument, is the SIGNATURE. */}
        <p style={{ marginTop: 10 }}>
          Explanations of automated decisions are already owed today: under the GDPR, the person
          concerned by a solely automated decision with legal or similarly significant effects
          (Art. 22(1)) is entitled to an explanation of the procedure and principles actually applied
          (Art. 15(1)(h) with Art. 22, as the CJEU held in C-203/22,
          Dun &amp; Bradstreet Austria, 27 February 2025). The AI Act&apos;s high-risk transparency
          duties follow in 2027/2028. Neither requires the explanation to be signed, so the signature
          here is this project&apos;s addition, not a mandate.
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
            <tr><td>Regulatory readiness</td><td>Crypto-agility lets an operator select ML-KEM-1024 and ML-DSA-87, the CNSA 2.0 algorithm sizes; XAI audit trails prepare for explanation duties</td><td>No selectable configuration is CNSA 2.0: ML-KEM-1024 always runs inside an X25519 hybrid, the combiner is always HKDF-SHA256, and the defaults are ML-KEM-768 and ML-DSA-65. Nothing here is FIPS 140-3- or NIAP-validated</td></tr>
            {/* THE LIMITS NAME WHAT THE CODE DOES (Round 84). "QML-IDS could reduce the breach
                probability" credited a quantum model with the browser's classical classifier (the
                qml_ids stage runs a softmax, plus the published random forest; telecom_ids.qml
                records quantum_beats_classical: false, RF macro F1 0.987 against QNN 0.467), and no
                IDS on QKD-link telemetry sees the passive recording HNDL is. The services row said
                "simulated" and stopped, leaving out the two keys that undo its privacy claims here:
                the HMAC key the PSI then used, which both banks held, and the session key every
                federated mask derived from, which the aggregator holds (docs/threat-model.md). Since
                Round 84 the PSI is RFC 9497's OPRF and each pair of clients agrees its mask seed by
                X25519, and the row says what each leaves out. */}
            <tr><td>Loss avoidance</td><td>The PQC hybrid key exchange addresses HNDL exposure of session keys; an attack classifier on QKD-link telemetry could flag attacks on the link itself (here it runs on synthetic telemetry and never reads the simulated link)</td><td>The classifier is classical: the page runs a softmax and the published random forest, both on synthetic QKD-attack telemetry, and the quantum-kernel SVC and the QNN in the recorded comparison, run offline, lose to the forest. It does not address HNDL, which is passive recording of classical traffic. The QKD layer is simulated here</td></tr>
            <tr><td>New privacy-preserving services</td><td>Cross-bank AML PSI, federated medical models, confidential genomics, free-space <b>satellite QKD</b> links</td><td>The PSI is semi-honest and not post-quantum: it runs RFC 9497&apos;s OPRF over ristretto255, so a bank that deviates from the protocol is not caught, and Bank A, with a quantum computer, could recover Bank B&apos;s list from the messages it receives. Each pair of federated clients agrees its mask seed by X25519, so an aggregator that follows the protocol cannot remove a mask alone, though every client runs in this tab and there is no dropout recovery or authentication of the public keys. Those seeds are not post-quantum either: the aggregator relays the public keys and the masked updates are not sealed, so with a quantum computer it, or anyone who recorded them, could recompute every seed and unmask each update. QKD and the satellite link are simulated, and with its default link budget, under a criterion that credits the eavesdropper with all of the channel&apos;s loss, this demo&apos;s model finds that QSDC cannot close at any range over a LEO downlink</td></tr>
            <tr><td>Operational savings</td><td><b>OptiQKD</b> (gradient-free coordinate ascent over decoy and basis parameters, with a convergence metric) points at automated QKD calibration; one config change migrates algorithms</td><td>The reference framework reports +14-25% in simulation; nothing here is field-validated, and no failover is implemented. In the browser the <b>QRNG</b>&apos;s bits feed only its own metrics, and the PQC+QKD combiner is seeded with SHAKE-256 of its own inputs and fed a fixed QKD-domain constant, so neither hardens a key</td></tr>
          </tbody>
        </table>
      </div>

      <div className="note" style={{ marginTop: 24 }}>
        Benefits are framed as a research/PoC investment thesis. Concrete ROI depends on
        field validation of the simulated quantum-channel results.
      </div>
    </div>
  );
}
