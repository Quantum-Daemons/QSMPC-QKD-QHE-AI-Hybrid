import { Link } from "react-router";

import ArchitectureDiagram from "../components/ArchitectureDiagram";
import Details from "../components/Details";
import StageBadge from "../components/StageBadge";
import { DEMO_BANNER, isClientDemo } from "../lib/demoMode";

export default function Overview() {
  // THE CIPHERTEXT-RATIO CARD THAT STOOD IN THIS ROW IS REMOVED, AND NOTHING REPLACES IT.
  //
  // It read "~1.32 Rate-1 QFHE ciphertext/plaintext ratio vs ~1483x for classical CKKS", from two
  // configuration constants, and it compared things that are not the same quantity:
  //
  //   1.32  was (bytes of a CLASSICALLY SIMULATED 2^n-amplitude state vector + a KEM-wrapped pad) /
  //         (bytes of that state vector). A quantum ciphertext has no size in bytes; the exponential
  //         denominator is a property of the simulation, and the code itself measures 1.2769 at 8
  //         qubits. The number is NOT in the Rate-1 QFHE paper it was attributed to
  //         (Chardouvelis, Doettling, Malavolta, TCC 2021, ePrint 2020/1454), which reports no
  //         concrete figure: its "rate 1" is an asymptotic limit for long messages, under quantum LWE.
  //   1483  was CKKS ciphertext bytes / (about 34 float64 inputs) - slot UTILISATION. The same
  //         N = 8192 ciphertext is ~12x against 4,096 packed values (Cheon, Kim, Kim, Song,
  //         ASIACRYPT 2017).
  //
  // And classical FHE reaches rate 1 too (Gentry-Halevi, TCC 2019; Brakerski-Doettling-Garg-Malavolta,
  // TCC 2019) - rate-1 QFHE is built on the latter - so a gap between these two numbers is not a
  // quantum advantage. A user's independent check flagged the card on 2026-10-06; it was removed on
  // 2026-10-23. `tests/test_the_retired_comparisons_stay_retired.py` keeps it from coming back.

  return (
    <div>
      {/* The banner states the two facts that make this project unusual - real CKKS and real
          garbled-circuit MPC execute in the visitor's own tab, and no computation reaches a server -
          and it rendered on every page EXCEPT the one visitors land on. */}
      {isClientDemo() && <div className="note" style={{ marginBottom: 14 }}>{DEMO_BANNER}</div>}
      <h1 className="page-title">Quantum-Safe Hybrid Orchestration</h1>
      <p className="page-sub">
        An end-to-end security layer that chains <b className="hl-c">QKD</b> →{" "}
        <b className="hl-c">QSMPC</b> → <b className="hl-q">QHE/FHE</b> →{" "}
        <b>privacy-preserving AI</b> on a <b>PQC-first hybrid</b> foundation —{" "}
        {/* Was "validated for finance, healthcare, genomics and telecom/defence". Nothing here is
            validated in any of those settings - the Research PoC note on this page says the
            quantum-channel results are "simulated, not hardware-validated" - and the list left out
            the fifth use case. */}
        demonstrated on finance, healthcare, genomics, telecom/defence and satellite use cases,
        with the quantum channel simulated.
      </p>

      <div className="grid cols-4">
        <div className="card"><div className="kpi c">4</div><p>workstreams: AI-QKD, QML-IDS, QSMPC×QKD, QHE×AI</p></div>
        {/* QUALIFIED, because the unqualified card claimed more than any run here achieves. The
            quantum one-time pad is information-theoretically hiding AS A SCHEME, given a uniform
            secret pad (Ambainis, Mosca, Tapp, de Wolf, FOCS 2000). In this demo the pad is drawn from
            a seeded, non-cryptographic generator, and on the Python path it is wrapped by
            ML-KEM-768 + X25519, which makes the composite computational (MLWE, FIPS 203). */}
        <div className="card"><div className="kpi q">2</div><p>security models compared: computational (RLWE, MLWE) vs information-theoretic <i>as a scheme</i> (quantum one-time pad). This demo seeds the pad, so no run here is information-theoretically secure.</p></div>
        {/* Both are a TARGET and a TIMELINE, not commitments. Cloudflare (blog "Cloudflare targets 2029 for full
            post-quantum security", 2026-04-07): "We now target 2029 to be fully post-quantum (PQ)
            secure including, crucially, post-quantum authentication", with intermediate milestones
            "subject to change". Google (2026-03-25): "We're setting a timeline for post-quantum
            cryptography migration to 2029". CNSA 2.0 binds national security systems only, and what
            CNSSP 15 requires from 2027 is CNSA 2.0 algorithms in new acquisitions (NSA CNSA 2.0 FAQ
            Ver. 2.2, Oct 2026) - "quantum-resistant" was looser than the rule. */}
        <div className="card"><div className="kpi c">2029</div><p>Cloudflare target for full PQC including authentication; Google has set 2029 as its PQC migration timeline. For national security systems, CNSA 2.0 algorithms are required in new acquisitions from 2027</p></div>
        {/* The novelty, stated where a first-time visitor reads it. Everything else on this page is
            context that other projects share; this is the part that is unusual, and it was legible
            only from the banner - which did not render here. */}
        <div className="card"><div className="kpi c">0</div><p><b>server-side compute</b> in this
        demo: CKKS, garbled-circuit MPC and the model inference all run in your tab - the two MPC
        parties are notional, both in this one tab. See{" "}
        <Link to="/value">what is and is not new</Link>.</p></div>
      </div>

      {/* Moved up from the bottom of the page in Round 58: the limits of what a visitor is about to read
          belong before it, not after the architecture and workstreams. Text unchanged. */}
      <div className="note" style={{ marginTop: 16 }}>
        {/* This note used to say QKD was an add-on "per BSI" and that "the hybrid framing here is
            BSI's". BSI's hybrid (TR-02102-1 v2026-01 section 2.2) is PQC + CLASSICAL key agreement and
            does not mention QKD; section 2.1 does not recommend QKD at all. So the QKD add-on is this
            project's choice alone, and only the ML-KEM + X25519 layer follows BSI. */}
        <b>Research PoC.</b> Quantum-channel results (QKD distance and QBER, QSDC) are <b>simulated</b>,
        not hardware-validated; photonic inference and full qOLE are not implemented here. QKD appears only as an optional add-on
        to the PQC hybrid, never standalone, and that is this project's choice, not a recommendation:{" "}
        <b>BSI TR-02102-1 v2026-01</b> states that QKD "is currently not ready for use from a security
        point of view" and does not recommend it (section 2.1). What BSI recommends is hybrid key
        agreement combining a quantum-safe and a classical mechanism (section 2.2), which the ML-KEM +
        X25519 layer implements. NSA answers "No" to using QKD to protect a national security system.
      </div>

      <h2 style={{ marginTop: 32 }}>Architecture at a glance</h2>
      {/* Replaces a row of text pills. They drew "QRNG entropy" as its own stage (it runs inside the
          PQC handshake), coloured the CKKS stage with the warning style used for scaffolds (CKKS is
          real), and put QKD on the key path, where nothing it produces goes in the browser. */}
      <div className="card"><ArchitectureDiagram /></div>

      <h2 style={{ marginTop: 32 }}>The four workstreams</h2>
      <div className="card">
        <table className="facts-table small">
          <thead><tr><th>Workstream</th><th>What it covers</th><th>In this demo</th></tr></thead>
          <tbody>
            <tr><td>WS1 · AI-QKD</td>
              <td>BB84 two-intensity decoy-state and E91/CHSH key agreement; OptiQKD tunes decoy and basis parameters; a QRNG with SP 800-90B health tests</td>
              <td><StageBadge kind="simulated" /> channel; the optimiser and the health tests really run. OptiQKD's reference framework reports +14–25% in simulation (arXiv:2603.04192v2)</td></tr>
            <tr><td>WS2 · QML-IDS</td>
              <td>Seven QKD attack types plus normal traffic - eight classes; classical RF baseline against a PennyLane variational quantum classifier</td>
              <td>Synthetic telemetry. The page runs a softmax and the random forest published as telecom_ids.qml; the quantum-kernel SVC and the QNN are offline figures, not run here, and the SVC&apos;s fidelity kernel equals a classical product kernel</td></tr>
            <tr><td>WS3 · QSMPC × QKD</td>
              <td>Shamir + d-level QFT secure summation, quantum oblivious transfer, quantum-safe private set intersection</td>
              <td><StageBadge kind="notional" /> classical PSI and additive-share secure sum, all parties in this tab. The PSI is an HMAC under a session key both banks hold, so over a small ID space either bank could recover the other&apos;s list; a production PSI would use an OPRF-based protocol. The QFT and QOT forms run in the full deployment, simulated</td></tr>
            <tr><td>WS4 · QHE × AI</td>
              <td>Quantum one-time pad, a QCNN classifier, Rate-1 QFHE transport; classical CKKS for comparison</td>
              <td><StageBadge kind="real" /> node-seal CKKS; an untrained QOTP + Clifford+T comparison circuit <StageBadge kind="seeded" /> that predicts nothing; on finance, medical and telecom an optional trained QCNN that runs unencrypted. Neither circuit is a private inference. TenSEAL / OpenFHE in the full deployment; HEonGPU is a build-gated adapter with no measured run</td></tr>
          </tbody>
        </table>
        <Details summary="The four workstreams in full">
          <div className="grid cols-2">
          <div className="card">
            <h3>WS1 · AI-QKD — parameter optimization + entropy</h3>
            {/* arXiv:2603.04192 v1 (OptiQKD) reported a median SKR gain of +20-30%; v2 (AML-QKD,
                2026-06) revised that down to +14-25%. Quoting ">20%" targets a figure the authors
                themselves superseded. Both versions are simulation-only. */}
            <p>BB84 <b>two-intensity decoy-state</b> (+ a selectable <b>E91/CHSH</b> protocol) with QBER/SKR telemetry; <b>OptiQKD</b> (gradient-free coordinate ascent, over the analytic decoy-state model in this browser and over a scikit-learn surrogate in the full deployment) tunes decoy/basis params to raise the secret-key rate, with a convergence metric. The reference framework reports a median gain of +14–25% over static operation in simulation (arXiv:2603.04192v2), and a 2026 survey reports +15–20% for deep-RL controllers (arXiv:2608.08280). A run here reports its gain over this demo's default parameters, which is not comparable with either. A <b>QRNG</b> entropy source (state-vector measurement + SP 800-90B min-entropy and section 4.4 health tests + Leftover-Hash-Lemma extractor) is <b>measured and reported alongside</b> the key material. Eavesdropper detection via QBER inflation.</p>
          </div>
          <div className="card">
            <h3>WS2 · QML-IDS — intrusion detection</h3>
            {/* The dataset is 8 classes = 7 attack types + normal traffic, per Al-Kuwari et al.
                (IET Quantum Communication 7(1), e70028, 2026). "Eight attack scenarios" miscounts it. */}
            <p>Seven QKD attack types plus normal traffic - eight classes in total (intercept-resend, PNS, Trojan, detector blinding and others). Classical RF baseline + PennyLane variational quantum classifier, shown as an architecture comparison rather than a performance claim. The browser runs a softmax and the published random forest; the quantum-kernel SVC and the QNN figures come from the full deployment's offline training, and the SVC&apos;s fidelity kernel equals the classical product kernel prod cos^2.</p>
          </div>
          <div className="card">
            <h3>WS3 · QSMPC × QKD</h3>
            <p>Shamir + d-level QFT secure summation (faithful to the paper example), quantum oblivious transfer, and quantum-safe private set intersection.</p>
          </div>
          <div className="card">
            <h3>WS4 · QHE × AI</h3>
            <p>Quantum one-time pad (perfectly hiding as a scheme; the demo applies the mask with a seeded key, so its structure and cost are measured and its secrecy is not). Two separate circuits, and neither is a private inference: an untrained QOTP + Clifford+T comparison circuit whose pad is never undone, so its output follows the key rather than the input; and, on finance, medical and telecom, a trained QCNN classifier that runs unencrypted. The trained circuit is built from two-qubit blocks on disjoint pairs (a tree, for the binary heads) on 7 to 9 qubits, so it is classically simulable (Bermejo et al., PRX Quantum 7, 020304, 2026), and logistic regression beats it on every use case. Rate-1 QFHE is implemented as transport only: a QOTP under a KEM, with no homomorphic evaluation. Classical CKKS (TenSEAL CPU / OpenFHE CPU) for comparison; HEonGPU is a build-gated adapter with no measured run here.</p>
          </div>
        </div>
        </Details>
      </div>

      <h2 style={{ marginTop: 32 }}>Crypto-agility: what can be swapped</h2>
      {/* Was one paragraph listing the PQC+QKD combiner as a run option. No run control selects it:
          it is a setting of the agent panel's MCP channel, and the browser pipeline mixes no QKD key
          at all. And TenSEAL / OpenFHE are full-deployment selectors - the demo's FHE is fixed. */}
      <div className="card">
        <table className="facts-table small">
          <thead><tr><th>Family</th><th>Options</th><th>Where it is selected</th></tr></thead>
          <tbody>
            <tr><td>KEM</td><td>ML-KEM-768 + X25519; ML-KEM-1024 + X25519</td><td>run controls, every run page</td></tr>
            <tr><td>Signature</td><td>ML-DSA-65; ML-DSA-87</td><td>run controls</td></tr>
            <tr><td>Key source</td><td>PQC + simulated QKD stage, whose session key in the browser still comes from the PQC handshake alone (the QKD stage yields a count, no key bytes); PQC only</td><td>run controls</td></tr>
            <tr><td>QKD protocol</td><td>BB84 decoy-state; E91/CHSH - both simulated</td><td>Parameters panel</td></tr>
            <tr><td>QRNG extractor</td><td>von Neumann; Toeplitz (LHL); seedless two-source</td><td>Parameters panel</td></tr>
            <tr><td>PQC+QKD combiner</td><td>HKDF (dual-PRF); a Toeplitz + QLHL extractor seeded with SHAKE-256 of its own inputs, so computational, not information-theoretic (the QLHL needs a seed independent of the input; Giestinhas and Spiller, arXiv:2603.26907). In the browser the QKD input is a fixed 32-byte constant</td><td>the agent panel's MCP channel on the use-case pages; browser pipeline runs mix no QKD key</td></tr>
            <tr><td>FHE / MPC backend</td><td>browser: node-seal CKKS + garbled-circuit 2PC (fixed); full deployment: TenSEAL or OpenFHE, with MPyC (the MPC selector's other options are not wired, so MPyC always runs); HEonGPU is a build-gated adapter, not a selectable backend</td><td>run controls in the full deployment only</td></tr>
          </tbody>
        </table>
        <p>Selection is operator-driven; there is no in-band cipher-suite negotiation. Open the{" "}
          {/* A raw <a> here triggered a full page reload out of the SPA. */}
          <Link to="/orchestration">E2E Orchestration</Link> page to run it live.</p>
      </div>
    </div>
  );
}
