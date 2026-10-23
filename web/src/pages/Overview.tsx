import { Link } from "react-router-dom";

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
        <b>privacy-preserving AI</b> on a <b>PQC-first hybrid</b> foundation —
        {/* Was "validated for finance, healthcare, genomics and telecom/defence". Nothing here is
            validated in any of those settings - the note at the bottom of this page says the
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
        {/* Both are TARGETS, not commitments. Cloudflare (blog "Cloudflare targets 2029 for full
            post-quantum security", 2026-04-07): "We now target 2029 to be fully post-quantum (PQ)
            secure including, crucially, post-quantum authentication", with intermediate milestones
            "subject to change". Google (2026-03-25): "We're setting a timeline for post-quantum
            cryptography migration to 2029". CNSA 2.0 binds national security systems only, and what
            CNSSP 15 requires from 2027 is CNSA 2.0 algorithms in new acquisitions (NSA CNSA 2.0 FAQ
            Ver. 2.2, Oct 2026) - "quantum-resistant" was looser than the rule. */}
        <div className="card"><div className="kpi c">2029</div><p>Cloudflare target for full PQC including authentication; Google targets PQC migration by the same year. For national security systems, CNSA 2.0 algorithms are required in new acquisitions from 2027</p></div>
        {/* The novelty, stated where a first-time visitor reads it. Everything else on this page is
            context that other projects share; this is the part that is unusual, and it was legible
            only from the banner - which did not render here. */}
        <div className="card"><div className="kpi c">0</div><p><b>server-side compute</b> in this
        demo: CKKS, garbled-circuit MPC and the model inference all run in your tab. See{" "}
        <Link to="/value">what is and is not new</Link>.</p></div>
      </div>

      <h2 style={{ marginTop: 32 }}>The four workstreams</h2>
      <div className="grid cols-2">
        <div className="card">
          <h3>WS1 · AI-QKD — parameter optimization + entropy</h3>
          {/* arXiv:2603.04192 v1 (OptiQKD) reported a median SKR gain of +20-30%; v2 (AML-QKD,
              2026-06) revised that down to +14-25%. Quoting ">20%" targets a figure the authors
              themselves superseded. Both versions are simulation-only. */}
          <p>BB84 <b>two-intensity decoy-state</b> (+ a selectable <b>E91/CHSH</b> protocol) with QBER/SKR telemetry; <b>OptiQKD</b> (scikit-learn surrogate + gradient-free search) tunes decoy/basis params to raise the secret-key rate, with a convergence metric. The reference framework reports a median gain of +14–25% in simulation (arXiv:2603.04192v2). A <b>QRNG</b> entropy source (state-vector measurement + SP 800-90B min-entropy and section 4.4 health tests + Leftover-Hash-Lemma extractor) is <b>measured and reported alongside</b> the key material. Eavesdropper detection via QBER inflation.</p>
        </div>
        <div className="card">
          <h3>WS2 · QML-IDS — intrusion detection</h3>
          {/* The dataset is 8 classes = 7 attack types + normal traffic, per Al-Kuwari et al.
              (IET Quantum Communication 7(1), e70028, 2026). "Eight attack scenarios" miscounts it. */}
          <p>Seven QKD attack types plus normal traffic — eight classes in total (intercept-resend, PNS, Trojan, detector blinding…). Classical RF baseline + PennyLane variational quantum classifier, shown as an architecture comparison rather than a performance claim.</p>
        </div>
        <div className="card">
          <h3>WS3 · QSMPC × QKD</h3>
          <p>Shamir + d-level QFT secure summation (faithful to the paper example), quantum oblivious transfer, and quantum-safe private set intersection.</p>
        </div>
        <div className="card">
          <h3>WS4 · QHE × AI</h3>
          <p>Quantum one-time pad (perfectly hiding as a scheme; the demo applies the mask with a seeded key, so its structure and cost are measured and its secrecy is not), Clifford+T QCNN private inference, and Rate-1 QFHE. Classical CKKS (TenSEAL CPU / OpenFHE CPU / HEonGPU GPU) for comparison.</p>
        </div>
      </div>

      <h2 style={{ marginTop: 32 }}>Pipeline</h2>
      <div className="card">
        <div className="pill-row" style={{ fontSize: 14 }}>
          <span className="tag">data</span> →
          <span className="tag deep">QRNG entropy</span> →
          <span className="tag deep">PQC handshake (ML-KEM + X25519)</span> →
          <span className="tag deep">QKD key agreement (BB84 decoy / E91, simulated)</span> →
          <span className="tag deep">QSMPC compute (PSI / secure sum)</span> →
          <span className="tag scaffold">QHE/FHE private inference</span> →
          <span className="tag">result + artifacts</span>
        </div>
        <p>Every algorithm is selectable at configuration time (crypto-agility): ML-KEM ↔ hybrid,
          QKD ↔ PQC-only (BB84 ↔ E91), TenSEAL ↔ OpenFHE, and the PQC+QKD key combiner
          (HKDF ↔ information-theoretic combiner). Selection is operator-driven; there is no
          in-band cipher-suite negotiation. Open the{" "}
          {/* A raw <a> here triggered a full page reload out of the SPA. */}
          <Link to="/orchestration">E2E Orchestration</Link> page to run it live.</p>
      </div>

      <div className="note" style={{ marginTop: 24 }}>
        {/* This note used to say QKD was an add-on "per BSI" and that "the hybrid framing here is
            BSI's". BSI's hybrid (TR-02102-1 v2026-01 section 2.2) is PQC + CLASSICAL key agreement and
            does not mention QKD; section 2.1 does not recommend QKD at all. So the QKD add-on is this
            project's choice alone, and only the ML-KEM + X25519 layer follows BSI. */}
        <b>Research PoC.</b> Quantum-channel results (QKD distance/QBER gains, photonic inference,
        full qOLE) are <b>simulated</b>, not hardware-validated. QKD appears only as an optional add-on
        to the PQC hybrid, never standalone, and that is this project's choice, not a recommendation:
        <b>BSI TR-02102-1 v2026-01</b> states that QKD "is currently not ready for use from a security
        point of view" and does not recommend it (section 2.1). What BSI recommends is hybrid key
        agreement combining a quantum-safe and a classical mechanism (section 2.2), which the ML-KEM +
        X25519 layer implements. NSA answers "No" to using QKD to protect a national security system.
      </div>
    </div>
  );
}
