import { Link } from "react-router-dom";

import { costPerOp } from "../lib/config";
import { DEMO_BANNER, isClientDemo } from "../lib/demoMode";

export default function Overview() {
  // Ciphertext expansion is a measured property of the two crypto paths, so it is read from the
  // config rather than written here. The page previously printed "~1.0" while the config said 1.32,
  // and README printed a third value again — one quantity, three numbers.
  const cost = costPerOp();
  const qRatio = cost.qsmpc_ct_pt_ratio;
  const cRatio = cost.demo_ct_pt_ratio;

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
        validated for finance, healthcare, genomics and telecom/defence.
      </p>

      <div className="grid cols-4">
        <div className="card"><div className="kpi c">4</div><p>workstreams: AI-QKD, QML-IDS, QSMPC×QKD, QHE×AI</p></div>
        <div className="card"><div className="kpi q">2</div><p>security models compared: computational (RLWE) vs information-theoretic</p></div>
        {/* Cloudflare (2026-04-07) is the one that committed to being "fully post-quantum secure
            including, crucially, post-quantum authentication". Google (2026-03-25) set "a timeline
            for post-quantum cryptography migration to 2029" — a migration timeline, not a
            full-PQC commitment. Attributing the stronger claim to both overstated Google. */}
        <div className="card"><div className="kpi c">2029</div><p>Cloudflare target for full PQC including authentication; Google targets PQC migration by the same year. CNSA 2.0 requires new NSS acquisitions to be quantum-resistant from 2027</p></div>
        <div className="card"><div className="kpi q">~{qRatio}</div><p>Rate-1 QFHE ciphertext/plaintext ratio vs ~{Math.round(cRatio)}× for classical CKKS</p></div>
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
          <p>Quantum one-time pad (perfect hiding), Clifford+T QCNN private inference, and Rate-1 QFHE. Classical CKKS (TenSEAL CPU / OpenFHE CPU / HEonGPU GPU) for comparison.</p>
        </div>
      </div>

      <h2 style={{ marginTop: 32 }}>Pipeline</h2>
      <div className="card">
        <div className="pill-row" style={{ fontSize: 14 }}>
          <span className="tag">data</span> →
          <span className="tag deep">QRNG entropy</span> →
          <span className="tag deep">PQC handshake (ML-KEM + X25519)</span> →
          <span className="tag deep">QKD key agreement (BB84 decoy / E91)</span> →
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
        <b>Research PoC.</b> Quantum-channel results (QKD distance/QBER gains, photonic inference,
        full qOLE) are <b>simulated</b>, not hardware-validated. QKD is presented strictly as a
        PQC-hybrid add-on, never standalone — per <b>BSI TR-02102-1 v2026-01</b>, which states that QKD
        "is currently not ready for use from a security point of view" and recommends hybrid key
        agreement. Not per NSA: NSA answers "No" to using QKD to protect a national security system
        and does not endorse a hybrid QKD add-on either, so the hybrid framing here is BSI's and this
        project's, not NSA's.
      </div>
    </div>
  );
}
