export default function Overview() {
  return (
    <div>
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
        <div className="card"><div className="kpi c">2029</div><p>target full PQC migration (Google/Cloudflare); CNSA 2.0 mandate 2027</p></div>
        <div className="card"><div className="kpi q">~1.0</div><p>Rate-1 QFHE ciphertext/plaintext ratio vs ~10³ for classical CKKS</p></div>
      </div>

      <h2 style={{ marginTop: 32 }}>The four workstreams</h2>
      <div className="grid cols-2">
        <div className="card">
          <h3>WS1 · AI-QKD — parameter optimization</h3>
          <p>BB84/decoy-state simulation with QBER/SKR telemetry; ML (RF/XGBoost, LSTM/TCN, DRL) tunes the control plane. Eavesdropper detection via QBER inflation.</p>
        </div>
        <div className="card">
          <h3>WS2 · QML-IDS — intrusion detection</h3>
          <p>Eight QKD attack scenarios (intercept-resend, PNS, Trojan, detector blinding…). Classical RF baseline + PennyLane variational quantum classifier.</p>
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
          <span className="tag deep">PQC handshake (ML-KEM + X25519)</span> →
          <span className="tag deep">QKD key agreement (BB84)</span> →
          <span className="tag deep">QSMPC compute (PSI / secure sum)</span> →
          <span className="tag scaffold">QHE/FHE private inference</span> →
          <span className="tag">result + artifacts</span>
        </div>
        <p>Every algorithm is swappable at runtime (crypto-agility): ML-KEM ↔ hybrid,
          QKD ↔ PQC-only, TenSEAL ↔ OpenFHE. Open the{" "}
          <a href="/orchestration">E2E Orchestration</a> page to run it live.</p>
      </div>

      <div className="note" style={{ marginTop: 24 }}>
        <b>Research PoC.</b> Quantum-channel results (QKD distance/QBER gains, photonic inference,
        full qOLE) are <b>simulated</b>, not hardware-validated. QKD is presented strictly as a
        PQC-hybrid add-on, never standalone (per NSA CNSA 2.0).
      </div>
    </div>
  );
}
