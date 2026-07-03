# Threat model

## Adversaries
- **Harvest-Now-Decrypt-Later (HNDL)**: records ciphertext today to decrypt once a CRQC
  exists. Mitigation: PQC-first hybrid (ML-KEM+X25519) on every channel — confidential
  as long as *either* primitive holds; the X25519 half covers today, ML-KEM the quantum future.
- **Quantum adversary at Q-Day**: breaks RSA/ECC. Mitigation: ML-KEM/ML-DSA (FIPS 203/204);
  QKD adds information-theoretic key material as a *hybrid add-on* (never standalone, per CNSA 2.0).
- **Honest-but-curious compute server** (delegated AI): would read inputs/model. Mitigation:
  QHE (QOTP perfect hiding) for the quantum path; CKKS for the classical path; secure
  aggregation hides per-client gradients in federated learning.
- **Malicious MPC participant**: injects bad inputs / aborts. Mitigation: identifiable-abort
  pattern (scaffold) and QOT test rounds; PSI reveals only the intersection.
- **QKD link attacker**: intercept-resend, PNS, Trojan-horse, detector-blinding, RNG
  manipulation. Mitigation: QBER monitoring + two-intensity decoy-state single-photon bounds; QRNG
  health + **SP 800-90B min-entropy** checks (arXiv:2603.26907) guard the entropy source; the config-gated
  **"secure-if-either" PQC+QKD combiner** keeps the session key safe if EITHER the PQC or the QKD input is
  compromised; + QML-IDS (WS2) classifying 8 attack signatures.
- **Network MITM / tampering**: Mitigation: AEAD (AES-256-GCM) record layer + ML-DSA
  message authentication over the hybrid session.

## Trust assumptions
- Endpoints (clients) are honest and hold their own secret keys.
- The QKD hardware is modelled by simulation; real deployments must address device
  imperfections (the WS2 IDS targets exactly these).
- The crypto/compute libraries are trusted as implemented (pin versions + review): server-side
  TenSEAL / OpenFHE / HEonGPU / liboqs / MPyC; client-side **node-seal** (CKKS) + **emp-wasm /
  mpc-framework** (garbled-circuit 2PC) for the in-browser demo; the **MCP Python SDK** (FastMCP) for
  the full-deployment MCP server.
- **MCP tool dispatch** (full deployment): the quantum-safe channel authenticates + seals the
  `tools/call` and its result (ML-DSA + AES-GCM), but the tool *logic* runs server-side and is trusted;
  the official filesystem MCP server is sandboxed to a use-case data dir. The public demo runs no
  server-side MCP (client mirror only), so it adds no server trust surface.

## Explicitly out of scope (PoC)
- Side-channel / fault attacks on the host.
- Formal verification of the protocols (referenced papers carry the proofs).
- Production key management / HSM integration (crypto-agility hooks are provided).

## Residual risk & honest limitations
- Quantum-channel results (QKD distance/QBER gains, photonic inference, full qOLE) are
  **simulated**, not hardware-validated.
- The QHE homomorphic evaluation is exact for Clifford circuits with key tracking;
  non-Clifford (trainable) layers require magic-state/T-gate corrections in a real device
  (their cost is reported as a T-count, not executed fault-tolerantly here).
- PSI is semi-honest keyed-hash over the quantum-safe channel; a malicious-secure OPRF
  PSI is future work.
