# Threat model

## Adversaries
- **Harvest-Now-Decrypt-Later (HNDL)**: records ciphertext today to decrypt once a CRQC
  exists. Mitigation: PQC-first hybrid (ML-KEM+X25519) on every channel — confidential
  as long as *either* primitive holds; the X25519 half covers today, ML-KEM the quantum future.
- **Quantum adversary at Q-Day**: breaks RSA/ECC. Mitigation: ML-KEM/ML-DSA (FIPS 203/204);
  QKD adds information-theoretic key material as a *hybrid add-on*, never standalone. The usual
  citation for that is NSA CNSA 2.0, but no NSA page is retrievable (HTTP 403 to every automated
  fetch), so the quotable source is **BSI TR-02102-1 v2026-01**: QKD *"is currently not ready for
  use from a security point of view"*.
- **Honest-but-curious compute server** (delegated AI): would read inputs/model. Mitigation:
  QHE (QOTP perfect hiding) for the quantum path; CKKS for the classical path; secure
  aggregation hides per-client gradients in federated learning.
- **Malicious MPC participant**: injects bad inputs / aborts. Mitigation: identifiable-abort
  pattern (scaffold) and QOT test rounds; PSI reveals only the intersection.
- **QKD link attacker**: intercept-resend, PNS, Trojan-horse, detector-blinding, RNG
  manipulation. Mitigation: QBER monitoring + two-intensity decoy-state single-photon bounds; QRNG
  health + **SP 800-90B min-entropy** checks (arXiv:2603.26907) guard the entropy source; the config-gated
  **leakage-resilient PQC+QKD combiner** keeps the session key safe if EITHER the PQC or the QKD input is
  compromised; + QML-IDS (WS2) classifying 7 attack types against normal traffic (8 classes).
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
- **Federated averaging is not a privacy guarantee.** The pairwise-mask secure aggregation in
  `qsh/ai/federated.py` hides an *individual client's* update from the aggregator, and that is all it
  claims. It does not bound what the trained model itself leaks: gradient inversion against a shared
  model is an established attack. A formal claim would require a stated DP-SGD budget, which this
  path does not provide, so none is made.
- **Not everything the demo displays was computed.** Three things cannot be done in a browser at all
  and are labelled `SIMULATED` rather than quietly approximated: QKD has no browser API and cannot
  have one, so every QKD figure comes from an analytic channel model; Zama's own documentation states
  the TFHE-rs WASM build *"does not support FHE computations"*; and no maintained browser MPC library
  exists (JIFF's last commit was June 2024, the PSE MPC Framework is sunset). Every result carries a
  `ProvenanceBadge` with one of exactly four states — `MEASURED (local WebGPU)`, `MEASURED (local
  WASM)`, `REMOTE`, `SIMULATED` — together with the runtime evidence for that claim (`adapter.info`,
  `crossOriginIsolated`, `ort.env.wasm.numThreads`, bytes transferred vs served from cache). A result
  labelled `MEASURED` that produced no compute or network activity is treated as a defect.
- **Published metrics are what this repository measured, not what the papers report.** Where a figure
  is carried from the literature instead of reproduced — the breast-cancer PRS AUROC being the clearest
  case, since 1000 Genomes has no case/control status to compute one from — it is labelled as external.
  Where a reproduction falls short of its published baseline, the gap and its cause are printed rather
  than closed by changing the metric.
