# PQC migration & threat roadmap (recommendations)

This project is designed to slot into the 2026–2030 post-quantum migration. The items
below are the recommended future tracks (PoC milestones M3/M4).

## Standards alignment (done / in progress)
- **NIST FIPS 203 (ML-KEM)** — implemented in the hybrid KEM. ✅
- **NIST FIPS 204 (ML-DSA)** — implemented for message authentication. ✅
- **NIST FIPS 205 (SLH-DSA)** — add as a hash-based signature option (not in this liboqs build). ☐
- **NSA CNSA 2.0** — 2027 procurement mandate; QKD non-recommended standalone. The
  architecture keeps QKD strictly as a PQC-hybrid add-on. ✅

## Crypto-agility (done)
Algorithms resolve through `CryptoRegistry` at runtime — ML-KEM-768 ↔ -1024, QKD ↔
PQC-only, TenSEAL ↔ OpenFHE. A future PQC algorithm is a registry entry + config change,
not a code rewrite. Add remote re-key + in-flight algorithm substitution next. ☐

## Recommended future workstreams
1. **NIST-aligned IDS against HNDL / Q-Day** — extend WS2 (QML-IDS) into a full network
   IDS plane: combine the QKD-attack classifier with classical NIDS (e.g. CICIDS-style
   flows), map detections to MITRE ATT&CK, and alert on harvesting patterns. Wire to
   SIEM/SOAR. (Report Phase-4 of WS2.)
2. **Hardware QKD field test** — replace the BB84 simulator with a real QKD testbed (or
   ETSI GS QKD 014 KME API) to validate the simulated QBER/SKR. (Milestone M4.)
3. **Malicious-secure MPC** — upgrade PSI to a post-quantum OPRF and add identifiable
   abort with ML-DSA-signed transcripts.
4. **Fault-tolerant QHE costing** — turn the measured T-counts into resource estimates
   under a chosen QEC code; track against the ~1.1M T-gate class.
5. **GPU FHE at scale** — build the HEonGPU + PIGEON/hpmpc path (`scripts/build_gpu_mpc.sh`)
   for batched CNN/Transformer private inference; compare HEonGPU (GPU) vs OpenFHE (CPU) CKKS.
6. **Continuous PQC inventory** — scan dependencies/protocols for quantum-vulnerable
   crypto (a "crypto bill of materials") to drive migration priority.

## Timeline anchors
- **2027** CNSA 2.0 new-procurement mandate → be compliant before bids.
- **2029** Google/Cloudflare full-PQC targets → supply-chain expectations.
- **2030–2035** NIST deprecation/removal of classical PKC; plan field validation by 2027 Q1.
