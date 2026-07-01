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

## AI-QKD (WS1) + entropy (done)
- **OptiQKD** — an AI parameter optimizer (scikit-learn GBM/GP surrogate + gradient-free coordinate
  ascent over the decoy-state / efficient-BB84 model) tunes `basis_prob` / `mu_signal` / `pp_threshold`
  to **raise the measured secret-key rate >20 %** vs the default baseline on the same simulated channel
  (`src/qsh/qkd/optimize.py`; report WS1 "OptiQKD" theme, arXiv:2603.04192). Opt-in per run. ✅
- **QRNG** — a local **state-vector quantum-measurement** entropy source with an SP 800-22-style health
  suite + von Neumann / Toeplitz post-processing (`src/qsh/qrng/`); it seeds the BB84 basis/bit choices.
  Optional real-hardware source = the **ANU QRNG** API (opt-in, online). Honestly labelled: sim, not HW. ✅

## Reference implementations (2026, commercial-OK)
Grounded, actively-maintained OSS to deepen each workstream (all MIT / Apache-2.0 / BSD; adopt as the
scaffold hardens). QRNG + OptiQKD above are built from this set; the rest are adoptable next steps:

| WS | Reference implementations |
|----|---------------------------|
| WS1 AI-QKD | TNO-Quantum `communication.qkd_key_rate` (Apache) · `sequence-toolbox/SeQUeNCe` (Argonne, BSD) · QKDNetSim · `DLR-RM/stable-baselines3` (RL) |
| WS2 QML-IDS | `rdisipio/qlstm` (PennyLane QLSTM) · `ACE07-Sev/QSVM` (12-variant QSVM) · `qiskit-community/qiskit-machine-learning` |
| WS4 QHE×AI / QFL | `elucidator8918/QFL-MLNCP-NeurIPS` (QFed+FHE) · `goldenskygiang/flower-fhe` · `ibarrond/Pyfhel` · OpenFHE · Flower |
| entropy | `dorahacksglobal/quantum-randomness-generator` · NIST SP 800-22 test suite |
| cross-cutting PQC/QKD | `open-quantum-safe/oqs-provider` · `rosenpass/rosenpass` · `arnika-project/arnika` · ETSI GS QKD 014 KME (`thomasarmel/qkd_kme_server`) |

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
