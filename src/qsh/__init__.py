"""qsh — Quantum-Safe Hybrid.

An end-to-end orchestration security layer that chains
QKD -> QSMPC -> QHE/FHE -> privacy-preserving AI on a PQC-first hybrid foundation.

Subpackages:
    orchestration  E2E pipeline engine (DAG of crypto stages, crypto-agility, events)
    pqc            post-quantum crypto (ML-KEM/ML-DSA hybrid transport)
    qkd            quantum key distribution (BB84/decoy-state sim) + parameter optimizer (WS1)
    qml_ids        quantum-ML intrusion detection (WS2)
    qsmpc          quantum secure multi-party computation (Shamir+QFT secure sum; a QOT sketch; OPRF PSI) (WS3)
    qhe            quantum homomorphic encryption + AI (QOTP, QCNN, Rate-1 QFHE) (WS4)
    fhe            classical/GPU homomorphic encryption backends
    ai             models + federated learning
    usecases       finance / medical / genomics / telecom-defense / satellite demos
    benchmark      classical MPC-FHE vs quantum QSMPC-FHE
    data           dataset loaders + synthetic generation
    monitoring     metrics export
"""

#: The release version, written here once. pyproject.toml reads it (a dynamic version), the API reports
#: it, and tests/test_r86_the_version_has_one_source.py holds web/package.json and its lockfile to it.
#: The minor is the verification round the release closes: 0.85.0 is the first release, of Round 85.
__version__ = "0.85.0"
__all__ = ["__version__"]
